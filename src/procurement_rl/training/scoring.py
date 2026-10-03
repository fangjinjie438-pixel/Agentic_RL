"""Teacher-forced gold likelihood using the existing distributed actor workers."""

import torch

from verl import DataProto
from verl.protocol import unpad_dataproto
from procurement_rl.training.batching import pad_for_workers
from procurement_rl.training.contracts import scoring_prefixes
from procurement_rl.training.generation import _pad, _positions


@torch.no_grad()
def score_information_gains(batch, actor_wg, tokenizer, config):
    """Score initial/post-observation states, before the actor optimizer update.

The neutral ``<answer>`` prefix is identical at every state. Only gold offer ID
tokens enter the mean, excluding delimiters and evidence. Gold never enters
rollout tensors or observations. Identical initial states are scored once.
    """
    import json

    records = []
    lookup = {}
    row_record_ids = []
    neutral = tokenizer.encode("\n<answer>", add_special_tokens=False)
    for i, trace in enumerate(batch.non_tensor_batch["procurement_trace"]):
        ground_truth = batch.non_tensor_batch["reward_model"][i]["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        gold = tokenizer.encode(ground_truth["gold_offer_id"], add_special_tokens=False)
        if not gold:
            raise ValueError("Gold answer has no tokens")
        states = scoring_prefixes(trace["prompt_token_ids"], trace["response_token_ids"],
                                  trace["observation_ends"])
        ids = []
        for state in states:
            prefix = state + neutral
            if len(prefix) + len(gold) > config.data.max_prompt_length:
                raise ValueError("Gold scoring exceeds full context budget")
            key = (tuple(prefix), tuple(gold))
            if key not in lookup:
                lookup[key] = len(records)
                records.append((prefix, gold))
            ids.append(lookup[key])
        row_record_ids.append(ids)
    world = actor_wg.world_size
    micro = int(config.procurement.get("gold_score_micro_batch_size", 1))
    if micro < 1:
        raise ValueError("gold_score_micro_batch_size must be at least one")
    chunk_size = world * micro
    likelihoods = []
    scored_states, scored_tokens, allocated_tokens = 0, 0, 0
    for begin in range(0, len(records), chunk_size):
        chunk = records[begin:begin + chunk_size]
        prompts, pmask = _pad([row[0] for row in chunk], tokenizer.pad_token_id, left=True)
        responses, rmask = _pad([row[1] for row in chunk], tokenizer.pad_token_id)
        attention = torch.cat((pmask, rmask), -1)
        data = DataProto.from_dict({"prompts": prompts, "responses": responses,
                                   "input_ids": torch.cat((prompts, responses), -1),
                                   "attention_mask": attention, "position_ids": _positions(attention)},
                                  meta_info={"temperature": 1.0, "micro_batch_size": micro,
                                             "use_dynamic_bsz": False,
                                             "max_token_len": config.data.max_prompt_length})
        padded, count = pad_for_workers(data, world)
        scored_states += len(padded)
        scored_tokens += int(padded.batch["attention_mask"].sum())
        allocated_tokens += padded.batch["input_ids"].numel()
        scores = unpad_dataproto(actor_wg.compute_log_prob(padded), count)
        logprobs = scores.batch["old_log_probs"].detach().cpu()
        means = (logprobs * rmask).sum(-1) / rmask.sum(-1)
        if not torch.isfinite(means).all():
            raise FloatingPointError("Nonfinite gold log probabilities")
        likelihoods.extend(means.tolist())
    gains = []
    for ids, trace in zip(row_record_ids, batch.non_tensor_batch["procurement_trace"]):
        values = [likelihoods[j] for j in ids]
        delta = [after - before for before, after in zip(values, values[1:])]
        trace["gold_log_probs"] = values
        trace["information_gains"] = delta
        gains.append(delta)
    return gains, {"ig/unique_scoring_states": len(records),
                   "ig/scored_states_with_repeats": scored_states,
                   "ig/unique_state_tokens": sum(len(a) + len(b) for a, b in records),
                   "ig/scored_tokens": scored_tokens,
                   "ig/allocated_tokens_with_padding": allocated_tokens}
