"""A provenance-preserving multi-turn rollout for bundled Search-R1.

Unlike upstream's decode/retokenize loop, sampled token IDs remain unchanged.
Environment tokens are recorded explicitly and always masked out of the loss.
No history or returned document is silently truncated.
"""

import re
from urllib.parse import urlsplit
import numpy as np
import requests
import torch

from verl import DataProto
from verl.protocol import unpad_dataproto
from procurement_rl.training.batching import pad_for_workers
from procurement_rl.training.contracts import pack_observation


def _pad(rows, pad_id, *, left=False):
    width = max(len(row) for row in rows)
    tensor = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(tensor)
    for i, row in enumerate(rows):
        start = width - len(row) if left else 0
        tensor[i, start:start + len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, start:start + len(row)] = 1
    return tensor, mask


def _positions(mask):
    return (mask.cumsum(-1) - 1).clamp(min=0) * mask


def _generated_action(tokenizer, token_ids):
    """Stop at the first complete action while preserving sampled token IDs."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    ends = [m.end() for m in re.finditer(r"</(?:search|answer)>", text)]
    if ends:
        end = min(ends)
        for n in range(1, len(token_ids) + 1):
            prefix = tokenizer.decode(token_ids[:n], skip_special_tokens=True)
            if len(prefix) >= end:
                return token_ids[:n], prefix
    return token_ids, text


class ProcurementGenerationManager:
    def __init__(self, tokenizer, actor_rollout_wg, config, procurement_config,
                 rollout_config, is_validation=False):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.procurement_config = procurement_config
        self.rollout_config = rollout_config
        self.is_validation = is_validation
        self.timing_raw = {}

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _retrieve(self, queries):
        if not queries:
            return []
        with requests.Session() as session:
            if urlsplit(self.config.search_url).hostname in {"localhost", "127.0.0.1", "::1"}:
                session.trust_env = False
            response = session.post(
                self.config.search_url,
                json={"queries": queries, "topk": self.config.topk, "return_scores": True},
                timeout=float(self.procurement_config.get("http_timeout", 30)),
            )
        # Infrastructure failures must fail the run, not become zero task reward.
        response.raise_for_status()
        rows = response.json()["result"]
        if len(rows) != len(queries):
            raise ValueError("Retriever returned an incorrect number of query results")
        return [[item["document"] for item in row] for row in rows]

    def run_llm_loop(self, gen_batch, initial_input_ids):
        from procurement_rl.protocol import parse_action

        # Read original masks; slicing to max_start_length could erase user constraints.
        prompts = [ids[mask.bool()].tolist() for ids, mask in zip(
            gen_batch.batch["input_ids"], gen_batch.batch["attention_mask"])]
        if any(len(row) > self.config.max_start_length for row in prompts):
            raise ValueError("Prompt exceeds max_start_length; regenerate/shorten prompts or raise the limit")
        n = len(prompts)
        response_rows = [[] for _ in prompts]
        token_turns = [[] for _ in prompts]
        traces = [{"assistant_messages": [], "observations": [], "observation_ends": [],
                   "valid_actions": 0, "search_calls": 0, "budget_exhausted": False,
                   "generated_tokens": 0, "dropped_documents": 0} for _ in prompts]
        active = list(range(n))
        for step in range(self.config.max_turns + 1):
            if not active:
                break
            context = [prompts[i] + response_rows[i] for i in active]
            if any(len(row) + self.config.max_response_length > self.config.max_prompt_length
                   for row in context):
                raise ValueError("Context limit would truncate history; increase max_prompt_length")
            ids, mask = _pad(context, self.tokenizer.pad_token_id, left=True)
            metadata = dict(gen_batch.meta_info)
            metadata.update(recompute_log_prob=False, do_sample=not self.is_validation)
            inputs = DataProto.from_dict({"input_ids": ids, "attention_mask": mask,
                                          "position_ids": _positions(mask)}, meta_info=metadata)
            padded, count = pad_for_workers(inputs, self.actor_rollout_wg.world_size)
            generated = unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), count)
            actions = []
            for j, index in enumerate(active):
                raw = generated.batch["responses"][j]
                rmask = generated.batch["attention_mask"][j, -raw.shape[-1]:].bool()
                all_ids = raw[rmask].tolist()
                sampled, text = _generated_action(self.tokenizer, all_ids)
                if not sampled:
                    raise ValueError("Worker returned no sampled token, including no EOS")
                traces[index]["generated_tokens"] += len(all_ids)
                turn_id = len(traces[index]["observation_ends"])
                response_rows[index].extend(sampled)
                token_turns[index].extend([turn_id] * len(sampled))
                traces[index]["assistant_messages"].append(text)
                action = parse_action(text)
                actions.append(action)
                if action.kind != "invalid":
                    traces[index]["valid_actions"] += 1

            # Search requests are issued only for active rows before the final attempt.
            queries = [a.query for a in actions if a.kind == "search"] if step < self.config.max_turns else []
            results = iter(self._retrieve(queries))
            next_active = []
            for index, action in zip(active, actions):
                if action.kind == "answer":
                    continue
                if step == self.config.max_turns:
                    traces[index]["budget_exhausted"] = True
                    continue
                if action.kind == "search":
                    documents = next(results)
                    observation, exposed = pack_observation(documents, self._encode, self.config.max_obs_length)
                    traces[index]["search_calls"] += 1
                    traces[index]["dropped_documents"] += len(documents) - len(exposed)
                    record = {"query": action.query, "documents": exposed, "text": observation}
                else:
                    observation = ("\n<information>Invalid action. Use <search>query</search>, or "
                                   "<evidence>doc_ids</evidence><answer>offer_id</answer>.</information>\n")
                    record = {"query": None, "documents": [], "text": observation, "invalid_action": True}
                observation_ids = self._encode(observation)
                if len(observation_ids) > self.config.max_obs_length:
                    raise ValueError("Observation exceeds configured budget")
                response_rows[index].extend(observation_ids)
                token_turns[index].extend([-1] * len(observation_ids))
                traces[index]["observations"].append(record)
                traces[index]["observation_ends"].append(len(response_rows[index]))
                next_active.append(index)
            active = next_active

        prompt_tensor, prompt_mask = _pad(prompts, self.tokenizer.pad_token_id, left=True)
        responses, response_mask = _pad(response_rows, self.tokenizer.pad_token_id)
        turns, _ = _pad(token_turns, -1)
        input_ids = torch.cat((prompt_tensor, responses), -1)
        attention = torch.cat((prompt_mask, response_mask), -1)
        info_mask = torch.cat((prompt_mask, (turns >= 0).long()), -1)
        for i, trace in enumerate(traces):
            trace["prompt_token_ids"] = prompts[i]
            trace["response_token_ids"] = response_rows[i]
        # Per-worker micro batch: upstream worker divides global config by world size.
        metadata = {
            "micro_batch_size": max(1, self.rollout_config.log_prob_micro_batch_size // self.config.num_gpus),
            "max_token_len": self.rollout_config.log_prob_max_token_len_per_gpu,
            "use_dynamic_bsz": self.rollout_config.log_prob_use_dynamic_bsz,
            "temperature": self.rollout_config.temperature,
            "turns_stats": [len(t["assistant_messages"]) for t in traces],
            "valid_action_stats": [t["valid_actions"] for t in traces],
            "valid_search_stats": [t["search_calls"] for t in traces],
            "active_mask": [t["budget_exhausted"] for t in traces],
        }
        return DataProto.from_dict({"prompts": prompt_tensor, "responses": responses,
                                    "input_ids": input_ids, "attention_mask": attention,
                                    "position_ids": _positions(attention), "info_mask": info_mask,
                                    "turn_ids": turns},
                                   non_tensors={"procurement_trace": np.array(traces, dtype=object)},
                                   meta_info=metadata)
