"""Pure-Python rollout invariants, usable by local tests and server preflight."""

import math


def max_trajectory_tokens(max_start_length, max_turns, max_response_length, max_obs_length):
    return max_start_length + max_turns * (max_response_length + max_obs_length) + max_response_length


def validate_config(config):
    """Fail before Ray/model initialization on unsafe or unsupported settings."""
    ar = config["actor_rollout_ref"]
    procurement = config["procurement"]
    world = config["trainer"]["n_gpus_per_node"] * config["trainer"]["nnodes"]
    if world < 1:
        raise ValueError("At least one GPU is required")
    if ar["actor"]["strategy"] != "fsdp":
        raise ValueError("This pinned adapter supports FSDP only")
    if config["algorithm"]["adv_estimator"] != "grpo" or config["algorithm"]["gamma"] != 1.0:
        raise ValueError("Use adv_estimator=grpo and gamma=1.0 for the controlled comparison")
    if not config["do_search"] or not ar["actor"]["state_masking"] or not ar["actor"]["use_kl_loss"]:
        raise ValueError("do_search, actor.state_masking, and actor.use_kl_loss must be true")
    if config["reward_model"]["enable"]:
        raise ValueError("Use the deterministic procurement reward, not a learned reward model")
    if ar["rollout"]["n"] != 1 or ar["rollout"]["tensor_model_parallel_size"] != 1:
        raise ValueError("Use rollout.n=1, tensor_model_parallel_size=1 and n_agent for groups")
    if ar["rollout"]["n_agent"] < 2 and not config["trainer"].get("val_only", False):
        raise ValueError("Training needs at least two samples per prompt")
    if ar["rollout"]["temperature"] != 1.0 or ar["rollout"]["top_p"] != 1.0:
        raise ValueError("Use temperature=1.0 and top_p=1.0 so rollout and teacher scoring agree")
    if ar["rollout"]["name"] != "vllm" or ar["rollout"].get("top_k", -1) != -1:
        raise ValueError("Use the pinned vLLM rollout with unrestricted top_k=-1")
    if ar["actor"].get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("This adapter requires actor sequence parallel size 1")
    if ar["actor"].get("ppo_epochs", 1) != 1:
        raise ValueError("The pinned actor implements one PPO epoch; require ppo_epochs=1")
    if procurement["mode"] not in {"grpo", "igpo", "trajectory_ig"}:
        raise ValueError("procurement.mode must be grpo, igpo or trajectory_ig")
    for name in ("ig_weight", "outcome_weight"):
        value = float(procurement.get(name, 1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"procurement.{name} must be nonnegative and finite")
    for section, key in (("actor", "ppo_mini_batch_size"), ("actor", "ppo_micro_batch_size"),
                         ("rollout", "log_prob_micro_batch_size"), ("ref", "log_prob_micro_batch_size")):
        value = ar[section][key]
        if value < world or value % world:
            raise ValueError(f"{section}.{key} must be a positive multiple of world size {world}")
    batch = config["data"]["train_batch_size"] * ar["rollout"]["n_agent"]
    if batch % world or batch % ar["actor"]["ppo_mini_batch_size"]:
        raise ValueError("Expanded train batch must divide world size and PPO mini batch exactly")
    if ar["actor"]["ppo_mini_batch_size"] % ar["actor"]["ppo_micro_batch_size"]:
        raise ValueError("PPO mini batch must divide into complete micro batches")
    data = config["data"]
    for key in ("max_start_length", "max_response_length", "max_obs_length", "max_prompt_length",
                "train_batch_size", "val_batch_size"):
        if not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if not isinstance(config["max_turns"], int) or config["max_turns"] < 0:
        raise ValueError("max_turns must be a nonnegative integer")
    upper = max_trajectory_tokens(data["max_start_length"], config["max_turns"],
                                  data["max_response_length"], data["max_obs_length"])
    if upper > data["max_prompt_length"]:
        raise ValueError(f"Full trajectory needs up to {upper} tokens; increase data.max_prompt_length")


def pack_observation(documents, encode, max_tokens):
    """Fit whole documents only; return exactly the documents exposed to the actor.

Never truncate a document then credit its complete evidence hash. A result that
does not fit is dropped; subsequent shorter documents may still be included.
    """
    selected = []
    body = ""
    for document in documents:
        piece = f"[{document['id']}] {document['contents']}\n"
        alone = f"\n<information>\n{piece}</information>\n"
        if not selected and not body and len(encode(alone)) > max_tokens:
            raise ValueError(f"Document {document['id']} exceeds max_obs_length={max_tokens}; "
                             "increase the observation and full-context budgets")
        candidate = f"\n<information>\n{body}{piece}</information>\n"
        if len(encode(candidate)) <= max_tokens:
            selected.append(document)
            body += piece
    if not selected:
        text = "\n<information>No complete document fits this query result.</information>\n"
    else:
        text = f"\n<information>\n{body}</information>\n"
    if len(encode(text)) > max_tokens:
        raise ValueError("Observation budget is too small even for the empty-result message")
    return text, selected


def scoring_prefixes(prompt_ids, response_ids, observation_ends):
    """Initial and post-observation states; never include the final model answer."""
    previous = 0
    prefixes = [list(prompt_ids)]
    for end in observation_ends:
        if end <= previous or end > len(response_ids):
            raise ValueError("Invalid observation boundary")
        prefixes.append(list(prompt_ids) + list(response_ids[:end]))
        previous = end
    return prefixes


def group_identifiers(batch_size, step):
    """Create a group per sampled prompt occurrence, before n_agent expansion."""
    return [f"{step}:{index}" for index in range(batch_size)]
