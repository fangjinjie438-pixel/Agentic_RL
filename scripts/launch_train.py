#!/usr/bin/env python3
"""Validate a profile, record the exact invocation, then run the pinned backend."""
import argparse
import copy
import datetime
import importlib.metadata
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODES = ("grpo", "igpo", "trajectory_ig")
sys.path.insert(0, str(ROOT / "src"))


def merge(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_profile(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Profile extends cycle")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Profile must be a YAML mapping")
    parent = data.pop("extends", None)
    return merge(load_profile(path.parent / parent, seen + (path,)), data) if parent else data


def set_key(mapping, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        mapping = mapping.setdefault(part, {})
    mapping[parts[-1]] = value


def flatten(mapping, prefix=""):
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, path)
        else:
            yield path, value


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum if maximum is not None else 'infinity'}]")


def _number(value, name, minimum=0, maximum=None, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            value < minimum or (positive and value == 0) or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a finite number within its allowed range")


def is_formal_profile(profile):
    return profile["name"].startswith("autodl_") or profile.get("experiment_kind") == "formal"


def validate_profile(profile):
    try:
        return _validate_profile(profile)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Incomplete or malformed profile: {error}") from error


def _validate_profile(profile):
    if not isinstance(profile["name"], str) or not profile["name"]:
        raise ValueError("Profile requires a name")
    u = profile["upstream"]
    a = u["actor_rollout_ref"]
    p = profile["procurement"]
    runtime = profile["runtime"]
    d = u["data"]
    integer_fields = {
        "runtime.gpu_count": runtime["gpu_count"],
        "trainer.n_gpus_per_node": u["trainer"]["n_gpus_per_node"],
        "trainer.nnodes": u["trainer"]["nnodes"],
        "rollout.tensor_model_parallel_size": a["rollout"]["tensor_model_parallel_size"],
        "rollout.n": a["rollout"]["n"], "rollout.n_agent": a["rollout"]["n_agent"],
        "rollout.max_num_seqs": a["rollout"]["max_num_seqs"],
        "rollout.max_num_batched_tokens": a["rollout"]["max_num_batched_tokens"],
        "gold_score_micro_batch_size": p["gold_score_micro_batch_size"],
    }
    for key in ("max_start_length", "max_prompt_length", "max_response_length", "max_obs_length",
                "train_batch_size", "val_batch_size"):
        integer_fields[f"data.{key}"] = d[key]
    for key in ("train_data_num", "val_data_num"):
        if d.get(key) is not None:
            integer_fields[f"data.{key}"] = d[key]
    for key in ("total_epochs", "total_training_steps"):
        if u["trainer"].get(key) is not None:
            integer_fields[f"trainer.{key}"] = u["trainer"][key]
    for name, value in integer_fields.items():
        _integer(value, name)
    _integer(u["max_turns"], "max_turns", minimum=0)
    _integer(p["seed"], "procurement.seed", minimum=0, maximum=2**32 - 1)
    _integer(profile.get("model_min_parameters", 0), "model_min_parameters", minimum=0)
    _integer(u["retriever"]["topk"], "retriever.topk", maximum=20)
    for key in ("save_freq", "test_freq"):
        value = u["trainer"][key]
        _integer(value, f"trainer.{key}", minimum=-1)
        if value == 0:
            raise ValueError(f"trainer.{key} must be -1 (disabled) or positive")
    for key in ("min_vram_gib_per_gpu", "min_free_vram_gib_per_gpu", "min_cpu_ram_gib"):
        _number(runtime[key], f"runtime.{key}", positive=True)
    if runtime["min_free_vram_gib_per_gpu"] > runtime["min_vram_gib_per_gpu"]:
        raise ValueError("Minimum free VRAM cannot exceed minimum total VRAM")
    for key in ("ig_weight", "outcome_weight"):
        _number(p[key], f"procurement.{key}")
    if p["outcome_weight"] == 0:
        raise ValueError("The controlled comparison requires a positive shared outcome reward weight")
    _number(p["http_timeout"], "procurement.http_timeout", positive=True)
    _number(a["actor"]["optim"]["lr"], "actor.optim.lr", positive=True)
    _number(a["actor"]["optim"]["lr_warmup_steps_ratio"], "actor.optim.lr_warmup_steps_ratio", maximum=1)
    _number(a["actor"]["kl_loss_coef"], "actor.kl_loss_coef")
    _number(a["actor"]["entropy_coeff"], "actor.entropy_coeff")
    _number(a["rollout"]["gpu_memory_utilization"], "rollout.gpu_memory_utilization", positive=True, maximum=1)
    boolean_fields = {"do_search": u["do_search"], "algorithm.no_think_rl": u["algorithm"]["no_think_rl"],
                      "hybrid_engine": a["hybrid_engine"], "data.shuffle_train_dataloader": d["shuffle_train_dataloader"]}
    for section, keys in ((a["model"], ("enable_gradient_checkpointing", "use_remove_padding")),
                          (a["actor"], ("use_dynamic_bsz", "state_masking", "use_kl_loss")),
                          (a["rollout"], ("enforce_eager", "free_cache_engine")),
                          (u["trainer"], ("val_before_train", "val_only"))):
        boolean_fields.update({key: section[key] for key in keys})
    for section in (a["actor"]["fsdp_config"], a["ref"]["fsdp_config"]):
        boolean_fields.update({key: value for key, value in section.items() if key.endswith("_offload")})
    for key, value in boolean_fields.items():
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
    if (not u["do_search"] or not a["hybrid_engine"] or u["algorithm"]["no_think_rl"] or
            u.get("reward_model", {}).get("enable", False) is not False):
        raise ValueError("Require hybrid search rollout, no_think_rl=false and the deterministic rule verifier")
    if a["actor"]["strategy"] != "fsdp" or a["rollout"]["name"] != "vllm" or a["rollout"]["dtype"] != "bfloat16":
        raise ValueError("The pinned implementation requires FSDP, vLLM and bfloat16")
    for key in ("temperature", "top_p"):
        _number(a["rollout"][key], f"rollout.{key}")
        if a["rollout"][key] != 1.0:
            raise ValueError("Use temperature=1 and top_p=1 for matching policy and teacher probabilities")
    for value, name in ((a["rollout"]["top_k"], "rollout.top_k"),
                        (a["actor"].get("ulysses_sequence_parallel_size", 1), "actor.ulysses_sequence_parallel_size"),
                        (a["actor"].get("ppo_epochs", 1), "actor.ppo_epochs")):
        _integer(value, name, minimum=-1)
        if value != (-1 if name == "rollout.top_k" else 1):
            raise ValueError(f"Unsupported {name}")
    model = a["model"]["path"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model path must be a nonempty string")
    if a["model"].get("override_config"):
        raise ValueError("Model config overrides bypass cached architecture/context checks; use the frozen model config")
    url = urllib.parse.urlsplit(u["retriever"]["url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.path != "/retrieve" or url.query or url.fragment:
        raise ValueError("Retriever URL must identify an HTTP(S) /retrieve endpoint")
    n_gpu = u["trainer"]["n_gpus_per_node"]
    if n_gpu != profile["runtime"]["gpu_count"] or n_gpu < 1:
        raise ValueError("runtime.gpu_count must match trainer.n_gpus_per_node")
    if u["trainer"]["nnodes"] != 1 or a["rollout"]["tensor_model_parallel_size"] != 1:
        raise ValueError("These verified launcher invariants require one node and TP=1")
    if a["rollout"]["n"] != 1 or a["rollout"]["n_agent"] < 2:
        raise ValueError("Search-R1 groups use n_agent >= 2 with rollout.n=1")
    sizes = [u["data"]["train_batch_size"], a["actor"]["ppo_mini_batch_size"],
             a["actor"]["ppo_micro_batch_size"], a["rollout"]["log_prob_micro_batch_size"], a["ref"]["log_prob_micro_batch_size"]]
    if any(isinstance(size, bool) or not isinstance(size, int) or size < n_gpu or size % n_gpu for size in sizes):
        raise ValueError("All global train/mini/micro batch sizes must be positive multiples of the GPU count")
    effective_batch = sizes[0] * a["rollout"]["n_agent"]
    if effective_batch % sizes[1] or sizes[1] % sizes[2]:
        raise ValueError("Sample batch must divide into PPO mini batches, which must divide into micro batches")
    required = d["max_start_length"] + u["max_turns"] * (d["max_response_length"] + d["max_obs_length"]) + d["max_response_length"]
    if d["max_prompt_length"] < required:
        raise ValueError(f"Context may truncate evidence: require max_prompt_length >= {required}")
    for key, expected in (("prompt_length", d["max_prompt_length"]), ("response_length", d["max_response_length"])):
        if key in a["rollout"] and a["rollout"][key] != expected:
            raise ValueError(f"rollout.{key} must match the data token budget")
    if a["rollout"]["max_num_batched_tokens"] < d["max_prompt_length"] + d["max_response_length"]:
        raise ValueError("vLLM max_num_batched_tokens must cover max_model_len (prompt + response)")
    if not a["actor"]["state_masking"] or not a["actor"]["use_kl_loss"]:
        raise ValueError("Require observation masking and the shared KL loss")
    _number(u["algorithm"]["gamma"], "algorithm.gamma")
    if u["algorithm"]["adv_estimator"] != "grpo" or u["algorithm"]["gamma"] != 1.0:
        raise ValueError("Backend dispatch requires grpo and gamma=1; choose method via procurement.mode")
    if profile["procurement"]["algorithm"] not in MODES:
        raise ValueError(f"algorithm must be one of {MODES}")
    if is_formal_profile(profile):
        if profile.get("model_min_parameters", 0) < 3_000_000_000:
            raise ValueError("AutoDL result profiles must require at least 3B parameters")
        known_size = re.search(r"(?:^|[-/])(\d+(?:\.\d+)?)B(?:-|$)", a["model"]["path"], re.I)
        if known_size and float(known_size.group(1)) < 3:
            raise ValueError("AutoDL results require >=3B; smaller model overrides are not allowed")
    return required


def hydra_command(profile):
    u = copy.deepcopy(profile["upstream"])
    procurement = copy.deepcopy(profile["procurement"])
    procurement["mode"] = procurement.pop("algorithm")
    command = [sys.executable, "-m", "procurement_rl.training.main"]
    additions = {"trainer.val_before_train", "trainer.val_only"}
    for key, value in flatten(u):
        command.append(("+" if key in additions else "") + key + "=" + json.dumps(value, ensure_ascii=False))
    for key, value in flatten(procurement, "procurement"):
        command.append("+" + key + "=" + json.dumps(value, ensure_ascii=False))
    return command


def resolve_args(args):
    profile = load_profile(args.profile)
    profile["procurement"]["algorithm"] = args.algorithm or profile["procurement"]["algorithm"]
    if args.model:
        profile["upstream"]["actor_rollout_ref"]["model"]["path"] = args.model
    if args.seed is not None:
        profile["procurement"]["seed"] = args.seed
    data_dir = Path(args.data_dir or profile["procurement"]["data_dir"]).resolve()
    profile["procurement"]["data_dir"] = str(data_dir)
    profile["upstream"]["data"]["train_files"] = str(data_dir / "train.parquet")
    profile["upstream"]["data"]["val_files"] = str(data_dir / "val.parquet")
    for override in args.overrides:
        if "=" not in override:
            raise ValueError("--set expects a dotted.key=value")
        key, value = override.split("=", 1)
        if not key.startswith(("upstream.", "procurement.", "runtime.")):
            raise ValueError("--set keys must start upstream., procurement., or runtime.")
        set_key(profile, key, yaml.safe_load(value))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.run_name or f"{profile['name']}-{profile['procurement']['algorithm']}-s{profile['procurement']['seed']}-{stamp}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("Run name must contain only letters, digits, underscore, period, and hyphen")
    output = ROOT / "outputs" / name
    profile["upstream"]["trainer"]["experiment_name"] = name
    profile["upstream"]["trainer"]["default_local_dir"] = str(output / "checkpoints")
    profile["procurement"]["trace_dir"] = str(output / "trajectories")
    validate_profile(profile)
    return profile, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(ROOT / "configs/school_smoke.yaml"))
    parser.add_argument("--algorithm", choices=MODES)
    parser.add_argument("--model")
    parser.add_argument("--data-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--hourly-cost-cny", type=float, help="Actual whole-instance hourly price; omit when unknown")
    parser.add_argument("--dry-run", action="store_true", help="Validate static config and print invocation; no files or GPU initialization.")
    args = parser.parse_args()
    if args.hourly_cost_cny is not None:
        _number(args.hourly_cost_cny, "hourly_cost_cny")
    profile, output = resolve_args(args)
    command = hydra_command(profile)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    from preflight import runtime_checks
    from procurement_rl.tracking import runtime_manifest, finish_manifest, write_json, file_digest
    output.mkdir(parents=True, exist_ok=False)
    manifest = runtime_manifest('training', profile)
    manifest['experiment_kind'] = 'formal' if is_formal_profile(profile) else 'smoke'
    manifest['launcher_source_sha256'] = {
        str(path.relative_to(ROOT)): file_digest(path)
        for folder in ('scripts', 'configs') for path in sorted((ROOT / folder).glob('*')) if path.is_file()}
    manifest['environment'] = {name: os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES', 'VLLM_ATTENTION_BACKEND')}
    write_json(output / 'manifest.json', manifest)
    started = time.perf_counter()
    status, error = 1, None
    process = None
    try:
        checks = runtime_checks(profile, check_retriever=True)
        manifest['preflight'] = checks
        # Pin the resolved snapshot path for this invocation, even if Hub main moves.
        profile['upstream']['actor_rollout_ref']['model']['path'] = checks['model_artifacts']['path']
        command = hydra_command(profile)
        write_json(output / 'resolved_profile.json', profile)
        write_json(output / 'invocation.json', command)
        write_json(output / 'manifest.json', manifest)
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (output / 'pip-freeze.txt').write_text(freeze, encoding='utf-8')
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
        environment.setdefault('SEARCH_R1_ROOT', str(ROOT / 'third_party/Search-R1'))
        environment.setdefault('VLLM_ATTENTION_BACKEND', 'XFORMERS')
        environment['TOKENIZERS_PARALLELISM'] = 'false'
        environment['PYTHONUNBUFFERED'] = '1'
        environment['PYTHONHASHSEED'] = str(profile['procurement']['seed'])
        with (output / 'train.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        if status:
            raise RuntimeError(f'Training subprocess exited with status {status}; see train.log')
    except BaseException as caught:
        error = caught
        if process is not None and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        manifest['returncode'] = status
        finish_manifest(manifest, time.perf_counter() - started, args.hourly_cost_cny, error)
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'exit_status.json', {'returncode': status, 'status': manifest['status']})


if __name__ == "__main__":
    main()
