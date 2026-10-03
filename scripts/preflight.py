#!/usr/bin/env python3
"""Fail early on incompatible profiles, cached models, hardware, or upstream code."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

from bootstrap_search_r1 import verify_checkout
from launch_train import ROOT, load_profile, validate_profile, is_formal_profile

EXPECTED = {"torch": "2.4.0", "vllm": "0.6.3", "transformers": "4.46.3", "tensordict": "0.5.0", "flash-attn": "2.7.0.post2"}


def qwen_parameters(config):
    """Exact Qwen2/Qwen2.5 dense decoder parameter count without loading weights."""
    if config.model_type != "qwen2":
        raise ValueError("This parameter-count guard supports Qwen2/Qwen2.5 dense models; add an audited counter for other architectures")
    hidden, intermediate, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    key_value = hidden // config.num_attention_heads * config.num_key_value_heads
    embedding = config.vocab_size * hidden * (1 if config.tie_word_embeddings else 2)
    attention = 2 * hidden * hidden + 2 * hidden * key_value + hidden + 2 * key_value
    return embedding + layers * (attention + 3 * hidden * intermediate + 2 * hidden) + hidden


def dataset_checks(profile, tokenizer=None):
    from procurement_rl.tracking import file_digest, read_jsonl, verified_dataset
    from procurement_rl.retrieval import BM25Retriever
    import pandas as pd

    root = Path(profile['procurement']['data_dir']).resolve()
    frozen = verified_dataset(root)
    exported = json.loads((root / 'parquet_manifest.json').read_text(encoding='utf-8'))
    if (exported['dataset_manifest_sha256'] != file_digest(root / 'manifest.json') or
            exported['source_files_sha256'] != frozen['files']):
        raise RuntimeError('Parquet export belongs to a different frozen dataset')
    budgets = {}
    for split in ('train', 'val', 'test'):
        path = root / f'{split}.parquet'
        if split != 'test' and Path(profile['upstream']['data'][f'{split}_files']).resolve() != path:
            raise RuntimeError('Training parquet paths must belong to procurement.data_dir')
        if file_digest(path) != exported['files'][path.name]:
            raise RuntimeError(f'Parquet hash mismatch: {path.name}')
        frame = pd.read_parquet(path)
        tasks = read_jsonl(root / split / 'tasks.jsonl')
        if len(frame) != len(tasks) or not tasks:
            raise RuntimeError(f'Invalid {split} row count')
        longest = 0
        for row, task in zip(frame.to_dict('records'), tasks):
            prompt = row['prompt'].tolist() if hasattr(row['prompt'], 'tolist') else row['prompt']
            if (row['task_id'] != task['task_id'] or prompt != [{'role': 'user', 'content': task['prompt']}] or
                    json.loads(row['reward_model']['ground_truth']) != task['ground_truth']):
                raise RuntimeError(f'Parquet content does not match frozen task {task["task_id"]}')
            if tokenizer is not None:
                longest = max(longest, len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True)))
        if tokenizer is not None:
            budgets[f'{split}_max_prompt_tokens'] = longest
            if longest > profile['upstream']['data']['max_start_length']:
                raise RuntimeError(f'{split} prompt requires {longest} tokens; increase max_start_length')
    retriever = BM25Retriever.from_jsonl([root / split / 'corpus.jsonl' for split in ('train', 'val', 'test')])
    if tokenizer is not None:
        longest = max(len(tokenizer.encode(
            f'\n<information>\n[{doc["id"]}] {doc["contents"]}\n</information>\n', add_special_tokens=False))
            for doc in retriever.documents)
        budgets['max_single_document_tokens'] = longest
        if longest > profile['upstream']['data']['max_obs_length']:
            raise RuntimeError(f'A complete document requires {longest} tokens; increase max_obs_length and rolling context')
    return {'dataset_manifest_sha256': file_digest(root / 'manifest.json'),
            'data_files_sha256': frozen['files'], 'parquet_sha256': exported['files'],
            'corpus_sha256': retriever.corpus_sha256, 'document_count': retriever.document_count,
            'token_budgets': budgets}


def model_artifacts(model):
    from procurement_rl.tracking import file_digest
    path = Path(model)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model, local_files_only=True))
    path = path.resolve()
    index = path / 'model.safetensors.index.json'
    binary_index = path / 'pytorch_model.bin.index.json'
    if index.exists() or binary_index.exists():
        source = index if index.exists() else binary_index
        required = set(json.loads(source.read_text())['weight_map'].values())
    elif (path / 'model.safetensors').exists():
        required = {'model.safetensors'}
    elif (path / 'pytorch_model.bin').exists():
        required = {'pytorch_model.bin'}
    else:
        raise RuntimeError(f'Model weights are missing at {path}; tokenizer-only downloads cannot train')
    for name in required:
        if Path(name).name != name or not (path / name).is_file():
            raise RuntimeError(f'Missing or invalid model weight shard: {name}')
    artifacts = [p for p in path.iterdir() if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt'}]
    return {'path': str(path), 'resolved_revision': path.name if path.parent.name == 'snapshots' else None,
            'files_sha256': {p.name: file_digest(p) for p in sorted(artifacts)}}


def retriever_health(url, expected):
    from urllib.parse import urlsplit
    health_url = url.rsplit('/', 1)[0] + '/health'
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if urlsplit(url).hostname in {'localhost', '127.0.0.1', '::1'} else urllib.request.build_opener())
    with opener.open(health_url, timeout=5) as response:
        health = json.load(response)
    if (health.get('status') != 'ok' or health.get('retriever') != 'bm25' or
            health.get('corpus_sha256') != expected['corpus_sha256'] or
            health.get('document_count') != expected['document_count']):
        raise RuntimeError('HTTP retriever does not serve the expected frozen shared corpus')
    return health


def runtime_checks(profile, check_retriever=False):
    validate_profile(profile)
    if platform.system() != "Linux":
        raise RuntimeError("GPU training requires Linux; Windows supports dataset, tests and --config-only checks")
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("Use the pinned Python 3.9 environment for this Search-R1 stack")
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    upstream = Path(os.environ.get("SEARCH_R1_ROOT", ROOT / lock["path"])).resolve()
    verify_checkout(upstream, lock["commit"])
    for package, expected in EXPECTED.items():
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{package} is {actual}; candidate stack requires {expected}")
    import torch
    import flash_attn  # Import catches CUDA/PyTorch ABI mismatches before starting Ray.
    import psutil
    from transformers import AutoConfig, AutoTokenizer

    if torch.version.cuda != "12.1":
        raise RuntimeError(f"Expected PyTorch cu121; found {torch.version.cuda}")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    if tuple(int(part) for part in driver.split(".")[:2]) < (525, 60):
        raise RuntimeError(f"Driver {driver} is below the CUDA 12.x Linux compatibility floor")
    count = profile["runtime"]["gpu_count"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise RuntimeError(f"Select exactly {count} GPUs with CUDA_VISIBLE_DEVICES; visible={torch.cuda.device_count()}")
    gpu_info = []
    for index in range(count):
        prop = torch.cuda.get_device_properties(index)
        if prop.major < 8 or prop.major >= 10:
            raise RuntimeError(f"{prop.name} compute capability {prop.major}.{prop.minor} unsupported by this pinned BF16/FlashAttention stack; use Ampere/Ada/Hopper")
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        total, free = total_bytes / 2**30, free_bytes / 2**30
        if total < profile["runtime"]["min_vram_gib_per_gpu"] or free < profile["runtime"]["min_free_vram_gib_per_gpu"]:
            raise RuntimeError(f"GPU {index} {prop.name}: {free:.1f}/{total:.1f} GiB free/total does not meet profile. Reserve idle GPUs or choose a smaller school profile.")
        gpu_info.append({"index": index, "name": prop.name, "free_gib": round(free, 2), "total_gib": round(total, 2)})
    memory = psutil.virtual_memory()
    available = memory.available / 2**30
    if available < profile["runtime"]["min_cpu_ram_gib"]:
        raise RuntimeError(f"Only {available:.1f} GiB CPU RAM available; profile requires {profile['runtime']['min_cpu_ram_gib']} for offloading")
    model = profile["upstream"]["actor_rollout_ref"]["model"]["path"]
    artifacts = model_artifacts(model)
    model = artifacts["path"]
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError("Model tokenizer must provide the Instruct chat template")
    parameters = qwen_parameters(config)
    minimum = max(profile.get("model_min_parameters", 0), 3_000_000_000 if is_formal_profile(profile) else 0)
    if parameters < minimum:
        raise RuntimeError(f"Model has {parameters:,} parameters; profile requires >= {minimum:,}. No small-model cloud fallback.")
    data = profile["upstream"]["data"]
    context = data["max_prompt_length"] + data["max_response_length"]
    if config.max_position_embeddings < context:
        raise RuntimeError(f"Model context limit {config.max_position_embeddings} < vLLM max_model_len {context}")
    dataset = dataset_checks(profile, tokenizer)
    health = retriever_health(profile['upstream']['retriever']['url'], dataset) if check_retriever else None
    result = {"profile": profile["name"], "upstream_commit": lock["commit"], "model": model,
              "model_parameters": parameters, "driver": driver, "gpus": gpu_info,
              "available_cpu_gib": round(available, 2), "retriever": health, "dataset": dataset,
              "model_artifacts": artifacts}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(ROOT / "configs/school_smoke.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--data-only", action="store_true", help="Check frozen parquet/corpus and local tokenizer budgets without GPU initialization or weights")
    parser.add_argument("--check-retriever", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.model:
        profile["upstream"]["actor_rollout_ref"]["model"]["path"] = args.model
    if args.data_dir:
        profile["procurement"]["data_dir"] = str(args.data_dir.resolve())
        for split in ("train", "val"):
            profile["upstream"]["data"][f"{split}_files"] = str(args.data_dir / f"{split}.parquet")
    required = validate_profile(profile)
    if args.config_only:
        print(json.dumps({"profile": profile["name"], "static_checks": "passed", "required_rolling_context": required, "gpu_runtime": "not tested"}))
    elif args.data_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(profile['upstream']['actor_rollout_ref']['model']['path'],
                                                  local_files_only=True, trust_remote_code=False)
        result = dataset_checks(profile, tokenizer)
        result['gpu_runtime'] = 'not tested'
        print(json.dumps(result, indent=2))
    else:
        runtime_checks(profile, check_retriever=args.check_retriever)


if __name__ == "__main__":
    main()
