#!/usr/bin/env python3
"""Download one immutable Qwen snapshot into the project; never initializes CUDA."""
import argparse
import json
from pathlib import Path

from launch_train import ROOT, load_profile, validate_profile, is_formal_profile
from preflight import qwen_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default=str(ROOT / 'configs/school_single.yaml'))
    parser.add_argument('--revision', required=True, help='Immutable Hugging Face commit SHA')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data/model-cache')
    parser.add_argument('--tokenizer-only', action='store_true')
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('Use an immutable 40-character commit SHA')
    profile = load_profile(args.profile)
    validate_profile(profile)
    model = profile['upstream']['actor_rollout_ref']['model']['path']
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig
    patterns = ['config.json', 'generation_config.json', 'tokenizer*.json', 'vocab.json', 'merges.txt']
    config_path = snapshot_download(model, revision=args.revision, cache_dir=args.cache_dir,
                                    allow_patterns=patterns)
    parameters = qwen_parameters(AutoConfig.from_pretrained(config_path, local_files_only=True))
    minimum = max(profile.get('model_min_parameters', 0), 3_000_000_000 if is_formal_profile(profile) else 0)
    if parameters < minimum:
        raise ValueError(f'Model has {parameters} parameters, requires >= {minimum}')
    if not args.tokenizer_only:
        config_path = snapshot_download(model, revision=args.revision, cache_dir=args.cache_dir,
                                        allow_patterns=patterns + ['*.safetensors', '*.safetensors.index.json'])
    print(json.dumps({'model': model, 'revision': args.revision, 'parameters': parameters,
                      'path': str(Path(config_path).resolve()), 'tokenizer_only': args.tokenizer_only}, indent=2))


if __name__ == '__main__':
    main()
