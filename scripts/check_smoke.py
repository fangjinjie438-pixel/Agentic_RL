#!/usr/bin/env python3
"""Require actual training and checkpoint reload evidence before accepting smoke."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from procurement_rl.tracking import file_digest, read_jsonl, write_json
from procurement_rl.protocol import parse_action


def check(run, reload_eval):
    run, reload_eval = Path(run).resolve(), Path(reload_eval).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['kind'] != 'training':
        raise ValueError('Training did not complete')
    profile = json.loads((run / 'resolved_profile.json').read_text())
    expected = profile['upstream']['trainer']['total_training_steps']
    rows = [row for row in read_jsonl(run / 'metrics.jsonl') if 'actor/pg_loss' in row]
    if [row['step'] for row in rows] != list(range(1, expected + 1)):
        raise ValueError('Missing or duplicate actor update iterations')
    for row in rows:
        for key in ('actor/pg_loss', 'actor/kl_loss', 'actor/grad_norm', 'actor/sampled_parameter_max_delta'):
            if not math.isfinite(row[key]):
                raise ValueError(f'Nonfinite {key}')
        if row['train/observation_loss_tokens'] != 0:
            raise ValueError('Tool or padding tokens received policy loss')
    if not any(row['train/advantage_abs_mean'] > 0 for row in rows):
        raise ValueError('No nonzero advantage was observed')
    if not any(row['actor/sampled_parameter_max_delta'] > 0 for row in rows):
        raise ValueError('Parameter update has no positive sample evidence')
    traces = read_jsonl(run / 'trajectories/train.jsonl')
    if not any(t['search_calls'] > 0 and any(o['documents'] for o in t['observations'])
               and parse_action(t['assistant_messages'][-1]).kind == 'answer' for t in traces):
        raise ValueError('No complete retrieval/continuation/answer trajectory was observed')
    evaluation = json.loads((reload_eval / 'manifest.json').read_text())
    if evaluation['status'] != 'completed' or evaluation['kind'] != 'model_evaluation':
        raise ValueError('A completed real-model checkpoint reload evaluation is required')
    model = Path(evaluation['policy']['model']).resolve()
    if run / 'checkpoints' not in model.parents:
        raise ValueError('Evaluation must load a checkpoint saved by this run')
    hashes = evaluation['policy'].get('local_artifact_sha256', {})
    if not any(name.endswith(('.safetensors', '.bin')) for name in hashes):
        raise ValueError('No checkpoint weight hash in evaluation record')
    for name, digest in hashes.items():
        if file_digest(model / name) != digest:
            raise ValueError('Evaluated checkpoint content changed')
    if evaluation.get('results_sha256') != file_digest(reload_eval / 'results.jsonl'):
        raise ValueError('Evaluation results hash mismatch')
    return {'accepted': True, 'train_iterations': len(rows),
            'optimizer_steps_per_worker': sum(row['actor/optimizer_steps'] for row in rows),
            'run': str(run), 'reload_evaluation': str(reload_eval),
            'scope': 'Engineering smoke only; does not demonstrate training improvement'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reload-eval', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.run, args.reload_eval)
    write_json(args.run / 'smoke_acceptance.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
