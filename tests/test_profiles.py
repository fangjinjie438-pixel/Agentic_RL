import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launch_train import load_profile, validate_profile
from preflight import dataset_checks, model_artifacts, qwen_parameters


class ProfileTests(unittest.TestCase):
    def test_profiles_accept_and_formal_rejects_small_model(self):
        for path in (ROOT / 'configs').glob('*.yaml'):
            if path.name == 'common.yaml':
                continue
            config = load_profile(path)
            validate_profile(config)
            if path.name.startswith('autodl'):
                config['upstream']['actor_rollout_ref']['model']['path'] = 'Qwen/Qwen2.5-1.5B-Instruct'
                with self.assertRaisesRegex(ValueError, '>=3B'):
                    validate_profile(config)

    def test_invalid_shapes_and_values_fail_before_launch(self):
        original = load_profile(ROOT / 'configs/school_smoke.yaml')
        for section, field, value in [('data', 'max_prompt_length', 1024), ('data', 'train_batch_size', True),
                                      ('data', 'val_batch_size', 0), ('algorithm', 'gamma', float('nan'))]:
            config = copy.deepcopy(original)
            config['upstream'][section][field] = value
            with self.assertRaises(ValueError):
                validate_profile(config)
        config = copy.deepcopy(original)
        config['procurement']['seed'] = -1
        with self.assertRaises(ValueError):
            validate_profile(config)
        config['procurement']['seed'] = 42
        config['upstream']['actor_rollout_ref']['rollout']['temperature'] = 0.7
        with self.assertRaises(ValueError):
            validate_profile(config)

    def test_missing_or_escaping_model_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'weights are missing'):
                model_artifacts(path)
            index = path / 'model.safetensors.index.json'
            for shard in ['missing.safetensors', '../outside.safetensors']:
                index.write_text(json.dumps({'weight_map': {'weight': shard}}))
                with self.assertRaisesRegex(RuntimeError, 'shard'):
                    model_artifacts(path)

    @unittest.skipUnless(importlib.util.find_spec('pandas') and importlib.util.find_spec('pyarrow'), 'parquet dependencies required')
    def test_corrupt_export_fails_closed(self):
        from procurement_rl.data import build_dataset
        from procurement_rl.cli import export_parquet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            build_dataset(path, 12, 6, 6, 42)
            export_parquet(path)
            profile = load_profile(ROOT / 'configs/school_single.yaml')
            profile['procurement']['data_dir'] = str(path)
            for split in ('train', 'val'):
                profile['upstream']['data'][f'{split}_files'] = str(path / f'{split}.parquet')
            report = dataset_checks(profile)
            self.assertGreater(report['document_count'], 0)
            with (path / 'train.parquet').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'Parquet hash mismatch'):
                dataset_checks(profile)


if __name__ == '__main__':
    unittest.main()
