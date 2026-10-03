"""Launch procurement training using only the locked Search-R1 bundled veRL.

The launcher supplies Hydra overrides over the upstream YAML. This module keeps
GPU imports lazy so CPU tools can verify the checkout without loading CUDA.
"""

import json
import os
from pathlib import Path
import subprocess
import sys


def activate_upstream():
    root = Path(__file__).resolve().parents[3]
    lock = json.loads((root / "upstream.lock.json").read_text(encoding="utf-8"))
    upstream = Path(os.environ.get("SEARCH_R1_ROOT", root / lock["path"])).resolve()
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(upstream), *args], text=True).strip()
    if not (upstream / ".git").exists() or git("rev-parse", "HEAD") != lock["commit"]:
        raise RuntimeError("SEARCH_R1_ROOT must be the checkout pinned in upstream.lock.json")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Pinned upstream has local changes; use an unmodified checkout")
    for name in ("verl", "search_r1"):
        module = sys.modules.get(name)
        if module is not None and upstream not in Path(module.__file__).resolve().parents:
            raise RuntimeError(f"{name} was already imported from a different installation")
    sys.path.insert(0, str(upstream))
    return upstream


def run_training(config):
    """Executed in a Ray driver task; worker code remains in project adapters."""
    activate_upstream()
    import random
    import numpy as np
    import torch
    import ray
    from omegaconf import OmegaConf, open_dict
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from procurement_rl.training.contracts import validate_config
    from procurement_rl.training.reward_manager import ProcurementRewardManager
    from procurement_rl.training.trainer import ProcurementTrainer
    from procurement_rl.training.workers import ProcurementActorRolloutRefWorker

    OmegaConf.resolve(config)
    validate_config(OmegaConf.to_container(config, resolve=True))
    seed = int(config.procurement.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open_dict(config):
        config.actor_rollout_ref.procurement_seed = seed
    path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(path)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The rollout requires explicit tokenizer pad and EOS IDs")
    roles = (Role.ActorRollout, Role.RefPolicy)
    pool = "procurement"
    resources = ResourcePoolManager(
        resource_pool_spec={pool: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={role: pool for role in roles})
    reward_fn = ProcurementRewardManager(config.procurement.trace_dir, phase="train")
    val_reward_fn = ProcurementRewardManager(config.procurement.trace_dir, phase="val")
    trainer = ProcurementTrainer(
        config=config, tokenizer=tokenizer,
        role_worker_mapping={role: ray.remote(ProcurementActorRolloutRefWorker) for role in roles},
        resource_pool_manager=resources, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn, val_reward_fn=val_reward_fn)
    reward_fn.trainer = val_reward_fn.trainer = trainer
    trainer.init_workers()
    trainer.fit()


def main():
    upstream = activate_upstream()
    import hydra
    import ray
    from omegaconf import OmegaConf
    from procurement_rl.training.contracts import validate_config

    @hydra.main(config_path=str(upstream / "verl/trainer/config"),
                config_name="ppo_trainer", version_base=None)
    def entry(config):
        validate_config(OmegaConf.to_container(config, resolve=True))
        # Workers need both project adapters and the pinned bundled package.
        package_root = str(Path(__file__).resolve().parents[2])
        paths = os.pathsep.join([package_root, str(upstream), os.environ.get("PYTHONPATH", "")])
        if ray.is_initialized():
            raise RuntimeError("Start training in a fresh process so the pinned Ray environment is applied")
        ray.init(runtime_env={"env_vars": {
            "PYTHONPATH": paths, "SEARCH_R1_ROOT": str(upstream),
            "TOKENIZERS_PARALLELISM": "false", "NCCL_DEBUG": "WARN",
            "PYTHONHASHSEED": str(config.procurement.seed)}})
        try:
            ray.get(ray.remote(num_cpus=1)(run_training).remote(config))
        finally:
            ray.shutdown()
    entry()


if __name__ == "__main__":
    main()
