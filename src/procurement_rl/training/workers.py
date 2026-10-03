"""Small worker adapter for experiment seeds; upstream files stay unchanged."""

import random
import numpy as np
import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.actor.dp_actor import DataParallelPPOActor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class AuditedPPOActor(DataParallelPPOActor):
    """Keep the pinned loss, but reject nonfinite gradients before optimizer.step."""

    def _optimizer_step(self):
        if isinstance(self.actor_module, FSDP):
            norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), self.config.grad_clip)
        if not torch.isfinite(norm).all():
            self.actor_optimizer.zero_grad()
            raise FloatingPointError("Nonfinite actor gradient norm; optimizer step was skipped")
        # A changed sample proves an update; an unchanged sample is inconclusive.
        samples = []
        for parameter in self.actor_module.parameters():
            if parameter.requires_grad and parameter.numel():
                flat = parameter.detach().view(-1)
                stride = max(1, flat.numel() // 256)
                samples.append((flat[::stride][:256], flat[::stride][:256].clone()))
        self.actor_optimizer.step()
        delta = max((float((after.float() - before.float()).abs().max()) for after, before in samples), default=0.0)
        self.sampled_deltas.append(delta)
        return norm

    def update_policy(self, data):
        self.sampled_deltas = []
        metrics = super().update_policy(data)
        metrics['actor/sampled_parameter_max_delta'] = self.sampled_deltas
        metrics['actor/optimizer_steps'] = len(self.sampled_deltas)
        metrics['gpu/peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        metrics['gpu/peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        return metrics


class ProcurementActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        seed = int(self.config.procurement_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        super().init_model()
        if self._is_actor:
            self.actor = AuditedPPOActor(config=self.config.actor,
                                        actor_module=self.actor_module_fsdp,
                                        actor_optimizer=self.actor_optimizer)
        # The pinned manager otherwise hardcodes rollout seed to DP rank+1000.
        # Preserve its swap/restore protocol, while using the recorded run seed.
        if self._is_rollout:
            manager = self.rollout_sharding_manager
            if manager.device_mesh is None:
                raise RuntimeError("Expected the pinned FSDP-vLLM data-parallel mesh")
            state = torch.cuda.get_rng_state()
            rank = manager.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(seed + rank)
            manager.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(state)
