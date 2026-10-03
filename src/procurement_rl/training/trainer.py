"""Procurement driver for [Search-R1 bundled veRL], without upstream patches.

Resource pools, FSDP workers, log-prob RPCs, PPO loss and model checkpoints are
inherited from the pinned backend. The project owns its rollout, validation,
reward and advantage flow; no process-wide monkey patch changes other trainers.
"""

from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, reduce_metrics
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from procurement_rl.training.advantages import turn_advantages, token_advantages
from procurement_rl.training.contracts import group_identifiers
from procurement_rl.training.generation import ProcurementGenerationManager
from procurement_rl.training.scoring import score_information_gains


class ProcurementTrainer(RayPPOTrainer):
    def _create_dataloader(self):
        # The old dataset uses one length for initial padding/truncation. Keep
        # it at max_start_length; the rollout owns the larger trajectory budget.
        def dataset(split):
            data = RLHFDataset(
                parquet_files=self.config.data[f"{split}_files"], tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_start_length,
                filter_prompts=False, return_raw_chat=False, truncation="error")
            count = self.config.data.get(f"{split}_data_num")
            if count is not None:
                if int(count) < 1 or int(count) > len(data):
                    raise ValueError(f"{split}_data_num={count} is outside dataset size {len(data)}")
                # Hold the selected subset fixed when the training seed changes.
                data.dataframe = data.dataframe.sample(int(count), random_state=42).reset_index(drop=True)
            return data

        self.train_dataset, self.val_dataset = dataset("train"), dataset("val")
        rng = torch.Generator().manual_seed(int(self.config.procurement.seed))
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config.data.train_batch_size,
            shuffle=self.config.data.shuffle_train_dataloader, drop_last=True,
            generator=rng, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.config.data.val_batch_size,
            shuffle=False, drop_last=False, collate_fn=collate_fn)
        if len(self.train_dataloader) < 1 or len(self.val_dataloader) < 1:
            raise ValueError("Training and validation must each contain a nonempty batch")
        capacity = len(self.train_dataloader) * self.config.trainer.total_epochs
        requested = self.config.trainer.total_training_steps
        self.total_training_steps = capacity if requested is None else int(requested)
        if not 1 <= self.total_training_steps <= capacity:
            raise ValueError("total_training_steps must be positive and fit within total_epochs")
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
            self.config.critic.optim.total_training_steps = self.total_training_steps

    def _generation_manager(self, validation=False):
        data = self.config.data
        generation = SimpleNamespace(
            max_turns=self.config.max_turns, max_start_length=data.max_start_length,
            max_prompt_length=data.max_prompt_length, max_response_length=data.max_response_length,
            max_obs_length=data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            search_url=self.config.retriever.url, topk=self.config.retriever.topk)
        return ProcurementGenerationManager(
            self.tokenizer, self.actor_rollout_wg, generation, self.config.procurement,
            self.config.actor_rollout_ref.rollout, is_validation=validation)

    @staticmethod
    def _worker_inputs(batch):
        # Keep private verifier facts/provenance on the driver. Only explicit
        # teacher-forcing batches in scoring.py contain the gold answer tokens.
        return batch.select(non_tensor_batch_keys=[], deepcopy=True)

    def _rollout(self, batch, manager):
        prompts = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        output = manager.run_llm_loop(prompts, initial_input_ids=None)
        return batch.union(output)

    def _validate(self):
        manager = self._generation_manager(validation=True)
        totals, count = defaultdict(float), 0
        for batch_dict in self.val_dataloader:
            batch = self._rollout(DataProto.from_single_dict(batch_dict), manager)
            rewards = self.val_reward_fn(batch).sum(-1)
            if hasattr(self.val_reward_fn, "write_traces"):
                self.val_reward_fn.write_traces(batch)
            size = len(batch)
            count += size
            # Weight by trajectories, including a final partial validation batch.
            for key, value in self.val_reward_fn.pending_metrics.items():
                totals[key] += size * value
            for source, reward in zip(batch.non_tensor_batch["data_source"], rewards.tolist()):
                totals[f"val/test_score/{source}"] += reward
                totals[f"val/count/{source}"] += 1
        if not count:
            raise ValueError("Validation produced no trajectories")
        metrics = {key: value / count for key, value in totals.items() if key.startswith("procurement/")}
        for key, value in totals.items():
            if key.startswith("val/test_score/"):
                metrics[key] = value / totals[key.replace("test_score", "count")]
        metrics["procurement/val/examples"] = count
        return metrics

    def _advantages(self, batch, gains):
        cfg = self.config.procurement
        values = turn_advantages(
            batch.non_tensor_batch["uid"], batch.batch["token_level_rewards"].sum(-1).tolist(),
            gains, mode=cfg.mode, ig_weight=float(cfg.ig_weight), outcome_weight=float(cfg.outcome_weight))
        rows = [token_advantages(value, ids.tolist())
                for value, ids in zip(values, batch.batch["turn_ids"])]
        advantages = torch.tensor(rows, dtype=torch.float32)
        if not torch.isfinite(advantages).all():
            raise FloatingPointError("Nonfinite procurement advantages")
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages.clone()
        response_length = batch.batch["responses"].shape[-1]
        loss_mask = batch.batch["info_mask"][:, -response_length:]
        expected = (batch.batch["turn_ids"] >= 0).long()
        if not torch.equal(loss_mask, expected):
            raise ValueError("Loss mask disagrees with sampled-token provenance")
        if torch.any(loss_mask > batch.batch["attention_mask"][:, -response_length:]):
            raise ValueError("Padding cannot receive policy loss")
        batch.batch["loss_mask"] = loss_mask
        for value, trace in zip(values, batch.non_tensor_batch["procurement_trace"]):
            trace["turn_advantages"] = value
        return batch

    def _log(self, metrics):
        self.logger.log(data=metrics, step=self.global_steps)
        path = Path(self.config.procurement.trace_dir).parent / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": self.global_steps, **metrics}, allow_nan=False) + "\n")

    def fit(self):
        self.global_steps = 0  # rollout/update iterations; each can contain several PPO optimizer steps
        if self.config.trainer.get("val_before_train", True) or self.config.trainer.get("val_only", False):
            self._log(self._validate())
        if self.config.trainer.get("val_only", False):
            return
        manager = self._generation_manager()
        last_validation, last_checkpoint = None, None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    break
                self.global_steps += 1
                metrics, timing = {"train/epoch": epoch}, {}
                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    group_identifiers(len(batch), self.global_steps), dtype=object)
                batch = batch.repeat(self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                with _timer("step", timing):
                    with _timer("gen", timing):
                        batch = self._rollout(batch, manager)
                    self._balance_batch(batch, metrics)
                    batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
                    with _timer("old_log_prob", timing):
                        batch = batch.union(self.actor_rollout_wg.compute_log_prob(self._worker_inputs(batch)))
                    with _timer("ref", timing):
                        batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(self._worker_inputs(batch)))
                    for key in ("old_log_probs", "ref_log_prob"):
                        if not torch.isfinite(batch.batch[key]).all():
                            raise FloatingPointError(f"Nonfinite {key}; actor update was not started")
                    with _timer("gold_score", timing):
                        if self.config.procurement.mode == "grpo":
                            gains = [[0.0] * len(t["observation_ends"])
                                     for t in batch.non_tensor_batch["procurement_trace"]]
                        else:
                            gains, gain_metrics = score_information_gains(
                                batch, self.actor_rollout_wg, self.tokenizer, self.config)
                            metrics.update(gain_metrics)
                    with _timer("adv", timing):
                        batch.batch["token_level_scores"] = self.reward_fn(batch)
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
                        batch = self._advantages(batch, gains)
                        if hasattr(self.reward_fn, "write_traces"):
                            self.reward_fn.write_traces(batch)
                        metrics.update(self.reward_fn.pending_metrics)
                    with _timer("update_actor", timing):
                        actor_output = self.actor_rollout_wg.update_actor(self._worker_inputs(batch))
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing):
                            metrics.update(self._validate())
                        last_validation = self.global_steps
                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing):
                            self._save_checkpoint()
                        last_checkpoint = self.global_steps
                mask = batch.batch["loss_mask"].bool()
                metrics["train/decision_tokens"] = int(mask.sum())
                metrics["train/advantage_mean"] = float(batch.batch["advantages"][mask].mean())
                metrics["train/advantage_abs_mean"] = float(batch.batch["advantages"][mask].abs().mean())
                groups = defaultdict(list)
                for uid, advantage in zip(batch.non_tensor_batch["uid"], batch.batch["advantages"]):
                    groups[str(uid)].append(bool(torch.any(advantage != 0)))
                metrics["train/zero_advantage_group_fraction"] = sum(not any(v) for v in groups.values()) / len(groups)
                metrics["train/observation_loss_tokens"] = int((batch.batch["loss_mask"] * (batch.batch["turn_ids"] < 0)).sum())
                metrics.update({f"timing_s/{key}": float(value) for key, value in timing.items()})
                self._log(metrics)
            if self.global_steps >= self.total_training_steps:
                break
        if self.global_steps != self.total_training_steps:
            raise RuntimeError("Training ended before the configured optimizer update count")
        if last_validation != self.global_steps:
            self._log(self._validate())
        if self.config.trainer.save_freq > 0 and last_checkpoint != self.global_steps:
            self._save_checkpoint()
