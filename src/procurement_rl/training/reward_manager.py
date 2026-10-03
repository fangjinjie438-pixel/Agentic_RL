"""Rule rewards are computed from trusted rollout provenance, never model text."""

import json
from pathlib import Path
import numpy as np
import torch

from procurement_rl.reward import score_trajectory


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ProcurementRewardManager:
    def __init__(self, trace_dir, phase="train"):
        self.phase = phase
        self.trace_dir = Path(trace_dir)
        self.trainer = None
        self.pending_metrics = {}

    def __call__(self, data):
        rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        outcomes = []
        for i, trace in enumerate(data.non_tensor_batch["procurement_trace"]):
            ground_truth = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            if isinstance(ground_truth, str):
                ground_truth = json.loads(ground_truth)
            identifiers = data.non_tensor_batch.get("task_id")
            if identifiers is None:
                identifiers = data.non_tensor_batch["index"]
            task_id = identifiers[i]
            task = {"task_id": str(task_id), "ground_truth": ground_truth}
            score = score_trajectory(task, trace["assistant_messages"], trace["observations"])
            # Terminal reward goes on the last generated token, never an observation.
            positions = torch.nonzero(data.batch["turn_ids"][i] >= 0).flatten()
            if len(positions) == 0:
                raise ValueError("Trajectory has no generated decision token")
            rewards[i, positions[-1]] = float(score["reward"])
            outcomes.append(score)
            trace["reward_metrics"] = score
            trace["task_id"] = str(task_id)
        for key in ("reward", "success", "format_valid", "constraints_satisfied", "optimal", "evidence_valid"):
            self.pending_metrics[f"procurement/{self.phase}/{key}"] = float(np.mean([m[key] for m in outcomes]))
        traces = data.non_tensor_batch["procurement_trace"]
        for key in ("search_calls", "generated_tokens", "budget_exhausted", "dropped_documents"):
            self.pending_metrics[f"procurement/{self.phase}/{key}"] = float(np.mean([t[key] for t in traces]))
        return rewards

    def write_traces(self, data):
        """Called after advantage construction so the persisted record is complete."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with (self.trace_dir / f"{self.phase}.jsonl").open("a", encoding="utf-8") as stream:
            for trace in data.non_tensor_batch["procurement_trace"]:
                log = {"phase": self.phase, "step": getattr(self.trainer, "global_steps", 0),
                       **{key: value for key, value in trace.items()
                          if key not in {"prompt_token_ids", "response_token_ids"}}}
                stream.write(json.dumps(log, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
