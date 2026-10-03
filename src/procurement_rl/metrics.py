"""Summaries and family-clustered paired bootstrap comparisons.

Scripted environment checks and model evaluations keep distinct result kinds.
Intervals measure task-family sampling variation for these fixed runs; they do
not estimate training-seed variance.
"""

import argparse
import collections
import json
import math
import random
import statistics

from .tracking import read_jsonl


SCORE_KEYS = ("success", "format_valid", "constraints_satisfied", "optimal", "evidence_valid", "reward")


def summarize(rows):
    rows = list(rows)
    kinds = {row["kind"] for row in rows}
    if len(kinds) > 1:
        raise ValueError("Cannot aggregate scripted checks and model evaluations")
    result = {"kind": next(iter(kinds), None), "tasks": len(rows),
              "families": len({row["family_id"] for row in rows})}
    for key in SCORE_KEYS:
        result[key] = statistics.mean(float(row["reward_metrics"][key]) for row in rows) if rows else None
    for key in ("search_calls", "generated_tokens", "budget_exhausted", "dropped_documents", "elapsed_seconds"):
        values = [row[key] for row in rows if row.get(key) is not None]
        result["mean_" + key] = statistics.mean(values) if values else None
    result["failure_reasons"] = dict(collections.Counter(row["reward_metrics"]["reason"] for row in rows
                                                        if not row["reward_metrics"]["success"]))
    return result


def _quantile(values, probability):
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_comparison(left, right, samples=2000, seed=42, confidence=0.95):
    """Compare right minus left with the same tasks and evaluation protocol."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    left, right = list(left), list(right)

    def index(rows):
        result = {row["task_id"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Repeated task IDs: compare one evaluation per task at a time")
        return result

    lhs, rhs = index(left), index(right)
    if not lhs or lhs.keys() != rhs.keys():
        raise ValueError("Paired comparisons require identical nonempty task ID sets")
    groups = collections.defaultdict(list)
    identities = ("kind", "family_id", "task_sha256", "corpus_sha256", "verifier_sha256", "evaluation_protocol")
    for task_id in sorted(lhs):
        a, b = lhs[task_id], rhs[task_id]
        for key in identities:
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"Incomparable {key} for task {task_id}")
        groups[a["family_id"]].append(float(b["reward_metrics"]["success"]) - float(a["reward_metrics"]["success"]))
    if len({row["kind"] for row in left}) != 1:
        raise ValueError("A comparison must contain exactly one evaluation kind")
    # Shared catalogues induce dependence. Resample families with replacement,
    # keeping every task of each sampled family and weighting tasks equally.
    cluster_values = list(groups.values())
    cluster_sums = [sum(values) for values in cluster_values]
    cluster_sizes = [len(values) for values in cluster_values]
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        selected = rng.choices(range(len(cluster_values)), k=len(cluster_values))
        bootstrap.append(sum(cluster_sums[i] for i in selected) / sum(cluster_sizes[i] for i in selected))
    bootstrap.sort()
    tail = (1 - confidence) / 2
    return {"kind": left[0]["kind"], "tasks": len(lhs), "families": len(groups),
            "left_success": statistics.mean(float(row["reward_metrics"]["success"]) for row in left),
            "right_success": statistics.mean(float(row["reward_metrics"]["success"]) for row in right),
            "success_difference_right_minus_left": sum(cluster_sums) / len(lhs),
            "confidence": confidence, "confidence_interval": [_quantile(bootstrap, tail), _quantile(bootstrap, 1 - tail)],
            "bootstrap_samples": samples, "bootstrap_seed": seed, "resampling_unit": "family_id",
            "limitation": "Fixed-run family sampling uncertainty; does not include training-seed variance"}


def estimate_budget(seconds_per_step, steps, hourly_instance_cny, repetitions=1):
    values = (seconds_per_step, hourly_instance_cny)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Measured time and instance rate must be finite and nonnegative")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (steps, repetitions)):
        raise ValueError("Steps and repetitions must be positive integers")
    seconds = seconds_per_step * steps * repetitions
    return {"estimated_seconds": seconds, "estimated_cost_cny": seconds / 3600 * hourly_instance_cny,
            "basis": "Measured step time; excludes setup, validation, billing rounding and idle time"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("results")
    paired = commands.add_parser("paired")
    paired.add_argument("left")
    paired.add_argument("right")
    paired.add_argument("--samples", type=int, default=2000)
    paired.add_argument("--seed", type=int, default=42)
    paired.add_argument("--confidence", type=float, default=0.95)
    budget = commands.add_parser("budget")
    budget.add_argument("--seconds-per-step", type=float, required=True)
    budget.add_argument("--steps", type=int, required=True)
    budget.add_argument("--hourly-instance-cny", type=float, required=True)
    budget.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = summarize(read_jsonl(args.results))
    elif args.command == "paired":
        result = paired_comparison(read_jsonl(args.left), read_jsonl(args.right), args.samples, args.seed, args.confidence)
    else:
        result = estimate_budget(args.seconds_per_step, args.steps, args.hourly_instance_cny, args.repetitions)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
