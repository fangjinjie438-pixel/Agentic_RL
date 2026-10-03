"""Reference implementation of IGPO v2 Equations 4–6, without GPU dependencies.

IG rewards are pooled over all *existing* interaction turns within each prompt
group; outcome rewards are normalized separately. Sample standard deviation is
used, matching the pinned Search-R1 GRPO baseline. Singleton/constant pools yield
zero. A final answer turn has no information-gain reward.
"""

from collections import defaultdict
import math


def standardize(values, epsilon=1e-6):
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Rewards must be finite")
    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return [(value - mean) / (std + epsilon) for value in values]


def turn_advantages(group_ids, outcome_rewards, interaction_gains, *, mode="igpo",
                    ig_weight=1.0, outcome_weight=1.0, epsilon=1e-6):
    """Return one value per interaction plus one terminal decision per trajectory.

``interaction_gains[i]`` lists actual environment interactions only, including
invalid-action feedback. A terminal generation is represented by the last value.
The trajectory_ig ablation broadcasts first-turn cumulative return without
renormalizing it. Nothing is padded into normalization pools.
    """
    if mode not in {"grpo", "igpo", "trajectory_ig"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if not (len(group_ids) == len(outcome_rewards) == len(interaction_gains)):
        raise ValueError("Batch lengths differ")
    groups = defaultdict(list)
    for i, key in enumerate(group_ids):
        groups[str(key)].append(i)
    result = [[] for _ in group_ids]
    for indices in groups.values():
        outcomes = standardize([outcome_rewards[i] for i in indices], epsilon)
        pooled = [gain for i in indices for gain in interaction_gains[i]]
        normalized_gains = iter(standardize(pooled, epsilon))
        for i, outcome in zip(indices, outcomes):
            n_turns = len(interaction_gains[i])
            gains = [next(normalized_gains) for _ in range(n_turns)]
            current = outcome_weight * outcome
            reversed_returns = [current]
            for gain in reversed(gains):
                current += ig_weight * gain if mode != "grpo" else 0.0
                reversed_returns.append(current)
            returns = reversed_returns[::-1]
            if mode == "trajectory_ig":
                returns = [returns[0]] * len(returns)
            result[i] = returns
    return result


def token_advantages(turn_values, token_turn_ids):
    """Map to model decisions; negative IDs identify environment/padding tokens."""
    result = []
    for turn_id in token_turn_ids:
        if turn_id < 0:
            result.append(0.0)
        elif turn_id >= len(turn_values):
            raise ValueError("Token references a missing turn")
        else:
            result.append(turn_values[turn_id])
    return result

