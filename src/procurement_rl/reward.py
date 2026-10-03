"""Deterministic outcome verifier, shared by every training algorithm."""

import hashlib
from typing import Dict, List

from .protocol import parse_action


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def offer_is_feasible(offer: dict, ground_truth: dict) -> bool:
    """Evaluate explicit business constraints in integer currency units."""
    quantity = ground_truth["quantity"]
    requirements = ground_truth["requirements"]
    # Invalid verifier data is an infrastructure error, not a model failure.
    # Floats and booleans must never be silently treated as integer money.
    _integer(quantity, "quantity", minimum=1)
    _integer(ground_truth["budget_cents"], "budget_cents")
    for key in ("stock", "unit_price_cents", "shipping_cents"):
        _integer(offer[key], key)
    for key, value in requirements["min_features"].items():
        _integer(value, f"min_features.{key}")
    for key, value in offer["features"].items():
        _integer(value, f"features.{key}")
    _integer(requirements["min_warranty_months"], "min_warranty_months")
    policy = offer["policy"]
    for key in ("warranty_months", "dispatch_days"):
        _integer(policy[key], key)
    if (not isinstance(requirements["invoice_required"], bool) or
            not isinstance(policy["invoice"], bool)):
        raise ValueError("Invoice requirements and supplier invoice facts must be booleans")
    max_dispatch = requirements.get("max_dispatch_days")
    if max_dispatch is not None:
        _integer(max_dispatch, "max_dispatch_days")
    if offer["stock"] < quantity:
        return False
    if offer["unit_price_cents"] * quantity + offer["shipping_cents"] > ground_truth["budget_cents"]:
        return False
    if any(offer["features"].get(key, -1) < minimum
           for key, minimum in requirements["min_features"].items()):
        return False
    if requirements["invoice_required"] and not policy["invoice"]:
        return False
    if policy["warranty_months"] < requirements["min_warranty_months"]:
        return False
    return max_dispatch is None or policy["dispatch_days"] <= max_dispatch


def optimal_offer_ids(ground_truth: dict) -> List[str]:
    """Recompute all optima; never trust a stored gold label as a reward."""
    feasible = [offer for offer in ground_truth["offers"] if offer_is_feasible(offer, ground_truth)]
    if not feasible:
        return []
    quantity = ground_truth["quantity"]
    best = min(offer["unit_price_cents"] * quantity + offer["shipping_cents"] for offer in feasible)
    return [offer["offer_id"] for offer in feasible
            if offer["unit_price_cents"] * quantity + offer["shipping_cents"] == best]


def score_trajectory(task: dict, assistant_messages: List[str], observations: List[dict]) -> Dict:
    """Reward a valid optimal answer backed by authentic retrieved documents.

    ``observations`` MUST be server-side receipts, not model-supplied JSON. The
    each receipt must identify the corresponding search query. The content hash
    check rejects fabricated/mutated evidence; document metadata is not evidence.
    Raw task labels remain entirely outside the visible rollout.
    """
    result = {"reward": 0.0, "success": False, "format_valid": False,
              "constraints_satisfied": False, "optimal": False,
              "evidence_valid": False, "reason": "no_final_answer"}
    if not assistant_messages:
        return result
    actions = [parse_action(message) for message in assistant_messages]
    if actions[-1].kind != "answer":
        result["reason"] = actions[-1].error or "no_final_answer"
        return result
    if any(action.kind != "search" for action in actions[:-1]):
        result["reason"] = "invalid_trajectory_actions"
        return result
    # One receipt per executed search, including empty result lists.
    if len(observations) != len(actions) - 1:
        result["reason"] = "observation_count_mismatch"
        return result
    for action, receipt in zip(actions[:-1], observations):
        if (not isinstance(receipt, dict) or receipt.get("query") != action.query or
                receipt.get("invalid_action", False) or
                not isinstance(receipt.get("documents"), list) or
                ("text" in receipt and not isinstance(receipt["text"], str))):
            result["reason"] = "invalid_observation_receipt"
            return result
    result["format_valid"] = True
    truth = task["ground_truth"]
    answer = actions[-1]
    chosen = next((offer for offer in truth["offers"] if offer["offer_id"] == answer.answer), None)
    if chosen is None:
        result["reason"] = "unknown_offer"
        return result
    result["constraints_satisfied"] = offer_is_feasible(chosen, truth)
    result["optimal"] = answer.answer in optimal_offer_ids(truth)
    authentic = set()
    for receipt in observations:
        for document in receipt["documents"]:
            if not isinstance(document, dict):
                continue
            # Search-R1 HTTP results may wrap the canonical document.
            document = document.get("document", document)
            if not isinstance(document, dict):
                continue
            identifier, contents = document.get("id"), document.get("contents")
            if (isinstance(identifier, str) and isinstance(contents, str) and
                    identifier in truth["document_hashes"] and
                    ("text" not in receipt or contents in receipt["text"])):
                digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
                if digest == truth["document_hashes"][identifier]:
                    authentic.add(identifier)
    cited = set(answer.evidence)
    required = set(truth["required_evidence_ids"])
    result["evidence_valid"] = required.issubset(cited) and cited.issubset(authentic)
    result["success"] = result["constraints_satisfied"] and result["optimal"] and result["evidence_valid"]
    result["reward"] = float(result["success"])
    result["reason"] = ("success" if result["success"] else "constraints_failed"
                        if not result["constraints_satisfied"] else "not_optimal"
                        if not result["optimal"] else "unverified_evidence")
    return result
