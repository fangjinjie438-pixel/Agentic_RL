"""Generate reproducible, family-disjoint simulated procurement tasks.

All amounts are integer CNY cents. Documents are public; ground_truth is private
verifier state and must never be included in the policy prompt or search index.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .reward import offer_is_feasible, optimal_offer_ids


SCHEMA_VERSION = 1
CATEGORIES = {
    "docks": {"display_hz": [30, 60, 120], "pd_w": [45, 65, 100], "usb_a_ports": [1, 2, 4]},
    "monitors": {"vertical_pixels": [1080, 1440, 2160], "refresh_hz": [60, 100, 144], "usb_c_power_w": [0, 65, 90]},
    "headsets": {"battery_hours": [12, 24, 40], "noise_cancellation": [0, 1, 1], "microphone": [0, 1, 1]},
}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identifier(rng, prefix):
    return f"{prefix}-{rng.getrandbits(64):016x}"


def _document(identifier, lines, kind, family):
    return {"id": identifier, "contents": f"[doc_id={identifier}]\n" + "\n".join(lines),
            "metadata": {"kind": kind, "family_id": family}}


def _make_family(rng, category):
    family_id = _identifier(rng, "FAM")
    scope_id = _identifier(rng, "CAT")
    quote_id = _identifier(rng, "QUOTE")
    policy_id = _identifier(rng, "POLICY")
    inline_quotes = rng.random() < 0.20
    inline_policies = inline_quotes and rng.random() < 0.40
    offers, sku_lines, quote_lines, policy_lines = [], [], [], []
    for _ in range(3):
        sku, offer_id, seller_id = (_identifier(rng, prefix) for prefix in ("SKU", "OFFER", "SELLER"))
        features = {key: rng.choice(values) for key, values in CATEGORIES[category].items()}
        policy = {"invoice": rng.random() < 0.75, "warranty_months": rng.choice([0, 12, 24, 36]),
                  "dispatch_days": rng.choice([1, 3, 7])}
        offer = {"offer_id": offer_id, "sku": sku, "seller_id": seller_id,
                 "unit_price_cents": rng.randrange(10000, 80000, 100),
                 "shipping_cents": rng.randrange(0, 3000, 100), "stock": rng.randint(2, 24),
                 "features": features, "policy": policy}
        offers.append(offer)
        sku_lines.append("SKU " + _json({"sku": sku, **features}))
        quote_lines.append("OFFER " + _json({key: value for key, value in offer.items()
                                            if key not in ("features", "policy")}))
        policy_lines.append("POLICY " + _json({"seller_id": seller_id, **policy}))
    catalog_lines = [f"Catalogue {scope_id}: {category}. These are the only authorized SKUs."] + sku_lines
    if inline_quotes:
        catalog_lines += ["All current offers (CNY cents; shipping is charged once):"] + quote_lines
        if inline_policies:
            catalog_lines += policy_lines
        else:
            catalog_lines += [f"Supplier terms: POLICY_BOOK={policy_id}"]
    else:
        catalog_lines += [f"Current prices and stock: QUOTE_BOOK={quote_id}"]
    documents = [_document(scope_id, catalog_lines, "catalogue", family_id)]
    quote_doc = scope_id
    policy_doc = scope_id if inline_policies else policy_id
    if not inline_quotes:
        quote_doc = quote_id
        documents.append(_document(quote_id, [f"Quote book {quote_id}. CNY cents; shipping charged once."]
                                   + quote_lines + [f"Supplier terms: POLICY_BOOK={policy_id}"], "quotes", family_id))
    if not inline_policies:
        documents.append(_document(policy_id, [f"Supplier policy book {policy_id}."] + policy_lines, "policies", family_id))
    return {"family_id": family_id, "scope_id": scope_id, "category": category, "offers": offers,
            "documents": documents, "quote_doc": quote_doc, "policy_doc": policy_doc}


def _make_task(rng, family, split, task_index):
    for _ in range(500):
        anchor = rng.choice(family["offers"])
        quantity = rng.randint(1, anchor["stock"])
        # No supplier constraints in roughly 35% of tasks: policy retrieval can
        # then be skipped. A high warranty, invoice, or dispatch requirement
        # makes the policy book necessary for the remaining tasks.
        needs_policy = rng.random() >= 0.35
        minimums = {}
        for key, values in CATEGORIES[family["category"]].items():
            minimums[key] = rng.choice([value for value in sorted(set(values)) if value <= anchor["features"][key]])
        requirements = {"min_features": minimums,
                        "invoice_required": bool(needs_policy and anchor["policy"]["invoice"] and rng.random() < 0.7),
                        "min_warranty_months": rng.choice([0, anchor["policy"]["warranty_months"]]) if needs_policy else 0,
                        "max_dispatch_days": anchor["policy"]["dispatch_days"] if needs_policy else None}
        budget = anchor["unit_price_cents"] * quantity + anchor["shipping_cents"] + rng.randrange(0, 30001, 100)
        truth = {"scope_id": family["scope_id"], "quantity": quantity, "budget_cents": budget,
                 "requirements": requirements, "offers": family["offers"]}
        winners = optimal_offer_ids(truth)
        # A task needs at least one realistic decoy rejected by its constraints.
        if len(winners) != 1 or all(offer_is_feasible(offer, truth) for offer in family["offers"]):
            continue
        required = {family["scope_id"], family["quote_doc"]}
        if requirements["invoice_required"] or requirements["min_warranty_months"] > 0 or requirements["max_dispatch_days"] is not None:
            required.add(family["policy_doc"])
        truth.update({"gold_offer_id": winners[0], "required_evidence_ids": sorted(required),
                      "document_hashes": {doc["id"]: _digest(doc["contents"]) for doc in family["documents"]}})
        constraints = "; ".join(f"{key}>={value}" for key, value in minimums.items())
        if requirements["invoice_required"]:
            constraints += "; enterprise invoice required"
        if requirements["min_warranty_months"]:
            constraints += f"; warranty>={requirements['min_warranty_months']} months"
        if requirements["max_dispatch_days"] is not None:
            constraints += f"; dispatch<={requirements['max_dispatch_days']} days"
        prompt = (f"Simulated procurement. Buy {quantity} {family['category']} from catalogue {family['scope_id']}. "
                  f"Requirements: {constraints}. Total budget including one shipping charge: {budget} CNY cents. "
                  "Select the lowest-total-cost eligible offer in this catalogue. Stock must cover the quantity. "
                  "Use integer cents; do not infer missing supplier terms. Search references discovered in documents "
                  "as needed. Cite evidence for the specifications, prices, stock and any required supplier terms.\n"
                  "Use <search>query</search> to retrieve documents. After receiving results, continue searching "
                  "or finish with <evidence>doc_id,doc_id</evidence><answer>offer_id</answer>. "
                  "Do not invent document IDs, offers, facts or tool results.")
        return {"task_id": f"{split}-{task_index:06d}", "split": split, "family_id": family["family_id"],
                "prompt": prompt, "ground_truth": truth}
    raise RuntimeError("Could not generate a unique feasible task for this family")


def build_dataset(output_dir: Path, train_size=2000, val_size=200, test_size=400, seed=42):
    """Write deterministic JSONL tasks/corpora and a hash manifest.

    Dataset defaults contain 333 disjoint families (999 SKUs). Tiny smoke
    datasets use fewer families. Existing files are replaced deterministically.
    """
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in sizes.values()):
        raise ValueError("Split sizes must be nonnegative integers")
    if not sum(sizes.values()):
        raise ValueError("At least one split must contain tasks")
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = sum(sizes.values())
    families_remaining = min(333, total)
    categories = list(CATEGORIES)
    manifest = {"schema_version": SCHEMA_VERSION, "seed": seed, "currency": "CNY cents",
                "simulation_only": True, "splits": {}, "files": {}, "sku_count": 0}
    nonempty_remaining = sum(bool(size) for size in sizes.values())
    for split, size in sizes.items():
        directory = output_dir / split
        directory.mkdir(exist_ok=True)
        if size:
            family_count = (families_remaining if nonempty_remaining == 1 else
                            max(1, min(size, families_remaining - nonempty_remaining + 1,
                                       round(min(333, total) * size / total))))
            families_remaining -= family_count
            nonempty_remaining -= 1
        else:
            family_count = 0
        families = [_make_family(rng, categories[index % len(categories)]) for index in range(family_count)]
        tasks, signatures = [], set()
        for index in range(size):
            family = families[index % len(families)]
            for _ in range(500):
                task = _make_task(rng, family, split, index)
                truth = task["ground_truth"]
                signature = _json([truth["scope_id"], truth["quantity"], truth["budget_cents"], truth["requirements"]])
                if signature not in signatures:
                    signatures.add(signature)
                    tasks.append(task)
                    break
            else:
                raise RuntimeError("Unable to generate distinct task constraints")
        documents = [doc for family in families for doc in family["documents"]]
        for filename, values in (("tasks.jsonl", tasks), ("corpus.jsonl", documents)):
            content = "".join(_json(value) + "\n" for value in values)
            (directory / filename).write_bytes(content.encode("utf-8"))
            manifest["files"][f"{split}/{filename}"] = _digest(content)
        manifest["splits"][split] = {"tasks": len(tasks), "families": len(families), "documents": len(documents),
                                    "required_document_counts": {str(count): sum(len(t["ground_truth"]["required_evidence_ids"]) == count for t in tasks)
                                                                 for count in (1, 2, 3)}}
        manifest["sku_count"] += 3 * len(families)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, args.train_size, args.val_size, args.test_size, args.seed), indent=2))


if __name__ == "__main__":
    main()
