"""CPU data preparation, bundled veRL parquet export, and scripted diagnostics."""

import argparse
import json
import re
from pathlib import Path

from .data import build_dataset
from .tracking import file_digest, read_jsonl, verified_dataset, write_json


def export_parquet(data_dir, output_dir=None):
    """Write the schema consumed by the pinned Search-R1 RLHFDataset.

Private facts live only in reward_model.ground_truth. JSON encoding avoids
Arrow rewriting category-specific nested dictionaries and nullable integers.
    """
    try:
        import pandas as pd
        import pyarrow  # noqa: F401; fail before partially exporting files
    except ImportError as error:
        raise RuntimeError("Parquet export needs the project's [data] optional dependencies") from error
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir is not None else data_dir
    frozen = verified_dataset(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "format": "Search-R1 bundled veRL RLHFDataset",
                "dataset_manifest_sha256": file_digest(data_dir / "manifest.json"),
                "source_files_sha256": frozen["files"], "files": {}, "rows": {}}
    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info", "task_id"]
    for split in ("train", "val", "test"):
        rows = []
        for index, task in enumerate(read_jsonl(data_dir / split / "tasks.jsonl")):
            rows.append({"data_source": "procurement", "prompt": [{"role": "user", "content": task["prompt"]}],
                         "ability": "procurement", "task_id": task["task_id"],
                         "reward_model": {"style": "rule", "ground_truth": json.dumps(task["ground_truth"],
                                                                                       ensure_ascii=False, sort_keys=True)},
                         "extra_info": {"index": index, "task_id": task["task_id"],
                                        "family_id": task["family_id"], "split": split}})
        target = output_dir / f"{split}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        pd.DataFrame(rows, columns=columns).to_parquet(temporary, engine="pyarrow", index=False)
        temporary.replace(target)
        manifest["files"][target.name] = file_digest(target)
        manifest["rows"][split] = len(rows)
    write_json(output_dir / "parquet_manifest.json", manifest)
    return manifest


class ScriptedProcurementPolicy:
    """Deterministic public-document parser for environment checks only.

This is deliberately not a language model. It parses the fixed simulator's
grammar and arithmetic, and uses no task dictionary or private verifier facts.
Its integer sequence represents Unicode characters, never model tokens.
    """

    metadata = {"backend": "scripted_public_document_parser", "model": None,
                "warning": "Environment diagnostic only; does not measure language-model ability",
                "context_units": "Unicode characters"}

    @staticmethod
    def encode(text):
        return [ord(character) for character in text]

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        return "".join(chr(identifier) for identifier in ids)

    def start(self, prompt):
        # The sole initial input is exactly the same public prompt the LM sees.
        return self.encode(prompt + "\n")

    def generate(self, context_ids, max_new_tokens):
        text = self.decode(context_ids)
        scope = re.search(r"from catalogue ([A-Za-z0-9-]+)\.", text).group(1)
        document_ids = list(dict.fromkeys(re.findall(r"\[doc_id=([^\]]+)\]", text)))
        if scope not in document_ids:
            message = f"<search>{scope}</search>"
        else:
            quantity = int(re.search(r"Buy (\d+) ", text).group(1))
            budget = int(re.search(r"Total budget including one shipping charge: (\d+) CNY cents", text).group(1))
            requirements = re.search(r"Requirements: (.*?)\. Total budget", text).group(1)
            invoice_required = "enterprise invoice required" in requirements
            warranty_match = re.search(r"warranty>=(\d+) months", requirements)
            dispatch_match = re.search(r"dispatch<=(\d+) days", requirements)
            needs_policy = invoice_required or warranty_match is not None or dispatch_match is not None
            references = re.findall(r"QUOTE_BOOK=([A-Za-z0-9-]+)", text)
            if needs_policy:
                references += re.findall(r"POLICY_BOOK=([A-Za-z0-9-]+)", text)
            missing = [reference for reference in references if reference not in document_ids]
            if missing:
                message = f"<search>{missing[0]}</search>"
            else:
                sku_rows, offer_rows, policy_rows = {}, {}, {}
                for kind, raw in re.findall(r"^(SKU|OFFER|POLICY) (\{[^\n]+\})$", text, re.M):
                    value = json.loads(raw)
                    table, key = {"SKU": (sku_rows, "sku"), "OFFER": (offer_rows, "offer_id"),
                                  "POLICY": (policy_rows, "seller_id")}[kind]
                    table[value[key]] = value
                feature_minima = {key: int(value) for key, value in re.findall(r"(\w+)>=(\d+)", requirements)
                                  if key != "warranty"}
                feasible = []
                for offer in offer_rows.values():
                    sku = sku_rows.get(offer["sku"])
                    total = offer["unit_price_cents"] * quantity + offer["shipping_cents"]
                    if sku is None or offer["stock"] < quantity or total > budget:
                        continue
                    if any(sku.get(key, -1) < value for key, value in feature_minima.items()):
                        continue
                    if needs_policy:
                        policy = policy_rows.get(offer["seller_id"])
                        if policy is None:
                            continue
                        if invoice_required and not policy["invoice"]:
                            continue
                        if warranty_match and policy["warranty_months"] < int(warranty_match.group(1)):
                            continue
                        if dispatch_match and policy["dispatch_days"] > int(dispatch_match.group(1)):
                            continue
                    feasible.append((total, offer["offer_id"]))
                if not feasible:
                    raise ValueError("Scripted parser could not find a feasible offer in the actual retrieved documents")
                offer_id = min(feasible)[1]
                message = f"<evidence>{','.join(document_ids)}</evidence><answer>{offer_id}</answer>"
        encoded = self.encode(message)
        if len(encoded) > max_new_tokens:
            raise ValueError("Scripted action exceeds diagnostic character budget")
        return encoded


def demo(data_dir, split, output, limit=None):
    from .evaluate import evaluate_dataset

    # Character budgets belong to this deterministic diagnostic only. Model
    # evaluation takes real tokenizer budgets through procurement_rl.evaluate.
    protocol = {"max_turns": 3, "max_new_tokens": 2048, "max_obs_tokens": 100000,
                "max_context_tokens": 400000, "topk": 2, "temperature": 0.0, "top_p": 1.0,
                "context_units": "Unicode characters"}
    return evaluate_dataset(data_dir, split, output, ScriptedProcurementPolicy, protocol,
                            kind="scripted_environment_check", limit=limit,
                            run_config={"purpose": "environment_check", "language_model_evaluation": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate frozen simulated facts and family-disjoint splits")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--train-size", type=int, default=2000)
    prepare.add_argument("--val-size", type=int, default=200)
    prepare.add_argument("--test-size", type=int, default=400)
    prepare.add_argument("--seed", type=int, default=42)
    export = commands.add_parser("export", help="Export the pinned bundled veRL parquet schema")
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--output", type=Path)
    check = commands.add_parser("demo", help="Scripted environment check; not a model evaluation")
    check.add_argument("--data-dir", type=Path, required=True)
    check.add_argument("--split", choices=("train", "val", "test"), default="test")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"Frozen dataset output must be new or empty: {args.output}")
        result = build_dataset(args.output, args.train_size, args.val_size, args.test_size, args.seed)
    elif args.command == "export":
        result = export_parquet(args.data_dir, args.output)
    else:
        result = demo(args.data_dir, args.split, args.output, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
