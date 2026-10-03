"""Evaluate a real Hugging Face causal LM with frozen local BM25 retrieval.

Only prompts, sampled tokens and actual tool observations enter the policy
context. Private task facts are supplied to the outcome verifier afterwards.
This module does not import torch/transformers until a real model is loaded.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

from .metrics import summarize
from .protocol import parse_action
from .retrieval import BM25Retriever
from .reward import score_trajectory
from .tracking import (file_digest, finish_manifest, json_digest, read_jsonl,
                       runtime_manifest, verified_dataset, write_json)
from .training.contracts import pack_observation


def trim_action(tokenizer, token_ids):
    """Preserve sampled token IDs while stopping at the first completed action."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    match = re.search(r"</(?:search|answer)>", text)
    if match:
        for count in range(1, len(token_ids) + 1):
            prefix = tokenizer.decode(token_ids[:count], skip_special_tokens=True)
            if len(prefix) >= match.end():
                return token_ids[:count], prefix
    return token_ids, text


def validate_protocol(protocol):
    for key in ("max_new_tokens", "max_obs_tokens", "max_context_tokens", "topk"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    turns = protocol["max_turns"]
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
        raise ValueError("max_turns must be a nonnegative integer")
    if protocol["topk"] > 20:
        raise ValueError("topk cannot exceed 20")
    if not math.isfinite(protocol["temperature"]) or protocol["temperature"] < 0:
        raise ValueError("temperature must be finite and nonnegative; zero means greedy")
    if not 0 < protocol["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")


def validate_model_size(parameters, purpose):
    if purpose not in {"smoke", "formal"}:
        raise ValueError("purpose must be smoke or formal")
    if purpose == "formal" and parameters < 3_000_000_000:
        raise ValueError(f"Formal evaluation requires at least 3B parameters; loaded {parameters}")


def run_trajectory(task, policy, retriever, protocol, kind="model_evaluation"):
    """Run one task; a policy only receives token IDs derived from public text.

The small policy interface is start(prompt), encode(text), decode(ids, ...),
and generate(context_ids, max_new_tokens). Its generation returns sampled IDs.
The scripted diagnostic explicitly reports no model-token measurement.
    """
    validate_protocol(protocol)
    started = time.perf_counter()
    context = list(policy.start(task["prompt"]))
    trace = {"kind": kind, "task_id": task["task_id"], "family_id": task["family_id"],
             "split": task["split"], "task_sha256": json_digest(task),
             "corpus_sha256": retriever.corpus_sha256,
             "verifier_sha256": json_digest({name: file_digest(Path(__file__).with_name(name))
                                              for name in ("reward.py", "protocol.py")}),
             "evaluation_protocol": dict(protocol), "assistant_messages": [], "observations": [],
             "search_calls": 0, "generated_tokens": 0 if kind == "model_evaluation" else None,
             "budget_exhausted": False, "dropped_documents": 0}
    for turn in range(protocol["max_turns"] + 1):
        if len(context) + protocol["max_new_tokens"] > protocol["max_context_tokens"]:
            raise ValueError("Evaluation context budget would truncate history; increase max_context_tokens")
        sampled = list(policy.generate(list(context), protocol["max_new_tokens"]))
        if len(sampled) > protocol["max_new_tokens"]:
            raise ValueError("Policy exceeded max_new_tokens")
        if trace["generated_tokens"] is not None:
            trace["generated_tokens"] += len(sampled)
        accepted, text = trim_action(policy, sampled)
        context.extend(accepted)
        trace["assistant_messages"].append(text)
        action = parse_action(text)
        if action.kind == "answer":
            break
        if turn == protocol["max_turns"]:
            trace["budget_exhausted"] = True
            break
        if action.kind == "search":
            documents = retriever.search(action.query, topk=protocol["topk"])
            # Store only complete documents actually exposed to the policy.
            observation, exposed = pack_observation(documents, policy.encode, protocol["max_obs_tokens"])
            trace["search_calls"] += 1
            trace["dropped_documents"] += len(documents) - len(exposed)
            receipt = {"query": action.query, "documents": exposed, "text": observation}
        else:
            observation = ("\n<information>Invalid action. Use <search>query</search>, or "
                           "<evidence>doc_ids</evidence><answer>offer_id</answer>.</information>\n")
            receipt = {"query": None, "documents": [], "text": observation, "invalid_action": True}
        observation_ids = policy.encode(observation)
        if len(observation_ids) > protocol["max_obs_tokens"]:
            raise ValueError("Observation exceeds max_obs_tokens")
        trace["observations"].append(receipt)
        context.extend(observation_ids)
    trace["reward_metrics"] = score_trajectory(task, trace["assistant_messages"], trace["observations"])
    trace["elapsed_seconds"] = time.perf_counter() - started
    return trace


class HuggingFacePolicy:
    """Single-device, batch-one inference for auditable smoke/formal evaluation."""

    def __init__(self, model_path, device="cuda:0", dtype="bfloat16", revision=None,
                 temperature=0.0, top_p=1.0, seed=42, purpose="smoke"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as error:
            raise RuntimeError("Real evaluation requires the Linux torch/transformers environment") from error
        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; GPU evaluation has not run")
        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad or EOS token")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, torch_dtype=getattr(torch, dtype), trust_remote_code=False,
        )
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        validate_model_size(parameters, purpose)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.temperature, self.top_p = temperature, top_p
        self.metadata = {"backend": "transformers", "model": str(model_path), "requested_revision": revision,
                         "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                         "tokenizer_resolved_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
                         "parameter_count": parameters, "device": str(self.device), "dtype": dtype,
                         "model_config_sha256": json_digest(self.model.config.to_dict()),
                         "tokenizer_vocabulary_sha256": json_digest(self.tokenizer.get_vocab()),
                         "tokenizer_chat_template_sha256": json_digest(self.tokenizer.chat_template),
                         "cuda_version": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None}
        # Local checkpoints need content hashes. For a Hub model, the resolved
        # commit above identifies its immutable snapshot and tokenizer files.
        local = Path(model_path)
        if local.is_dir():
            artifacts = [path for path in local.iterdir() if path.is_file()
                         and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}]
            self.metadata["local_artifact_sha256"] = {path.name: file_digest(path) for path in sorted(artifacts)}

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def start(self, prompt):
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                       add_generation_prompt=True, tokenize=True)
        return self.encode(prompt)

    def generate(self, context_ids, max_new_tokens):
        from transformers import StoppingCriteria, StoppingCriteriaList

        tokenizer = self.tokenizer
        start = len(context_ids)

        class EndOfAction(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, start:].tolist(), skip_special_tokens=True)
                return bool(re.search(r"</(?:search|answer)>", text))

        ids = self.torch.tensor([context_ids], device=self.device, dtype=self.torch.long)
        generation = {"max_new_tokens": max_new_tokens, "do_sample": self.temperature > 0,
                      "pad_token_id": self.tokenizer.pad_token_id,
                      "eos_token_id": self.tokenizer.eos_token_id,
                      "stopping_criteria": StoppingCriteriaList([EndOfAction()])}
        if self.temperature > 0:
            generation.update(temperature=self.temperature, top_p=self.top_p, top_k=0)
        with self.torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids), **generation)
        return output[0, start:].tolist()


def evaluate_dataset(data_dir, split, output_dir, policy_factory, protocol, *, kind="model_evaluation",
                     limit=None, run_config=None, hourly_cost_cny=None):
    """Write flushed per-task results plus a manifest, including failed runs."""
    validate_protocol(protocol)
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown dataset split")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if hourly_cost_cny is not None and (not math.isfinite(hourly_cost_cny) or hourly_cost_cny < 0):
        raise ValueError("Hourly cost must be finite and nonnegative")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    dataset_manifest = verified_dataset(data_dir)
    tasks = read_jsonl(data_dir / split / "tasks.jsonl")
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("Evaluation requires a nonempty split")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(run_config or {})
    config.update(data_dir=str(data_dir.resolve()), split=split, limit=limit, protocol=protocol)
    manifest = runtime_manifest(kind, config)
    manifest["dataset_manifest_sha256"] = file_digest(data_dir / "manifest.json")
    manifest["dataset_files_sha256"] = dataset_manifest["files"]
    manifest["task_ids"] = [task["task_id"] for task in tasks]
    write_json(output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    rows = []
    try:
        # Training serves the same frozen public catalogue across all splits.
        # Keep BM25 IDF and distractors identical during separate evaluation;
        # task families remain disjoint and private labels are never indexed.
        retriever = BM25Retriever.from_jsonl([data_dir / name / "corpus.jsonl"
                                            for name in ("train", "val", "test")])
        manifest["corpus_sha256"] = retriever.corpus_sha256
        policy = policy_factory()
        manifest["policy"] = getattr(policy, "metadata", {})
        write_json(output_dir / "manifest.json", manifest)
        if hasattr(policy, "model"):
            position_limit = getattr(policy.model.config, "max_position_embeddings", None)
            if position_limit and protocol["max_context_tokens"] > position_limit:
                raise ValueError("Requested context exceeds the loaded model's position limit")
        with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
            for task in tasks:
                row = run_trajectory(task, policy, retriever, protocol, kind=kind)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        summary = summarize(rows)
        write_json(output_dir / "summary.json", summary)
        manifest["results_sha256"] = file_digest(output_dir / "results.jsonl")
        finish_manifest(manifest, time.perf_counter() - started, hourly_cost_cny)
        write_json(output_dir / "manifest.json", manifest)
        return summary
    except BaseException as error:
        manifest["completed_tasks"] = len(rows)
        if (output_dir / "results.jsonl").exists():
            manifest["results_sha256"] = file_digest(output_dir / "results.jsonl")
        finish_manifest(manifest, time.perf_counter() - started, hourly_cost_cny, error=error)
        write_json(output_dir / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--initialization", help="Original initialization model/revision for the controlled comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--purpose", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int, default=3, help="Maximum executed search/feedback rounds")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-obs-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hourly-cost-cny", type=float, help="Actual whole-instance price per hour, not per GPU")
    args = parser.parse_args(argv)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol = {key: config[key] for key in ("max_turns", "max_new_tokens", "max_obs_tokens", "max_context_tokens",
                                             "topk", "temperature", "top_p")}
    factory = lambda: HuggingFacePolicy(args.model, args.device, args.dtype, args.revision,
                                        args.temperature, args.top_p, args.seed, args.purpose)
    summary = evaluate_dataset(args.data_dir, args.split, args.output, factory, protocol,
                               limit=args.limit, run_config=config, hourly_cost_cny=args.hourly_cost_cny)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
