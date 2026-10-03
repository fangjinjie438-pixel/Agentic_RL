import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from procurement_rl.cli import ScriptedProcurementPolicy, demo, export_parquet
from procurement_rl.data import build_dataset
from procurement_rl.evaluate import evaluate_dataset, run_trajectory, validate_model_size
from procurement_rl.retrieval import BM25Retriever
from procurement_rl.tracking import read_jsonl, verified_dataset


PROTOCOL = {"max_turns": 3, "max_new_tokens": 2048, "max_obs_tokens": 100000,
            "max_context_tokens": 400000, "topk": 2, "temperature": 0.0, "top_p": 1.0,
            "context_units": "Unicode characters"}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        build_dataset(self.data, 30, 6, 6, 42)

    def tearDown(self):
        self.temp.cleanup()

    def test_scripted_demo_retrieves_authentic_documents_without_model_claims(self):
        summary = demo(self.data, "train", self.root / "demo")
        self.assertEqual(summary["kind"], "scripted_environment_check")
        self.assertEqual(summary["tasks"], 30)
        self.assertEqual(summary["success"], 1)
        self.assertIsNone(summary["mean_generated_tokens"])
        manifest = json.loads((self.root / "demo/manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertIsNone(manifest["policy"]["model"])
        self.assertIsNone(manifest["cost"]["estimated_compute_cost"])
        shared_corpus = BM25Retriever.from_jsonl([self.data / name / "corpus.jsonl"
                                                for name in ("train", "val", "test")])
        self.assertEqual(manifest["corpus_sha256"], shared_corpus.corpus_sha256)
        rows = read_jsonl(self.root / "demo/results.jsonl")
        for row in rows:
            self.assertEqual(row["corpus_sha256"], shared_corpus.corpus_sha256)
            self.assertEqual(row["search_calls"], len(row["observations"]))
            for message, receipt in zip(row["assistant_messages"], row["observations"]):
                self.assertIn(receipt["query"], message)
                for document in receipt["documents"]:
                    self.assertIn(document["contents"], receipt["text"])

    def test_private_facts_never_enter_policy_context_or_result(self):
        class RecordingPolicy(ScriptedProcurementPolicy):
            def __init__(self):
                self.contexts = []

            def generate(self, context_ids, max_new_tokens):
                self.contexts.append(self.decode(context_ids))
                return super().generate(context_ids, max_new_tokens)

        task = read_jsonl(self.data / "train/tasks.jsonl")[0]
        poisoned = copy.deepcopy(task)
        poisoned["ground_truth"]["gold_offer_id"] = "PRIVATE-SENTINEL-NOT-A-PUBLIC-OFFER"
        poisoned["ground_truth"]["private_note"] = "PRIVATE-SENTINEL-NOT-A-PUBLIC-OFFER"
        retriever = BM25Retriever.from_jsonl(self.data / "train/corpus.jsonl")
        policy = RecordingPolicy()
        original = run_trajectory(task, ScriptedProcurementPolicy(), retriever, PROTOCOL,
                                  kind="scripted_environment_check")
        result = run_trajectory(poisoned, policy, retriever, PROTOCOL, kind="scripted_environment_check")
        self.assertEqual(original["assistant_messages"], result["assistant_messages"])
        self.assertTrue(result["reward_metrics"]["success"])
        self.assertNotIn("PRIVATE-SENTINEL", json.dumps(policy.contexts))
        self.assertNotIn("PRIVATE-SENTINEL", json.dumps(result))

    def test_dataset_tampering_fails_before_policy_load(self):
        corpus = self.data / "test/corpus.jsonl"
        corpus.write_text(corpus.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Dataset hash mismatch"):
            verified_dataset(self.data)

    def test_failure_is_recorded_and_output_is_not_overwritten(self):
        def failed_model_load():
            raise RuntimeError("deliberate load failure")

        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "deliberate load failure"):
            evaluate_dataset(self.data, "test", output, failed_model_load, PROTOCOL, hourly_cost_cny=2.0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_tasks"], 0)
        self.assertEqual(manifest["cost"]["hourly_instance_rate"], 2.0)
        self.assertGreaterEqual(manifest["cost"]["estimated_compute_cost"], 0)
        before = (output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            demo(self.data, "test", output)
        self.assertEqual(before, (output / "manifest.json").read_bytes())

    def test_context_and_observation_limits_fail_without_silent_truncation(self):
        task = read_jsonl(self.data / "test/tasks.jsonl")[0]
        retriever = BM25Retriever.from_jsonl(self.data / "test/corpus.jsonl")
        for updates in ({"max_context_tokens": 2}, {"max_obs_tokens": 20}):
            with self.assertRaises(ValueError):
                run_trajectory(task, ScriptedProcurementPolicy(), retriever, {**PROTOCOL, **updates})

    def test_zero_search_budget_does_not_execute_or_credit_tool_results(self):
        task = read_jsonl(self.data / "test/tasks.jsonl")[0]
        retriever = BM25Retriever.from_jsonl(self.data / "test/corpus.jsonl")
        trace = run_trajectory(task, ScriptedProcurementPolicy(), retriever, {**PROTOCOL, "max_turns": 0})
        self.assertTrue(trace["budget_exhausted"])
        self.assertEqual(trace["search_calls"], 0)
        self.assertEqual(trace["observations"], [])
        self.assertFalse(trace["reward_metrics"]["success"])

    def test_formal_model_floor_is_three_billion_parameters(self):
        validate_model_size(1_500_000_000, "smoke")
        validate_model_size(3_000_000_000, "formal")
        with self.assertRaisesRegex(ValueError, "at least 3B"):
            validate_model_size(1_500_000_000, "formal")

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"),
                         "Parquet optional dependencies are absent")
    def test_parquet_preserves_private_json_integer_types_and_public_prompt(self):
        import pandas as pd

        manifest = export_parquet(self.data)
        self.assertEqual(manifest["rows"], {"train": 30, "val": 6, "test": 6})
        source = read_jsonl(self.data / "train/tasks.jsonl")[0]
        row = pd.read_parquet(self.data / "train.parquet").iloc[0].to_dict()
        self.assertEqual(row["prompt"][0]["content"], source["prompt"])
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertNotIn("gold_offer_id", row["prompt"][0]["content"])
        truth = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(truth, source["ground_truth"])
        self.assertIs(type(truth["budget_cents"]), int)
        self.assertEqual(row["extra_info"]["family_id"], source["family_id"])


if __name__ == "__main__":
    unittest.main()
