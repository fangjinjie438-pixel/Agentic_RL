import copy
import unittest

from procurement_rl.metrics import estimate_budget, paired_comparison, summarize


def result(task_id, success, family="family-a", kind="model_evaluation"):
    return {"task_id": task_id, "family_id": family, "kind": kind,
            "task_sha256": task_id + "-hash", "corpus_sha256": "corpus-hash",
            "verifier_sha256": "verifier-hash", "evaluation_protocol": {"max_turns": 3},
            "search_calls": 2, "generated_tokens": 30, "budget_exhausted": False,
            "dropped_documents": 0, "elapsed_seconds": 1,
            "reward_metrics": {"success": success, "format_valid": True,
                               "constraints_satisfied": success, "optimal": success,
                               "evidence_valid": True, "reward": float(success),
                               "reason": "success" if success else "not_optimal"}}


class MetricsTests(unittest.TestCase):
    def test_summary_does_not_mix_demo_and_model_runs(self):
        summary = summarize([result("a", True), result("b", False)])
        self.assertEqual(summary["success"], 0.5)
        self.assertEqual(summary["failure_reasons"], {"not_optimal": 1})
        self.assertIsNone(summarize([])["success"])
        with self.assertRaises(ValueError):
            summarize([result("a", True), result("b", True, kind="scripted_environment_check")])

    def test_bootstrap_pairs_by_task_and_clusters_by_family(self):
        left = [result("a", False), result("b", False), result("c", True, family="family-b")]
        right = [result("c", True, family="family-b"), result("a", True), result("b", True)]
        comparison = paired_comparison(left, right, samples=200, seed=7)
        self.assertEqual(comparison["tasks"], 3)
        self.assertEqual(comparison["families"], 2)
        self.assertEqual(comparison["success_difference_right_minus_left"], 2 / 3)
        self.assertEqual(comparison["confidence_interval"], [0.0, 1.0])
        self.assertEqual(comparison, paired_comparison(left, right, samples=200, seed=7))

    def test_mismatched_data_protocol_or_verifier_fails_closed(self):
        baseline = result("a", True)
        for field in ("family_id", "task_sha256", "corpus_sha256", "verifier_sha256", "kind", "evaluation_protocol"):
            changed = copy.deepcopy(baseline)
            changed[field] = "changed"
            with self.assertRaisesRegex(ValueError, "Incomparable"):
                paired_comparison([baseline], [changed], samples=5)
        with self.assertRaises(ValueError):
            paired_comparison([baseline], [result("b", True)], samples=5)
        with self.assertRaises(ValueError):
            paired_comparison([baseline, baseline], [baseline, baseline], samples=5)

    def test_budget_uses_whole_instance_rate_and_measured_step_time(self):
        estimate = estimate_budget(18, 200, 3, repetitions=3)
        self.assertEqual(estimate["estimated_seconds"], 10800)
        self.assertEqual(estimate["estimated_cost_cny"], 9)
        with self.assertRaises(ValueError):
            estimate_budget(float("nan"), 200, 3)


if __name__ == "__main__":
    unittest.main()
