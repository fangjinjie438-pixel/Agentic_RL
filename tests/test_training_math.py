"""Algorithm and provenance contracts runnable without any GPU dependencies."""

import math
import unittest

from procurement_rl.training.advantages import standardize, turn_advantages, token_advantages
from procurement_rl.training.contracts import group_identifiers, pack_observation, scoring_prefixes


class AdvantageTests(unittest.TestCase):
    def test_sample_standard_deviation_and_degenerate_groups(self):
        self.assertEqual(standardize([]), [])
        self.assertEqual(standardize([3]), [0])
        self.assertEqual(standardize([3, 3]), [0, 0])
        scale = 1 / (math.sqrt(2) + 1e-6)
        self.assertEqual(standardize([0, 2]), [-scale, scale])
        with self.assertRaises(ValueError):
            standardize([0, float("nan")])

    def test_igpo_pools_existing_turns_and_terminal_has_outcome_only(self):
        gains = [[1, 3], [2], []]
        values = turn_advantages(["a", "a", "b"], [0, 1, 1], gains)
        outcome = standardize([0, 1])
        ig = standardize([1, 3, 2])
        expected = [[outcome[0] + ig[0] + ig[1], outcome[0] + ig[1], outcome[0]],
                    [outcome[1] + ig[2], outcome[1]], [0]]
        for actual, reference in zip(values, expected):
            for a, b in zip(actual, reference):
                self.assertAlmostEqual(a, b)

    def test_broadcast_uses_same_signal_and_grpo_ignores_it(self):
        args = (["a", "a"], [0, 1], [[1, 3], [2]])
        turns = turn_advantages(*args, mode="igpo")
        broadcast = turn_advantages(*args, mode="trajectory_ig")
        baseline = turn_advantages(*args, mode="grpo")
        for i in range(2):
            self.assertEqual(broadcast[i], [turns[i][0]] * len(turns[i]))
            self.assertEqual(baseline[i], [standardize([0, 1])[i]] * len(turns[i]))

    def test_reordering_keeps_groups_and_no_loss_on_environment(self):
        first = turn_advantages(["a", "b", "a", "b"], [0, 1, 1, 0], [[1], [], [], [3]])
        second = turn_advantages(["b", "a", "b", "a"], [0, 1, 1, 0], [[3], [], [], [1]])
        self.assertEqual(first, list(reversed(second)))
        self.assertEqual(token_advantages([2, 4], [0, 0, -1, -1, 1, -1]), [2, 2, 0, 0, 4, 0])
        with self.assertRaises(ValueError):
            token_advantages([2], [1])
        self.assertEqual(len(set(group_identifiers(4, 1))), 4)
        self.assertNotEqual(group_identifiers(4, 1), group_identifiers(4, 2))

    def test_prefixes_exclude_terminal_answer(self):
        self.assertEqual(scoring_prefixes([10], [11, 12, 13, 14, 99], [2, 4]),
                         [[10], [10, 11, 12], [10, 11, 12, 13, 14]])
        for ends in ([3, 2], [0], [10]):
            with self.assertRaises(ValueError):
                scoring_prefixes([10], [11, 12, 13], ends)

    def test_observation_receipts_match_whole_visible_documents(self):
        documents = [{"id": "one", "contents": "short"}, {"id": "two", "contents": "x" * 100}]
        text, exposed = pack_observation(documents, list, 65)
        self.assertEqual(exposed, documents[:1])
        self.assertIn(documents[0]["contents"], text)
        self.assertNotIn(documents[1]["contents"], text)
        with self.assertRaises(ValueError):
            pack_observation(documents[1:], list, 65)


if __name__ == "__main__":
    unittest.main()
