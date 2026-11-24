"""Verify the fold planner's training-only calibration, decisions, and native evaluator calls."""

import unittest
from unittest.mock import patch

import numpy as np
from experiment import TRAIN
from folding import action, geometry
from guarded_planner import HERE, NO_MOVE, GuardedFoldPlanner


class TestGuardedFoldPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = GuardedFoldPlanner(HERE / "fold-response.npz")
        self.g = geometry(22042)
        self.actions = np.stack([action(22042 + j) for j in range(5)])

    def test_reconstructs_train_domain_without_labels(self):
        for i in range(TRAIN):
            self.assertEqual(
                self.planner.outside(geometry(22000 + i), action(22000 + i)[None]), []
            )
        for i in range(8):
            self.assertEqual(
                self.planner.outside(geometry(28000 + i), NO_MOVE[None]), []
            )
        self.assertEqual(self.planner.center_count, 200)
        self.assertEqual(self.planner.no_move_count, 8)

    def test_extrapolation_uses_every_native_score_without_predicting(self):
        self.g[2] = 1e5
        checked = []

        def native_cost(index):
            checked.append(index)
            return len(self.actions) - index

        with patch(
            "guarded_planner.FoldResponse",
            side_effect=AssertionError("unexpected model load"),
        ):
            decision = self.planner.select(self.g, self.actions, 0.1, native_cost)
        self.assertEqual(checked, list(range(len(self.actions))))
        self.assertEqual(decision["outside"], ["membrane_stiffness"])
        self.assertEqual(decision["selected"], len(self.actions) - 1)
        self.assertFalse(decision["used_model"])

    def test_one_unknown_program_forces_full_bank(self):
        self.actions[-1, 1] = 2
        decision = self.planner.select(self.g, self.actions, 0.12, float)
        self.assertEqual(decision["native_evaluations"], len(self.actions))
        self.assertEqual(decision["outside"], ["lift_fraction"])

    def test_shortlist_and_small_banks(self):
        for n in (1, 2, 5):
            decision = self.planner.select(self.g, self.actions[:n], 0.14, float)
            self.assertTrue(decision["used_model"])
            self.assertEqual(decision["native_evaluations"], min(n, 3))
            self.assertEqual(decision["selected"], min(decision["checked"]))

    def test_invalid_native_scores_and_inputs(self):
        with self.assertRaises(ValueError):
            self.planner.select(self.g, self.actions, 0.14, lambda _i: float("nan"))
        with self.assertRaises(ValueError):
            self.planner.select(self.g, np.empty((0, 3)), 0.14, float)
        invalid = self.g.copy()
        invalid[0] = np.nan
        with self.assertRaises(ValueError):
            self.planner.select(invalid, self.actions, 0.14, float)


if __name__ == "__main__":
    unittest.main()
