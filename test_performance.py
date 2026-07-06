#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the performance-metrics helpers (standard library only)."""
import json
import unittest

import performance
from performance import xg_proxy, compute_team_performance
from build_performance_metrics import opponent_points_lookup


def _match(gf, ga, shots, sot, opp_shots, opp_sot):
    return {
        "goals_for": gf, "goals_against": ga,
        "team_stats": {"shots": shots, "shots_on_target": sot},
        "opponent_stats": {"shots": opp_shots, "shots_on_target": opp_sot},
    }


class XgProxyTest(unittest.TestCase):
    def test_formula_matches_hand_calculation(self):
        # 8 shots, 5 on target -> 0.10*3 + 0.34*5 = 2.00
        self.assertEqual(xg_proxy(8, 5), 2.0)
        # 13 shots, 4 on target -> 0.10*9 + 0.34*4 = 2.26
        self.assertEqual(xg_proxy(13, 4), 2.26)
        # 16 shots, 5 on target -> 2.80 ; 3 shots, 2 on target -> 0.78
        self.assertEqual(xg_proxy(16, 5), 2.8)
        self.assertEqual(xg_proxy(3, 2), 0.78)

    def test_all_shots_on_target(self):
        self.assertEqual(xg_proxy(4, 4), round(0.34 * 4, 3))

    def test_rejects_more_on_target_than_total(self):
        with self.assertRaises(ValueError):
            xg_proxy(3, 5)


class TeamPerformanceTest(unittest.TestCase):
    def test_below_threshold_keeps_form_unchanged(self):
        # One match only: correction must stay off, form untouched.
        matches = [_match(1, 1, 8, 5, 13, 4)]
        out = compute_team_performance(matches, form_raw_base=9.4,
                                       points=7, goal_difference=6)
        self.assertFalse(out["correction_active"])
        self.assertEqual(out["form_raw_adjusted"], 9.4)
        self.assertEqual(out["xg_for_total"], 2.0)
        self.assertEqual(out["xg_against_total"], 2.26)
        self.assertEqual(out["xg_diff_total"], -0.26)
        self.assertEqual(out["matches_covered"], 1)

    def test_full_coverage_activates_correction(self):
        # Three matches, full coverage -> correction on, form recomputed.
        matches = [
            _match(2, 0, 16, 5, 3, 2),   # xgF 2.80 xgA 0.78
            _match(1, 1, 8, 5, 13, 4),   # xgF 2.00 xgA 2.26
            _match(3, 0, 12, 6, 4, 1),   # xgF 0.6+2.04=2.64 xgA 0.30+0.34=0.64
        ]
        beta, points, gd = 0.4, 7, 6
        out = compute_team_performance(matches, form_raw_base=9.4,
                                       points=points, goal_difference=gd, beta=beta)
        self.assertTrue(out["correction_active"])
        xg_diff = out["xg_diff_total"]
        expected = round(points + performance.GD_WEIGHT
                         * ((1 - beta) * gd + beta * xg_diff), 3)
        self.assertEqual(out["form_raw_adjusted"], expected)

    def test_empty_matches_is_safe(self):
        out = compute_team_performance([], form_raw_base=5.0,
                                       points=3, goal_difference=1)
        self.assertFalse(out["correction_active"])
        self.assertEqual(out["form_raw_adjusted"], 5.0)
        self.assertEqual(out["xg_diff_per_match"], 0.0)


class DatasetInvariantTest(unittest.TestCase):
    """Phase-1 invariant checks against the committed dataset."""

    @classmethod
    def setUpClass(cls):
        with open("worldcup2026_r32_dataset.json", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_stored_xg_and_form_are_consistent(self):
        for t in self.data["teams"]:
            perf = t["world_cup_2026_performance"]
            matches = perf.get("group_stage_matches", [])
            pm = t.get("performance_metrics")
            if not matches:
                # Teams without detailed matches carry no performance block.
                self.assertIsNone(pm)
                continue
            self.assertIsNotNone(pm)
            for m in matches:
                self.assertEqual(m["xg_proxy_for"],
                                 xg_proxy(m["team_stats"]["shots"],
                                          m["team_stats"]["shots_on_target"]))
                self.assertEqual(m["xg_proxy_against"],
                                 xg_proxy(m["opponent_stats"]["shots"],
                                          m["opponent_stats"]["shots_on_target"]))
            # Phase 1: with partial coverage the correction stays off and the
            # form is byte-for-byte the original results-only value.
            if pm["matches_covered"] < pm["coverage_threshold"]:
                self.assertFalse(pm["correction_active"])
                self.assertEqual(pm["form_raw_adjusted"], perf["form_raw_index"])

    def test_opponent_fifa_points_are_populated(self):
        points = opponent_points_lookup(self.data)
        for t in self.data["teams"]:
            for m in t["world_cup_2026_performance"].get("group_stage_matches", []):
                self.assertIn("opponent_fifa_points", m)
                self.assertEqual(m["opponent_fifa_points"],
                                 points[m["opponent_code"]])


if __name__ == "__main__":
    unittest.main()
