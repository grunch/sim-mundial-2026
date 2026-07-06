#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the performance-metrics helpers (standard library only)."""
import json
import unittest

import performance
from performance import xg_proxy, compute_team_performance
from build_performance_metrics import opponent_points_lookup


def _match(gf, ga, in_box, out_box, opp_in_box, opp_out_box):
    return {
        "goals_for": gf, "goals_against": ga,
        "team_stats": {"shots_in_box": in_box, "shots_out_box": out_box},
        "opponent_stats": {"shots_in_box": opp_in_box, "shots_out_box": opp_out_box},
    }


class XgProxyTest(unittest.TestCase):
    def test_formula_matches_hand_calculation(self):
        # 9 in box, 7 out of box -> 0.13*9 + 0.035*7 = 1.415
        self.assertEqual(xg_proxy(9, 7), 1.415)
        # 1 in box, 2 out of box -> 0.13 + 0.07 = 0.20
        self.assertEqual(xg_proxy(1, 2), 0.2)

    def test_in_box_worth_more_than_out_box(self):
        self.assertGreater(xg_proxy(1, 0), xg_proxy(0, 1))

    def test_rejects_negative_counts(self):
        with self.assertRaises(ValueError):
            xg_proxy(-1, 2)


class TeamPerformanceTest(unittest.TestCase):
    def test_below_threshold_keeps_form_unchanged(self):
        # One match only: correction must stay off, form untouched.
        # team 9 in / 7 out -> xgF 1.415 ; opp 1 in / 2 out -> xgA 0.20
        matches = [_match(2, 0, 9, 7, 1, 2)]
        out = compute_team_performance(matches, form_raw_base=9.4,
                                       points=7, goal_difference=6)
        self.assertFalse(out["correction_active"])
        self.assertEqual(out["form_raw_adjusted"], 9.4)
        self.assertEqual(out["xg_for_total"], 1.415)
        self.assertEqual(out["xg_against_total"], 0.2)
        self.assertEqual(out["xg_diff_total"], 1.215)
        self.assertEqual(out["matches_covered"], 1)

    def test_full_coverage_activates_correction(self):
        # Three matches, full coverage -> correction on, form recomputed.
        matches = [
            _match(2, 0, 9, 7, 1, 2),
            _match(1, 1, 5, 3, 8, 5),
            _match(3, 0, 8, 4, 2, 2),
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
                                 xg_proxy(m["team_stats"]["shots_in_box"],
                                          m["team_stats"]["shots_out_box"]))
                self.assertEqual(m["xg_proxy_against"],
                                 xg_proxy(m["opponent_stats"]["shots_in_box"],
                                          m["opponent_stats"]["shots_out_box"]))
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
