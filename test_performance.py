#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the performance-metrics helpers (standard library only)."""
import json
import os
import unittest

import performance
from performance import xg_proxy, compute_team_performance
from build_performance_metrics import opponent_points_lookup

_DATASET_PATH = os.path.join(os.path.dirname(__file__),
                             "worldcup2026_r32_dataset.json")


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
        # xGD: for 3.35 - against 1.745 = 1.605 -> gd_adjusted 4.242 ->
        # form 7 + 0.4*4.242 = 8.697 (hand-calculated, independent of the impl).
        self.assertEqual(out["xg_diff_total"], 1.605)
        self.assertEqual(out["form_raw_adjusted"], 8.697)
        # Cross-check against the formula so a wrong beta/GD_WEIGHT is caught.
        expected = round(points + performance.GD_WEIGHT
                         * ((1 - beta) * gd + beta * out["xg_diff_total"]), 3)
        self.assertEqual(out["form_raw_adjusted"], expected)

    def test_empty_matches_is_safe(self):
        out = compute_team_performance([], form_raw_base=5.0,
                                       points=3, goal_difference=1)
        self.assertFalse(out["correction_active"])
        self.assertEqual(out["form_raw_adjusted"], 5.0)
        self.assertIsNone(out["xg_diff_per_match"])


class DatasetInvariantTest(unittest.TestCase):
    """Phase-1 invariant checks against the committed dataset."""

    @classmethod
    def setUpClass(cls):
        with open(_DATASET_PATH, encoding="utf-8") as f:
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
            if pm["matches_covered"] < pm["coverage_threshold"]:
                # Partial coverage: correction stays off and the form is the
                # original results-only value.
                self.assertFalse(pm["correction_active"])
                self.assertEqual(pm["form_raw_adjusted"], perf["form_raw_index"])
            else:
                # Full coverage: correction on and the stored adjusted form
                # matches the blend of actual GD and xGD.
                self.assertTrue(pm["correction_active"])
                gd_adjusted = ((1 - pm["beta"]) * perf["goal_difference"]
                               + pm["beta"] * pm["xg_diff_total"])
                expected = round(perf["points"]
                                 + performance.GD_WEIGHT * gd_adjusted, 3)
                self.assertEqual(pm["form_raw_adjusted"], expected)

    def test_match_log_matches_group_aggregates(self):
        # After the build recomputes aggregates, the per-match log and the
        # stored points/goal_difference must agree for fully covered teams.
        for t in self.data["teams"]:
            perf = t["world_cup_2026_performance"]
            matches = perf.get("group_stage_matches", [])
            if len(matches) != perf.get("group_matches_played"):
                continue
            points = sum({"W": 3, "D": 1, "L": 0}[m["result"]] for m in matches)
            gd = sum(m["goals_for"] - m["goals_against"] for m in matches)
            self.assertEqual(perf["points"], points, t["code"])
            self.assertEqual(perf["goal_difference"], gd, t["code"])

    def test_opponent_fifa_points_are_populated(self):
        points = opponent_points_lookup(self.data)
        for t in self.data["teams"]:
            for m in t["world_cup_2026_performance"].get("group_stage_matches", []):
                self.assertIn("opponent_fifa_points", m)
                self.assertEqual(m["opponent_fifa_points"],
                                 points[m["opponent_code"]])


if __name__ == "__main__":
    unittest.main()
