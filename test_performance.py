#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the performance-metrics helpers (standard library only)."""
import json
import os
import unittest

import math
from typing import ClassVar

import performance
from performance import (adjust_xgd, compute_team_performance, effective_elo,
                        fit_opponent_adjustment, minmax, strength_index,
                        xg_proxy, zscore)
from build_performance_metrics import (build, effective_form,
                                       opponent_points_lookup)

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

    def test_opponent_adjustment_changes_the_signal(self):
        # Same match log, but crediting a stronger schedule (negative slope,
        # opponent below the reference) should lift the adjusted differential.
        matches = [_match(2, 0, 9, 7, 1, 2)]
        for m in matches:
            m["opponent_fifa_points"] = 1400.0
        raw = compute_team_performance(matches, 9.4, 7, 6)
        adj = compute_team_performance(matches, 9.4, 7, 6,
                                       opp_slope=-0.004, opp_ref=1561.0)
        self.assertFalse(raw["opponent_adjusted"])
        self.assertTrue(adj["opponent_adjusted"])
        # Facing a weaker-than-reference opponent discounts the differential.
        self.assertLess(adj["xg_diff_adjusted_total"], raw["xg_diff_total"])


class OpponentAdjustmentTest(unittest.TestCase):
    def test_slope_is_negative_when_stronger_opponents_lower_xgd(self):
        rows = [(1400, 2.0), (1500, 1.0), (1600, 0.0), (1700, -1.0)]
        slope, ref = fit_opponent_adjustment(rows)
        self.assertLess(slope, 0)
        self.assertEqual(ref, 1550)

    def test_no_variance_yields_zero_slope(self):
        slope, ref = fit_opponent_adjustment([(1500, 1.0), (1500, -1.0)])
        self.assertEqual(slope, 0.0)
        self.assertEqual(ref, 1500)

    def test_adjust_xgd_discounts_weak_and_credits_strong(self):
        # slope < 0: weaker-than-ref opponent -> lower; stronger -> higher.
        self.assertLess(adjust_xgd(1.0, 1400, -0.004, 1561), 1.0)
        self.assertGreater(adjust_xgd(1.0, 1700, -0.004, 1561), 1.0)
        self.assertEqual(adjust_xgd(1.0, 1561, -0.004, 1561), 1.0)


class StrengthModelTest(unittest.TestCase):
    """Assembly validated against the spec's worked example (France)."""

    WEIGHTS: ClassVar[dict] = {"fifa": 0.470588, "value": 0.235294, "form": 0.294118}
    COEFFS: ClassVar[dict] = {"form": 40.0, "value": 25.0}

    def test_strength_index_reproduces_france(self):
        # Pedigree removed: the three remaining weights are renormalised to 1.
        si = strength_index(norm_fifa=0.9887, norm_value=1.0, norm_form=1.0,
                            weights=self.WEIGHTS)
        self.assertEqual(round(si, 2), 99.47)

    def test_effective_elo_reproduces_france(self):
        z_form = zscore(12.2, 6.4625, 2.6093)
        z_value = zscore(math.log10(1520.0), 2.5337, 0.3861)
        elo = effective_elo(1871.0, z_form, z_value, self.COEFFS)
        self.assertEqual(round(elo, 1), 2000.9)

    def test_minmax_and_zscore_edges(self):
        self.assertEqual(minmax(5, 0, 10), 0.5)
        self.assertEqual(minmax(5, 5, 5), 0.0)   # degenerate range
        self.assertEqual(zscore(5, 5, 0), 0.0)   # degenerate spread


class DatasetInvariantTest(unittest.TestCase):
    """Phase-2 invariant checks against the committed dataset."""

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
                # matches the blend of actual GD and the opponent-adjusted xGD.
                self.assertTrue(pm["correction_active"])
                self.assertTrue(pm["opponent_adjusted"])
                gd_adjusted = ((1 - pm["beta"]) * perf["goal_difference"]
                               + pm["beta"] * pm["xg_diff_adjusted_total"])
                expected = round(perf["points"]
                                 + performance.GD_WEIGHT * gd_adjusted, 3)
                self.assertEqual(pm["form_raw_adjusted"], expected)

    def test_derived_metrics_match_adjusted_form(self):
        # The stored effective_elo/strength_index must be reproducible from the
        # effective (adjusted) form and the pool aggregates -> the Phase-2
        # rewrite of derived_metrics is internally consistent.
        meta = self.data["meta"]
        weights, coeffs = (meta["weights_strength_index"],
                           meta["elo_adjustment_coeffs"])
        agg = meta["pool_aggregates"]
        fstats = agg["form_effective"]
        lv = agg["log10_value"]
        for t in self.data["teams"]:
            dm = t["derived_metrics"]
            form = effective_form(t)
            norm_form = minmax(form, fstats["min"], fstats["max"])
            si = strength_index(dm["norm_fifa"], dm["norm_value"], norm_form,
                                weights)
            log10v = math.log10(
                t["squad_value_transfermarkt"]["value_eur_millions"])
            elo = effective_elo(
                t["fifa_ranking"]["points_official_2026_06_11"],
                zscore(form, fstats["mean"], fstats["pstdev"]),
                zscore(log10v, lv["mean"], lv["pstdev"]),
                coeffs)
            self.assertAlmostEqual(dm["strength_index_0_100"], round(si, 2),
                                   places=2, msg=t["code"])
            self.assertAlmostEqual(dm["effective_elo"], round(elo, 1),
                                   places=1, msg=t["code"])

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


class BuildIdempotencyTest(unittest.TestCase):
    """The committed dataset must be a fixed point of the build."""

    def test_committed_dataset_equals_a_fresh_build(self):
        # Rebuilding from the committed dataset must reproduce it byte for byte:
        # proves the build is idempotent and that no hand-edit has drifted from
        # what build_performance_metrics.py would generate.
        with open(_DATASET_PATH, encoding="utf-8") as f:
            original = f.read()
        rebuilt = build(json.loads(original))
        serialized = json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n"
        self.assertEqual(serialized, original)


if __name__ == "__main__":
    unittest.main()
