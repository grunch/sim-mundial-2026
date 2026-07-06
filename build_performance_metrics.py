#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute and store performance metrics (xG proxies) into the dataset.

Phase 1: purely additive. For every group-stage match it stores
``xg_proxy_for``/``xg_proxy_against``, and for every team with detailed
matches it stores a ``performance_metrics`` block. It does NOT touch
``derived_metrics`` or match probabilities.

Idempotent: recomputes everything from the raw match stats, so it is safe to
re-run after adding new matches. Run from the repository root:

    python3 build_performance_metrics.py
"""
import json

from performance import (BETA, COVERAGE_THRESHOLD, XG_PER_SHOT_IN_BOX,
                         XG_PER_SHOT_OUT_BOX, compute_team_performance,
                         xg_proxy)

PATH = "worldcup2026_r32_dataset.json"

# FIFA ranking points (official, 2026-06-11) for opponents that did NOT reach
# the Round of 32 and are therefore absent from the dataset. Same reference
# date and scale as the qualified teams' points_official_2026_06_11.
# Source: football-ranking.com / FIFA.
EXTERNAL_OPPONENT_FIFA_POINTS = {
    "QAT": 1438.82,
    "TUR": 1605.73,
    "TUN": 1476.41,
    "IRQ": 1419.24,
    "JOR": 1387.74,
    "PAN": 1522.88,
    "CUW": 1294.77,
    "UZB": 1458.73,
    "KOR": 1591.63,
    "SCO": 1503.34,
    "HAI": 1277.67,
    "KSA": 1426.71,
    "IRN": 1619.58,
}


def opponent_points_lookup(data):
    """Map every team code to its pre-tournament FIFA points.

    Qualified teams come from the dataset; non-qualified opponents come from
    ``EXTERNAL_OPPONENT_FIFA_POINTS``.
    """
    points = {t["code"]: t["fifa_ranking"]["points_official_2026_06_11"]
              for t in data["teams"]}
    points.update(EXTERNAL_OPPONENT_FIFA_POINTS)
    return points


def build(data):
    """Mutate ``data`` in place with performance intermediates; return it."""
    data["meta"]["performance_model"] = {
        "phase": 1,
        "xg_per_shot_in_box": XG_PER_SHOT_IN_BOX,
        "xg_per_shot_out_box": XG_PER_SHOT_OUT_BOX,
        "beta_form_correction": BETA,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "opponent_adjustment": ("planned (Phase 2): the expected xGD given the "
                                "opponent's pre-tournament FIFA points is the "
                                "baseline, and the residual (actual - expected) "
                                "is the strength-of-schedule-adjusted signal. "
                                "opponent_fifa_points is captured per match now."),
        "note": ("Phase 1 stores xG-proxy intermediates and per-match "
                 "opponent_fifa_points only; match probabilities are unchanged. "
                 "The form correction activates per team once matches_covered "
                 ">= coverage_threshold."),
    }
    points = opponent_points_lookup(data)
    for t in data["teams"]:
        perf = t["world_cup_2026_performance"]
        matches = perf.get("group_stage_matches", [])
        for m in matches:
            m["xg_proxy_for"] = xg_proxy(m["team_stats"]["shots_in_box"],
                                         m["team_stats"]["shots_out_box"])
            m["xg_proxy_against"] = xg_proxy(m["opponent_stats"]["shots_in_box"],
                                             m["opponent_stats"]["shots_out_box"])
            opp = m["opponent_code"]
            if opp not in points:
                raise KeyError(
                    f"No FIFA points for opponent {opp}; add it to "
                    "EXTERNAL_OPPONENT_FIFA_POINTS in build_performance_metrics.py")
            m["opponent_fifa_points"] = points[opp]
        if matches:
            t["performance_metrics"] = compute_team_performance(
                matches, perf["form_raw_index"], perf["points"],
                perf["goal_difference"])
        else:
            t.pop("performance_metrics", None)
    return data


def main():
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)
    build(data)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    covered = sum(1 for t in data["teams"] if "performance_metrics" in t)
    active = sum(1 for t in data["teams"]
                 if t.get("performance_metrics", {}).get("correction_active"))
    print(f"Performance metrics computed for {covered} teams "
          f"({active} with the form correction active).")


if __name__ == "__main__":
    main()
