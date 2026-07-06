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

from performance import (BETA, COVERAGE_THRESHOLD, GD_WEIGHT,
                         XG_PER_SHOT_IN_BOX, XG_PER_SHOT_OUT_BOX,
                         compute_team_performance, xg_proxy)

RESULT_POINTS = {"W": 3, "D": 1, "L": 0}

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
    "URU": 1673.07,
    "NZL": 1197.00,  # approx pre-tournament (2026-06-11); sources vary
    "CZE": 1505.00,  # approx pre-tournament (2026-06-11, rank ~40)
}


def opponent_points_lookup(data):
    """Map every team code to its pre-tournament FIFA points.

    Qualified teams come from the dataset; non-qualified opponents come from
    ``EXTERNAL_OPPONENT_FIFA_POINTS``.
    """
    points = {t["code"]: t["fifa_ranking"]["points_official_2026_06_11"]
              for t in data["teams"]}
    overlap = points.keys() & EXTERNAL_OPPONENT_FIFA_POINTS.keys()
    if overlap:
        raise ValueError(
            "EXTERNAL_OPPONENT_FIFA_POINTS overlaps qualified teams "
            f"(would overwrite real points): {sorted(overlap)}")
    points.update(EXTERNAL_OPPONENT_FIFA_POINTS)
    return points


def recompute_group_aggregates(perf, matches):
    """Rebuild the aggregate group-stage fields from the per-match log.

    The dataset's original ``points``/``goal_difference``/``form_raw_index``
    came from a different source and can disagree with the (authoritative)
    match-by-match data. Once a team's group stage is fully covered, recompute
    those fields from the matches so they cannot drift from the log. Leaves the
    aggregates untouched under partial coverage.
    """
    played = perf.get("group_matches_played")
    if not played or len(matches) != played:
        return
    points = sum(RESULT_POINTS[m["result"]] for m in matches)
    gd = sum(m["goals_for"] - m["goals_against"] for m in matches)
    perf["points"] = points
    perf["goal_difference"] = gd
    perf["points_per_match"] = round(points / played, 3)
    perf["gd_per_match"] = round(gd / played, 3)
    perf["form_raw_index"] = round(points + GD_WEIGHT * gd, 3)


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
        recompute_group_aggregates(perf, matches)
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
