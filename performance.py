#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance metrics derived from group-stage match statistics.

The results-only "form" (``form_raw = points + 0.4 * goal_difference``) is
noisy: a team can win with few chances or lose while dominating. These helpers
add a chance-quality signal, an expected-goals proxy computed from shots and
shots on target, that can later correct the goal-difference component of form.

Phase 1 (current): only compute and store these intermediates. Match
probabilities are NOT changed. The correction activates for a team only once
it has the full group stage covered with detailed stats (``COVERAGE_THRESHOLD``);
until then ``form_raw_adjusted`` equals the untouched ``form_raw`` base.
"""

# Expected goals contributed per shot by location. Shot location is the single
# strongest determinant of chance quality, so with the richer source we weight
# by zone instead of by on/off target. Coefficients approximate average
# conversion rates and are tunable (stored in meta.performance_model).
XG_PER_SHOT_IN_BOX = 0.13
XG_PER_SHOT_OUT_BOX = 0.035

# Blend weight applied to the expected goal difference when correcting form.
BETA = 0.4
# Minimum group-stage matches with detailed stats before the correction turns on.
COVERAGE_THRESHOLD = 3
# Weight of goal difference inside form_raw (mirrors form_raw = points + 0.4*gd).
GD_WEIGHT = 0.4


def xg_proxy(shots_in_box, shots_out_box):
    """Expected-goals proxy for one team in one match, from shot location.

    Location (inside vs outside the box) is the dominant driver of shot
    quality, mirroring how real xG is built, so this is closer to real
    expected goals than an on-target-based estimate.
    """
    if shots_in_box < 0 or shots_out_box < 0:
        raise ValueError("shot counts cannot be negative")
    return round(XG_PER_SHOT_IN_BOX * shots_in_box
                 + XG_PER_SHOT_OUT_BOX * shots_out_box, 3)


def compute_team_performance(matches, form_raw_base, points, goal_difference,
                             beta=BETA, threshold=COVERAGE_THRESHOLD):
    """Aggregate per-match xG proxies into a team performance block.

    ``matches`` is a list of ``group_stage_matches`` entries, each holding
    ``team_stats`` and ``opponent_stats`` with ``shots`` and
    ``shots_on_target``, plus ``goals_for``/``goals_against``.

    The form correction only activates when ``len(matches) >= threshold``;
    otherwise ``form_raw_adjusted`` equals ``form_raw_base`` so downstream
    probabilities stay identical to the results-only model.
    """
    covered = len(matches)
    xg_for = 0.0
    xg_against = 0.0
    actual_gd_covered = 0
    opp_points = []
    for m in matches:
        xg_for += xg_proxy(m["team_stats"]["shots_in_box"],
                           m["team_stats"]["shots_out_box"])
        xg_against += xg_proxy(m["opponent_stats"]["shots_in_box"],
                               m["opponent_stats"]["shots_out_box"])
        actual_gd_covered += m["goals_for"] - m["goals_against"]
        if m.get("opponent_fifa_points") is not None:
            opp_points.append(m["opponent_fifa_points"])
    xg_diff_total = round(xg_for - xg_against, 3)
    # Average opponent strength (pre-tournament FIFA points) over covered
    # matches. This feeds the Phase-2 opponent adjustment (strength of
    # schedule): the same xGD is worth more against stronger opponents.
    opponent_fifa_points_avg = (round(sum(opp_points) / len(opp_points), 2)
                                if opp_points else None)

    active = covered >= threshold
    if active:
        gd_adjusted = (1 - beta) * goal_difference + beta * xg_diff_total
        form_raw_adjusted = round(points + GD_WEIGHT * gd_adjusted, 3)
    else:
        form_raw_adjusted = form_raw_base

    return {
        "matches_covered": covered,
        "coverage_threshold": threshold,
        "correction_active": active,
        "beta": beta,
        "xg_for_total": round(xg_for, 3),
        "xg_against_total": round(xg_against, 3),
        "xg_diff_total": xg_diff_total,
        "xg_diff_per_match": round(xg_diff_total / covered, 3) if covered else None,
        "opponent_fifa_points_avg": opponent_fifa_points_avg,
        "actual_gd_covered": actual_gd_covered,
        "form_raw_base": form_raw_base,
        "form_raw_adjusted": form_raw_adjusted,
    }
