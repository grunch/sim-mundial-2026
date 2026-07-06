#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance metrics derived from group-stage match statistics.

The results-only "form" (``form_raw = points + 0.4 * goal_difference``) is
noisy: a team can win with few chances or lose while dominating. These helpers
add a chance-quality signal — an expected-goals proxy from shot location — and,
in Phase 2, feed it back into the strength metrics that drive the predictions.

Phase 2 (current):
  * ``xg_proxy`` turns a team's shots into expected goals by location.
  * The per-match xG differential is **adjusted for the opponent's strength**
    (strength of schedule): dominating a weak side is worth less than the same
    output against a strong one. The adjustment slope is fit from the data.
  * The opponent-adjusted xGD blends with the actual goal difference to form
    ``form_raw_adjusted``, which then replaces ``form_raw`` in the strength
    index and effective Elo (see ``build_performance_metrics.py``).
  * The correction only activates once a team has the full group stage covered
    (``COVERAGE_THRESHOLD``); below that its form is left untouched.
"""

# Expected goals contributed per shot by location. Shot location is the single
# strongest determinant of chance quality, so with the richer source we weight
# by zone instead of by on/off target. Coefficients approximate average
# conversion rates and are tunable (stored in meta.performance_model).
XG_PER_SHOT_IN_BOX = 0.13
XG_PER_SHOT_OUT_BOX = 0.035

# Blend weight applied to the (opponent-adjusted) expected goal difference when
# correcting form.
BETA = 0.8
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


def fit_opponent_adjustment(rows):
    """Fit the strength-of-schedule slope from ``(opponent_points, xgd)`` rows.

    Returns ``(slope, ref)`` where ``slope`` is the ordinary-least-squares slope
    of per-match xG differential on opponent strength (expected to be negative:
    a stronger opponent depresses your xGD) and ``ref`` is the mean opponent
    strength, used as the neutral baseline. Returns ``(0.0, ref)`` when the
    opponent strengths carry no variance (nothing to adjust for).
    """
    n = len(rows)
    if n == 0:
        return 0.0, 0.0
    mean_opp = sum(o for o, _ in rows) / n
    mean_xgd = sum(x for _, x in rows) / n
    var = sum((o - mean_opp) ** 2 for o, _ in rows) / n
    if var == 0:
        return 0.0, mean_opp
    cov = sum((o - mean_opp) * (x - mean_xgd) for o, x in rows) / n
    return cov / var, mean_opp


def adjust_xgd(xgd, opponent_points, slope, ref):
    """Neutralise a match xGD to a reference opponent strength.

    Removes the portion of ``xgd`` explained by the opponent being weaker or
    stronger than ``ref``. With ``slope < 0``, output against a weaker-than-ref
    opponent is discounted and output against a stronger one is credited.
    """
    return xgd - slope * (opponent_points - ref)


def compute_team_performance(matches, form_raw_base, points, goal_difference,
                             beta=BETA, threshold=COVERAGE_THRESHOLD,
                             opp_slope=None, opp_ref=None):
    """Aggregate per-match xG proxies into a team performance block.

    ``matches`` is a list of ``group_stage_matches`` entries with ``team_stats``
    and ``opponent_stats`` (shots by location), ``goals_for``/``goals_against``
    and ``opponent_fifa_points``.

    When ``opp_slope``/``opp_ref`` are given, each match xGD is opponent-adjusted
    (strength of schedule) before aggregation, and the adjusted total drives the
    form correction. Without them the raw xGD is used (Phase-1 behaviour).

    The correction only activates when ``len(matches) >= threshold``; otherwise
    ``form_raw_adjusted`` equals ``form_raw_base`` so downstream probabilities
    stay identical to the results-only model.
    """
    covered = len(matches)
    xg_for = 0.0
    xg_against = 0.0
    xg_diff_adjusted = 0.0
    actual_gd_covered = 0
    opp_points = []
    for m in matches:
        f = xg_proxy(m["team_stats"]["shots_in_box"],
                     m["team_stats"]["shots_out_box"])
        a = xg_proxy(m["opponent_stats"]["shots_in_box"],
                     m["opponent_stats"]["shots_out_box"])
        xg_for += f
        xg_against += a
        actual_gd_covered += m["goals_for"] - m["goals_against"]
        opp = m.get("opponent_fifa_points")
        if opp is not None:
            opp_points.append(opp)
        if opp_slope is not None and opp is not None:
            xg_diff_adjusted += adjust_xgd(f - a, opp, opp_slope, opp_ref)
        else:
            xg_diff_adjusted += f - a

    xg_diff_total = round(xg_for - xg_against, 3)
    xg_diff_adjusted_total = round(xg_diff_adjusted, 3)
    opponent_fifa_points_avg = (round(sum(opp_points) / len(opp_points), 2)
                                if opp_points else None)

    # The signal blended into form: opponent-adjusted when a slope is supplied,
    # otherwise the raw differential.
    perf_signal = (xg_diff_adjusted_total if opp_slope is not None
                   else xg_diff_total)

    active = covered >= threshold
    if active:
        gd_adjusted = (1 - beta) * goal_difference + beta * perf_signal
        form_raw_adjusted = round(points + GD_WEIGHT * gd_adjusted, 3)
    else:
        form_raw_adjusted = form_raw_base

    return {
        "matches_covered": covered,
        "coverage_threshold": threshold,
        "correction_active": active,
        "beta": beta,
        "opponent_adjusted": opp_slope is not None,
        "xg_for_total": round(xg_for, 3),
        "xg_against_total": round(xg_against, 3),
        "xg_diff_total": xg_diff_total,
        "xg_diff_adjusted_total": xg_diff_adjusted_total,
        "xg_diff_per_match": round(xg_diff_total / covered, 3) if covered else None,
        "opponent_fifa_points_avg": opponent_fifa_points_avg,
        "actual_gd_covered": actual_gd_covered,
        "form_raw_base": form_raw_base,
        "form_raw_adjusted": form_raw_adjusted,
    }


# --- Strength model (recomputes derived metrics from the adjusted form) -------

def minmax(x, lo, hi):
    """Min-max normalise ``x`` to [0, 1] over a pool range ``[lo, hi]``."""
    return (x - lo) / (hi - lo) if hi > lo else 0.0


def zscore(x, mean, pstdev):
    """Standard score of ``x`` relative to a pool ``mean``/``pstdev``."""
    return (x - mean) / pstdev if pstdev else 0.0


def strength_index(norm_fifa, norm_value, norm_form, norm_pedigree, weights):
    """Composite Strength Index (0-100) from the four normalised axes."""
    return 100.0 * (weights["fifa"] * norm_fifa
                    + weights["value"] * norm_value
                    + weights["form"] * norm_form
                    + weights["pedigree"] * norm_pedigree)


def effective_elo(fifa_points, z_form, z_value, z_pedigree, coeffs):
    """Effective Elo: real FIFA Elo adjusted by the form/value/pedigree axes."""
    return (fifa_points + coeffs["form"] * z_form
            + coeffs["value"] * z_value + coeffs["pedigree"] * z_pedigree)
