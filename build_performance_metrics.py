#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute performance metrics and fold them into the strength model.

Phase 2. For every group-stage match it stores ``xg_proxy_for``/``against`` and
``opponent_fifa_points``; for every team it stores a ``performance_metrics``
block whose opponent-adjusted expected-goal differential corrects the form; and
it then **recomputes** ``norm_form``, ``strength_index_0_100`` and
``effective_elo`` from that adjusted form, so the correction reaches the
predictions. The results-only values are kept per team for audit.

Idempotent: recomputes everything from the raw match stats and the pool
aggregates, so it is safe to re-run after adding new matches. Run from the
repository root:

    python3 build_performance_metrics.py
"""
import json
import math

from performance import (BETA, COVERAGE_THRESHOLD, GD_WEIGHT,
                         XG_PER_SHOT_IN_BOX, XG_PER_SHOT_OUT_BOX,
                         compute_team_performance, effective_elo,
                         fit_opponent_adjustment, minmax, strength_index,
                         xg_proxy, zscore)

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


def populate_match_stats(matches, points):
    """Store xG proxies and opponent FIFA points on each match, in place."""
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


def fit_opponent_slope(data):
    """Fit the strength-of-schedule slope over every stored team-match row."""
    rows = []
    for t in data["teams"]:
        for m in t["world_cup_2026_performance"].get("group_stage_matches", []):
            xgd = m["xg_proxy_for"] - m["xg_proxy_against"]
            rows.append((m["opponent_fifa_points"], xgd))
    return fit_opponent_adjustment(rows)


def set_performance_model_meta(data, slope, ref):
    data["meta"]["performance_model"] = {
        "phase": 2,
        "xg_per_shot_in_box": XG_PER_SHOT_IN_BOX,
        "xg_per_shot_out_box": XG_PER_SHOT_OUT_BOX,
        "beta_form_correction": BETA,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "opponent_adjustment": {
            "method": ("neutralise each match xGD to the mean opponent "
                       "strength using a slope fit from the data"),
            "slope_xgd_per_fifa_point": round(slope, 6),
            "reference_opponent_fifa_points": round(ref, 2),
        },
        "note": ("Phase 2: form_raw_adjusted (opponent-adjusted xGD blended "
                 "with actual goal difference) replaces form_raw in norm_form, "
                 "strength_index and effective_elo, so it now affects "
                 "predictions. Per-team results-only values are kept under "
                 "performance_metrics for audit."),
    }


def effective_form(t):
    """The form that feeds the strength model: adjusted when active, else raw."""
    pm = t.get("performance_metrics")
    if pm and pm.get("correction_active"):
        return pm["form_raw_adjusted"]
    return t["world_cup_2026_performance"]["form_raw_index"]


def _pool_stats(values):
    n = len(values)
    mean = sum(values) / n
    pstdev = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return {"min": round(min(values), 4), "max": round(max(values), 4),
            "mean": round(mean, 4), "pstdev": round(pstdev, 4)}


def _team_si_elo(t, form, form_stats, weights, coeffs, log10_value_agg,
                 pedigree_agg):
    """Strength index, effective Elo and norm_form for one team at ``form``."""
    dm = t["derived_metrics"]
    log10v = math.log10(t["squad_value_transfermarkt"]["value_eur_millions"])
    ped = t["world_cup_history"]["pedigree_raw"]
    norm_form = minmax(form, form_stats["min"], form_stats["max"])
    si = strength_index(dm["norm_fifa"], dm["norm_value"], norm_form,
                        dm["norm_pedigree"], weights)
    elo = effective_elo(
        t["fifa_ranking"]["points_official_2026_06_11"],
        zscore(form, form_stats["mean"], form_stats["pstdev"]),
        zscore(log10v, log10_value_agg["mean"], log10_value_agg["pstdev"]),
        zscore(ped, pedigree_agg["mean"], pedigree_agg["pstdev"]),
        coeffs)
    return norm_form, round(si, 2), round(elo, 1)


def _validate_norms(data, agg):
    """Assert the recompute reproduces the stored fifa/value/pedigree norms.

    These axes never change, so this stays true on every re-run and proves the
    min-max normalisation and pool aggregates match how the dataset was built.
    """
    fifa, lv, pd = agg["fifa_points"], agg["log10_value"], agg["pedigree_raw"]
    for t in data["teams"]:
        dm = t["derived_metrics"]
        checks = (
            ("norm_fifa", minmax(t["fifa_ranking"]["points_official_2026_06_11"],
                                 fifa["min"], fifa["max"])),
            ("norm_value", minmax(math.log10(
                t["squad_value_transfermarkt"]["value_eur_millions"]),
                lv["min"], lv["max"])),
            ("norm_pedigree", minmax(t["world_cup_history"]["pedigree_raw"],
                                     pd["min"], pd["max"])),
        )
        for name, got in checks:
            if abs(round(got, 4) - dm[name]) > 1e-3:
                raise AssertionError(
                    f"{t['code']} {name}: recomputed {got:.4f} != stored {dm[name]}")


def recompute_strength_and_elo(data):
    """Recompute norm_form/strength_index/effective_elo from the adjusted form.

    Overwrites ``derived_metrics`` with performance-adjusted values and keeps
    the results-only baseline (computed deterministically from ``form_raw_index``
    and the original form pool) per team for audit.
    """
    meta = data["meta"]
    weights, coeffs = meta["weights_strength_index"], meta["elo_adjustment_coeffs"]
    agg = meta["pool_aggregates"]
    lv, pd, raw_form = agg["log10_value"], agg["pedigree_raw"], agg["form_raw"]
    _validate_norms(data, agg)

    forms = [effective_form(t) for t in data["teams"]]
    adj_stats = _pool_stats(forms)
    agg["form_effective"] = adj_stats

    for t, form in zip(data["teams"], forms):
        dm = t["derived_metrics"]
        raw_index = t["world_cup_2026_performance"]["form_raw_index"]
        _, ro_si, ro_elo = _team_si_elo(t, raw_index, raw_form, weights,
                                        coeffs, lv, pd)
        norm_form, si, elo = _team_si_elo(t, form, adj_stats, weights,
                                          coeffs, lv, pd)
        pm = t.get("performance_metrics")
        if pm is not None:
            pm["strength_index_results_only"] = ro_si
            pm["effective_elo_results_only"] = ro_elo
            pm["strength_index_adjusted"] = si
            pm["effective_elo_adjusted"] = elo
            pm["elo_shift"] = round(elo - ro_elo, 1)
        dm["norm_form"] = round(norm_form, 4)
        dm["strength_index_0_100"] = si
        dm["effective_elo"] = elo


def build(data):
    """Mutate ``data`` in place: performance metrics + adjusted strength model."""
    points = opponent_points_lookup(data)
    for t in data["teams"]:
        perf = t["world_cup_2026_performance"]
        matches = perf.get("group_stage_matches", [])
        populate_match_stats(matches, points)
        recompute_group_aggregates(perf, matches)

    slope, ref = fit_opponent_slope(data)
    set_performance_model_meta(data, slope, ref)

    for t in data["teams"]:
        perf = t["world_cup_2026_performance"]
        matches = perf.get("group_stage_matches", [])
        if matches:
            t["performance_metrics"] = compute_team_performance(
                matches, perf["form_raw_index"], perf["points"],
                perf["goal_difference"], opp_slope=slope, opp_ref=ref)
        else:
            t.pop("performance_metrics", None)

    recompute_strength_and_elo(data)
    return data


def main():
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)
    build(data)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    active = sum(1 for t in data["teams"]
                 if t.get("performance_metrics", {}).get("correction_active"))
    shifts = [t["performance_metrics"]["elo_shift"] for t in data["teams"]
              if "performance_metrics" in t]
    biggest = max(shifts, key=abs) if shifts else 0.0
    print(f"Strength model rebuilt: {active} teams with the form correction "
          f"active; largest effective-Elo shift {biggest:+.1f}.")


if __name__ == "__main__":
    main()
