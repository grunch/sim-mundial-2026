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

from performance import (BETA, COVERAGE_THRESHOLD, XG_PER_OFF_TARGET_SHOT,
                         XG_PER_SHOT_ON_TARGET, compute_team_performance,
                         xg_proxy)

PATH = "worldcup2026_r32_dataset.json"


def build(data):
    """Mutate ``data`` in place with performance intermediates; return it."""
    data["meta"]["performance_model"] = {
        "phase": 1,
        "xg_per_off_target_shot": XG_PER_OFF_TARGET_SHOT,
        "xg_per_shot_on_target": XG_PER_SHOT_ON_TARGET,
        "beta_form_correction": BETA,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "note": ("Phase 1 stores xG-proxy intermediates only; match "
                 "probabilities are unchanged. The form correction activates "
                 "per team once matches_covered >= coverage_threshold."),
    }
    for t in data["teams"]:
        perf = t["world_cup_2026_performance"]
        matches = perf.get("group_stage_matches", [])
        for m in matches:
            m["xg_proxy_for"] = xg_proxy(m["team_stats"]["shots"],
                                         m["team_stats"]["shots_on_target"])
            m["xg_proxy_against"] = xg_proxy(m["opponent_stats"]["shots"],
                                             m["opponent_stats"]["shots_on_target"])
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
