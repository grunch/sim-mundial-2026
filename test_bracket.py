#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for bracket.py (stdlib unittest, no dependencies).

They cover the probability engine (p_adv/favorite), the analytic goal model
(lambdas/top_scorelines/fmt_scorelines), tie resolution with random penalties
(decide/resolve), real-result loading (load_real_results), CLI parsing (parse_args)
and the printed output (main).

Run FROM the repo folder (bracket.py reads the dataset via a relative path):
    python3 -m unittest -v test_bracket
    python3 test_bracket.py
"""
import io
import os
import random
import unittest
from contextlib import redirect_stdout

import bracket

# Pair with an overwhelming favorite (wins in regular time, does NOT go to penalties)
FAV, DOG = "ARG", "CPV"     # Argentina vs Cape Verde
# Even pair whose most likely scoreline is a draw (decided on penalties)
EVEN_A, EVEN_B = "NED", "MAR"   # Netherlands vs Morocco -> 1-1 modal


class _StubRng:
    """Fake rng with a fixed .random(), to force each branch of the penalty draw."""
    def __init__(self, value):
        self._value = value

    def random(self):
        return self._value


class TestMatchProbability(unittest.TestCase):
    """p_adv and favorite: the head-to-head engine."""

    def test_p_adv_is_symmetric_and_sums_to_one(self):
        self.assertAlmostEqual(bracket.p_adv(FAV, DOG) + bracket.p_adv(DOG, FAV), 1.0, places=9)

    def test_p_adv_of_a_team_against_itself_is_one_half(self):
        self.assertAlmostEqual(bracket.p_adv(FAV, FAV), 0.5, places=9)

    def test_favorite_returns_the_stronger_team_with_prob_ge_one_half(self):
        fav, dog, p_fav = bracket.favorite(FAV, DOG)
        self.assertEqual((fav, dog), (FAV, DOG))
        self.assertGreaterEqual(p_fav, 0.5)
        self.assertAlmostEqual(p_fav, bracket.p_adv(FAV, DOG), places=9)

    def test_favorite_is_independent_of_argument_order(self):
        self.assertEqual(bracket.favorite(FAV, DOG)[0], bracket.favorite(DOG, FAV)[0])


class TestGoalModel(unittest.TestCase):
    """lambdas, top_scorelines and fmt_scorelines: analytic scoreline (Poisson)."""

    def test_lambdas_sum_to_the_base_total_T(self):
        lf, ld = bracket.lambdas(FAV, DOG)
        self.assertAlmostEqual(lf + ld, bracket.T, places=9)

    def test_lambdas_favorite_not_below_the_underdog(self):
        lf, ld = bracket.lambdas(FAV, DOG)
        self.assertGreaterEqual(lf, ld)
        self.assertGreater(ld, 0.0)

    def test_lambdas_equal_when_supremacy_is_negative(self):
        lf, ld = bracket.lambdas(DOG, FAV)
        self.assertAlmostEqual(lf, ld, places=9)
        self.assertAlmostEqual(lf, bracket.T / 2.0, places=9)

    def test_top_scorelines_returns_n_rows_sorted_desc(self):
        rows = bracket.top_scorelines(FAV, DOG, 5)
        self.assertEqual(len(rows), 5)
        probs = [p for p, _, _ in rows]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top_scorelines_caps_at_the_enumerable_scorelines(self):
        rows = bracket.top_scorelines(FAV, DOG, 1_000_000)
        self.assertEqual(len(rows), (bracket.MAX_GOALS + 1) ** 2)

    def test_top_scorelines_sum_to_almost_one(self):
        total = sum(p for p, _, _ in bracket.top_scorelines(FAV, DOG, 1_000_000))
        self.assertGreater(total, 0.99)
        self.assertLessEqual(total, 1.0 + 1e-9)

    def test_most_likely_scoreline_of_the_favorite_puts_it_ahead(self):
        _, gf, gd = bracket.top_scorelines(FAV, DOG, 1)[0]
        self.assertGreater(gf, gd)

    def test_even_pair_has_a_draw_as_most_likely_scoreline(self):
        _, gf, gd = bracket.top_scorelines(EVEN_A, EVEN_B, 1)[0]
        self.assertEqual(gf, gd)

    def test_fmt_scorelines_flags_penalties_on_a_draw(self):
        self.assertIn("->pen", bracket.fmt_scorelines(EVEN_A, EVEN_B, 1))

    def test_fmt_scorelines_no_penalties_with_a_clear_favorite(self):
        text = bracket.fmt_scorelines(FAV, DOG, 1)
        self.assertNotIn("->pen", text)
        self.assertIn("%", text)

    def test_fmt_scorelines_orients_goals_to_the_shown_order(self):
        # Mexico (home) vs England (away): England wins -> must read 0-1, not 1-0
        self.assertEqual(bracket.favorite("MEX", "ENG")[0], "ENG")   # favorite = away team
        self.assertTrue(bracket.fmt_scorelines("MEX", "ENG", 1).startswith("0-1"))
        # swapping the team order swaps the scoreline
        self.assertTrue(bracket.fmt_scorelines("ENG", "MEX", 1).startswith("1-0"))

    def test_poisson_sums_to_one_over_its_support(self):
        self.assertAlmostEqual(sum(bracket._poisson(k, 2.4) for k in range(50)), 1.0, places=9)


class TestDecidePenalties(unittest.TestCase):
    """decide: draw -> penalties at random (the underdog can go through)."""

    def test_clear_favorite_wins_in_regular_time_without_penalties(self):
        win, prob, pens = bracket.decide(FAV, DOG, _StubRng(0.999))
        self.assertEqual(win, FAV)
        self.assertFalse(pens)            # the coin is not flipped
        self.assertGreaterEqual(prob, 0.5)

    def test_draw_with_low_coin_is_won_by_the_favorite(self):
        # rng.random()=0.0 < p_fav -> the favorite advances
        win, prob, pens = bracket.decide(EVEN_A, EVEN_B, _StubRng(0.0))
        fav, _, p_fav = bracket.favorite(EVEN_A, EVEN_B)
        self.assertTrue(pens)
        self.assertEqual(win, fav)
        self.assertAlmostEqual(prob, p_fav, places=9)

    def test_draw_with_high_coin_is_won_by_the_underdog(self):
        # rng.random()=0.999 > p_fav -> the underdog advances (the WORSE percentage)
        win, prob, pens = bracket.decide(EVEN_A, EVEN_B, _StubRng(0.999))
        fav, dog, p_fav = bracket.favorite(EVEN_A, EVEN_B)
        self.assertTrue(pens)
        self.assertEqual(win, dog)
        self.assertLess(prob, 0.5)        # the winner had less than 50%
        self.assertAlmostEqual(prob, 1 - p_fav, places=9)


class TestResolveBracket(unittest.TestCase):
    """resolve: play the whole bracket with random penalties."""

    def test_resolves_all_matches(self):
        W, PROB, MATCH, PENS, REAL = bracket.resolve(random.Random(bracket.DEFAULT_SEED))
        self.assertEqual(set(W), set(bracket.ORDER))
        self.assertEqual(set(PENS), set(bracket.ORDER))
        self.assertEqual(set(REAL), set(bracket.ORDER))
        # Without real results, every tie is predicted (REAL[mid] is None)
        self.assertTrue(all(REAL[mid] is None for mid in bracket.ORDER))

    def test_same_seed_gives_the_same_bracket(self):
        a = bracket.resolve(random.Random(7))[0]
        b = bracket.resolve(random.Random(7))[0]
        self.assertEqual(a, b)

    def test_different_seeds_can_give_different_brackets(self):
        brackets = {tuple(sorted(bracket.resolve(random.Random(s))[0].items())) for s in range(40)}
        self.assertGreater(len(brackets), 1)   # penalty randomness moves results

    def test_each_winner_is_one_of_the_two_in_the_tie(self):
        W, _, MATCH, _, _ = bracket.resolve(random.Random(7))
        for mid in bracket.ORDER:
            self.assertIn(W[mid], MATCH[mid])

    def test_some_tie_is_decided_on_penalties(self):
        _, _, _, PENS, _ = bracket.resolve(random.Random(bracket.DEFAULT_SEED))
        self.assertTrue(any(PENS.values()))

    def test_an_underdog_can_advance_on_penalties(self):
        # In some tie decided on penalties the winner has prob < 0.5 (was not the favorite)
        W, PROB, _, PENS, _ = bracket.resolve(random.Random(bracket.DEFAULT_SEED))
        upsets = [mid for mid in bracket.ORDER if PENS[mid] and PROB[mid] < 0.5]
        self.assertTrue(upsets)


class TestParseArgs(unittest.TestCase):
    """parse_args: --scoreline [N], --seed S and --results [FILE] (in any order)."""

    def test_no_arguments_uses_defaults(self):
        self.assertEqual(bracket.parse_args([]), (False, 1, bracket.DEFAULT_SEED, None))

    def test_scoreline_flag_without_n(self):
        self.assertEqual(bracket.parse_args(["--scoreline"]), (True, 1, bracket.DEFAULT_SEED, None))

    def test_scoreline_flag_with_n(self):
        self.assertEqual(bracket.parse_args(["--scoreline", "3"]), (True, 3, bracket.DEFAULT_SEED, None))

    def test_short_alias_s(self):
        self.assertEqual(bracket.parse_args(["-s", "2"]), (True, 2, bracket.DEFAULT_SEED, None))

    def test_seed_changes_the_seed(self):
        self.assertEqual(bracket.parse_args(["--seed", "7"]), (False, 1, 7, None))

    def test_scoreline_and_seed_combined_in_any_order(self):
        self.assertEqual(bracket.parse_args(["--seed", "7", "-s", "3"]), (True, 3, 7, None))

    def test_results_without_file_uses_the_default(self):
        self.assertEqual(
            bracket.parse_args(["--results"]),
            (False, 1, bracket.DEFAULT_SEED, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_results_with_explicit_file(self):
        self.assertEqual(
            bracket.parse_args(["--results", "other.json"]),
            (False, 1, bracket.DEFAULT_SEED, "other.json"),
        )

    def test_short_alias_r(self):
        self.assertEqual(
            bracket.parse_args(["-r"]),
            (False, 1, bracket.DEFAULT_SEED, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_results_combined_with_seed_and_scoreline(self):
        self.assertEqual(
            bracket.parse_args(["--results", "--seed", "7", "-s", "2"]),
            (True, 2, 7, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_results_does_not_consume_a_following_flag(self):
        # after --results comes another flag, not a file -> uses the default
        self.assertEqual(
            bracket.parse_args(["--results", "--scoreline"]),
            (True, 1, bracket.DEFAULT_SEED, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_n_zero_is_invalid(self):
        with self.assertRaises(SystemExit):
            bracket.parse_args(["--scoreline", "0"])

    def test_non_integer_seed_is_invalid(self):
        with self.assertRaises(SystemExit):
            bracket.parse_args(["--seed", "x"])

    def test_unknown_argument_is_invalid(self):
        with self.assertRaises(SystemExit):
            bracket.parse_args(["--bogus"])


class TestRealResults(unittest.TestCase):
    """load_real_results and resolve/main with real results (--results)."""

    def _write(self, matches):
        """Write a minimal results JSON and return its temp path."""
        import json
        import tempfile
        payload = {"round_of_32": {"matches": matches}, "round_of_16": {"matches": []}}
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self.addCleanup(os.remove, path)
        return path

    def _played(self, home, hc, hg, away, ac, ag, winner, decided_by="regular", pens_h=None, pens_a=None):
        return {
            "home": {"team": home, "code": hc, "score": hg, "penalties": pens_h},
            "away": {"team": away, "code": ac, "score": ag, "penalties": pens_a},
            "decided_by": decided_by, "played": True, "winner": winner,
        }

    def test_reads_only_played_matches(self):
        path = self._write([
            self._played("Brazil", "BRA", 2, "Japan", "JPN", 1, "Brazil"),
            {"home": {"team": "Portugal", "code": "POR", "score": None, "penalties": None},
             "away": {"team": "Croatia", "code": "CRO", "score": None, "penalties": None},
             "decided_by": None, "played": False, "winner": None},
        ])
        real = bracket.load_real_results(path)
        self.assertIn(frozenset(("BRA", "JPN")), real)
        self.assertNotIn(frozenset(("POR", "CRO")), real)

    def test_extracts_winner_scoreline_and_penalties(self):
        path = self._write([
            self._played("Germany", "GER", 1, "Paraguay", "PAR", 1, "Paraguay",
                         decided_by="penalties", pens_h=3, pens_a=4),
        ])
        real = bracket.load_real_results(path)
        r = real[frozenset(("GER", "PAR"))]
        self.assertEqual(r["winner"], "PAR")
        self.assertTrue(r["pens"])
        self.assertEqual(r["goals"], {"GER": 1, "PAR": 1})

    def test_ignores_inconsistent_winner(self):
        path = self._write([
            self._played("Brazil", "BRA", 2, "Japan", "JPN", 1, "Mars"),
        ])
        self.assertEqual(bracket.load_real_results(path), {})

    def test_reads_results_from_the_quarterfinals_stage(self):
        # A played quarterfinal (round_of_8) must be loaded like any earlier round
        import json
        import tempfile
        payload = {
            "round_of_32": {"matches": []},
            "round_of_16": {"matches": []},
            "round_of_8": {"matches": [
                self._played("France", "FRA", 2, "Morocco", "MAR", 0, "France"),
            ]},
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self.addCleanup(os.remove, path)
        real = bracket.load_real_results(path)
        self.assertIn(frozenset(("FRA", "MAR")), real)
        self.assertEqual(real[frozenset(("FRA", "MAR"))]["winner"], "FRA")

    def test_resolve_uses_the_real_winner_no_prediction(self):
        # Morocco advances by real result even though it is NOT the model's favorite
        real = {frozenset(("NED", "MAR")): {"winner": "MAR", "pens": True,
                                            "goals": {"NED": 1, "MAR": 1}}}
        W, PROB, MATCH, PENS, REAL = bracket.resolve(random.Random(7), real)
        # match 75 is NED vs MAR in the official BRACKET
        self.assertEqual(W[75], "MAR")
        self.assertEqual(PROB[75], 1.0)
        self.assertIsNotNone(REAL[75])
        self.assertTrue(PENS[75])

    def test_resolve_without_real_is_identical_to_previous_behavior(self):
        W_real = bracket.resolve(random.Random(7), {})[0]
        W_none = bracket.resolve(random.Random(7))[0]
        self.assertEqual(W_real, W_none)

    def test_main_with_results_marks_the_real_ties(self):
        path = self._write([
            self._played("Germany", "GER", 1, "Paraguay", "PAR", 1, "Paraguay",
                         decided_by="penalties", pens_h=3, pens_a=4),
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            bracket.main(["--results", path, "--seed", "7"])
        out = buf.getvalue()
        self.assertIn("(real)", out)
        self.assertIn("real result already played", out)


class TestMainOutput(unittest.TestCase):
    """Smoke tests of the printed output."""

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bracket.main(argv)
        return buf.getvalue()

    def test_default_mode_prints_champion_and_seed(self):
        out = self._run(["--seed", "7"])
        self.assertIn("CHAMPION:", out)
        self.assertIn("penalty seed: 7", out)
        self.assertNotIn("most likely scoreline", out)

    def test_scoreline_mode_prints_header_and_flags_penalties(self):
        out = self._run(["--scoreline", "--seed", "7"])
        self.assertIn("most likely scoreline", out)
        self.assertIn("(pen)", out)

    def test_scoreline_top3_shows_three_results_in_a_tie(self):
        out = self._run(["--scoreline", "3", "--seed", "7"])
        line = next(l for l in out.splitlines() if l.strip().startswith("[86]"))
        scorelines = [tok for tok in line.split() if "-" in tok and tok[0].isdigit()]
        self.assertEqual(len(scorelines), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
