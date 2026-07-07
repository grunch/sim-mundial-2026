#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World Cup 2026 bracket: from the Round of 32 to the Final.
Reuses the dataset and the SAME per-match probability logic as
predict.py / simulate_bracket.py (average of effective Elo and Strength Index).

Advancing in each tie:
  - If the most likely scoreline (Poisson model, see below) is NOT a draw, the team
    ahead (the favorite) advances in regular time.
  - If the most likely scoreline is a DRAW, the tie goes to PENALTIES and the winner
    is decided AT RANDOM: a coin weighted by strength (p_adv), so the underdog CAN
    also go through. By default the randomness is REAL (system seed): every run may
    yield a different champion. With --seed S the bracket is reproducible.

Scoreline mode (--scoreline [N]):
    Predicts the MOST LIKELY scoreline of each tie with a Poisson goal model derived
    from the Elo. The scoreline and its probability are computed ANALYTICALLY and
    EXACTLY (not by simulation): for two independent Poisson variables,
    P(i-j) = Poisson(i; lambda_fav) * Poisson(j; lambda_rival).
    N = how many top scorelines to show per tie (default 1).
    A drawn scoreline is flagged "->pen": that tie is decided on penalties (random).

Real results (--results [FILE]):
    By default the WHOLE bracket is predicted (historical behavior). With --results
    it reads 'results_bracket.json' (or the given FILE) and every tie that ALREADY
    has a real result (played=true) is NOT predicted: the team that really advanced
    goes through and the real scoreline is printed. Ties without a result are still
    predicted as usual (favorite / penalties at random).

(For "who is champion most often" across thousands of tournaments, use
 simulate_bracket.py: that question aggregates many realizations like this one.)

Usage:
    python3 bracket.py                       # NEW bracket every run (penalties at random)
    python3 bracket.py --seed 7              # reproducible: same seed, same bracket
    python3 bracket.py --scoreline           # + most likely scoreline of each tie
    python3 bracket.py --scoreline 3 --seed 7 # top-3 scorelines and fixed seed 7
    python3 bracket.py --results             # use real results already played
    python3 bracket.py --results data.json    # read real results from another file
"""
import json, math, random, sys

D = 600.0   # Elo denominator (FIFA) — identical to predict.py / simulate_bracket.py
S = 12.0    # logistic scale of the Strength Index
T = 2.6     # base expected total goals per match (knockout World Cup average)
C = 200.0   # Elo points per "supremacy" goal (maps Elo edge -> goals)
MAX_GOALS = 8        # cap for enumerating scorelines (the Poisson tail is negligible)
DEFAULT_SEED = None  # without --seed -> REAL system randomness: every run may differ
DEFAULT_RESULTS_FILE = "results_bracket.json"  # real-results file (--results)
# Knockout stages read from the results file, in bracket order. Stages missing from
# the file are ignored, so later rounds can be added as they are played.
KNOCKOUT_STAGES = ("round_of_32", "round_of_16", "round_of_8", "semifinals", "final")

with open("worldcup2026_r32_dataset.json", encoding="utf-8") as f:
    DATA = json.load(f)
ELO  = {t["code"]: t["derived_metrics"]["effective_elo"]        for t in DATA["teams"]}
SI   = {t["code"]: t["derived_metrics"]["strength_index_0_100"] for t in DATA["teams"]}
NAME = {t["code"]: t["team"] for t in DATA["teams"]}

def team_name(code):
    """Display name of a team code (falls back to the code itself)."""
    return NAME.get(code, code)

def p_adv(a, b):
    """P(A advances over B), including extra time/penalties. Same engine formula."""
    pe = 1.0 / (1.0 + 10 ** (-(ELO[a] - ELO[b]) / D))
    ps = 1.0 / (1.0 + 10 ** (-(SI[a]  - SI[b])  / S))
    return (pe + ps) / 2.0

def favorite(a, b):
    """Return (favorite, underdog, prob_favorite) according to p_adv."""
    pa = p_adv(a, b)
    return (a, b, pa) if pa >= 0.5 else (b, a, 1 - pa)

# ---------------------------------------------------------------------------
# Goal model (most likely scoreline, analytic and exact)
# ---------------------------------------------------------------------------
def _poisson(k, lam):
    """P(exactly k goals) under Poisson(lam)."""
    return math.exp(-lam) * lam ** k / math.factorial(k)

def lambdas(fav, dog):
    """Expected goals (lambda) of favorite and underdog from the Elo.

    Supremacy is clamped to >= 0 so the favorite never ends up below the underdog,
    keeping the scoreline consistent with who the favorite is.
    """
    sup = max(0.0, (ELO[fav] - ELO[dog]) / C)   # supremacy in goals (>= 0)
    sup = min(sup, T - 0.4)                      # avoid a non-positive underdog lambda
    return (T + sup) / 2.0, (T - sup) / 2.0

def top_scorelines(fav, dog, n):
    """Top-n most likely (fav-dog) scorelines with their EXACT probability.

    Returns a list of (prob, goals_fav, goals_dog) sorted from highest to lowest.
    """
    lf, ld = lambdas(fav, dog)
    rows = [(_poisson(i, lf) * _poisson(j, ld), i, j)
            for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    rows.sort(reverse=True)
    return rows[:n]

def fmt_scorelines(home, away, n):
    """Compact text '2-0 (21.4%) . 1-0 (17.8%)' in the 'home vs away' order.

    The model computes the scoreline favorite-first, but here it is reoriented so the
    goals match the ORDER IN WHICH THE TEAMS ARE SHOWN (home-away). If the favorite is
    the away team, the favorite's '1-0' is printed '0-1'.
    """
    fav, dog, _ = favorite(home, away)
    parts = []
    for prob, gf, gd in top_scorelines(fav, dog, n):
        g_home, g_away = (gf, gd) if home == fav else (gd, gf)
        pen = "->pen" if g_home == g_away else ""   # draw -> decided on penalties
        parts.append(f"{g_home}-{g_away} ({prob*100:.1f}%){pen}")
    return " . ".join(parts)

def decide(a, b, rng):
    """Resolve the a-vs-b tie. Returns (winner, prob_winner, was_penalties).

    - If the most likely scoreline is NOT a draw -> the favorite wins in regular time.
    - If it is a DRAW -> penalties: the winner is drawn with a coin weighted by
      strength (favorite prob = p_adv). The underdog can also go through.
    """
    fav, dog, p_fav = favorite(a, b)
    _, gf, gd = top_scorelines(fav, dog, 1)[0]   # most likely scoreline
    if gf != gd:
        return fav, p_fav, False
    win = fav if rng.random() < p_fav else dog   # draw -> penalties (random)
    return win, (p_fav if win == fav else 1 - p_fav), True

# Official bracket structure (match_id -> (source_a, source_b)).
# Source = team code (Round of 32) or "W<id>" (winner of a previous match).
BRACKET = {
    73:("CAN","RSA"), 74:("GER","PAR"), 75:("NED","MAR"), 76:("BRA","JPN"),
    77:("FRA","SWE"), 78:("CIV","NOR"), 79:("MEX","ECU"), 80:("ENG","COD"),
    81:("USA","BIH"), 82:("BEL","SEN"), 83:("ESP","AUT"), 84:("POR","CRO"),
    85:("SUI","ALG"), 86:("ARG","CPV"), 87:("COL","GHA"), 88:("AUS","EGY"),
    89:("W74","W77"), 90:("W73","W75"), 91:("W76","W78"), 92:("W79","W80"),
    93:("W83","W84"), 94:("W81","W82"), 95:("W86","W88"), 96:("W85","W87"),
    97:("W89","W90"), 98:("W93","W94"), 99:("W91","W92"), 100:("W95","W96"),
    101:("W97","W98"), 102:("W99","W100"),
    104:("W101","W102"),
}
ORDER = sorted(BRACKET)

ROUNDS = [
    ("ROUND OF 32",   list(range(73, 89))),
    ("ROUND OF 16",   list(range(89, 97))),
    ("QUARTERFINALS", list(range(97, 101))),
    ("SEMIFINALS",    [101, 102]),
    ("FINAL",         [104]),
]

def source_team(src, W):
    """Resolve a source ('CAN' or 'W74') to the code of the present team."""
    return W[int(src[1:])] if src[0] == "W" else src

def load_real_results(path):
    """Read real results from a JSON file (results_bracket.json format).

    Returns a dict {frozenset({codeA, codeB}): result} with ONLY the ties already
    played (played=true). Each 'result' is:
        {"winner": code, "pens": bool, "goals": {code: goals, ...}}
    The key is the pair of codes, so the bracket recognizes the match regardless of
    order or internal ids. Matches without codes, not marked as played, or whose
    winner does not match either team are ignored.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    real = {}
    for stage in KNOCKOUT_STAGES:
        for m in data.get(stage, {}).get("matches", []):
            if not m.get("played"):
                continue
            home, away = m.get("home", {}), m.get("away", {})
            hc, ac = home.get("code"), away.get("code")
            if not hc or not ac:
                continue
            winner_name = m.get("winner")
            if winner_name == home.get("team"):
                wc = hc
            elif winner_name == away.get("team"):
                wc = ac
            else:
                continue   # winner inconsistent with the teams: ignored
            real[frozenset((hc, ac))] = {
                "winner": wc,
                "pens": m.get("decided_by") == "penalties",
                "goals": {hc: home.get("score"), ac: away.get("score")},
            }
    return real

def resolve(rng, real=None):
    """Play the whole bracket with the generator 'rng' (penalty randomness).

    If 'real' (dict from load_real_results) has an already-played tie, its real result
    is used and NOT predicted; everything else is predicted as usual.

    Returns winner[id], prob[id], match[id]=(a,b), pens[id]=bool,
    real[id]=(real-result dict or None).
    """
    real = real or {}
    W, PROB, MATCH, PENS, REAL = {}, {}, {}, {}, {}
    for mid in ORDER:
        sa, sb = BRACKET[mid]
        a, b = source_team(sa, W), source_team(sb, W)
        result = real.get(frozenset((a, b)))
        if result is not None:
            win, prob, pens = result["winner"], 1.0, result["pens"]
        else:
            win, prob, pens = decide(a, b, rng)
        W[mid], PROB[mid], MATCH[mid], PENS[mid], REAL[mid] = win, prob, (a, b), pens, result
    return W, PROB, MATCH, PENS, REAL

def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def parse_args(argv):
    """Read the CLI (flags in any order). Returns (scoreline, n, seed, real_path).

        --scoreline | -s [N]   enable scorelines; N = how many to show (default 1)
        --seed S               seed for the penalty randomness (default DEFAULT_SEED)
        --results | -r [FILE]  use real results; FILE = file (default
                               DEFAULT_RESULTS_FILE). Without the flag, real_path=None.
    """
    scoreline, n, seed, real_path = False, 1, DEFAULT_SEED, None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--scoreline", "-s"):
            scoreline = True
            if i + 1 < len(argv) and _is_int(argv[i + 1]):
                n = int(argv[i + 1]); i += 1
        elif tok == "--seed":
            if i + 1 >= len(argv) or not _is_int(argv[i + 1]):
                sys.exit("--seed requires an integer")
            seed = int(argv[i + 1]); i += 1
        elif tok in ("--results", "-r"):
            real_path = DEFAULT_RESULTS_FILE
            # optional FILE: the next token, if not another flag nor an integer
            if i + 1 < len(argv) and not argv[i + 1].startswith("-") and not _is_int(argv[i + 1]):
                real_path = argv[i + 1]; i += 1
        else:
            sys.exit(f"Unrecognized argument: {tok!r}")
        i += 1
    if n < 1:
        sys.exit("N must be >= 1")
    return scoreline, n, seed, real_path

def _match_line(mid, MATCH, W, PROB, PENS, REAL, scoreline, n):
    """Build the printed line of a tie.

    Tie with a real result: shows the real scoreline and the (real) tag.
    Predicted tie: shows the probability (and, with --scoreline, the Poisson model).
    """
    a, b = MATCH[mid]
    win = W[mid]
    head = f"  [{mid}] {team_name(a):<20} vs {team_name(b):<20}  ->  {team_name(win):<18}"
    result = REAL[mid]
    if result is not None:
        goals = result["goals"]
        tag = " (real, pen)" if PENS[mid] else " (real)"
        return f"{head} {goals[a]}-{goals[b]}{tag}"
    tag = " (pen)" if PENS[mid] else ""
    line = f"{head} ({PROB[mid]*100:4.1f}%){tag}"
    if scoreline:
        line += f"   | {fmt_scorelines(a, b, n)}"   # shown order: a vs b
    return line

def main(argv=None):
    scoreline, n, seed, real_path = parse_args(sys.argv[1:] if argv is None else argv)
    real = load_real_results(real_path) if real_path else {}
    # Without --seed: random system seed -> each run may give a different champion.
    used_seed = seed if seed is not None else random.randrange(1_000_000_000)
    W, PROB, MATCH, PENS, REAL = resolve(random.Random(used_seed), real)

    print("=" * 72)
    print("WORLD CUP 2026 - PREDICTED BRACKET")
    print("favorite in regular time; DRAW -> penalties at random (the underdog can pass)")
    if seed is None:
        print(f"penalty seed: {used_seed} (random - reproduce this bracket with --seed {used_seed})")
    else:
        print(f"penalty seed: {used_seed} (fixed)")
    print("    (pen) = tie decided on penalties")
    if real:
        n_real = sum(1 for mid in ORDER if REAL[mid] is not None)
        print(f"    (real) = real result already played ({n_real} ties from {real_path}; not predicted)")
    if scoreline:
        print(f"+ most likely scoreline (Poisson, top-{n}); favorite's goals first")
    print("method: average(effective Elo D=600, Strength Index S=12)")
    print("=" * 72)

    round_of = {mid: lbl for lbl, ids in ROUNDS for mid in ids}
    for label, ids in ROUNDS:
        print(f"\n{label}")
        print("-" * 72)
        for mid in ids:
            print(_match_line(mid, MATCH, W, PROB, PENS, REAL, scoreline, n))

    champ = W[104]
    print("\n" + "=" * 72)
    print(f"  CHAMPION: {team_name(champ)}")
    print("=" * 72)

    print("\nChampion's path:")
    for mid in ORDER:
        if W[mid] != champ:
            continue
        a, b = MATCH[mid]
        rival = b if champ == a else a
        result = REAL[mid]
        if result is not None:
            goals = result["goals"]
            tag = " (real, pen)" if PENS[mid] else " (real)"
            line = f"  {round_of[mid]:<26} vs {team_name(rival):<20} {goals[champ]}-{goals[rival]}{tag}"
        else:
            tag = " (pen)" if PENS[mid] else ""
            line = f"  {round_of[mid]:<26} vs {team_name(rival):<20} ({PROB[mid]*100:4.1f}%){tag}"
            if scoreline:
                line += f"   | {fmt_scorelines(champ, rival, n)}"   # shown order: champion vs rival
        print(line)

if __name__ == "__main__":
    main()
