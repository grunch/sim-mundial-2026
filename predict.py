#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head-to-head prediction engine for the World Cup 2026 (Round of 32).
Consumes worldcup2026_r32_dataset.json and returns probabilities for an A vs B tie.

Two methods:
  A) Effective Elo (recommended): uses 'effective_elo' (FIFA Elo ranking + adjustments
     for form, squad value and honours). FIFA/Elo formula, denominator D=600.
  B) Logistic Strength Index: uses 'strength_index_0_100' with scale S=12.

In a direct tie (single match with extra time and penalties) P(advance) is reported.
For regular time, a draw band is split out with the draw_band parameter.
"""
import json, sys, math

D = 600.0   # FIFA-style Elo denominator
S = 12.0    # logistic scale of the Strength Index

with open("worldcup2026_r32_dataset.json", encoding="utf-8") as f:
    DATA = json.load(f)
BY_CODE = {t["code"]: t for t in DATA["teams"]}
BY_NAME = {t["team"].lower(): t for t in DATA["teams"]}

def find(key):
    k = key.strip()
    if k.upper() in BY_CODE: return BY_CODE[k.upper()]
    if k.lower() in BY_NAME: return BY_NAME[k.lower()]
    for t in DATA["teams"]:
        if k.lower() in t["team"].lower(): return t
    raise KeyError(f"Team not found: {key}")

def p_elo(ra, rb):
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / D))

def p_si(sa, sb):
    return 1.0 / (1.0 + 10 ** (-(sa - sb) / S))

def predict(a_key, b_key, draw_band=0.0):
    A, B = find(a_key), find(b_key)
    ra = A["derived_metrics"]["effective_elo"]
    rb = B["derived_metrics"]["effective_elo"]
    sa = A["derived_metrics"]["strength_index_0_100"]
    sb = B["derived_metrics"]["strength_index_0_100"]
    pa_elo = p_elo(ra, rb)
    pa_si  = p_si(sa, sb)
    # combined probability (average of both methods) to advance (incl. extra time/penalties)
    pa_adv = (pa_elo + pa_si) / 2.0
    res = {
      "A": A["team"], "B": B["team"],
      "elo_A": ra, "elo_B": rb, "SI_A": sa, "SI_B": sb,
      "P_A_advance_elo": round(pa_elo, 3),
      "P_A_advance_si":  round(pa_si, 3),
      "P_A_advance":     round(pa_adv, 3),
      "P_B_advance":     round(1 - pa_adv, 3),
    }
    if draw_band > 0:  # regular-time version with a draw
        # split a draw band proportional to how close the tie is
        closeness = 1 - abs(pa_adv - 0.5) * 2  # 1 if even, 0 if a rout
        pdraw = draw_band * closeness
        res["P_draw_90"] = round(pdraw, 3)
        res["P_A_win_90"] = round(pa_adv * (1 - pdraw), 3)
        res["P_B_win_90"] = round((1 - pa_adv) * (1 - pdraw), 3)
    return res

def fmt(r):
    line = (f"{r['A']:<22} {r['P_A_advance']*100:5.1f}%  "
            f"vs  {r['P_B_advance']*100:5.1f}% {r['B']:>22}   "
            f"[Elo {r['P_A_advance_elo']*100:4.1f}/{(1-r['P_A_advance_elo'])*100:4.1f}  "
            f"SI {r['P_A_advance_si']*100:4.1f}/{(1-r['P_A_advance_si'])*100:4.1f}]")
    fav = r['A'] if r['P_A_advance'] >= 0.5 else r['B']
    return line + f"  -> {fav}"

# 16 official Round of 32 ties
TIES = [
 ("CAN","RSA"), ("GER","PAR"), ("NED","MAR"), ("BRA","JPN"),
 ("FRA","SWE"), ("CIV","NOR"), ("MEX","ECU"), ("ENG","COD"),
 ("USA","BIH"), ("BEL","SEN"), ("POR","CRO"), ("ESP","AUT"),
 ("SUI","ALG"), ("ARG","CPV"), ("COL","GHA"), ("AUS","EGY"),
]

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        print(json.dumps(predict(sys.argv[1], sys.argv[2],
              draw_band=float(sys.argv[3]) if len(sys.argv)>3 else 0.0),
              ensure_ascii=False, indent=2))
    else:
        print("="*104)
        print("ROUND OF 32 FORECAST - WORLD CUP 2026  (P of advancing, incl. extra time/penalties)")
        print("="*104)
        for a,b in TIES:
            print(fmt(predict(a,b)))
        print("\nSingle use:  python3 predict.py ARG CPV         (direct tie)")
        print("             python3 predict.py ARG CPV 0.26    (with a regular-time draw band)")
