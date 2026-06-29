#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de prediccion cabeza a cabeza para el Mundial 2026 (dieciseisavos).
Consume mundial2026_r32_dataset.json y devuelve probabilidades para un duelo A vs B.

Dos metodos:
  A) Elo efectivo (recomendado): usa 'effective_elo' (ranking FIFA Elo + ajustes por
     forma, valor de plantilla y palmares). Formula FIFA/Elo, denominador D=600.
  B) Strength Index logistico: usa 'strength_index_0_100' con escala S=12.

En llave directa (a partido unico con prorroga y penales) se reporta P(avanzar).
Para 90' se separa una banda de empate con el parametro draw_band.
"""
import json, sys, math

D = 600.0   # denominador Elo estilo FIFA
S = 12.0    # escala logistica del Strength Index

with open("mundial2026_r32_dataset.json", encoding="utf-8") as f:
    DATA = json.load(f)
BY_CODE = {t["code"]: t for t in DATA["teams"]}
BY_NAME = {t["team"].lower(): t for t in DATA["teams"]}

def find(key):
    k = key.strip()
    if k.upper() in BY_CODE: return BY_CODE[k.upper()]
    if k.lower() in BY_NAME: return BY_NAME[k.lower()]
    for t in DATA["teams"]:
        if k.lower() in t["team"].lower(): return t
    raise KeyError(f"Equipo no encontrado: {key}")

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
    # probabilidad combinada (media de ambos metodos) para avanzar (incluye prorroga/penales)
    pa_adv = (pa_elo + pa_si) / 2.0
    res = {
      "A": A["team"], "B": B["team"],
      "elo_A": ra, "elo_B": rb, "SI_A": sa, "SI_B": sb,
      "P_A_advance_elo": round(pa_elo, 3),
      "P_A_advance_si":  round(pa_si, 3),
      "P_A_advance":     round(pa_adv, 3),
      "P_B_advance":     round(1 - pa_adv, 3),
    }
    if draw_band > 0:  # version 90' con empate
        # reparte una banda de empate proporcional a la cercania
        closeness = 1 - abs(pa_adv - 0.5) * 2  # 1 si parejo, 0 si paliza
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

# 16 cruces oficiales de dieciseisavos
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
        print("PRONOSTICO DIECISEISAVOS - MUNDIAL 2026  (P de avanzar, incl. prorroga/penales)")
        print("="*104)
        for a,b in TIES:
            print(fmt(predict(a,b)))
        print("\nUso individual:  python3 predict.py ARG CPV         (llave directa)")
        print("                 python3 predict.py ARG CPV 0.26    (con banda de empate a 90')")
