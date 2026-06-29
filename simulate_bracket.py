#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador Montecarlo del cuadro eliminatorio del Mundial 2026 (de dieciseisavos a la final).
Consume mundial2026_r32_dataset.json y, partiendo de la probabilidad por partido del modelo
(misma de predict.py: promedio de Elo efectivo y Strength Index), juega el torneo N veces
y estima, para cada equipo, la probabilidad de alcanzar cada ronda y de ser campeon.

Uso:
    python3 simulate_bracket.py            # 100.000 simulaciones (por defecto)
    python3 simulate_bracket.py 500000     # numero de simulaciones a eleccion
"""
import json, random, sys
from collections import defaultdict

random.seed(2026)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

D = 600.0   # denominador Elo (FIFA)
S = 12.0    # escala logistica del Strength Index

with open("mundial2026_r32_dataset.json", encoding="utf-8") as f:
    DATA = json.load(f)
ELO = {t["code"]: t["derived_metrics"]["effective_elo"]     for t in DATA["teams"]}
SI  = {t["code"]: t["derived_metrics"]["strength_index_0_100"] for t in DATA["teams"]}
NAME= {t["code"]: t["team"] for t in DATA["teams"]}

# probabilidad de que A avance sobre B (incluye prorroga/penales) -> cacheada por par
_cache = {}
def p_adv(a, b):
    k = (a, b)
    if k in _cache: return _cache[k]
    pe = 1.0 / (1.0 + 10 ** (-(ELO[a]-ELO[b]) / D))
    ps = 1.0 / (1.0 + 10 ** (-(SI[a]-SI[b])  / S))
    p = (pe + ps) / 2.0
    _cache[k] = p
    _cache[(b, a)] = 1 - p
    return p

# Estructura del cuadro (id_partido -> (fuente_a, fuente_b))
# Fuente = codigo de equipo (dieciseisavos) o "W<id>" (ganador de partido previo)
BRACKET = {
    # Dieciseisavos (Round of 32)
    73:("CAN","RSA"), 74:("GER","PAR"), 75:("NED","MAR"), 76:("BRA","JPN"),
    77:("FRA","SWE"), 78:("CIV","NOR"), 79:("MEX","ECU"), 80:("ENG","COD"),
    81:("USA","BIH"), 82:("BEL","SEN"), 83:("ESP","AUT"), 84:("POR","CRO"),
    85:("SUI","ALG"), 86:("ARG","CPV"), 87:("COL","GHA"), 88:("AUS","EGY"),
    # Octavos (Round of 16)
    89:("W74","W77"), 90:("W73","W75"), 91:("W76","W78"), 92:("W79","W80"),
    93:("W83","W84"), 94:("W81","W82"), 95:("W86","W88"), 96:("W85","W87"),
    # Cuartos
    97:("W89","W90"), 98:("W93","W94"), 99:("W91","W92"), 100:("W95","W96"),
    # Semifinales
    101:("W97","W98"), 102:("W99","W100"),
    # Final
    104:("W101","W102"),
}
ORDER = sorted(BRACKET)            # procesar por id ascendente
# que ronda "alcanza" el GANADOR de cada partido
ROUND_OF = ({m:"r16" for m in range(73,89)} | {m:"qf" for m in range(89,97)} |
            {m:"sf" for m in range(97,101)} | {m:"final" for m in (101,102)} |
            {104:"champ"})
ROUNDS = ["r16","qf","sf","final","champ"]

count = {r: defaultdict(int) for r in ROUNDS}

for _ in range(N):
    W = {}
    for mid in ORDER:
        sa, sb = BRACKET[mid]
        a = W[int(sa[1:])] if sa[0] == "W" else sa
        b = W[int(sb[1:])] if sb[0] == "W" else sb
        win = a if random.random() < p_adv(a, b) else b
        W[mid] = win
        count[ROUND_OF[mid]][win] += 1

# resultados por equipo
res = []
for c in NAME:
    res.append({
        "team": NAME[c], "code": c,
        "p_round_of_16": count["r16"][c]/N,
        "p_quarterfinal": count["qf"][c]/N,
        "p_semifinal":   count["sf"][c]/N,
        "p_final":       count["final"][c]/N,
        "p_champion":    count["champ"][c]/N,
    })
res.sort(key=lambda r: r["p_champion"], reverse=True)

json.dump({"simulations": N, "method": "avg(Elo D=600, StrengthIndex S=12)", "teams": res},
          open("mundial2026_montecarlo.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)

# tabla
print(f"\nMONTECARLO MUNDIAL 2026 — {N:,} simulaciones del cuadro completo\n")
print(f"{'#':>2} {'EQUIPO':<22}{'Octavos':>9}{'Cuartos':>9}{'Semis':>8}{'Final':>8}{'CAMPEON':>9}")
print("-"*67)
for i, r in enumerate(res, 1):
    print(f"{i:>2} {r['team']:<22}"
          f"{r['p_round_of_16']*100:>8.1f}%{r['p_quarterfinal']*100:>8.1f}%"
          f"{r['p_semifinal']*100:>7.1f}%{r['p_final']*100:>7.1f}%{r['p_champion']*100:>8.1f}%")
print("\nOK -> mundial2026_montecarlo.json")
