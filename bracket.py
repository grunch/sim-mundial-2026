#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cuadro (bracket) del Mundial 2026: de dieciseisavos a la final.
Reutiliza el dataset y la MISMA logica de probabilidad por partido que
predict.py / simulate_bracket.py (promedio de Elo efectivo y Strength Index).

Avance de cada cruce:
  - Si el marcador mas probable (modelo Poisson, ver abajo) NO es empate, avanza
    el equipo que va por delante (el favorito) en el tiempo reglamentario.
  - Si el marcador mas probable es EMPATE, se va a PENALES y el ganador se decide
    POR AZAR: una moneda ponderada por la fuerza (p_adv), de modo que el menos
    favorito TAMBIEN puede pasar. Por defecto el azar es REAL (semilla del sistema):
    cada corrida puede dar otro campeon. Con --seed S el cuadro es reproducible.

Modo marcador (--marcador [N]):
    Predice el MARCADOR mas probable de cada cruce con un modelo de goles de
    Poisson derivado del Elo. El marcador y su porcentaje se calculan de forma
    ANALITICA y EXACTA (no por simulacion): para dos Poisson independientes,
    P(i-j) = Poisson(i; lambda_fav) * Poisson(j; lambda_rival).
    N = cuantos marcadores mas probables mostrar por cruce (default 1).
    Un marcador empatado se marca "→pen": ese cruce se decide en penales (azar).

(Para "quien es campeon con mas frecuencia" sobre miles de torneos, usa
 simulate_bracket.py: esa pregunta agrega muchas realizaciones como esta.)

Uso:
    python3 bracket.py                       # cuadro NUEVO cada vez (penales al azar)
    python3 bracket.py --seed 7              # reproducible: misma semilla, mismo cuadro
    python3 bracket.py --marcador            # + marcador mas probable de cada cruce
    python3 bracket.py --marcador 3 --seed 7 # top-3 marcadores y semilla fija 7
"""
import json, math, random, sys

D = 600.0   # denominador Elo (FIFA) — identico a predict.py / simulate_bracket.py
S = 12.0    # escala logistica del Strength Index
T = 2.6     # goles totales esperados base por partido (media de knockout mundialista)
C = 200.0   # puntos de Elo por gol de "supremacia" (mapea ventaja Elo -> goles)
MAX_GOALS = 8        # tope para enumerar marcadores (la cola de Poisson es despreciable)
DEFAULT_SEED = None  # sin --seed -> azar REAL del sistema: cada corrida puede diferir

with open("mundial2026_r32_dataset.json", encoding="utf-8") as f:
    DATA = json.load(f)
ELO  = {t["code"]: t["derived_metrics"]["effective_elo"]        for t in DATA["teams"]}
SI   = {t["code"]: t["derived_metrics"]["strength_index_0_100"] for t in DATA["teams"]}
NAME = {t["code"]: t["team"] for t in DATA["teams"]}

# Nombres en espanol para la impresion (cae al nombre del dataset si falta)
ES = {
 "FRA":"Francia","ARG":"Argentina","ESP":"España","BRA":"Brasil","ENG":"Inglaterra",
 "GER":"Alemania","NED":"Países Bajos","POR":"Portugal","MAR":"Marruecos","BEL":"Bélgica",
 "MEX":"México","CRO":"Croacia","USA":"Estados Unidos","SUI":"Suiza","COL":"Colombia",
 "JPN":"Japón","NOR":"Noruega","CIV":"Costa de Marfil","ECU":"Ecuador","AUT":"Austria",
 "SEN":"Senegal","SWE":"Suecia","CAN":"Canadá","ALG":"Argelia","EGY":"Egipto",
 "AUS":"Australia","PAR":"Paraguay","COD":"RD Congo","GHA":"Ghana","BIH":"Bosnia y Herzegovina",
 "RSA":"Sudáfrica","CPV":"Cabo Verde",
}
def es(code): return ES.get(code, NAME.get(code, code))

def p_adv(a, b):
    """P(A avanza sobre B), incluye prorroga/penales. Misma formula del motor."""
    pe = 1.0 / (1.0 + 10 ** (-(ELO[a] - ELO[b]) / D))
    ps = 1.0 / (1.0 + 10 ** (-(SI[a]  - SI[b])  / S))
    return (pe + ps) / 2.0

def favorite(a, b):
    """Devuelve (favorito, rival, prob_favorito) segun p_adv."""
    pa = p_adv(a, b)
    return (a, b, pa) if pa >= 0.5 else (b, a, 1 - pa)

# ---------------------------------------------------------------------------
# Modelo de goles (marcador mas probable, analitico y exacto)
# ---------------------------------------------------------------------------
def _poisson(k, lam):
    """P(exactamente k goles) bajo Poisson(lam)."""
    return math.exp(-lam) * lam ** k / math.factorial(k)

def lambdas(fav, dog):
    """Goles esperados (lambda) de favorito y rival a partir del Elo.

    La supremacia se acota a >= 0 para que el favorito nunca quede por debajo del
    rival, manteniendo el marcador coherente con quien es favorito.
    """
    sup = max(0.0, (ELO[fav] - ELO[dog]) / C)   # supremacia en goles (>= 0)
    sup = min(sup, T - 0.4)                      # evita lambda no positiva del rival
    return (T + sup) / 2.0, (T - sup) / 2.0

def top_scorelines(fav, dog, n):
    """Top-n marcadores (fav-dog) mas probables con su probabilidad EXACTA.

    Devuelve lista de (prob, goles_fav, goles_dog) ordenada de mayor a menor.
    """
    lf, ld = lambdas(fav, dog)
    rows = [(_poisson(i, lf) * _poisson(j, ld), i, j)
            for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    rows.sort(reverse=True)
    return rows[:n]

def fmt_scorelines(home, away, n):
    """Texto compacto '2-0 (21.4%) · 1-0 (17.8%)' en el orden 'home vs away'.

    El modelo calcula el marcador favorito-primero, pero aqui se reorienta para que
    los goles correspondan al ORDEN EN QUE SE MUESTRAN los equipos (home-away). Si
    el favorito es el visitante, '1-0' del favorito se imprime '0-1'.
    """
    fav, dog, _ = favorite(home, away)
    parts = []
    for prob, gf, gd in top_scorelines(fav, dog, n):
        g_home, g_away = (gf, gd) if home == fav else (gd, gf)
        pen = "→pen" if g_home == g_away else ""   # empate -> se define en penales
        parts.append(f"{g_home}-{g_away} ({prob*100:.1f}%){pen}")
    return " · ".join(parts)

def decide(a, b, rng):
    """Resuelve el cruce a-vs-b. Devuelve (ganador, prob_ganador, fue_penales).

    - Si el marcador mas probable NO es empate -> gana el favorito en los 90'.
    - Si es EMPATE -> penales: el ganador se sortea con una moneda ponderada por
      la fuerza (prob del favorito = p_adv). El menos favorito tambien puede pasar.
    """
    fav, dog, p_fav = favorite(a, b)
    _, gf, gd = top_scorelines(fav, dog, 1)[0]   # marcador mas probable
    if gf != gd:
        return fav, p_fav, False
    win = fav if rng.random() < p_fav else dog   # empate -> penales (azar)
    return win, (p_fav if win == fav else 1 - p_fav), True

# Estructura oficial del cuadro (id_partido -> (fuente_a, fuente_b)).
# Fuente = codigo de equipo (dieciseisavos) o "W<id>" (ganador de partido previo).
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
    ("DIECISEISAVOS DE FINAL  (Round of 32)", list(range(73, 89))),
    ("OCTAVOS DE FINAL  (Round of 16)",       list(range(89, 97))),
    ("CUARTOS DE FINAL",                      list(range(97, 101))),
    ("SEMIFINALES",                           [101, 102]),
    ("FINAL",                                 [104]),
]

def source_team(src, W):
    """Resuelve una fuente ('CAN' o 'W74') al codigo del equipo presente."""
    return W[int(src[1:])] if src[0] == "W" else src

def resolve(rng):
    """Juega el cuadro entero con el generador 'rng' (azar de penales).

    Devuelve winner[id], prob[id], match[id]=(a,b), pens[id]=bool.
    """
    W, PROB, MATCH, PENS = {}, {}, {}, {}
    for mid in ORDER:
        sa, sb = BRACKET[mid]
        a, b = source_team(sa, W), source_team(sb, W)
        win, prob, pens = decide(a, b, rng)
        W[mid], PROB[mid], MATCH[mid], PENS[mid] = win, prob, (a, b), pens
    return W, PROB, MATCH, PENS

def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def parse_args(argv):
    """Lee la CLI (flags en cualquier orden). Devuelve (marcador, n, seed).

        --marcador | -m [N]   activa marcadores; N = cuantos mostrar (default 1)
        --seed S              semilla del azar de penales (default DEFAULT_SEED)
    """
    marcador, n, seed = False, 1, DEFAULT_SEED
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--marcador", "-m"):
            marcador = True
            if i + 1 < len(argv) and _is_int(argv[i + 1]):
                n = int(argv[i + 1]); i += 1
        elif tok == "--seed":
            if i + 1 >= len(argv) or not _is_int(argv[i + 1]):
                sys.exit("--seed requiere un entero")
            seed = int(argv[i + 1]); i += 1
        else:
            sys.exit(f"Argumento no reconocido: {tok!r}")
        i += 1
    if n < 1:
        sys.exit("N debe ser >= 1")
    return marcador, n, seed

def main(argv=None):
    marcador, n, seed = parse_args(sys.argv[1:] if argv is None else argv)
    # Sin --seed: semilla aleatoria del sistema -> cada corrida puede dar otro campeon.
    used_seed = seed if seed is not None else random.randrange(1_000_000_000)
    W, PROB, MATCH, PENS = resolve(random.Random(used_seed))

    print("=" * 72)
    print("MUNDIAL 2026 — CUADRO PREDICHO")
    print("favorito en los 90'; EMPATE -> penales por azar (el menos favorito puede pasar)")
    if seed is None:
        print(f"semilla de penales: {used_seed} (aleatoria — repite este cuadro con --seed {used_seed})")
    else:
        print(f"semilla de penales: {used_seed} (fija)")
    print("    (pen) = cruce decidido en penales")
    if marcador:
        print(f"+ marcador mas probable (Poisson, top-{n}); goles del favorito primero")
    print("metodo: promedio(Elo efectivo D=600, Strength Index S=12)")
    print("=" * 72)

    round_of = {mid: lbl.split("  ")[0] for lbl, ids in ROUNDS for mid in ids}
    for label, ids in ROUNDS:
        print(f"\n{label}")
        print("-" * 72)
        for mid in ids:
            a, b = MATCH[mid]
            win = W[mid]
            tag = " (pen)" if PENS[mid] else ""
            line = (f"  [{mid}] {es(a):<20} vs {es(b):<20}"
                    f"  ->  {es(win):<18} ({PROB[mid]*100:4.1f}%){tag}")
            if marcador:
                line += f"   | {fmt_scorelines(a, b, n)}"   # orden mostrado: a vs b
            print(line)

    champ = W[104]
    print("\n" + "=" * 72)
    print(f"  CAMPEON: {es(champ)}")
    print("=" * 72)

    print("\nCamino del campeon:")
    for mid in ORDER:
        if W[mid] != champ:
            continue
        a, b = MATCH[mid]
        rival = b if champ == a else a
        tag = " (pen)" if PENS[mid] else ""
        line = f"  {round_of[mid]:<26} vs {es(rival):<20} ({PROB[mid]*100:4.1f}%){tag}"
        if marcador:
            line += f"   | {fmt_scorelines(champ, rival, n)}"   # orden mostrado: campeon vs rival
        print(line)

if __name__ == "__main__":
    main()
