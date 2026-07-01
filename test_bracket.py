#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de bracket.py (stdlib unittest, sin dependencias).

Cubren el motor de probabilidad (p_adv/favorite), el modelo de goles analitico
(lambdas/top_scorelines/fmt_scorelines), la resolucion de cruces con penales por
azar (decide/resolve), el parseo de CLI (parse_args) y la salida impresa (main).

Ejecutar DESDE la carpeta del repo (bracket.py lee el dataset por ruta relativa):
    python3 -m unittest -v test_bracket
    python3 test_bracket.py
"""
import io
import os
import random
import unittest
from contextlib import redirect_stdout

import bracket

# Pareja con favorito aplastante (gana en los 90', NO va a penales)
FAV, DOG = "ARG", "CPV"     # Argentina vs Cabo Verde
# Pareja pareja cuyo marcador mas probable es empate (se define en penales)
EVEN_A, EVEN_B = "NED", "MAR"   # Paises Bajos vs Marruecos -> 1-1 modal


class _StubRng:
    """rng falso con .random() fijo, para forzar cada rama del sorteo de penales."""
    def __init__(self, value):
        self._value = value

    def random(self):
        return self._value


class TestProbabilidadPartido(unittest.TestCase):
    """p_adv y favorite: el motor cabeza a cabeza."""

    def test_p_adv_es_simetrica_y_suma_uno(self):
        self.assertAlmostEqual(bracket.p_adv(FAV, DOG) + bracket.p_adv(DOG, FAV), 1.0, places=9)

    def test_p_adv_de_un_equipo_contra_si_mismo_es_un_medio(self):
        self.assertAlmostEqual(bracket.p_adv(FAV, FAV), 0.5, places=9)

    def test_favorite_devuelve_al_mas_fuerte_con_prob_mayor_o_igual_a_medio(self):
        fav, dog, p_fav = bracket.favorite(FAV, DOG)
        self.assertEqual((fav, dog), (FAV, DOG))
        self.assertGreaterEqual(p_fav, 0.5)
        self.assertAlmostEqual(p_fav, bracket.p_adv(FAV, DOG), places=9)

    def test_favorite_es_independiente_del_orden_de_los_argumentos(self):
        self.assertEqual(bracket.favorite(FAV, DOG)[0], bracket.favorite(DOG, FAV)[0])


class TestModeloDeGoles(unittest.TestCase):
    """lambdas, top_scorelines y fmt_scorelines: marcador analitico (Poisson)."""

    def test_lambdas_suman_el_total_base_T(self):
        lf, ld = bracket.lambdas(FAV, DOG)
        self.assertAlmostEqual(lf + ld, bracket.T, places=9)

    def test_lambdas_favorito_no_queda_por_debajo_del_rival(self):
        lf, ld = bracket.lambdas(FAV, DOG)
        self.assertGreaterEqual(lf, ld)
        self.assertGreater(ld, 0.0)

    def test_lambdas_iguales_cuando_la_supremacia_es_negativa(self):
        lf, ld = bracket.lambdas(DOG, FAV)
        self.assertAlmostEqual(lf, ld, places=9)
        self.assertAlmostEqual(lf, bracket.T / 2.0, places=9)

    def test_top_scorelines_devuelve_n_filas_ordenadas_desc(self):
        rows = bracket.top_scorelines(FAV, DOG, 5)
        self.assertEqual(len(rows), 5)
        probs = [p for p, _, _ in rows]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top_scorelines_se_topa_en_los_81_marcadores_enumerables(self):
        rows = bracket.top_scorelines(FAV, DOG, 1_000_000)
        self.assertEqual(len(rows), (bracket.MAX_GOALS + 1) ** 2)

    def test_top_scorelines_suman_casi_uno(self):
        total = sum(p for p, _, _ in bracket.top_scorelines(FAV, DOG, 1_000_000))
        self.assertGreater(total, 0.99)
        self.assertLessEqual(total, 1.0 + 1e-9)

    def test_marcador_mas_probable_del_favorito_lo_pone_ganando(self):
        _, gf, gd = bracket.top_scorelines(FAV, DOG, 1)[0]
        self.assertGreater(gf, gd)

    def test_pareja_pareja_tiene_empate_como_marcador_mas_probable(self):
        _, gf, gd = bracket.top_scorelines(EVEN_A, EVEN_B, 1)[0]
        self.assertEqual(gf, gd)

    def test_fmt_scorelines_marca_penales_en_empate(self):
        self.assertIn("→pen", bracket.fmt_scorelines(EVEN_A, EVEN_B, 1))

    def test_fmt_scorelines_sin_penales_cuando_hay_favorito_claro(self):
        texto = bracket.fmt_scorelines(FAV, DOG, 1)
        self.assertNotIn("→pen", texto)
        self.assertIn("%", texto)

    def test_fmt_scorelines_orienta_los_goles_al_orden_mostrado(self):
        # México (home) vs Inglaterra (away): gana Inglaterra -> debe verse 0-1, no 1-0
        self.assertEqual(bracket.favorite("MEX", "ENG")[0], "ENG")   # favorito = visitante
        self.assertTrue(bracket.fmt_scorelines("MEX", "ENG", 1).startswith("0-1"))
        # invertir el orden de los equipos invierte el marcador
        self.assertTrue(bracket.fmt_scorelines("ENG", "MEX", 1).startswith("1-0"))

    def test_poisson_suma_uno_sobre_su_soporte(self):
        self.assertAlmostEqual(sum(bracket._poisson(k, 2.4) for k in range(50)), 1.0, places=9)


class TestDecidePenales(unittest.TestCase):
    """decide: empate -> penales por azar (el menos favorito puede pasar)."""

    def test_favorito_claro_gana_en_los_90_sin_penales(self):
        win, prob, pens = bracket.decide(FAV, DOG, _StubRng(0.999))
        self.assertEqual(win, FAV)
        self.assertFalse(pens)            # no se juega la moneda
        self.assertGreaterEqual(prob, 0.5)

    def test_empate_con_moneda_baja_lo_gana_el_favorito(self):
        # rng.random()=0.0 < p_fav -> avanza el favorito
        win, prob, pens = bracket.decide(EVEN_A, EVEN_B, _StubRng(0.0))
        fav, _, p_fav = bracket.favorite(EVEN_A, EVEN_B)
        self.assertTrue(pens)
        self.assertEqual(win, fav)
        self.assertAlmostEqual(prob, p_fav, places=9)

    def test_empate_con_moneda_alta_lo_gana_el_menos_favorito(self):
        # rng.random()=0.999 > p_fav -> avanza el rival (el de PEOR porcentaje)
        win, prob, pens = bracket.decide(EVEN_A, EVEN_B, _StubRng(0.999))
        fav, dog, p_fav = bracket.favorite(EVEN_A, EVEN_B)
        self.assertTrue(pens)
        self.assertEqual(win, dog)
        self.assertLess(prob, 0.5)        # el ganador tenia menos del 50%
        self.assertAlmostEqual(prob, 1 - p_fav, places=9)


class TestResolveCuadro(unittest.TestCase):
    """resolve: juega el cuadro entero con azar de penales."""

    def test_resuelve_todos_los_partidos(self):
        W, PROB, MATCH, PENS, REAL = bracket.resolve(random.Random(bracket.DEFAULT_SEED))
        self.assertEqual(set(W), set(bracket.ORDER))
        self.assertEqual(set(PENS), set(bracket.ORDER))
        self.assertEqual(set(REAL), set(bracket.ORDER))
        # Sin resultados reales, todos los cruces son predichos (REAL[mid] is None)
        self.assertTrue(all(REAL[mid] is None for mid in bracket.ORDER))

    def test_misma_semilla_da_el_mismo_cuadro(self):
        a = bracket.resolve(random.Random(7))[0]
        b = bracket.resolve(random.Random(7))[0]
        self.assertEqual(a, b)

    def test_semillas_distintas_pueden_dar_cuadros_distintos(self):
        brackets = {tuple(sorted(bracket.resolve(random.Random(s))[0].items())) for s in range(40)}
        self.assertGreater(len(brackets), 1)   # el azar de penales mueve resultados

    def test_cada_ganador_es_uno_de_los_dos_del_cruce(self):
        W, _, MATCH, _, _ = bracket.resolve(random.Random(7))
        for mid in bracket.ORDER:
            self.assertIn(W[mid], MATCH[mid])

    def test_algun_cruce_se_decide_en_penales(self):
        _, _, _, PENS, _ = bracket.resolve(random.Random(bracket.DEFAULT_SEED))
        self.assertTrue(any(PENS.values()))

    def test_un_menos_favorito_puede_avanzar_por_penales(self):
        # En algun cruce a penales el ganador tiene prob < 0.5 (no era el favorito)
        W, PROB, _, PENS, _ = bracket.resolve(random.Random(bracket.DEFAULT_SEED))
        upsets = [mid for mid in bracket.ORDER if PENS[mid] and PROB[mid] < 0.5]
        self.assertTrue(upsets)


class TestParseArgs(unittest.TestCase):
    """parse_args: --marcador [N] y --seed S (en cualquier orden)."""

    def test_sin_argumentos_usa_defaults(self):
        self.assertEqual(bracket.parse_args([]), (False, 1, bracket.DEFAULT_SEED, None))

    def test_flag_marcador_sin_n(self):
        self.assertEqual(bracket.parse_args(["--marcador"]), (True, 1, bracket.DEFAULT_SEED, None))

    def test_flag_marcador_con_n(self):
        self.assertEqual(bracket.parse_args(["--marcador", "3"]), (True, 3, bracket.DEFAULT_SEED, None))

    def test_alias_corto_m(self):
        self.assertEqual(bracket.parse_args(["-m", "2"]), (True, 2, bracket.DEFAULT_SEED, None))

    def test_seed_cambia_la_semilla(self):
        self.assertEqual(bracket.parse_args(["--seed", "7"]), (False, 1, 7, None))

    def test_marcador_y_seed_combinados_en_cualquier_orden(self):
        self.assertEqual(bracket.parse_args(["--seed", "7", "-m", "3"]), (True, 3, 7, None))

    def test_reales_sin_archivo_usa_el_default(self):
        self.assertEqual(
            bracket.parse_args(["--reales"]),
            (False, 1, bracket.DEFAULT_SEED, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_reales_con_archivo_explicito(self):
        self.assertEqual(
            bracket.parse_args(["--reales", "otros.json"]),
            (False, 1, bracket.DEFAULT_SEED, "otros.json"),
        )

    def test_alias_corto_r(self):
        self.assertEqual(
            bracket.parse_args(["-r"]),
            (False, 1, bracket.DEFAULT_SEED, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_reales_combinado_con_seed_y_marcador(self):
        self.assertEqual(
            bracket.parse_args(["--reales", "--seed", "7", "-m", "2"]),
            (True, 2, 7, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_reales_no_consume_una_bandera_siguiente(self):
        # tras --reales viene otra bandera, no un archivo -> usa el default
        self.assertEqual(
            bracket.parse_args(["--reales", "--marcador"]),
            (True, 1, bracket.DEFAULT_SEED, bracket.DEFAULT_RESULTS_FILE),
        )

    def test_n_cero_es_invalido(self):
        with self.assertRaises(SystemExit):
            bracket.parse_args(["--marcador", "0"])

    def test_seed_no_entero_es_invalido(self):
        with self.assertRaises(SystemExit):
            bracket.parse_args(["--seed", "x"])

    def test_argumento_desconocido_es_invalido(self):
        with self.assertRaises(SystemExit):
            bracket.parse_args(["--bogus"])


class TestResultadosReales(unittest.TestCase):
    """load_real_results y resolve/main con resultados reales (--reales)."""

    def _write(self, matches):
        """Escribe un JSON minimo de resultados y devuelve su ruta temporal."""
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

    def test_lee_solo_partidos_jugados(self):
        path = self._write([
            self._played("Brasil", "BRA", 2, "Japon", "JPN", 1, "Brasil"),
            {"home": {"team": "Portugal", "code": "POR", "score": None, "penalties": None},
             "away": {"team": "Croacia", "code": "CRO", "score": None, "penalties": None},
             "decided_by": None, "played": False, "winner": None},
        ])
        real = bracket.load_real_results(path)
        self.assertIn(frozenset(("BRA", "JPN")), real)
        self.assertNotIn(frozenset(("POR", "CRO")), real)

    def test_extrae_ganador_marcador_y_penales(self):
        path = self._write([
            self._played("Alemania", "GER", 1, "Paraguay", "PAR", 1, "Paraguay",
                         decided_by="penalties", pens_h=3, pens_a=4),
        ])
        real = bracket.load_real_results(path)
        r = real[frozenset(("GER", "PAR"))]
        self.assertEqual(r["winner"], "PAR")
        self.assertTrue(r["pens"])
        self.assertEqual(r["goals"], {"GER": 1, "PAR": 1})

    def test_ignora_ganador_inconsistente(self):
        path = self._write([
            self._played("Brasil", "BRA", 2, "Japon", "JPN", 1, "Marte"),
        ])
        self.assertEqual(bracket.load_real_results(path), {})

    def test_resolve_usa_el_ganador_real_no_predice(self):
        # Marruecos avanza por resultado real aunque NO sea el favorito del modelo
        real = {frozenset(("NED", "MAR")): {"winner": "MAR", "pens": True,
                                            "goals": {"NED": 1, "MAR": 1}}}
        W, PROB, MATCH, PENS, REAL = bracket.resolve(random.Random(7), real)
        # cruce 75 es NED vs MAR en el BRACKET oficial
        self.assertEqual(W[75], "MAR")
        self.assertEqual(PROB[75], 1.0)
        self.assertIsNotNone(REAL[75])
        self.assertTrue(PENS[75])

    def test_resolve_sin_real_es_identico_al_comportamiento_previo(self):
        W_real = bracket.resolve(random.Random(7), {})[0]
        W_none = bracket.resolve(random.Random(7))[0]
        self.assertEqual(W_real, W_none)

    def test_main_con_reales_marca_los_cruces_reales(self):
        path = self._write([
            self._played("Alemania", "GER", 1, "Paraguay", "PAR", 1, "Paraguay",
                         decided_by="penalties", pens_h=3, pens_a=4),
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            bracket.main(["--reales", path, "--seed", "7"])
        out = buf.getvalue()
        self.assertIn("(real)", out)
        self.assertIn("resultado real ya jugado", out)


class TestSalidaMain(unittest.TestCase):
    """Smoke tests de la salida impresa."""

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bracket.main(argv)
        return buf.getvalue()

    def test_modo_default_imprime_campeon_y_semilla(self):
        out = self._run(["--seed", "7"])
        self.assertIn("CAMPEON:", out)
        self.assertIn("semilla de penales: 7", out)
        self.assertNotIn("marcador mas probable", out)

    def test_modo_marcador_imprime_cabecera_y_marca_penales(self):
        out = self._run(["--marcador", "--seed", "7"])
        self.assertIn("marcador mas probable", out)
        self.assertIn("(pen)", out)

    def test_modo_marcador_top3_muestra_tres_resultados_en_un_cruce(self):
        out = self._run(["--marcador", "3", "--seed", "7"])
        linea = next(l for l in out.splitlines() if l.strip().startswith("[86]"))
        marcadores = [tok for tok in linea.split() if "-" in tok and tok[0].isdigit()]
        self.assertEqual(len(marcadores), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
