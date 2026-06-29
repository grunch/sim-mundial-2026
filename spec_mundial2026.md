# SPEC — Dataset y motor de predicción · Dieciseisavos del Mundial FIFA 2026

**Versión:** 1.0 · **Fecha de corte de datos:** 28–29 jun 2026 · **Etapa:** Round of 32 (32 equipos)
**Entregables:** `mundial2026_r32_dataset.json` (datos por equipo) · `mundial2026_r32_matchups.json` (16 cruces) · `build_dataset.py` (generador) · `predict.py` (motor de pronóstico)

---

## 1. Objetivo

Construir, para cada uno de los **32 equipos** clasificados a dieciseisavos del Mundial 2026, un objeto JSON autocontenido con datos relevantes y métricas normalizadas, de modo que puedan alimentar un **modelo de predicción cabeza a cabeza** entre cualquier par de selecciones (no solo el cruce que les tocó). El diseño prioriza que las cifras sean **reproducibles** y **auditables**: cada métrica derivada se calcula con una fórmula explícita a partir de datos crudos que también quedan guardados.

## 2. Universo de equipos (cómo clasificaron)

Formato nuevo de 48 equipos → 12 grupos de 4. Avanzan los **2 primeros de cada grupo (24)** + los **8 mejores terceros**. Los 32 clasificados:

| Grupo | 1º (group_winner) | 2º (runner_up) | Mejor 3º (best_third) |
|---|---|---|---|
| A | México | Sudáfrica | — |
| B | Suiza | Canadá | Bosnia y Herzegovina |
| C | Brasil | Marruecos | — |
| D | Estados Unidos | Australia | Paraguay |
| E | Alemania | Costa de Marfil | Ecuador |
| F | Países Bajos | Japón | Suecia |
| G | Bélgica | Egipto | — |
| H | España | Cabo Verde | — |
| I | Francia | Noruega | Senegal |
| J | Argentina | Austria | Argelia |
| K | Colombia | Portugal | RD Congo |
| L | Inglaterra | Croacia | Ghana |

## 3. Métricas (definición y fuente)

Por cada equipo se recogen **cuatro ejes** de datos crudos más el historial:

1. **Ranking FIFA** — `points_official_2026_06_11` (puntos oficiales del 11-jun-2026, base pre-Mundial; sistema Elo de la FIFA, denominador 600) y `points_live_in_tournament` (proyección que ya incorpora resultados del torneo). *Se usa el oficial como base para no doblar el conteo con el eje de rendimiento.*
   Fuente: FIFA / football-ranking.com.
2. **Valor de plantilla (Transfermarkt)** — `value_eur_millions`, valuación acumulada pre-torneo (~27-jun-2026), en EUR.
   Fuente: Transfermarkt vía PlanetFootball/OneFootball.
3. **Rendimiento en el Mundial 2026** — fase de grupos: `points`, `goal_difference`, derivados por partido.
   Fuente: tablas finales de grupos (NBC Sports / FIFA).
4. **Experiencia / palmarés mundialista** — `titles`, `title_years`, `best_finish_pre_2026`, `appearances_incl_2026`, `defending_champion`.
   Fuente: registro histórico de Copas del Mundo.

## 4. Esquema del JSON por equipo

```jsonc
{
  "team": "Argentina", "code": "ARG", "confederation": "CONMEBOL",
  "qualification": { "group": "J", "finish_position": 1,
                     "finish_type": "group_winner", "finish_label_es": "1ro de grupo" },
  "fifa_ranking": { "rank_official_2026_06_11": 1,
                    "points_official_2026_06_11": 1877.0,
                    "points_live_in_tournament": 1907.41, "source_note": "..." },
  "squad_value_transfermarkt": { "value_eur_millions": 807.5, "as_of": "2026-06-27", "currency": "EUR" },
  "world_cup_2026_performance": { "stage_reached": "Round of 32", "group_matches_played": 3,
                    "points": 9, "goal_difference": 7,
                    "points_per_match": 3.0, "gd_per_match": 2.333, "form_raw_index": 11.8 },
  "world_cup_history": { "titles": 3, "title_years": [1978,1986,2022],
                    "best_finish_pre_2026": "Campeon", "appearances_incl_2026": 19,
                    "defending_champion": true, "pedigree_score_0_100": 100.0 },
  "derived_metrics": { "norm_fifa": 1.0, "norm_value": ..., "norm_form": ...,
                    "norm_pedigree": ..., "strength_index_0_100": 95.57, "effective_elo": 2025.4 },
  "round_of_32": { "opponent": "Cape Verde", "opponent_code": "CPV",
                    "match_id": 86, "venue": "Hard Rock Stadium, Miami", "date": "2026-06-30" },
  "data_notes": []
}
```

## 5. Métricas derivadas (fórmulas exactas)

Todas las normalizaciones se calculan **sobre el pool de los 32 equipos**.

**5.1 Normalizaciones min-max → [0,1]**
- `norm_fifa  = minmax(points_official)`
- `norm_value = minmax(log10(value_eur_millions))`  ← escala logarítmica porque el valor está muy sesgado (Francia €1 520 M vs Cabo Verde €49 M).
- `norm_form  = minmax(form_raw)`, con `form_raw = points + 0.4 · goal_difference`
- `norm_pedigree = minmax(pedigree_raw)`

**5.2 Pedigrí (palmarés mundialista), proxy reproducible**
```
best_pts:  Campeón=30 · Subcampeón=22 · Semifinal/3º-4º=16 · Cuartos=10 · Octavos=5 · Grupos=0
pedigree_raw = titles*20 + best_pts + min(appearances,25)*0.8     (tope 100)
```

**5.3 Strength Index (SI, 0–100)** — índice compuesto, pesos configurables:
```
SI = 100 · ( 0.40·norm_fifa + 0.20·norm_value + 0.25·norm_form + 0.15·norm_pedigree )
```

**5.4 Elo efectivo** — toma el Elo FIFA real y lo ajusta con los otros 3 ejes (en *puntos Elo por desviación estándar*):
```
effective_elo = points_official
              + 40·z(form_raw) + 25·z(log10_value) + 20·z(pedigree_raw)
```
(`z` = puntaje estándar respecto al pool).

## 6. Motor de predicción cabeza a cabeza (`predict.py`)

Dos métodos independientes; el script promedia ambos para la probabilidad de **avanzar** (llave directa, ya incluye prórroga y penales):

**Método A — Elo efectivo (recomendado).** El ranking FIFA ya es Elo, así que la probabilidad usa la fórmula estándar:
```
P(A avanza) = 1 / ( 1 + 10^(-(elo_A - elo_B)/600) )
```

**Método B — Strength Index logístico.**
```
P(A avanza) = 1 / ( 1 + 10^(-(SI_A - SI_B)/12) )
```

**Combinada:** `P_A_advance = (P_elo + P_SI) / 2`.

**Versión a 90' con empate** (para fase de grupos o mercados 1X2): se reparte una banda de empate proporcional a la paridad del duelo:
```
closeness = 1 - |P_A - 0.5|·2
P_draw = draw_band · closeness        (sugerido draw_band ≈ 0.24–0.28)
P_A_win = P_A·(1 - P_draw) ;  P_B_win = (1 - P_A)·(1 - P_draw)
```

**Uso:**
```
python3 predict.py ARG CPV          # llave directa (P de avanzar)
python3 predict.py BEL SEN 0.28     # a 90' con banda de empate
python3 predict.py                  # corre los 16 cruces de dieciseisavos
```

## 7. Ranking por Strength Index (los 32)

| # | Equipo | Grp | Pos | FIFA | TM €M | WC26 (pts/DG) | Pedigrí | SI | eELO |
|--:|--------|:--:|:--:|--:|--:|:--:|--:|--:|--:|
| 1 | Francia | I | 1º | 1871 | 1520 | 9/+8 | 84 | 97.1 | 2038 |
| 2 | Argentina | J | 1º | 1877 | 808 | 9/+7 | 100 | 95.2 | 2032 |
| 3 | España | H | 1º | 1875 | 1220 | 7/+5 | 64 | 84.4 | 1974 |
| 4 | Brasil | C | 1º | 1766 | 928 | 7/+6 | 100 | 81.1 | 1888 |
| 5 | Inglaterra | L | 1º | 1828 | 1360 | 7/+4 | 64 | 80.4 | 1924 |
| 6 | Alemania | E | 1º | 1736 | 947 | 6/+6 | 100 | 76.3 | 1843 |
| 7 | Países Bajos | F | 1º | 1754 | 754 | 7/+6 | 32 | 68.7 | 1824 |
| 8 | Marruecos | C | 2º | 1755 | 448 | 7/+3 | 22 | 60.9 | 1785 |
| 9 | Portugal | K | 2º | 1768 | 1010 | 4/+5 | 23 | 60.9 | 1788 |
| 10 | México | A | 1º | 1687 | 192 | 9/+6 | 24 | 60.0 | 1744 |
| 11 | Bélgica | G | 1º | 1742 | 548 | 5/+3 | 28 | 56.7 | 1751 |
| 12 | Estados Unidos | D | 1º | 1671 | 386 | 6/+4 | 26 | 52.7 | 1690 |
| 13 | Suiza | B | 1º | 1650 | 332 | 7/+4 | 20 | 52.2 | 1677 |
| 14 | Croacia | L | 2º | 1715 | 387 | 6/+0 | 28 | 52.0 | 1711 |
| 15 | Colombia | K | 1º | 1698 | 302 | 6/+3 | 16 | 50.7 | 1698 |
| 16 | Japón | F | 2º | 1662 | 271 | 5/+4 | 11 | 45.1 | 1646 |
| 17 | Senegal | I | 3º | 1684 | 478 | 3/+3 | 13 | 43.8 | 1575 |
| 18 | Noruega | I | 2º | 1557 | 590 | 6/+1 | 8 | 40.7 | 1558 |
| 19 | Costa de Marfil | E | 2º | 1541 | 522 | 6/+2 | 3 | 39.1 | 1542 |
| 20 | Ecuador | E | 3º | 1599 | 369 | 4/+0 | 9 | 34.7 | 1551 |
| 21 | Austria | J | 2º | 1597 | 245 | 4/+0 | 22 | 34.2 | 1546 |
| 22 | Canadá | B | 2º | 1559 | 199 | 4/+5 | 2 | 32.5 | 1520 |
| 23 | Suecia | F | 3º | 1510 | 406 | 4/+0 | 32 | 32.1 | 1480 |
| 24 | Egipto | G | 2º | 1562 | 116 | 5/+2 | 3 | 29.2 | 1505 |
| 25 | Argelia | J | 3º | 1571 | 257 | 4/-2 | 9 | 28.3 | 1500 |
| 26 | Australia | D | 2º | 1579 | 77 | 4/+0 | 11 | 24.3 | 1488 |
| 27 | Paraguay | D | 3º | 1505 | 154 | 4/-2 | 17 | 21.6 | 1425 |
| 28 | RD Congo | K | 3º | 1474 | 144 | 4/+1 | 2 | 19.8 | 1400 |
| 29 | Ghana | L | 3º | 1347 | 234 | 4/+0 | 13 | 13.7 | 1289 |
| 30 | Bosnia y Herzegovina | B | 3º | 1387 | 146 | 4/-1 | 2 | 11.1 | 1302 |
| 31 | Sudáfrica | A | 2º | 1428 | 49 | 4/-1 | 3 | 8.1 | 1313 |
| 32 | Cabo Verde | H | 2º | 1371 | 49 | 3/+0 | 1 | 1.8 | 1245 |

## 8. Pronóstico de los 16 cruces (demostración del motor)

P de avanzar (incluye prórroga/penales). Cruces parejos donde el modelo invierte al favorito nominal marcados con ⚑.

| Cruce | Favorito (P avanzar) |
|---|---|
| Canadá vs Sudáfrica | Canadá **84%** |
| Alemania vs Paraguay | Alemania **92%** |
| Países Bajos vs Marruecos | Países Bajos **68%** |
| Brasil vs Japón | Brasil **86%** |
| Francia vs Suecia | Francia **95%** |
| Costa de Marfil vs Noruega ⚑ | Noruega **55%** |
| México vs Ecuador | México **84%** |
| Inglaterra vs RD Congo | Inglaterra **94%** |
| Estados Unidos vs Bosnia | EE. UU. **91%** |
| Bélgica vs Senegal | Bélgica **79%** |
| Portugal vs Croacia | Portugal **71%** |
| España vs Austria | España **92%** |
| Suiza vs Argelia | Suiza **83%** |
| Argentina vs Cabo Verde | Argentina **98%** |
| Colombia vs Ghana | Colombia **91%** |
| Australia vs Egipto ⚑ | Egipto **62%** |

## 9. Cómo ajustar el modelo

- **Pesos del SI** → diccionario `W` en `build_dataset.py`. Subir `form` da más peso al estado de forma en el torneo; subir `fifa` lo hace más conservador.
- **Coeficientes Elo** → `ELO_ADJ` (puntos Elo por desviación estándar de forma/valor/pedigrí).
- **Denominador Elo `D` y escala `S`** → en `predict.py`. `D` más alto = predicciones más planas (favoritos menos dominantes).
- Tras cambiar pesos: re-ejecutar `build_dataset.py` y luego `predict.py`.

## 10. Limitaciones y advertencias (importante)

- **Torneo en curso:** las cifras `live` y las posiciones cambian con cada jornada; el dataset es una foto al 28–29 jun 2026.
- **Pedigrí:** es un proxy reproducible (títulos + mejor resultado + presencias). **Subestima** a finalistas/semifinalistas sin título (Países Bajos, Suecia, Portugal, Croacia). Si te importa el palmarés, suבe el peso o reemplaza la fórmula.
- **El modelo no captura**: lesiones/suspensiones del partido concreto, descanso entre rondas, sede/localía real, motivación, ni el estilo táctico del rival. Trátalo como una **línea base cuantitativa**, no como verdad.
- **Transfermarkt en EUR**: hay listados equivalentes en USD con cifras distintas; no mezclar monedas.

## 11. Simulación Montecarlo del cuadro completo (`simulate_bracket.py`)

`predict.py` resuelve **un** partido; Montecarlo resuelve **el torneo entero** miles de veces para obtener probabilidades a nivel de ronda (octavos → campeón).

**Cómo funciona:** para cada partido se toma `p = P(A avanza)` del modelo y se "tira un dado" (`random()`); si cae bajo `p` gana A, si no, B. Se resuelven los 16 cruces de dieciseisavos, se arman los octavos con los ganadores, y así hasta la final — eso es **un torneo simulado**. Se repite N veces (por defecto 100 000) y se cuenta en qué fracción de torneos cada equipo alcanza cada ronda. Por la ley de los grandes números, a más simulaciones, más estable el número.

**Estructura del cuadro** (predefinida por la FIFA, IDs de partido 73→104):
```
Octavos:  89:(G74,G77) 90:(G73,G75) 91:(G76,G78) 92:(G79,G80)
          93:(G83,G84) 94:(G81,G82) 95:(G86,G88) 96:(G85,G87)
Cuartos:  97:(89,90)   98:(93,94)   99:(91,92)   100:(95,96)
Semis:    101:(97,98)  102:(99,100)            Final: 104:(101,102)
```

**Resultado (200 000 simulaciones) — probabilidad de ser campeón (top):**

| Equipo | Octavos | Cuartos | Semis | Final | **Campeón** |
|---|--:|--:|--:|--:|--:|
| Argentina | 97.6% | 92.0% | 82.6% | 65.3% | **39.1%** |
| Francia | 94.8% | 79.6% | 68.7% | 54.0% | **34.5%** |
| España | 91.8% | 77.1% | 66.7% | 27.9% | **11.3%** |
| Inglaterra | 94.1% | 78.7% | 46.5% | 15.7% | **5.6%** |
| Brasil | 85.8% | 76.6% | 43.6% | 14.2% | **4.9%** |
| Alemania | 91.6% | 19.1% | 13.6% | 6.2% | **1.7%** |
| Países Bajos | 67.8% | 60.4% | 12.0% | 4.6% | **1.0%** |
| Portugal | 70.8% | 16.1% | 10.8% | 2.2% | **0.4%** |

> **Lectura del cuadro:** Argentina y Francia están en mitades opuestas (solo pueden cruzarse en la final), de ahí sus altas probabilidades de finalista. **Alemania** ilustra la asimetría del bracket: 91% de pasar octavos pero solo 20% de llegar a cuartos, porque su rival de octavos (partido 89) es muy probablemente **Francia**. El sorteo importa tanto como la calidad.

## 12. Reproducir
```bash
python3 build_dataset.py            # genera el dataset + tabla por equipo
python3 predict.py                  # 16 pronósticos de dieciseisavos
python3 predict.py ARG CPV          # pronóstico de un duelo puntual
python3 simulate_bracket.py 200000  # Montecarlo del cuadro (campeón, finalista, etc.)
```
