# SPEC — Dataset and prediction engine · Round of 32 of the FIFA World Cup 2026

**Version:** 1.0 · **Data cutoff date:** Jun 28–29, 2026 · **Stage:** Round of 32 (32 teams)
**Deliverables:** `worldcup2026_r32_dataset.json` (per-team data) · `worldcup2026_r32_matchups.json` (16 matchups) · `build_dataset.py` (generator) · `predict.py` (prediction engine)

---

## 1. Objective

Build, for each of the **32 teams** qualified to the Round of 32 of the 2026 World Cup, a self-contained JSON object with relevant data and normalized metrics, so that they can feed a **head-to-head prediction model** between any pair of national teams (not just the matchup they were drawn into). The design prioritizes that the figures be **reproducible** and **auditable**: each derived metric is computed with an explicit formula from raw data that is also stored.

## 2. Team universe (how they qualified)

New 48-team format → 12 groups of 4. The **top 2 of each group (24)** advance + the **8 best third-placed teams**. The 32 qualified teams:

| Group | 1st (group_winner) | 2nd (runner_up) | Best 3rd (best_third) |
|---|---|---|---|
| A | Mexico | South Africa | — |
| B | Switzerland | Canada | Bosnia and Herzegovina |
| C | Brazil | Morocco | — |
| D | United States | Australia | Paraguay |
| E | Germany | Ivory Coast | Ecuador |
| F | Netherlands | Japan | Sweden |
| G | Belgium | Egypt | — |
| H | Spain | Cape Verde | — |
| I | France | Norway | Senegal |
| J | Argentina | Austria | Algeria |
| K | Colombia | Portugal | DR Congo |
| L | England | Croatia | Ghana |

## 3. Metrics (definition and source)

For each team, **four axes** of raw data are collected plus the history:

1. **FIFA Ranking** — `points_official_2026_06_11` (official points as of Jun 11, 2026, pre-World Cup base; FIFA Elo system, denominator 600) and `points_live_in_tournament` (projection that already incorporates tournament results). *The official one is used as the base so as not to double-count with the performance axis.*
   Source: FIFA / football-ranking.com.
2. **Squad value (Transfermarkt)** — `value_eur_millions`, cumulative pre-tournament valuation (~Jun 27, 2026), in EUR.
   Source: Transfermarkt via PlanetFootball/OneFootball.
3. **Performance in the 2026 World Cup** — Group stage: `points`, `goal_difference`, per-match derivatives.
   Source: final group tables (NBC Sports / FIFA).
4. **Experience / World Cup honours** — `titles`, `title_years`, `best_finish_pre_2026`, `appearances_incl_2026`, `defending_champion`.
   Source: historical World Cup record.

## 4. Per-team JSON schema

```jsonc
{
  "team": "Argentina", "code": "ARG", "confederation": "CONMEBOL",
  "qualification": { "group": "J", "finish_position": 1,
                     "finish_type": "group_winner", "finish_label": "1st in group" },
  "fifa_ranking": { "rank_official_2026_06_11": 1,
                    "points_official_2026_06_11": 1877.0,
                    "points_live_in_tournament": 1907.41, "source_note": "..." },
  "squad_value_transfermarkt": { "value_eur_millions": 807.5, "as_of": "2026-06-27", "currency": "EUR" },
  "world_cup_2026_performance": { "stage_reached": "Round of 32", "group_matches_played": 3,
                    "points": 9, "goal_difference": 7,
                    "points_per_match": 3.0, "gd_per_match": 2.333, "form_raw_index": 11.8 },
  "world_cup_history": { "titles": 3, "title_years": [1978,1986,2022],
                    "best_finish_pre_2026": "Champion", "appearances_incl_2026": 19,
                    "defending_champion": true, "pedigree_score_0_100": 100.0 },
  "derived_metrics": { "norm_fifa": 1.0, "norm_value": ..., "norm_form": ...,
                    "norm_pedigree": ..., "strength_index_0_100": 95.57, "effective_elo": 2025.4 },
  "round_of_32": { "opponent": "Cape Verde", "opponent_code": "CPV",
                    "match_id": 86, "venue": "Hard Rock Stadium, Miami", "date": "2026-06-30" },
  "data_notes": []
}
```

## 5. Derived metrics (exact formulas)

All normalizations are computed **over the pool of the 32 teams**.

**5.1 Min-max normalizations → [0,1]**
- `norm_fifa  = minmax(points_official)`
- `norm_value = minmax(log10(value_eur_millions))`  ← logarithmic scale because value is heavily skewed (France €1,520M vs Cape Verde €49M).
- `norm_form  = minmax(form_raw)`, with `form_raw = points + 0.4 · goal_difference`
- `norm_pedigree = minmax(pedigree_raw)`

**5.2 Pedigree (World Cup honours), reproducible proxy**
```
best_pts:  Champion=30 · Runner-up=22 · Semifinal/3rd-4th=16 · Quarterfinals=10 · Round of 16=5 · Group stage=0
pedigree_raw = titles*20 + best_pts + min(appearances,25)*0.8     (cap 100)
```

**5.3 Strength Index (SI, 0–100)** — composite index, configurable weights:
```
SI = 100 · ( 0.40·norm_fifa + 0.20·norm_value + 0.25·norm_form + 0.15·norm_pedigree )
```

**5.4 Effective Elo** — takes the real FIFA Elo and adjusts it with the other 3 axes (in *Elo points per standard deviation*):
```
effective_elo = points_official
              + 40·z(form_raw) + 25·z(log10_value) + 20·z(pedigree_raw)
```
(`z` = standard score relative to the pool).

**5.5 Performance metrics from per-match statistics (Phase 1)**

Each group-stage match in `world_cup_2026_performance.group_stage_matches`
stores detailed statistics (shots, shots on target, possession, passes, pass
accuracy, fouls, cards, offsides, corners) for the team and its opponent. From
the shot figures we derive an expected-goals proxy per team per match:
```
xg_proxy = 0.10 · (shots − shots_on_target) + 0.34 · shots_on_target
```
Aggregated per team (over the matches with detailed stats) this yields
`xg_for_total`, `xg_against_total` and `xg_diff_total`, stored in a
`performance_metrics` block. The intended form correction blends the actual
goal difference with the expected one:
```
gd_adjusted       = (1 − β) · goal_difference + β · xg_diff_total
form_raw_adjusted = points + 0.4 · gd_adjusted
```
with `β = 0.4`. To avoid small-sample bias, the correction only activates for a
team once it has the full group stage covered (`matches_covered ≥ 3`).

*Strength of schedule (Phase 2).* Raw stats are opponent-dependent: 26 shots
against a weak side are worth less than 7 against a strong one. Each match
therefore stores `opponent_fifa_points` (the opponent's pre-tournament FIFA
points — from the dataset for qualified opponents, from
`EXTERNAL_OPPONENT_FIFA_POINTS` for eliminated ones). The pre-tournament rating
is used, not the derived `effective_elo`, to avoid circularity. Phase 2 will
measure the *residual* xGD (actual minus the value expected given the
opponent's strength) instead of the raw xGD.

**Phase 1 (current) is additive only:** `xg_proxy_*`, `opponent_fifa_points`
and `performance_metrics` are computed and stored, but `form_raw`, `norm_form`,
`SI` and `effective_elo` are left untouched, so match probabilities are
unchanged. Constants live in `meta.performance_model`; the values are
(re)generated by `build_performance_metrics.py`.

## 6. Head-to-head prediction engine (`predict.py`)

Two independent methods; the script averages both for the probability to **advance** (direct knockout, already includes extra time and penalties):

**Method A — Effective Elo (recommended).** The FIFA ranking is already Elo, so the probability uses the standard formula:
```
P(A advances) = 1 / ( 1 + 10^(-(elo_A - elo_B)/600) )
```

**Method B — Logistic Strength Index.**
```
P(A advances) = 1 / ( 1 + 10^(-(SI_A - SI_B)/12) )
```

**Combined:** `P_A_advance = (P_elo + P_SI) / 2`.

**90' version with draw** (for group stage or 1X2 markets): a draw band is allocated proportional to the parity of the duel:
```
closeness = 1 - |P_A - 0.5|·2
P_draw = draw_band · closeness        (suggested draw_band ≈ 0.24–0.28)
P_A_win = P_A·(1 - P_draw) ;  P_B_win = (1 - P_A)·(1 - P_draw)
```

**Usage:**
```
python3 predict.py ARG CPV          # direct knockout (P to advance)
python3 predict.py BEL SEN 0.28     # at 90' with draw band
python3 predict.py                  # runs the 16 Round of 32 matchups
```

## 7. Ranking by Strength Index (the 32)

| # | Team | Grp | Pos | FIFA | TM €M | WC26 (pts/GD) | Pedigree | SI | eELO |
|--:|--------|:--:|:--:|--:|--:|:--:|--:|--:|--:|
| 1 | France | I | 1st | 1871 | 1520 | 9/+8 | 84 | 97.1 | 2038 |
| 2 | Argentina | J | 1st | 1877 | 808 | 9/+7 | 100 | 95.2 | 2032 |
| 3 | Spain | H | 1st | 1875 | 1220 | 7/+5 | 64 | 84.4 | 1974 |
| 4 | Brazil | C | 1st | 1766 | 928 | 7/+6 | 100 | 81.1 | 1888 |
| 5 | England | L | 1st | 1828 | 1360 | 7/+4 | 64 | 80.4 | 1924 |
| 6 | Germany | E | 1st | 1736 | 947 | 6/+6 | 100 | 76.3 | 1843 |
| 7 | Netherlands | F | 1st | 1754 | 754 | 7/+6 | 32 | 68.7 | 1824 |
| 8 | Morocco | C | 2nd | 1755 | 448 | 7/+3 | 22 | 60.9 | 1785 |
| 9 | Portugal | K | 2nd | 1768 | 1010 | 4/+5 | 23 | 60.9 | 1788 |
| 10 | Mexico | A | 1st | 1687 | 192 | 9/+6 | 24 | 60.0 | 1744 |
| 11 | Belgium | G | 1st | 1742 | 548 | 5/+3 | 28 | 56.7 | 1751 |
| 12 | United States | D | 1st | 1671 | 386 | 6/+4 | 26 | 52.7 | 1690 |
| 13 | Switzerland | B | 1st | 1650 | 332 | 7/+4 | 20 | 52.2 | 1677 |
| 14 | Croatia | L | 2nd | 1715 | 387 | 6/+0 | 28 | 52.0 | 1711 |
| 15 | Colombia | K | 1st | 1698 | 302 | 6/+3 | 16 | 50.7 | 1698 |
| 16 | Japan | F | 2nd | 1662 | 271 | 5/+4 | 11 | 45.1 | 1646 |
| 17 | Senegal | I | 3rd | 1684 | 478 | 3/+3 | 13 | 43.8 | 1575 |
| 18 | Norway | I | 2nd | 1557 | 590 | 6/+1 | 8 | 40.7 | 1558 |
| 19 | Ivory Coast | E | 2nd | 1541 | 522 | 6/+2 | 3 | 39.1 | 1542 |
| 20 | Ecuador | E | 3rd | 1599 | 369 | 4/+0 | 9 | 34.7 | 1551 |
| 21 | Austria | J | 2nd | 1597 | 245 | 4/+0 | 22 | 34.2 | 1546 |
| 22 | Canada | B | 2nd | 1559 | 199 | 4/+5 | 2 | 32.5 | 1520 |
| 23 | Sweden | F | 3rd | 1510 | 406 | 4/+0 | 32 | 32.1 | 1480 |
| 24 | Egypt | G | 2nd | 1562 | 116 | 5/+2 | 3 | 29.2 | 1505 |
| 25 | Algeria | J | 3rd | 1571 | 257 | 4/-2 | 9 | 28.3 | 1500 |
| 26 | Australia | D | 2nd | 1579 | 77 | 4/+0 | 11 | 24.3 | 1488 |
| 27 | Paraguay | D | 3rd | 1505 | 154 | 4/-2 | 17 | 21.6 | 1425 |
| 28 | DR Congo | K | 3rd | 1474 | 144 | 4/+1 | 2 | 19.8 | 1400 |
| 29 | Ghana | L | 3rd | 1347 | 234 | 4/+0 | 13 | 13.7 | 1289 |
| 30 | Bosnia and Herzegovina | B | 3rd | 1387 | 146 | 4/-1 | 2 | 11.1 | 1302 |
| 31 | South Africa | A | 2nd | 1428 | 49 | 4/-1 | 3 | 8.1 | 1313 |
| 32 | Cape Verde | H | 2nd | 1371 | 49 | 3/+0 | 1 | 1.8 | 1245 |

## 8. Forecast of the 16 matchups (engine demonstration)

P to advance (includes extra time/penalties). Close matchups where the model flips the nominal favorite are marked with ⚑.

| Matchup | Favorite (P advance) |
|---|---|
| Canada vs South Africa | Canada **84%** |
| Germany vs Paraguay | Germany **92%** |
| Netherlands vs Morocco | Netherlands **68%** |
| Brazil vs Japan | Brazil **86%** |
| France vs Sweden | France **95%** |
| Ivory Coast vs Norway ⚑ | Norway **55%** |
| Mexico vs Ecuador | Mexico **84%** |
| England vs DR Congo | England **94%** |
| United States vs Bosnia | USA **91%** |
| Belgium vs Senegal | Belgium **79%** |
| Portugal vs Croatia | Portugal **71%** |
| Spain vs Austria | Spain **92%** |
| Switzerland vs Algeria | Switzerland **83%** |
| Argentina vs Cape Verde | Argentina **98%** |
| Colombia vs Ghana | Colombia **91%** |
| Australia vs Egypt ⚑ | Egypt **62%** |

## 9. How to tune the model

- **SI weights** → dictionary `W` in `build_dataset.py`. Raising `form` gives more weight to tournament form; raising `fifa` makes it more conservative.
- **Elo coefficients** → `ELO_ADJ` (Elo points per standard deviation of form/value/pedigree).
- **Elo denominator `D` and scale `S`** → in `predict.py`. Higher `D` = flatter predictions (less dominant favorites).
- After changing weights: re-run `build_dataset.py` and then `predict.py`.

## 10. Limitations and warnings (important)

- **Tournament in progress:** the `live` figures and the standings change with each matchday; the dataset is a snapshot as of Jun 28–29, 2026.
- **Pedigree:** it is a reproducible proxy (titles + best finish + appearances). It **underestimates** finalists/semifinalists without a title (Netherlands, Sweden, Portugal, Croatia). If you care about honours, raise the weight or replace the formula.
- **The model does not capture**: injuries/suspensions for the specific match, rest between rounds, actual venue/home advantage, motivation, or the opponent's tactical style. Treat it as a **quantitative baseline**, not as truth.
- **Transfermarkt in EUR**: there are equivalent listings in USD with different figures; do not mix currencies.

## 11. Monte Carlo simulation of the full bracket (`simulate_bracket.py`)

`predict.py` resolves **one** match; Monte Carlo resolves **the entire tournament** thousands of times to obtain round-level probabilities (Round of 16 → champion).

**How it works:** for each match, `p = P(A advances)` is taken from the model and a "die is rolled" (`random()`); if it falls below `p`, A wins, otherwise B. The 16 Round of 32 matchups are resolved, the Round of 16 is assembled with the winners, and so on to the Final — that is **one simulated tournament**. This is repeated N times (100,000 by default) and it is counted in what fraction of tournaments each team reaches each round. By the law of large numbers, the more simulations, the more stable the number.

**Bracket structure** (predefined by FIFA, match IDs 73→104):
```
Round of 16:  89:(G74,G77) 90:(G73,G75) 91:(G76,G78) 92:(G79,G80)
              93:(G83,G84) 94:(G81,G82) 95:(G86,G88) 96:(G85,G87)
Quarterfinals:  97:(89,90)   98:(93,94)   99:(91,92)   100:(95,96)
Semifinals:    101:(97,98)  102:(99,100)            Final: 104:(101,102)
```

**Result (200,000 simulations) — probability of being champion (top):**

| Team | Round of 16 | Quarterfinals | Semifinals | Final | **Champion** |
|---|--:|--:|--:|--:|--:|
| Argentina | 97.6% | 92.0% | 82.6% | 65.3% | **39.1%** |
| France | 94.8% | 79.6% | 68.7% | 54.0% | **34.5%** |
| Spain | 91.8% | 77.1% | 66.7% | 27.9% | **11.3%** |
| England | 94.1% | 78.7% | 46.5% | 15.7% | **5.6%** |
| Brazil | 85.8% | 76.6% | 43.6% | 14.2% | **4.9%** |
| Germany | 91.6% | 19.1% | 13.6% | 6.2% | **1.7%** |
| Netherlands | 67.8% | 60.4% | 12.0% | 4.6% | **1.0%** |
| Portugal | 70.8% | 16.1% | 10.8% | 2.2% | **0.4%** |

> **Reading the bracket:** Argentina and France are in opposite halves (they can only meet in the Final), hence their high probabilities of reaching the Final. **Germany** illustrates the asymmetry of the bracket: 91% to pass the Round of 16 but only 20% to reach the Quarterfinals, because its Round of 16 opponent (match 89) is very likely **France**. The draw matters as much as quality.

## 12. Reproduce
```bash
python3 build_dataset.py            # generates the dataset + per-team table
python3 predict.py                  # 16 Round of 32 forecasts
python3 predict.py ARG CPV          # forecast of a specific duel
python3 simulate_bracket.py 200000  # Monte Carlo of the bracket (champion, finalist, etc.)
```
