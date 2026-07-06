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

For each team, **three prediction axes** of raw data are collected, plus World Cup
honours which are stored as descriptive history but **do not feed the model**:

1. **FIFA Ranking** — `points_official_2026_06_11` (official points as of Jun 11, 2026, pre-World Cup base; FIFA Elo system, denominator 600) and `points_live_in_tournament` (projection that already incorporates tournament results). *The official one is used as the base so as not to double-count with the performance axis.*
   Source: FIFA / football-ranking.com.
2. **Squad value (Transfermarkt)** — `value_eur_millions`, cumulative pre-tournament valuation (~Jun 27, 2026), in EUR.
   Source: Transfermarkt via PlanetFootball/OneFootball.
3. **Performance in the 2026 World Cup** — Group stage: `points`, `goal_difference`, per-match derivatives.
   Source: final group tables (NBC Sports / FIFA).
4. **Experience / World Cup honours** *(descriptive only — not a prediction input; see §5.2)* — `titles`, `title_years`, `best_finish_pre_2026`, `appearances_incl_2026`, `defending_champion`.
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
                    "strength_index_0_100": 93.77, "effective_elo": 1974.6 },
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

**5.2 Pedigree (World Cup honours) — retained as history, NOT a model input**
```
best_pts:  Champion=30 · Runner-up=22 · Semifinal/3rd-4th=16 · Quarterfinals=10 · Round of 16=5 · Group stage=0
pedigree_raw = titles*20 + best_pts + min(appearances,25)*0.8     (cap 100)
```
`pedigree_score` is still stored per team as descriptive history, but the honours
axis was **removed from the prediction**: it no longer enters `strength_index` or
`effective_elo`. The proxy skewed toward past champions and underestimated
finalists/semifinalists without a title (Netherlands, Sweden, Portugal, Croatia).

**5.3 Strength Index (SI, 0–100)** — composite index over the three remaining
axes; the weights are renormalized to sum to 1 after dropping pedigree (0.15):
```
SI = 100 · ( 0.4706·norm_fifa + 0.2353·norm_value + 0.2941·norm_form )
```
(exact weights: fifa 8/17, value 4/17, form 5/17.)

**5.4 Effective Elo** — takes the real FIFA Elo and adjusts it with the other 2 axes (in *Elo points per standard deviation*):
```
effective_elo = points_official
              + 40·z(form_raw) + 25·z(log10_value)
```
(`z` = standard score relative to the pool).

**5.5 Performance metrics from per-match statistics (Phase 2)**

Each group-stage match in `world_cup_2026_performance.group_stage_matches`
stores detailed statistics for the team and its opponent: shot breakdown by
outcome (on target, off target, blocked) and by location (in box, out of box),
goalkeeper saves, possession, passes, completed passes, pass accuracy, corners,
offsides, fouls and cards. Shot **location** is the dominant driver of chance
quality, so the expected-goals proxy per team per match is:
```text
xg_proxy = 0.13 · shots_in_box + 0.035 · shots_out_box
```
*Strength of schedule.* Raw stats are opponent-dependent: 26 shots against a
weak side are worth less than 7 against a strong one. Each match stores
`opponent_fifa_points` (the opponent's pre-tournament FIFA points — from the
dataset for qualified opponents, from `EXTERNAL_OPPONENT_FIFA_POINTS` for
eliminated ones; the pre-tournament rating is used, not the derived
`effective_elo`, to avoid circularity). We fit, by ordinary least squares over
every team-match row, the slope `b` of per-match xGD on opponent strength (it
comes out negative — a stronger opponent depresses your xGD), then neutralise
each match to the mean opponent strength `ref`:
```text
xgd            = xg_proxy_for − xg_proxy_against
adj_xgd        = xgd − b · (opponent_fifa_points − ref)
```
So beating a below-`ref` opponent is discounted and performing against an
above-`ref` one is credited. `b` and `ref` are stored in
`meta.performance_model.opponent_adjustment`.

Per team we aggregate `xg_for_total`, `xg_against_total`, the raw `xg_diff_total`
and the opponent-adjusted `xg_diff_adjusted_total` into a `performance_metrics`
block. The form correction blends the actual goal difference with the
opponent-adjusted differential:
```text
gd_adjusted       = (1 − β) · goal_difference + β · xg_diff_adjusted_total
form_raw_adjusted = points + 0.4 · gd_adjusted
```
with `β = 0.8`, so the opponent-adjusted expected-goal differential carries most
of the correction and the actual goal difference only a fifth. To avoid
small-sample bias, the correction only activates for a team once it has the full
group stage covered (`matches_covered ≥ 3`).

Because the detailed match log is the authoritative record, once a team's group
stage is fully covered `build_performance_metrics.py` recomputes the aggregate
fields (`points`, `goal_difference`, `points_per_match`, `gd_per_match`,
`form_raw_index`) from the matches so they cannot disagree with the log.

**Phase 2 folds the correction into the predictions.** `form_raw_adjusted`
replaces `form_raw` in the form axis: the build recomputes `norm_form`,
`strength_index_0_100` and `effective_elo` (§5.3–5.4) from the adjusted form
over a freshly measured `form_effective` pool (`meta.pool_aggregates`), so
`predict.py` uses the performance-adjusted ratings. The `fifa` and `value` axes
are untouched (a sanity check asserts their normalisations still match). Each
team keeps its results-only baseline
(`strength_index_results_only`, `effective_elo_results_only`) and the resulting
`elo_shift` under `performance_metrics` for audit. `build_performance_metrics.py`
is idempotent; constants live in `meta.performance_model`.

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

*(Pedigree / World Cup honours is no longer a model input, so it is not shown.)*

| # | Team | Grp | Pos | FIFA | TM €M | WC26 (pts/GD) | SI | eELO |
|--:|--------|:--:|:--:|--:|--:|:--:|--:|--:|
| 1 | France | I | 1st | 1871 | 1520 | 9/+8 | 99.5 | 1996 |
| 2 | Argentina | J | 1st | 1877 | 808 | 9/+7 | 93.8 | 1975 |
| 3 | Spain | H | 1st | 1875 | 1220 | 7/+5 | 91.1 | 1958 |
| 4 | England | L | 1st | 1828 | 1360 | 7/+4 | 87.8 | 1914 |
| 5 | Brazil | C | 1st | 1766 | 928 | 7/+6 | 76.3 | 1825 |
| 6 | Netherlands | F | 1st | 1754 | 754 | 7/+6 | 74.9 | 1813 |
| 7 | Germany | E | 1st | 1736 | 947 | 6/+6 | 71.8 | 1786 |
| 8 | Morocco | C | 2nd | 1755 | 448 | 7/+3 | 70.5 | 1794 |
| 9 | Portugal | K | 2nd | 1768 | 1010 | 5/+5 | 68.3 | 1786 |
| 10 | Mexico | A | 1st | 1687 | 192 | 9/+6 | 66.2 | 1740 |
| 11 | Belgium | G | 1st | 1742 | 548 | 5/+4 | 66.0 | 1764 |
| 12 | Colombia | K | 1st | 1698 | 302 | 7/+3 | 65.2 | 1738 |
| 13 | United States | D | 1st | 1671 | 386 | 6/+4 | 60.1 | 1697 |
| 14 | Switzerland | B | 1st | 1650 | 332 | 7/+4 | 58.6 | 1679 |
| 15 | Croatia | L | 2nd | 1715 | 387 | 6/+0 | 57.0 | 1706 |
| 16 | Senegal | I | 3rd | 1684 | 478 | 3/+2 | 50.5 | 1655 |
| 17 | Japan | F | 2nd | 1662 | 271 | 5/+4 | 50.4 | 1646 |
| 18 | Norway | I | 2nd | 1557 | 590 | 6/+1 | 49.6 | 1579 |
| 19 | Ivory Coast | E | 2nd | 1541 | 522 | 6/+2 | 46.1 | 1553 |
| 20 | Ecuador | E | 3rd | 1599 | 369 | 4/+0 | 42.7 | 1570 |
| 21 | Canada | B | 2nd | 1559 | 199 | 4/+5 | 39.5 | 1536 |
| 22 | Austria | J | 2nd | 1597 | 245 | 4/+0 | 37.9 | 1548 |
| 23 | Algeria | J | 3rd | 1571 | 257 | 4/-2 | 37.6 | 1532 |
| 24 | Sweden | F | 3rd | 1510 | 406 | 4/+0 | 36.9 | 1491 |
| 25 | Egypt | G | 2nd | 1562 | 116 | 5/+2 | 34.9 | 1518 |
| 26 | Australia | D | 2nd | 1579 | 77 | 4/+0 | 27.4 | 1492 |
| 27 | DR Congo | K | 3rd | 1474 | 144 | 4/+1 | 25.1 | 1419 |
| 28 | Paraguay | D | 3rd | 1505 | 154 | 4/-2 | 22.6 | 1423 |
| 29 | Ghana | L | 3rd | 1347 | 234 | 4/+0 | 15.5 | 1297 |
| 30 | Bosnia and Herzegovina | B | 3rd | 1387 | 146 | 4/-1 | 13.9 | 1315 |
| 31 | South Africa | A | 2nd | 1428 | 49 | 4/-1 | 10.8 | 1328 |
| 32 | Cape Verde | H | 2nd | 1371 | 49 | 3/+0 | 2.1 | 1254 |

## 8. Forecast of the 16 matchups (engine demonstration)

P to advance (includes extra time/penalties). Close matchups where the model flips the nominal favorite are marked with ⚑.

| Matchup | Favorite (P advance) |
|---|---|
| Canada vs South Africa | Canada **84%** |
| Germany vs Paraguay | Germany **90%** |
| Netherlands vs Morocco | Netherlands **61%** |
| Brazil vs Japan | Brazil **83%** |
| France vs Sweden | France **94%** |
| Ivory Coast vs Norway ⚑ | Norway **59%** |
| Mexico vs Ecuador | Mexico **82%** |
| England vs DR Congo | England **94%** |
| United States vs Bosnia | USA **91%** |
| Belgium vs Senegal | Belgium **78%** |
| Portugal vs Croatia | Portugal **74%** |
| Spain vs Austria | Spain **91%** |
| Switzerland vs Algeria | Switzerland **81%** |
| Argentina vs Cape Verde | Argentina **97%** |
| Colombia vs Ghana | Colombia **92%** |
| Australia vs Egypt ⚑ | Egypt **67%** |

## 9. How to tune the model

- **SI weights** → `weights_strength_index` in the dataset meta (`fifa`, `value`, `form`; they should sum to 1). Raising `form` gives more weight to tournament form; raising `fifa` makes it more conservative.
- **Elo coefficients** → `elo_adjustment_coeffs` (Elo points per standard deviation of form/value).
- **Elo denominator `D` and scale `S`** → in `predict.py`. Higher `D` = flatter predictions (less dominant favorites).
- After changing weights: re-run `build_dataset.py` and then `predict.py`.

## 10. Limitations and warnings (important)

- **Tournament in progress:** the `live` figures and the standings change with each matchday; the dataset is a snapshot as of Jun 28–29, 2026.
- **World Cup honours are not a model input.** `pedigree_score` is kept only as descriptive history: the honours axis was removed from `strength_index`/`effective_elo` because the proxy skewed toward past champions and underestimated finalists/semifinalists without a title (Netherlands, Sweden, Portugal, Croatia). To restore it, re-add a `pedigree` weight/coefficient and the `norm_pedigree` term.
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
| France | 93.8% | 79.9% | 67.6% | 50.5% | **35.3%** |
| Argentina | 97.0% | 90.0% | 77.9% | 56.4% | **29.1%** |
| Spain | 91.3% | 76.1% | 64.8% | 30.7% | **16.4%** |
| England | 93.5% | 78.1% | 61.2% | 27.7% | **11.2%** |
| Brazil | 83.0% | 71.6% | 26.2% | 8.5% | **2.4%** |
| Netherlands | 61.0% | 53.7% | 13.9% | 5.3% | **1.4%** |
| Germany | 90.0% | 18.3% | 9.3% | 3.3% | **0.8%** |
| Morocco | 39.0% | 34.1% | 7.8% | 2.7% | **0.7%** |

> **Reading the bracket:** Argentina and France are in opposite halves (they can only meet in the Final). Argentina's easier half sends it to the Final more often (**56%** vs France's **50%**), but France is the Final favorite and converts its knockouts better, so **France is champion more often** (35% vs 29%). **Germany** illustrates the asymmetry of the bracket: 90% to pass the Round of 16 but only 18% to reach the Quarterfinals, because its Round of 16 opponent (match 89) is very likely **France**. The draw decides who reaches the Final; quality decides who wins it.

## 12. Reproduce
```bash
python3 build_dataset.py            # generates the dataset + per-team table
python3 predict.py                  # 16 Round of 32 forecasts
python3 predict.py ARG CPV          # forecast of a specific duel
python3 simulate_bracket.py 200000  # Monte Carlo of the bracket (champion, finalist, etc.)
```
