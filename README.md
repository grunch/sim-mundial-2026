# World Cup 2026 Simulator — Round of 32 to the Final

A quantitative, reproducible and auditable model to forecast the knockout ties of the
**FIFA World Cup 2026** (Canada / Mexico / USA), from the **32-team stage
(Round of 32)** to the champion.

Each national team has a profile with raw data (FIFA ranking, squad value, group
performance, honours) and derived metrics with **explicit formulas**. From there, a
head-to-head engine estimates the probability that one team eliminates another, and
two simulators resolve the full bracket.

---

## 📁 Project files

| File | What it is |
|---|---|
| `worldcup2026_r32_dataset.json` | **Source of truth.** The 32 teams with raw data and derived metrics. |
| `worldcup2026_montecarlo.json` | Output of `simulate_bracket.py`: each team's probability of reaching each round. |
| `predict.py` | Head-to-head engine: probability that A eliminates B in a direct tie. |
| `performance.py` | Performance model: xG proxy from shot location, opponent (strength-of-schedule) adjustment, and the strength-index / effective-Elo assembly. |
| `build_performance_metrics.py` | Recomputes the per-match xG intermediates and folds the opponent-adjusted form into `norm_form` / `strength_index` / `effective_elo`. Idempotent. |
| `simulate_bracket.py` | **Aggregate Monte Carlo:** plays the tournament N times and counts how often each team is champion. |
| `bracket.py` | **Single predicted bracket:** draws the bracket round by round; the favorite advances and **draws are decided on penalties at random**, so **every run can give a different champion** (`--seed` to reproduce one). With `--scoreline [N]` it adds the most likely scoreline of each match (Poisson, exact %). With `--results` it honors the ties already played and only predicts what is left. |
| `results_bracket.json` | **Real results** of the knockout stage (Round of 32, Round of 16 and Quarterfinals, with Semifinals/Final added as they are played): scoreline, winner and qualified teams of each tie already played. Consumed by `bracket.py --results`. |
| `spec_worldcup2026.md` | Full technical specification (formulas, sources, JSON schema). |

> ⚙️ The dataset generator (`build_dataset.py`, cited in the spec) is not included;
> the source of truth is the JSON, which already carries all computed metrics.

---

## 🚀 Quick start

Requirements: **Python 3** (standard library only, no external dependencies).

```bash
# 1) Forecast a direct tie (P of advancing, incl. extra time/penalties)
python3 predict.py ARG CPV          # Argentina vs Cape Verde
python3 predict.py BEL SEN 0.28     # Belgium vs Senegal, regular-time version with a draw band

# 2) The 16 official Round of 32 ties
python3 predict.py

# 3) Monte Carlo: who is champion most often? (N tournaments)
python3 simulate_bracket.py 200000

# 4) Full bracket drawn, round by round to the final
python3 bracket.py                  # predicted bracket; draws to penalties at random (--seed)
python3 bracket.py --scoreline      # + most likely scoreline of each tie (2-0, 21%)
python3 bracket.py --scoreline 3    # + the 3 most likely scorelines per tie
python3 bracket.py --results        # use real results already played; predict the rest
```

---

## 🧠 How the model works

### Raw data (3 prediction axes + history)

1. **FIFA ranking** — official points (FIFA Elo system, denominator 600).
2. **Squad value** — Transfermarkt, in millions of EUR.
3. **2026 group performance** — `points`, `goal_difference` and a **per-match
   statistics log** (shots by location/outcome, possession, passes, cards…) that
   feeds the performance model below.

> **World Cup honours** (titles, best historical finish, appearances) are still
> stored per team as `pedigree_score`, but they **no longer feed the prediction** —
> the honours axis was removed from the strength model. The three axes above are
> the only inputs to `strength_index` / `effective_elo`.

### Derived metrics (all normalized over the pool of 32)

```text
norm_fifa  = minmax(official_FIFA_points)
norm_value = minmax(log10(squad_value))             # log: value is heavily skewed
norm_form  = minmax(form_raw_adjusted)              # performance-adjusted, see below

# The three weights are renormalized to sum to 1 after dropping pedigree (0.15).
Strength Index (0–100) = 100 · (0.4706·norm_fifa + 0.2353·norm_value
                                + 0.2941·norm_form)

effective_elo = FIFA_points + 40·z(form_raw_adjusted) + 25·z(log10_value)
```

(`minmax` = rescale to [0,1]; `z` = standard score relative to the pool.)

### Performance model — the "form" axis is opponent-adjusted

The results-only form (`points + 0.4·goal_difference`) is noisy: a team can win
with few chances or lose while dominating. The performance model corrects it
from the per-match statistics.

1. **Chance quality (xG proxy).** Shot **location** drives chance quality, so per
   team per match: `xg_proxy = 0.13·shots_in_box + 0.035·shots_out_box`. The
   match differential is `xgd = xg_for − xg_against`.
2. **Strength of schedule.** 26 shots against a weak side are worth less than 7
   against a strong one. We fit (least squares over every match) the slope `b`
   of `xgd` on the opponent's pre-tournament FIFA points, then neutralise each
   match to the mean opponent `ref`: `adj_xgd = xgd − b·(opp_points − ref)`. So
   beating weak sides is discounted and performing against strong ones is
   credited.
3. **Corrected form.** Blend the actual and the opponent-adjusted differential
   (`β = 0.8`, so the opponent-adjusted xGD carries most of the correction),
   only once a team has its three group games:
   `form_raw_adjusted = points + 0.4·[(1−β)·goal_difference + β·xg_diff_adjusted_total]`.

`build_performance_metrics.py` then recomputes `norm_form`, `strength_index` and
`effective_elo` from `form_raw_adjusted`, so the correction reaches the
probabilities. Each team keeps its results-only baseline and the resulting
`elo_shift` under `performance_metrics` for audit (e.g. Portugal −19.7 Elo: a
+5 goal difference not backed by chances; Colombia +18.4, England +16.1). The run
is idempotent and the FIFA/value axes are untouched.

### Tie probability (engine)

**Two independent methods** are averaged:

```text
Method A (Elo):  P(A) = 1 / (1 + 10^(-(eELO_A − eELO_B)/600))
Method B (SI):   P(A) = 1 / (1 + 10^(-(SI_A − SI_B)/12))
P(A advances)  =  (P_A_elo + P_A_si) / 2          # includes extra time and penalties
```

Full details and sources in [`spec_worldcup2026.md`](./spec_worldcup2026.md).

---

## ⚔️ `simulate_bracket.py` vs `bracket.py` — they read the tournament differently

Both use **the same per-match probability**, but answer different questions.
It is normal that **the champion does not match** between them.

### `simulate_bracket.py` — "who wins MORE tournaments?"

Plays the full bracket N times. In each match it rolls a die (`random() < P`), so
**sometimes the favorite loses** (as in real life). It counts in what fraction of the
N tournaments each team reaches each round and is champion.

It captures the **path**: a team with an easy draw goes further even if it is not the
best in a direct tie.

### `bracket.py` — "one predicted bracket, with penalties played at random"

Resolves **a single bracket** round by round. In each tie it looks at the **most
likely scoreline** (Poisson model, see below):

- If it is **not** a draw → the favorite advances (wins in regular time, deterministic).
- If it **is a draw** → it goes to **penalties**, and the winner is decided **at
  random**: a coin weighted by strength (`p_adv`), so **the underdog can also go
  through**. Those ties are flagged `(pen)`.

Since penalties are random, **every run can give a different champion**: if the final
ends 1-1, sometimes France wins and sometimes Argentina. By default the randomness is
**real** (system seed); the header prints the seed used so you can **reproduce** that
exact bracket with `--seed`.

```bash
python3 bracket.py                  # NEW bracket every run (penalties at random)
python3 bracket.py                  # ...run again and the champion may change
python3 bracket.py --seed 7         # reproducible: same seed → same bracket
```

*(For "who wins MORE tournaments?" across thousands of realizations like this one, use
`simulate_bracket.py`.)*

#### Scorelines with `--scoreline [N]`

It also predicts the **scoreline** of each tie with a **Poisson** goal model derived
from the Elo: `λ_fav` and `λ_rival` come from the Elo supremacy, and the **exact**
probability of each result is `P(i-j) = Poisson(i; λ_fav)·Poisson(j; λ_rival)`.

- `N` = how many top scorelines to show per match (default 1).
- It is **analytic, not iterative**: the `%` is exact and instant. Simulating N
  matches would only approximate that number with noise (with `N=3`, `2-0 (66%)` would be false statistics).
- The most likely scoreline of an overwhelming favorite is **not** 4-0: the
  probability spreads out. Argentina vs Cape Verde → most likely **2-0, but only 21%** (wins 98%).
- A draw as the most likely scoreline → flagged `->pen`: that tie is decided on
  **penalties at random** (see above), not automatically to the favorite.

```bash
python3 bracket.py --scoreline      # [86] Argentina vs Cape Verde -> Argentina (97.7%) | 2-0 (21.4%)
python3 bracket.py --scoreline 3    # ... | 2-0 (21.4%) · 1-0 (17.8%) · 3-0 (17.1%)
```

#### Real results with `--results [FILE]`

By default the **whole** bracket is predicted. With `--results` it reads
`results_bracket.json` (or the given `FILE`) and every tie that **already has a real
result** (`played: true`) is **not predicted**: the team that really advanced goes
through and the real scoreline is printed, flagged `(real)` / `(real, pen)`. Ties
without a result are still predicted as usual (favorite / penalties at random),
including the Round of 16 ties that depend on those real winners.

- Matching is done by **team code** (`code` in the JSON), not by name.
- The file contains `round_of_32` and `round_of_16`; each match carries `home`/`away`
  (`team`, `code`, `score`, `penalties`), `decided_by`, `played` and `winner`.

```bash
python3 bracket.py --results             # [74] Germany vs Paraguay -> Paraguay 1-1 (real, pen)
python3 bracket.py --results data.json   # read the real results from another file
python3 bracket.py --results --scoreline # marks the real ties; predicts the rest with scorelines
```

### A real example of the difference

| Question | Tool | Answer |
|---|---|---|
| Who lifts the cup most often? | `simulate_bracket.py` | **France** (~35%) |
| Who reaches the final most often? | `simulate_bracket.py` | **Argentina** (~56%) |
| How does ONE bracket end (penalties at random)? | `bracket.py --seed 13` | **Argentina** |

Why? Argentina and France are in opposite halves (they only meet in the final).
Argentina's half is easier, so **it reaches the final more often (~56%) than France
(~50%)** — France's half is brutal (Spain, England, Netherlands). But in the final
**France is the favorite (~64%)**, and it converts its knockouts better across the
board, so **France lifts the cup more often (~35% vs ~29%)**. With draws played on
penalties, a single bracket can still crown Argentina (`--seed 13`) or Spain
(`--seed 5`).

> 📌 **Reaching the final most often ≠ winning it most often.** The draw decides who
> *gets there*; quality decides who *wins it*.

---

## 🔁 Reproduce everything

```bash
python3 predict.py                  # 16 Round of 32 forecasts
python3 simulate_bracket.py 200000  # Monte Carlo → regenerates worldcup2026_montecarlo.json
python3 bracket.py                  # full bracket drawn (draws to penalties at random)
```

---

## ⚠️ Limitations

- **Ongoing tournament:** figures and standings are a snapshot at **28–29 Jun 2026**.
- The model is a **quantitative baseline**, not a truth: it does not capture injuries,
  suspensions, rest between rounds, real home advantage, motivation or the opponent's
  tactics.
- **World Cup honours are not a prediction input.** `pedigree_score` is kept as
  descriptive history only; the honours axis was removed from the strength model
  because the proxy underestimated finalists/semifinalists without a title
  (Netherlands, Sweden, Portugal, Croatia) and skewed toward past champions.
- Transfermarkt values in EUR (do not mix with USD listings).

---

## 📐 Team codes

3-letter (FIFA) codes are used in `predict.py` and in the dataset: `ARG`, `FRA`,
`ESP`, `BRA`, `ENG`, `GER`, `NED`, `POR`, `MAR`, `BEL`, `MEX`, `CRO`, `USA`, `SUI`,
`COL`, `JPN`, `NOR`, `CIV`, `ECU`, `AUT`, `SEN`, `SWE`, `CAN`, `ALG`, `EGY`, `AUS`,
`PAR`, `COD`, `GHA`, `BIH`, `RSA`, `CPV`.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the
[`LICENSE`](./LICENSE) file for the full text.
