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
| `simulate_bracket.py` | **Aggregate Monte Carlo:** plays the tournament N times and counts how often each team is champion. |
| `bracket.py` | **Single predicted bracket:** draws the bracket round by round; the favorite advances and **draws are decided on penalties at random**, so **every run can give a different champion** (`--seed` to reproduce one). With `--scoreline [N]` it adds the most likely scoreline of each match (Poisson, exact %). With `--results` it honors the ties already played and only predicts what is left. |
| `results_bracket.json` | **Real results** of the knockout stage (Round of 32 and Round of 16): scoreline, winner and qualified teams of each tie already played. Consumed by `bracket.py --results`. |
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

### Raw data (4 axes + history)

1. **FIFA ranking** — official points (FIFA Elo system, denominator 600).
2. **Squad value** — Transfermarkt, in millions of EUR.
3. **2026 group performance** — `points` and `goal_difference`.
4. **World Cup honours** — titles, best historical finish, appearances.

### Derived metrics (all normalized over the pool of 32)

```
norm_fifa  = minmax(official_FIFA_points)
norm_value = minmax(log10(squad_value))             # log: value is heavily skewed
norm_form  = minmax(form_raw),  form_raw = points + 0.4 · goal_difference
norm_pedigree = minmax(pedigree_raw)

Strength Index (0–100) = 100 · (0.40·norm_fifa + 0.20·norm_value
                                + 0.25·norm_form + 0.15·norm_pedigree)

effective_elo = FIFA_points + 40·z(form_raw) + 25·z(log10_value) + 20·z(pedigree_raw)
```

(`minmax` = rescale to [0,1]; `z` = standard score relative to the pool.)

### Tie probability (engine)

**Two independent methods** are averaged:

```
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

| Question | Tool | Champion |
|---|---|---|
| Who lifts the cup most often? | `simulate_bracket.py` | **Argentina** (~39%) |
| How does ONE bracket end (penalties at random)? | `bracket.py --seed 2026` | **France** |

Why? Argentina and France are in opposite halves (they only meet in the final). In
the final **France is a slight favorite (~55%)**, so in many seeds France wins — but
with draws played on penalties, other seeds give a different champion. But France's
half is brutal (Spain, Germany, Netherlands), so **France only reaches the final 54%
of the time**, while **Argentina reaches it 65%** thanks to an easier path. Across
thousands of tournaments, Argentina lifts the cup more often.

> 📌 **Being champion of the favorites' bracket ≠ being the most likely champion.** The
> draw matters as much as quality.

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
- The `pedigree_score` is a reproducible proxy; it underestimates finalists/semifinalists
  without a title (Netherlands, Sweden, Portugal, Croatia).
- Transfermarkt values in EUR (do not mix with USD listings).

---

## 📐 Team codes

3-letter (FIFA) codes are used in `predict.py` and in the dataset: `ARG`, `FRA`,
`ESP`, `BRA`, `ENG`, `GER`, `NED`, `POR`, `MAR`, `BEL`, `MEX`, `CRO`, `USA`, `SUI`,
`COL`, `JPN`, `NOR`, `CIV`, `ECU`, `AUT`, `SEN`, `SWE`, `CAN`, `ALG`, `EGY`, `AUS`,
`PAR`, `COD`, `GHA`, `BIH`, `RSA`, `CPV`.
