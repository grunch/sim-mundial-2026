# AGENTS.md

Guidance for AI coding agents (and human contributors) working in this repository.

## Project overview

This is a small, dependency-free Python project that models the knockout stage of
the 2026 FIFA World Cup. It predicts head-to-head matchups from
a team strength dataset and can draw a full bracket, run aggregate simulations, and
respect real results that have already been played.

Everything runs on the Python standard library — there is no build step, no virtual
environment requirement, and no third-party packages.

## Repository layout

| Path | Purpose |
|---|---|
| `worldcup2026_r32_dataset.json` | Source of truth: the 32 teams with raw data and derived metrics. |
| `worldcup2026_montecarlo.json` | Output of `simulate_bracket.py`: per-team probability of reaching each round. |
| `results_bracket.json` | Real knockout results (round of 32, round of 16 and quarterfinals; semifinals/final added as played): score, winner, and qualified teams per match. Consumed by `bracket.py --results`. |
| `performance.py` | Performance model: xG proxy from shot location, opponent (strength-of-schedule) adjustment, form correction, and the strength-index / effective-Elo assembly. |
| `build_performance_metrics.py` | Recomputes the xG intermediates and folds the opponent-adjusted form into `norm_form`/`strength_index`/`effective_elo` (so it affects predictions). Idempotent; re-run after adding matches. |
| `predict.py` | Head-to-head engine: probability that team A eliminates team B in a single tie. |
| `simulate_bracket.py` | Aggregate Monte Carlo: plays the tournament N times and counts champions. |
| `bracket.py` | Single predicted bracket, round by round to the final. Supports `--seed`, `--scoreline [N]`, and `--results [FILE]`. |
| `test_bracket.py` | Standard-library `unittest` suite for `bracket.py`. |
| `spec_worldcup2026.md` | Full technical specification (formulas, sources, JSON schema). |
| `README.md` | User-facing documentation. |

## Language policy (MANDATORY)

**All project artifacts MUST be written in English.** This applies to:

- Source code (identifiers of every kind)
- Variable, function, class, and constant names
- Parameter names and CLI flag names
- Code comments and docstrings
- Documentation files (including `README.md`)
- Commit messages (subject and body)
- Test names and test descriptions
- Human-readable content and file names

Rationale: a single working language keeps the codebase consistent, reviewable, and
accessible to any contributor or tool.

Notes and exceptions:

- Real-world proper nouns and data values (team names, stadium names) are data, not
  code. Use the standard English name where one exists (e.g. "Germany", "Mexico City")
  and keep proper nouns that have no translation as-is (e.g. "Estadio Azteca").
- Team display names come from the dataset (`team` field) and are matched to results
  by 3-letter **code** (e.g. `BRA`), never by localized name.

## How to run

```bash
# Head-to-head prediction and aggregate simulation
python3 predict.py
python3 simulate_bracket.py

# Single predicted bracket
python3 bracket.py                  # new bracket each run (penalties decided at random)
python3 bracket.py --seed 7         # reproducible: same seed -> same bracket
python3 bracket.py --scoreline      # + most likely scoreline per tie (Poisson, exact %)
python3 bracket.py --scoreline 3    # + top-3 scorelines per tie
python3 bracket.py --results        # honor already-played real results; predict the rest
python3 bracket.py --results FILE   # read real results from another file
```

Run all commands from the repository root — the scripts read the JSON datasets via
relative paths.

## Testing

```bash
python3 -m unittest -v test_bracket
# or
python3 test_bracket.py
```

Requirements for changes:

- Write tests first (TDD): add or update a failing test, then implement.
- Keep the suite green before committing.
- When you change a public function signature (e.g. `parse_args`, `resolve`), update
  the affected tests in the same change.
- Tests use only the standard library (`unittest`); do not introduce test dependencies.

## Coding conventions

- Python standard library only — do not add third-party dependencies.
- Keep functions small and focused; prefer many small, cohesive units over large ones.
- Validate inputs at boundaries (CLI parsing, file loading) and fail fast with a
  clear message.
- Keep the model logic identical across `predict.py`, `simulate_bracket.py`, and
  `bracket.py` (shared Elo denominator `D`, Strength Index scale `S`, etc.). If you
  change a shared formula or constant, change it consistently and update the tests.
- Match teams to the dataset by team **code** (e.g. `BRA`), not by display name.
- Do not commit secrets. There are none today; keep it that way.

## Commit and PR workflow

- Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`,
  `perf:`, `ci:`.
- Write commit subject and body in English.
- Keep commits focused; do not mix unrelated changes.
- Ensure the test suite passes and any touched JSON stays valid
  (`python3 -m json.tool <file> > /dev/null`) before committing.

## Data files

- `worldcup2026_r32_dataset.json` is the source of truth for team strength; other
  outputs derive from it. Treat it as authoritative.
- `results_bracket.json` records real match outcomes. Each match carries
  `home`/`away` (with `team`, `code`, `score`, `penalties`), `decided_by`, `played`,
  and `winner`; dates use `YYYY-MM-DD`. Unplayed matches have `played: false` and
  `winner: null`.
- `worldcup2026_montecarlo.json` is generated by `simulate_bracket.py`; regenerate it
  rather than editing by hand.
- Keep JSON valid and formatted; validate before committing.
