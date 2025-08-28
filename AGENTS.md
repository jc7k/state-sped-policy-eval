# Repository Guidelines

Clear, practical guidance for contributing to state-sped-policy-eval.

## Project Structure & Module Organization
- `src/analysis/`: main entry points (`01_descriptive.py`, `02_causal_models.py`, `03_robustness.py`) and helpers (`panel_setup.py`, `policy_database.py`).
- `src/collection/`, `src/cleaning/`, `src/validation/`: data acquisition, integration, and QA.
- `src/visualization/`, `src/reporting/`: plots, dashboards, briefs, and technical docs.
- `src/**/tests/` and `src/fixtures/`: pytest suites and test data.
- `docs/prds/`: product/design docs. `data/`: raw/processed/final (contents gitignored). `output/`: figures, tables, reports, dashboards.

## Build, Test, and Development Commands
- Install deps: `uv sync --dev` (Python 3.12+; `.python-version` pins 3.13).
- Lint: `uv run ruff check src/`  |  Format: `uv run ruff format src/`.
- Type check (optional): `uv run mypy src/ --ignore-missing-imports`.
- Run analysis: `uv run python src/analysis/02_causal_models.py` (see README for more).
- Tests: `uv run pytest -q` (configured via `pytest.ini` to discover in `src/`; coverage ≥ 80%).
  - Examples: `-m "not slow"`, `-m integration`, `-k name_substring`.

## Coding Style & Naming Conventions
- Formatting: 4-space indent, max line length 100, double quotes (ruff formatter).
- Linting: ruff rulesets enabled (pyflakes, pycodestyle subset, bugbear, comprehensions, simplify, pyupgrade).
- Naming: `snake_case` for modules/functions/vars, `PascalCase` for classes; prefer explicit imports.
- Types: add type hints for new/changed public functions; keep functions small and pure where feasible.

## Testing Guidelines
- Framework: pytest with markers `slow`, `integration`, `performance`; fixtures in `src/fixtures/`.
- Conventions: files `test_*.py`, classes `Test*`, functions `test_*` (see `pytest.ini`).
- Coverage: project minimum 80%; add tests with new features/bug fixes and update existing snapshots/fixtures if behavior changes.

## Commit & Pull Request Guidelines
- Commits: use imperative mood; prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) and reference issues (`#123`).
- PRs: include a clear description, linked issues, test evidence (e.g., key paths under `output/figures/` or `output/tables/` when relevant), and notes on data/assumptions.
- Quality gate: run `ruff` (check + format) and `pytest` locally before pushing; do not commit secrets or large raw data.

## Security & Configuration Tips
- Secrets: copy `.env.example` to `.env`; never commit API keys (see `.gitignore`).
- Data: place external/raw files under `data/raw/` (gitignored); generated artifacts belong in `output/`.

