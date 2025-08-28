# Contributing to state-sped-policy-eval

Thanks for contributing! This project uses a PR-driven workflow with CI gates to maintain quality and reproducibility.

## Getting Started
- Prereqs: Python 3.12+ (repo pins 3.13 via `.python-version`), `uv` for env/deps.
- Install: `uv sync --dev`
- Run checks locally:
  - Lint: `uv run ruff check src/`
  - Format: `uv run ruff format src/`
  - Type check (optional): `uv run mypy src/ --ignore-missing-imports`
  - Tests: `uv run pytest -q` (see `pytest.ini` and markers)

## Branching & Commits
- Branches: `feat/...`, `fix/...`, `chore/...`, `docs/...`, `refactor/...` from `main`.
- Commits: Prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`). Reference issues (e.g., `#123`). Keep diffs focused.

## Pull Requests
- Open a PR from your feature branch into `main`.
- Required before merge:
  - `ruff check` passes and code formatted.
  - Tests pass with coverage ≥ 80% (unit + integration where applicable).
  - No secrets or large raw data committed.
- Include in the PR description:
  - Summary of changes and motivation.
  - Any assumptions or data notes.
  - Evidence: key output paths (e.g., under `output/figures/` or `output/tables/`).
  - Screenshots of plots if helpful.
- Reviews: At least one reviewer. Use Draft PRs early for feedback.
- Merge strategy: Prefer squash merge for a clean history.

## Coding Style
- Formatting: 4-space indent, double quotes, max line length 100 (ruff formatter).
- Linting: Ruff rulesets enabled (pyflakes, bugbear, comprehensions, simplify, pyupgrade).
- Naming: `snake_case` for modules/functions/vars, `PascalCase` for classes.
- Types: Add type hints for new or changed public functions.
- Functions: Favor small, pure helpers when feasible.

## Tests
- Framework: `pytest` with markers `slow`, `integration`, `performance`.
- Location: `src/**/tests/` and fixtures in `src/fixtures/`.
- Conventions: files `test_*.py`, classes `Test*`, functions `test_*`.
- Coverage: keep project-wide coverage ≥ 80%. Add/update tests when fixing bugs or adding features.

## Data & Security
- Secrets: copy `.env.example` to `.env` locally; never commit `.env` or keys.
- Data: place external/raw in `data/raw/` (gitignored). Generated artifacts go to `output/`.
- Reproducibility: pin seeds where applicable; avoid committing large, frequently changing binaries.

## CI/CD
- GitHub Actions lint, type-check, and test. Fix failures before merge.
- Optionally use Draft PRs while iterating to get early CI feedback.

## Project Structure (quick ref)
- `src/analysis/`: main analysis entry points and helpers.
- `src/collection/`, `src/cleaning/`, `src/validation/`: data acquisition, integration, QA.
- `src/visualization/`, `src/reporting/`: plots, dashboards, briefs, tech docs.
- `src/**/tests/`, `src/fixtures/`: pytest suites and test data.
- `docs/prds/`: product/design docs.
- `data/`: raw/processed/final (gitignored). `output/`: figures, tables, reports, dashboards.

## Releasing
- Tag versions and update CHANGELOG when publishing notable analysis updates.
- Attach built reports/figures to releases if needed.

## Questions
Open an issue or mention a maintainer in the PR.

