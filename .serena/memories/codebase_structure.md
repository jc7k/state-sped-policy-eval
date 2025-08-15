# Codebase Structure

## Root Directory Layout
```
state-sped-policy-eval/
├── main.py                    # Entry point
├── pyproject.toml            # Project configuration and dependencies
├── ruff.toml                 # Ruff linting/formatting configuration
├── pytest.ini               # Pytest configuration
├── CLAUDE.md                 # Comprehensive development guidelines
├── README.md                 # Project description
├── .gitignore               # Git ignore patterns
├── .pre-commit-config.yaml  # Pre-commit hooks (outdated - using ruff now)
└── uv.lock                  # UV dependency lock file
```

## Core Code Structure
```
code/
├── __init__.py              # Package initialization (__version__, __author__)
├── config.py                # Configuration management
├── collection/              # Data collection modules
│   ├── __init__.py         # Exports all collectors and utilities
│   ├── base_collector.py   # Base classes for data collectors
│   ├── common.py           # Shared utilities (StateUtils, APIClient, etc.)
│   ├── naep_collector.py   # NAEP achievement data collector
│   ├── census_collector.py # Census finance data collector
│   ├── census_data_parser.py # Excel file parser for Census data
│   ├── census_file_downloader.py # Direct file downloader
│   ├── edfacts_collector.py # EdFacts special education data
│   └── ocr_collector.py    # OCR civil rights data
├── cleaning/               # Data cleaning and standardization
├── analysis/               # Econometric analysis modules
├── visualization/          # Plotting and reporting
└── validation/             # Data validation utilities
```

## Data Directory Structure
```
data/
├── raw/                    # Downloaded/original datasets
│   ├── naep_state_swd_data.csv
│   ├── census_education_finance_parsed.csv
│   └── census_f33_*.xls
├── processed/              # Cleaned individual files
└── final/                  # Merged analysis datasets
```

## Test Structure
```
tests/
├── conftest.py             # Shared test fixtures
├── unit/                   # Unit tests (mirror code structure)
│   ├── collection/
│   │   ├── test_naep_collector.py
│   │   ├── test_census_collector.py
│   │   └── test_validation.py
│   ├── cleaning/
│   └── analysis/
├── integration/            # Integration tests
├── performance/            # Performance tests
└── fixtures/               # Test data files
    ├── naep_sample_response.json
    ├── census_sample_response.json
    └── policy_test_data.csv
```

## Output Structure
```
output/
├── tables/                 # LaTeX regression tables
├── figures/                # Event studies and plots
└── reports/                # Policy briefs and papers
```

## Key Architecture Notes
- **Modular Design**: Each collector is independent but shares common utilities
- **Base Classes**: `BaseDataCollector`, `APIBasedCollector`, `FileBasedCollector`
- **Common Utilities**: `StateUtils`, `APIClient`, `DataValidator`, `SafeTypeConverter`
- **Test Coverage**: 72 unit tests with 80%+ coverage requirement
- **Configuration**: Centralized in `config.py` with environment variable support