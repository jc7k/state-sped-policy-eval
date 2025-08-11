# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project analyzing state-level special education policies and their impact on student outcomes. The project uses quasi-experimental methods to identify causal effects of policy reforms on achievement, inclusion, and equity for students with disabilities (SWD).

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (preferred) or pip
uv sync
# or
pip install -e .

# Run the main analysis
python main.py

# Create project directory structure for full analysis
mkdir -p data/{raw,processed,final} code/{collection,cleaning,analysis,visualization} output/{tables,figures,reports} docs
```

### Testing and Quality
```bash
# Run any tests (if implemented)
python -m pytest

# Format code (if needed)
python -m black .
python -m isort .

# Type checking (if implemented)
python -m mypy main.py
```

## Project Architecture

### Current Structure
- `main.py` - Entry point with basic "Hello World" functionality
- `pyproject.toml` - Project configuration requiring Python >=3.13
- `state-sped-policy-eval.md` - Detailed research proposal and methodology document
- `README.md` - Empty project description

### Research Design (from state-sped-policy-eval.md)
The project employs multiple identification strategies:
1. **Staggered Difference-in-Differences**: Exploiting timing of state funding formula reforms (2009-2023)
2. **Instrumental Variables**: Using court-ordered funding increases and federal monitoring changes
3. **COVID Natural Experiment**: Triple-difference design comparing pre/post COVID by reform status
4. **Event Studies**: Analyzing effects around policy implementation dates

### Key Data Sources
- NAEP State Assessments (2009-2022) for achievement outcomes
- EdFacts/IDEA Reports (2009-2023) for inclusion rates and graduation
- Census F-33 Finance Data for per-pupil spending
- Hand-collected state policy database (funding formulas, court orders, federal monitoring)

## Development Workflow

### Phase 1: Data Collection (Month 1)
- Implement API collectors for NAEP, EdFacts, Census, and OCR data
- Build comprehensive state policy database with reform dates
- Validate data completeness and quality

### Phase 2: Data Cleaning (Month 2)  
- Standardize variable names and state codes across datasets
- Handle missing data through interpolation where appropriate
- Create master analysis dataset with state-year panel structure

### Phase 3: Analysis (Months 3-4)
- Implement staggered DiD using Callaway-Sant'Anna estimator
- Run IV specifications with multiple instruments
- Conduct COVID triple-difference analysis
- Generate event study plots and robustness checks

### Phase 4: Output Generation (Months 5-6)
- Create regression tables and publication-ready figures
- Generate policy brief with state-specific recommendations
- Validate all results through comprehensive checks

## Key Implementation Notes

### Statistical Methods
- Use `linearmodels` for panel data estimation and fixed effects models
- Use `statsmodels` for standard econometric specifications
- **Manual DiD Implementation**: Implement Callaway-Sant'Anna and staggered DiD using statsmodels instead of `did` package (due to dependency conflicts)
- Cluster standard errors at state level for all specifications
- Use wild cluster bootstrap for robust inference with small N

### Automation Strategy
- Full pipeline executable with single command: `python run_analysis.py --stage all`
- Modular design allows running individual components
- Built-in validation checks ensure data quality
- Reproducible with fixed random seeds

### Data Handling
- State-year panel structure (N ≈ 500 observations)
- Handle staggered treatment adoption across 15+ states
- Missing data patterns vary by source (NAEP biennial, EdFacts annual)
- Policy coding requires manual verification of reform dates

## File Organization for Full Implementation

```
state-sped-policy-eval/
├── main.py                    # Current entry point
├── run_analysis.py           # Master pipeline script
├── code/
│   ├── collection/           # API data collectors
│   ├── cleaning/            # Data standardization
│   ├── analysis/            # Econometric models
│   └── visualization/       # Plotting functions
├── data/
│   ├── raw/                 # Downloaded datasets
│   ├── processed/           # Cleaned individual files
│   └── final/              # Merged analysis data
└── output/
    ├── tables/             # LaTeX regression tables
    ├── figures/            # Event studies and plots
    └── reports/            # Policy briefs and papers
```

## Research Context

This project contributes to the $190B IDEA reauthorization debate by providing first causal estimates using post-COVID special education data. The state-level focus enables clean identification while remaining directly relevant for state policymakers.

### Key Innovation
First study combining COVID-19 disruption with state policy variation to identify which approaches proved most resilient and effective for students with disabilities.

## Dependencies and Requirements

- **Python >=3.12** (updated from 3.13 for better package compatibility)
- **Core packages**: pandas, numpy, statsmodels, scipy, linearmodels
- **Econometric analysis**: statsmodels + linearmodels for manual DiD implementation
- **Data collection**: requests, beautifulsoup4 for API and web scraping
- **Visualization**: matplotlib, seaborn
- **Development**: econtools, pyrddl for additional econometric utilities
- **Notebook support**: jupyter, ipykernel

### Dependency Notes
- **Removed `did` package**: Requires system Kerberos libraries (gssapi) causing build failures
- **Removed `synthdid` package**: Has build issues with missing files  
- **Solution**: Implement Callaway-Sant'Anna and synthetic control methods manually using statsmodels/linearmodels
- **Python version**: Downgraded from 3.13 to 3.12 for broader package support

## Validation and Quality Assurance

- Comprehensive validation script checks data quality, effect size reasonableness, and output completeness
- Leave-one-state-out robustness analysis
- Specification curve across multiple model variants
- Permutation tests for inference with small N

When working on this project, prioritize automation and reproducibility while maintaining econometric rigor. The research design is ambitious but feasible given the state-level focus and systematic approach.
- Always use ruff as the linter and formatter instead of black, isort, and flake8