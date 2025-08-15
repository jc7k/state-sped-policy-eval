# CLAUDE.md

This file provides comprehensive guidance to Claude Code when working with Python code in this repository.

## Project Overview

This is a research project analyzing state-level special education policies and their impact on student outcomes. The project uses quasi-experimental methods to identify causal effects of policy reforms on achievement, inclusion, and equity for students with disabilities (SWD).

### Key Research Innovation
First study combining COVID-19 disruption with state policy variation to identify which approaches proved most resilient and effective for students with disabilities. The project contributes to the $190B IDEA reauthorization debate by providing first causal estimates using post-COVID special education data.

### Research Methods
- **Staggered Difference-in-Differences**: Exploiting timing of state funding formula reforms (2009-2023)
- **Instrumental Variables**: Using court-ordered funding increases and federal monitoring changes
- **COVID Natural Experiment**: Triple-difference design comparing pre/post COVID by reform status
- **Event Studies**: Analyzing effects around policy implementation dates

### Key Data Sources
- NAEP State Assessments (2009-2022) for achievement outcomes
- EdFacts/IDEA Reports (2009-2023) for inclusion rates and graduation
- Census F-33 Finance Data for per-pupil spending
- Hand-collected state policy database (funding formulas, court orders, federal monitoring)

## Core Development Philosophy

### KISS (Keep It Simple, Stupid)
Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)
Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

### Design Principles
- **Dependency Inversion**: High-level modules should not depend on low-level modules. Both should depend on abstractions.
- **Open/Closed Principle**: Software entities should be open for extension but closed for modification.
- **Single Responsibility**: Each function, class, and module should have one clear purpose.
- **Fail Fast**: Check for potential errors early and raise exceptions immediately when issues occur.

## 🧱 Code Structure & Modularity

### File and Function Limits
- **Never create a file longer than 500 lines of code**. If approaching this limit, refactor by splitting into modules.
- **Functions should be under 50 lines** with a single, clear responsibility.
- **Classes should be under 100 lines** and represent a single concept or entity.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Line length should be max 100 characters** (configured in ruff.toml)
- **Use .venv** (the UV virtual environment) whenever executing Python commands, including for unit tests.

### Project Architecture
Follow strict vertical slice architecture with tests living next to the code they test:

```
src/
    __init__.py
    conftest.py
    fixtures/
        naep_sample_response.json
        census_sample_response.json
        ...

    # Core modules
    collection/
        __init__.py
        naep_collector.py
        census_collector.py
        tests/
            test_naep_collector.py
            test_census_collector.py

    cleaning/
        __init__.py
        data_loaders.py
        panel_creator.py
        tests/
            test_refactored_integration.py

    analysis/
        __init__.py
        staggered_did.py
        instrumental_variables.py
        tests/
            test_analysis.py

    visualization/
        __init__.py
        event_study_plots.py
        treatment_dashboard.py
        tests/
            test_visualization.py
```

## 🛠️ Development Environment

### UV Package Management
This project uses UV for blazing-fast Python package and environment management.

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Sync dependencies
uv sync

# Add a package ***NEVER UPDATE A DEPENDENCY DIRECTLY IN PYPROJECT.toml***
# ALWAYS USE UV ADD
uv add requests

# Add development dependency
uv add --dev pytest ruff mypy

# Remove a package
uv remove requests

# Run commands in the environment
uv run python script.py
uv run pytest
uv run ruff check .

# Install specific Python version
uv python install 3.12
```

### Development Commands

```bash
# Run all tests
uv run pytest

# Run specific tests with verbose output
uv run pytest src/collection/tests/test_naep_collector.py -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Type checking
uv run mypy src/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Testing and Quality Commands

```bash
# Run tests with coverage (main command)
PYTHONPATH=/home/user/projects/state-sped-policy-eval uv run pytest src/ -v --cov=src --cov-report=term-missing

# Format and lint code with ruff (NOT black/isort/flake8)
uv run ruff check src/
uv run ruff format src/

# Type checking (if implemented)
python -m mypy src/
```

## 📋 Style & Conventions

### Python Style Guide
- **Follow PEP8** with these specific choices:
  - Line length: 100 characters (set by ruff in ruff.toml)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures
- **Always use type hints** for function signatures and class attributes
- **Format with `ruff format`** (faster alternative to Black)
- **Use `pydantic` v2** for data validation and settings management

### Docstring Standards
Use Google-style docstrings for all public functions, classes, and modules:

```python
def calculate_discount(
    price: Decimal,
    discount_percent: float,
    min_amount: Decimal = Decimal("0.01")
) -> Decimal:
    """
    Calculate the discounted price for a product.

    Args:
        price: Original price of the product
        discount_percent: Discount percentage (0-100)
        min_amount: Minimum allowed final price

    Returns:
        Final price after applying discount

    Raises:
        ValueError: If discount_percent is not between 0 and 100
        ValueError: If final price would be below min_amount

    Example:
        >>> calculate_discount(Decimal("100"), 20)
        Decimal('80.00')
    """
```

### Naming Conventions
- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes/methods**: `_leading_underscore`
- **Type aliases**: `PascalCase`
- **Enum values**: `UPPER_SNAKE_CASE`

## 🧪 Testing Strategy

### Test-Driven Development (TDD)
1. **Write the test first** - Define expected behavior before implementation
2. **Watch it fail** - Ensure the test actually tests something
3. **Write minimal code** - Just enough to make the test pass
4. **Refactor** - Improve code while keeping tests green
5. **Repeat** - One test at a time

### Testing Best Practices
```python
# Always use pytest fixtures for setup
import pytest
from datetime import datetime

@pytest.fixture
def sample_user():
    """Provide a sample user for testing."""
    return User(
        id=123,
        name="Test User",
        email="test@example.com",
        created_at=datetime.now()
    )

# Use descriptive test names
def test_user_can_update_email_when_valid(sample_user):
    """Test that users can update their email with valid input."""
    new_email = "newemail@example.com"
    sample_user.update_email(new_email)
    assert sample_user.email == new_email

# Test edge cases and error conditions
def test_user_update_email_fails_with_invalid_format(sample_user):
    """Test that invalid email formats are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        sample_user.update_email("not-an-email")
    assert "Invalid email format" in str(exc_info.value)
```

### Test Organization
- Unit tests: Test individual functions/methods in isolation
- Integration tests: Test component interactions
- End-to-end tests: Test complete user workflows
- Keep test files next to the code they test
- Use `conftest.py` for shared fixtures
- Aim for 80%+ code coverage, but focus on critical paths

## 🚨 Error Handling

### Exception Best Practices
```python
# Create custom exceptions for your domain
class PaymentError(Exception):
    """Base exception for payment-related errors."""
    pass

class InsufficientFundsError(PaymentError):
    """Raised when account has insufficient funds."""
    def __init__(self, required: Decimal, available: Decimal):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient funds: required {required}, available {available}"
        )

# Use specific exception handling
try:
    process_payment(amount)
except InsufficientFundsError as e:
    logger.warning(f"Payment failed: {e}")
    return PaymentResult(success=False, reason="insufficient_funds")
except PaymentError as e:
    logger.error(f"Payment error: {e}")
    return PaymentResult(success=False, reason="payment_error")
```

### Logging Strategy
```python
import logging
from functools import wraps

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log function entry/exit for debugging
def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

## 🔧 Configuration Management

### Environment Variables and Settings
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation."""
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379"
    api_key: str
    max_connections: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Usage
settings = get_settings()
```

## 🔄 Git Workflow

### Branch Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring
- `test/*` - Test additions or fixes

### Commit Message Format
Never include claude code, or written by claude code in commit messages

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: feat, fix, docs, style, refactor, test, chore

Example:
```
feat(auth): add two-factor authentication

- Implement TOTP generation and validation
- Add QR code generation for authenticator apps
- Update user model with 2FA fields

Closes #123
```

### Daily Workflow:
1. git checkout main && git pull origin main
2. git checkout -b feature/new-feature
3. Make changes + tests
4. git push origin feature/new-feature
5. Create PR → Review → Merge to main

## 🔍 Search Command Requirements

**CRITICAL**: Always use `rg` (ripgrep) instead of traditional `grep` and `find` commands:

```bash
# ❌ Don't use grep
grep -r "pattern" .

# ✅ Use rg instead
rg "pattern"

# ❌ Don't use find with name
find . -name "*.py"

# ✅ Use rg with file filtering
rg --files | rg "\.py$"
# or
rg --files -g "*.py"
```

## Dependencies and Requirements

- **Python >=3.12** (downgraded from 3.13 for better package compatibility)
- **Core packages**: pandas, numpy, statsmodels, scipy, linearmodels
- **Econometric analysis**: statsmodels + linearmodels for manual DiD implementation
- **Data collection**: requests, beautifulsoup4 for API and web scraping
- **Visualization**: matplotlib, seaborn
- **Development**: ruff, pytest, mypy
- **Notebook support**: jupyter, ipykernel

### Dependency Notes
- **Removed `did` package**: Requires system Kerberos libraries (gssapi) causing build failures
- **Removed `synthdid` package**: Has build issues with missing files
- **Solution**: Implement Callaway-Sant'Anna and synthetic control methods manually using statsmodels/linearmodels
- **Python version**: Downgraded from 3.13 to 3.12 for broader package support

## Statistical Methods
- Use `linearmodels` for panel data estimation and fixed effects models
- Use `statsmodels` for standard econometric specifications
- **Manual DiD Implementation**: Implement Callaway-Sant'Anna and staggered DiD using statsmodels instead of `did` package (due to dependency conflicts)
- Cluster standard errors at state level for all specifications
- Use wild cluster bootstrap for robust inference with small N

## Current Project Status (2025-08-15)

### Data Collection Progress
1. **NAEP Achievement Data** ✅ COMPLETE
   - Successfully collected 1,200 records (100% coverage)
   - Years: 2017, 2019, 2022
   - Validated data quality with comprehensive checks
   - Achievement gaps properly calculated (~39 point gap)
   - Data file: `data/raw/naep_state_swd_data.csv`

2. **Census F-33 Finance Data** ✅ COMPLETE
   - Downloaded and parsed Excel files for 2019-2021
   - Successfully extracted 153 state-year records
   - All financial metrics captured: revenue, expenditure by source
   - 100% data coverage for all 51 states (50 + DC)
   - Data file: `data/raw/census_education_finance_parsed.csv`

3. **EdFacts Special Education Data** 🔜 NEXT
   - Will collect enrollment, placement, and outcomes data
   - API endpoint: https://www2.ed.gov/data/

4. **OCR Civil Rights Data** 🔜 PENDING
   - Will collect discipline and access data
   - Direct CSV downloads from ocrdata.ed.gov

### Infrastructure Updates
- ✅ Migrated from flake8/black/isort to ruff for better performance
- ✅ Fixed GitHub Actions CI/CD pipeline (updated to actions/upload-artifact@v4)
- ✅ All 72 unit tests passing
- ✅ Proper rate limiting implemented across all collectors
- ✅ Excel parsing capability added (xlrd dependency)
- ✅ Refactored code/ to src/ with tests next to source code
- ✅ Updated line length to 100 characters in ruff configuration

### Contact Information
- Anywhere contact info or author info is needed, use "Jeff Chen, jeffreyc1@alumni.cmu.edu"
- Where credit is given, mention the reports and code were created in collaboration with Claude Code.

## Validation and Quality Assurance

- Comprehensive validation script checks data quality, effect size reasonableness, and output completeness
- Leave-one-state-out robustness analysis
- Specification curve across multiple model variants
- Permutation tests for inference with small N

## File Organization

```
state-sped-policy-eval/
├── main.py                    # Entry point
├── pyproject.toml            # Project configuration
├── ruff.toml                 # Ruff configuration
├── pytest.ini               # Pytest configuration
├── src/                      # Source code with embedded tests
│   ├── collection/           # API data collectors
│   │   ├── tests/           # Collection tests
│   ├── cleaning/            # Data standardization
│   │   ├── tests/           # Cleaning tests
│   ├── analysis/            # Econometric models
│   │   ├── tests/           # Analysis tests
│   └── visualization/       # Plotting functions
│       └── tests/           # Visualization tests
├── data/
│   ├── raw/                 # Downloaded datasets
│   ├── processed/           # Cleaned individual files
│   └── final/              # Merged analysis data
└── output/
    ├── tables/             # LaTeX regression tables
    ├── figures/            # Event studies and plots
    └── reports/            # Policy briefs and papers
```

## ⚠️ Important Notes

- **NEVER ASSUME OR GUESS** - When in doubt, ask for clarification
- **Always verify file paths and module names** before use
- **Keep CLAUDE.md updated** when adding new patterns or dependencies
- **Test your code** - No feature is complete without tests
- **Document your decisions** - Future developers (including yourself) will thank you
- **Always use ruff as the linter and formatter** instead of black, isort, and flake8

When working on this project, prioritize automation and reproducibility while maintaining econometric rigor. The research design is ambitious but feasible given the state-level focus and systematic approach.