# Essential Development Commands

## Package Management (UV)
```bash
# Sync dependencies
uv sync

# Add a package (NEVER edit pyproject.toml directly)
uv add requests

# Add development dependency
uv add --dev pytest ruff mypy

# Remove a package
uv remove requests

# Run commands in the environment
uv run python script.py
```

## Testing Commands
```bash
# Run all tests with coverage (main command)
PYTHONPATH=/home/user/projects/state-sped-policy-eval uv run pytest tests/unit/ -v --cov=code --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/collection/test_naep_collector.py -v

# Run tests without coverage for faster execution
uv run pytest tests/unit/ -v --tb=short

# Run integration tests
uv run pytest tests/integration/ -v

# Run performance tests
uv run pytest tests/performance/ -v
```

## Code Quality Commands
```bash
# Format code (primary formatter)
uv run ruff format code/

# Check linting
uv run ruff check code/

# Fix linting issues automatically
uv run ruff check --fix code/

# Type checking (if implemented)
uv run mypy code/
```

## Data Collection Commands
```bash
# Run main analysis pipeline
python main.py

# Set Python path for module imports
export PYTHONPATH=/home/user/projects/state-sped-policy-eval

# Run specific data collectors
PYTHONPATH=$PWD uv run python code/collection/naep_collector.py
PYTHONPATH=$PWD uv run python code/collection/census_data_parser.py
```

## Git Commands
```bash
# Standard workflow
git checkout main && git pull origin main
git checkout -b feature/new-feature
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Check status
git status
git diff
```

## Search Commands (Use ripgrep)
```bash
# Search for patterns (preferred over grep)
rg "pattern"

# Search in specific file types
rg "pattern" -g "*.py"

# List files matching pattern
rg --files -g "*.py"
```

## System Commands
```bash
# List directories
ls -la

# Navigate directories
cd /path/to/directory

# Check disk space
df -h

# Check memory usage
free -h

# Process monitoring
top
htop
```

## Project-Specific Shortcuts
```bash
# Quick test run (unit tests only)
uv run pytest tests/unit/ -x --tb=short -q

# Full validation pipeline
uv run pytest tests/unit/ -v --cov=code --cov-report=term-missing --cov-fail-under=80

# Format and lint check
uv run ruff format code/ && uv run ruff check code/
```