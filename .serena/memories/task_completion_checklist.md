# Task Completion Checklist

## Before Starting Any Task
1. **Read CLAUDE.md** - Review project-specific guidelines
2. **Understand the codebase** - Use symbolic tools to explore relevant modules
3. **Check existing tests** - Understand current test patterns and coverage
4. **Set Python path** - Ensure `PYTHONPATH=/home/user/projects/state-sped-policy-eval`

## During Development
1. **Follow TDD approach** - Write tests first when possible
2. **Use type hints** - All function signatures must have type annotations
3. **Write docstrings** - Google-style for all public functions/classes
4. **Keep functions small** - Under 50 lines with single responsibility
5. **Error handling** - Use specific exceptions and proper logging

## Code Quality Checks (MANDATORY)
```bash
# 1. Format code with ruff
uv run ruff format code/

# 2. Check and fix linting issues
uv run ruff check --fix code/

# 3. Run type checking (if mypy is set up)
uv run mypy code/

# 4. Run all unit tests with coverage
PYTHONPATH=/home/user/projects/state-sped-policy-eval uv run pytest tests/unit/ -v --cov=code --cov-report=term-missing

# 5. Ensure coverage meets minimum threshold (80%)
PYTHONPATH=/home/user/projects/state-sped-policy-eval uv run pytest tests/unit/ -v --cov=code --cov-report=term-missing --cov-fail-under=80
```

## Testing Requirements
1. **Unit tests** - Test individual functions/methods in isolation
2. **Integration tests** - Test component interactions when applicable
3. **Mock external dependencies** - Use pytest-mock for API calls
4. **Test edge cases** - Handle error conditions and boundary cases
5. **Maintain test fixtures** - Keep sample data in tests/fixtures/

## Documentation Updates
1. **Update docstrings** - Ensure all new functions have complete documentation
2. **Update CLAUDE.md** - Add new patterns or conventions if introduced
3. **Commit messages** - Follow conventional commit format (feat/fix/docs/refactor)

## Final Validation
1. **All tests pass** - No failing unit or integration tests
2. **Coverage maintained** - 80%+ code coverage requirement
3. **No linting errors** - Clean ruff check output
4. **Type hints complete** - All new code properly typed
5. **Documentation current** - README and docstrings updated

## Git Workflow
1. **Create feature branch** - `git checkout -b feature/description`
2. **Make atomic commits** - Small, focused commits with clear messages
3. **Push to remote** - `git push origin feature/description`
4. **Create pull request** - Only when explicitly requested by user

## Performance Considerations
1. **Profile when needed** - Use cProfile for performance-critical code
2. **Use generators** - For large datasets to minimize memory usage
3. **Cache expensive operations** - Use `@lru_cache` when appropriate
4. **Rate limiting** - Respect API limits (2s for NAEP, 1s for others)

## Security Checklist
1. **No hardcoded secrets** - Use environment variables
2. **Input validation** - Validate all external data with proper error handling
3. **Secure API calls** - Use proper authentication and error handling
4. **Log security events** - Track data access and potential issues

## Special Project Notes
- **Author attribution**: Use "Jeff Chen, jeffreyc1@alumni.cmu.edu"
- **Collaboration credit**: Mention "created in collaboration with Claude Code"
- **Rate limiting**: 2 seconds for NAEP API, 1 second for other APIs
- **Python path**: Always set PYTHONPATH for imports to work correctly
- **Virtual environment**: Use UV for all Python commands