# Code Style and Conventions

## Core Principles
- **KISS (Keep It Simple, Stupid)**: Choose straightforward solutions over complex ones
- **YAGNI (You Aren't Gonna Need It)**: Implement features only when needed
- **Single Responsibility**: Each function, class, and module should have one clear purpose
- **Fail Fast**: Check for potential errors early and raise exceptions immediately

## Code Structure Limits
- **Files**: Never exceed 500 lines of code
- **Functions**: Under 50 lines with single, clear responsibility
- **Classes**: Under 100 lines representing a single concept
- **Line Length**: Maximum 100 characters (enforced by ruff)

## Python Style Guide
- **Follow PEP8** with specific choices:
  - Line length: 100 characters (set by ruff in ruff.toml)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures
- **Always use type hints** for function signatures and class attributes
- **Format with ruff format** (faster alternative to Black)

## Naming Conventions
- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes/methods**: `_leading_underscore`
- **Type aliases**: `PascalCase`
- **Enum values**: `UPPER_SNAKE_CASE`

## Docstring Standards
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

    Example:
        >>> calculate_discount(Decimal("100"), 20)
        Decimal('80.00')
    """
```

## Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Use explicit imports when possible

## Error Handling
- Create custom exceptions for domain-specific errors
- Use specific exception handling, not bare except clauses
- Log errors appropriately with context
- Use context managers for resource management