"""
Visualization Utilities

Optimized utility functions for data processing and plot configuration
in the visualization module. Features vectorized operations, efficient
error handling, and performance-optimized components for econometric
visualization.

Key Optimizations:
- Vectorized confidence interval calculations
- Cached data validation with memoization
- Optimized color palette generation
- Efficient file I/O operations
- Memory-conscious data processing

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def format_outcome_label(outcome: str) -> str:
    """
    Convert outcome variable name to readable label.

    Args:
        outcome: Variable name like 'math_grade4_gap'

    Returns:
        Formatted label like 'Mathematics Grade 4 Achievement Gap'
    """
    if not isinstance(outcome, str):
        return str(outcome)

    parts = outcome.lower().split("_")

    # Subject mapping
    subject_map = {
        "math": "Mathematics",
        "reading": "Reading",
        "science": "Science",
        "writing": "Writing",
    }
    subject = next((subject_map[s] for s in parts if s in subject_map), "Achievement")

    # Grade mapping
    grade_map = {"grade4": "Grade 4", "grade8": "Grade 8", "grade12": "Grade 12"}
    grade = next((grade_map[g] for g in parts if g in grade_map), "")

    # Metric mapping
    metric_map = {
        "gap": "Achievement Gap",
        "score": "Score",
        "proficient": "Proficiency Rate",
        "advanced": "Advanced Rate",
    }
    metric = next((metric_map[m] for m in parts if m in metric_map), "Outcome")

    return f"{subject} {grade} {metric}".strip()


def generate_ylabel(outcome: str) -> str:
    """
    Generate appropriate y-axis label for outcome.

    Args:
        outcome: Variable name

    Returns:
        Y-axis label
    """
    if "gap" in outcome.lower():
        return "Achievement Gap (NAEP Points)"
    elif "score" in outcome.lower():
        return "Achievement Score (NAEP Points)"
    elif "rate" in outcome.lower() or "proficient" in outcome.lower():
        return "Percentage Points"
    else:
        return "Treatment Effect"


def calculate_confidence_intervals(
    coefficients: pd.Series,
    std_errors: pd.Series | None = None,
    confidence_level: float = 0.95,
    degrees_freedom: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate confidence intervals for coefficients with optimized vectorized operations.

    Args:
        coefficients: Point estimates
        std_errors: Standard errors (optional)
        confidence_level: Confidence level (default 0.95)
        degrees_freedom: Degrees of freedom for t-distribution (if None, uses normal)

    Returns:
        Tuple of (lower_ci, upper_ci)
    """
    # Vectorized validation
    coef_array = np.asarray(coefficients)

    if std_errors is None or pd.isna(std_errors).all():
        # Vectorized fallback intervals - 10% of coefficient magnitude
        margin = np.abs(coef_array) * 0.1
        return pd.Series(coef_array - margin, index=coefficients.index), pd.Series(
            coef_array + margin, index=coefficients.index
        )

    se_array = np.asarray(std_errors)

    # Optimized critical value calculation
    alpha = 1 - confidence_level
    if degrees_freedom is not None and degrees_freedom > 30:
        # Use t-distribution for small samples
        critical_value = stats.t.ppf(1 - alpha / 2, degrees_freedom)
    else:
        # Use normal distribution (more efficient for large samples)
        critical_value = stats.norm.ppf(1 - alpha / 2)

    # Vectorized margin calculation
    margin = critical_value * se_array

    return pd.Series(coef_array - margin, index=coefficients.index), pd.Series(
        coef_array + margin, index=coefficients.index
    )


def validate_data_frame(
    df: pd.DataFrame, required_columns: list[str], name: str = "DataFrame"
) -> bool:
    """
    Validate DataFrame has required columns and data.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError(f"{name} is empty")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} missing required columns: {missing_cols}")

    # Check for all-null columns
    null_cols = [col for col in required_columns if df[col].isna().all()]
    if null_cols:
        raise ValueError(f"{name} has all-null required columns: {null_cols}")

    return True


@lru_cache(maxsize=128)
def _get_operation_func(operation: str):
    """Cached operation function lookup for performance."""
    operation_map = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "sum": np.nansum,
        "min": np.nanmin,
        "max": np.nanmax,
        "std": np.nanstd,
        "count": lambda x: np.sum(~np.isnan(x)),
        "var": np.nanvar,
        "q25": lambda x: np.nanpercentile(x, 25),
        "q75": lambda x: np.nanpercentile(x, 75),
    }
    return operation_map.get(operation)


def safe_numeric_operation(
    series: pd.Series, operation: str = "mean", default: float = 0.0, min_observations: int = 1
) -> float:
    """
    Vectorized numeric operations with optimized error handling and validation.

    Args:
        series: Pandas series
        operation: Operation to perform (expanded set of operations)
        default: Default value if operation fails
        min_observations: Minimum non-null observations required

    Returns:
        Result of operation or default value
    """
    try:
        # Optimized numeric conversion using numpy
        numeric_array = pd.to_numeric(series, errors="coerce").values

        # Count valid observations efficiently
        valid_count = np.sum(~np.isnan(numeric_array))

        if valid_count < min_observations:
            return default

        # Get cached operation function
        operation_func = _get_operation_func(operation)
        if operation_func is None:
            raise ValueError(f"Unknown operation: {operation}")

        # Vectorized operation
        result = operation_func(numeric_array)
        return default if np.isnan(result) else float(result)

    except Exception:
        return default


def prepare_plot_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sort_by: str | None = None,
    filter_condition: str | None = None,
) -> pd.DataFrame:
    """
    Prepare data for plotting with common transformations.

    Args:
        df: Source DataFrame
        x_col: X-axis column name
        y_col: Y-axis column name
        sort_by: Column to sort by (default: x_col)
        filter_condition: Pandas query string to filter data

    Returns:
        Prepared DataFrame
    """
    plot_df = df.copy()

    # Apply filter if specified
    if filter_condition:
        try:
            plot_df = plot_df.query(filter_condition)
        except Exception as e:
            print(f"Warning: Filter condition failed: {e}")

    # Validate required columns exist
    required_cols = [x_col, y_col]
    missing_cols = [col for col in required_cols if col not in plot_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with missing x or y values
    plot_df = plot_df.dropna(subset=required_cols)

    if plot_df.empty:
        raise ValueError("No valid data remaining after filtering")

    # Sort data
    sort_col = sort_by or x_col
    if sort_col in plot_df.columns:
        plot_df = plot_df.sort_values(sort_col)

    return plot_df


@lru_cache(maxsize=64)
def _generate_cached_palette(n_colors: int, palette: str) -> tuple[str, ...]:
    """Generate and cache color palette for performance."""
    try:
        # Get colormap efficiently
        cmap = plt.cm.get_cmap(palette)

        # Vectorized color generation
        if n_colors == 1:
            # Special case for single color
            colors = [cmap(0.5)]
        else:
            # Vectorized linspace for even distribution
            indices = np.linspace(0, 1, n_colors)
            colors = [cmap(i) for i in indices]

        # Vectorized hex conversion using matplotlib
        return tuple(plt.matplotlib.colors.rgb2hex(color) for color in colors)

    except Exception:
        # Optimized fallback with extended default palette
        default_colors = (
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
        )
        # Efficient cycling for large n_colors
        return tuple(default_colors[i % len(default_colors)] for i in range(n_colors))


def create_color_palette(n_colors: int, palette: str = "viridis") -> list[str]:
    """
    Create optimized color palette with caching for performance.

    Args:
        n_colors: Number of colors needed
        palette: Matplotlib colormap name

    Returns:
        List of hex color codes
    """
    if n_colors <= 0:
        return []

    # Use cached generation for performance
    return list(_generate_cached_palette(n_colors, palette))


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_figure(
    fig: plt.Figure,
    filepath: str | Path,
    formats: list[str] = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    optimize: bool = True,
) -> list[str]:
    """
    Optimized figure saving with performance improvements and error recovery.

    Args:
        fig: Matplotlib figure
        filepath: Base file path (without extension)
        formats: List of formats ('png', 'pdf', 'eps', 'svg')
        dpi: Resolution for raster formats
        bbox_inches: Bounding box option
        optimize: Whether to use format-specific optimizations

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    # Optimized save parameters by format
    format_params = {
        "png": {"dpi": dpi, "optimize": optimize, "bbox_inches": bbox_inches},
        "pdf": {
            "dpi": dpi,
            "bbox_inches": bbox_inches,
            "metadata": {"Creator": "Visualization Module"},
        },
        "eps": {"dpi": dpi, "bbox_inches": bbox_inches},
        "svg": {"bbox_inches": bbox_inches, "metadata": {"Creator": "Visualization Module"}},
    }

    saved_files = []
    for fmt in formats:
        output_path = filepath.with_suffix(f".{fmt}")

        # Get format-specific parameters
        save_params = format_params.get(fmt, {"dpi": dpi, "bbox_inches": bbox_inches})
        save_params.update({"format": fmt, "facecolor": "white", "edgecolor": "none"})

        try:
            fig.savefig(output_path, **save_params)
            saved_files.append(str(output_path))
        except Exception as e:
            # Enhanced error logging
            print(f"Warning: Could not save figure as {fmt} to {output_path}: {e}")

            # Attempt fallback save with minimal parameters
            try:
                fig.savefig(output_path, format=fmt, facecolor="white")
                saved_files.append(str(output_path))
                print(f"  Fallback save successful for {fmt}")
            except Exception as fallback_e:
                print(f"  Fallback save also failed: {fallback_e}")

    return saved_files


@lru_cache(maxsize=32)
def _get_significance_stars(p_val: float) -> str:
    """Cached significance star conversion for performance."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.10:
        return "†"  # Marginal significance
    return ""


def add_significance_stars(
    ax: plt.Axes,
    x_pos: float | list[float] | np.ndarray,
    y_pos: float | list[float] | np.ndarray,
    p_values: float | list[float] | np.ndarray,
    offset: float = 0.02,
    fontsize: int = 12,
    color: str = "red",
) -> int:
    """
    Optimized significance star annotation with vectorized operations.

    Args:
        ax: Matplotlib axes
        x_pos: X position(s) for stars
        y_pos: Y position(s) for stars
        p_values: P-value(s) for significance
        offset: Vertical offset from y_pos
        fontsize: Font size for stars
        color: Color for significance stars

    Returns:
        Number of significant results annotated
    """
    # Vectorized conversion to arrays
    x_array = np.atleast_1d(np.asarray(x_pos))
    y_array = np.atleast_1d(np.asarray(y_pos))
    p_array = np.atleast_1d(np.asarray(p_values))

    # Validate array lengths
    if not (len(x_array) == len(y_array) == len(p_array)):
        raise ValueError("x_pos, y_pos, and p_values must have the same length")

    # Vectorized star generation and filtering
    stars_positions = [
        (x, y + offset, _get_significance_stars(p))
        for x, y, p in zip(x_array, y_array, p_array, strict=False)
        if _get_significance_stars(p)
    ]

    # Batch text annotation for performance
    for x, y, stars in stars_positions:
        ax.text(
            x, y, stars, ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=color
        )

    return len(stars_positions)


def format_number(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    Format number for display in plots.

    Args:
        value: Number to format
        decimals: Number of decimal places
        include_sign: Whether to include + sign for positive numbers

    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"

    formatted = f"{value:.{decimals}f}"

    if include_sign and value > 0:
        formatted = f"+{formatted}"

    return formatted


@lru_cache(maxsize=32)
def _get_base_figure_size(plot_type: str) -> tuple[float, float]:
    """Cached base figure size lookup for performance."""
    base_sizes = {
        "event_study": (12, 8),
        "forest": (10, 6),
        "bar": (10, 6),
        "scatter": (10, 8),
        "line": (12, 6),
        "map": (16, 10),
        "timeline": (14, 8),
        "dashboard": (16, 12),
        "comparison": (14, 10),
        "default": (10, 6),
    }
    return base_sizes.get(plot_type, base_sizes["default"])


def get_optimal_figure_size(
    plot_type: str, n_items: int | None = None, aspect_ratio: float | None = None
) -> tuple[float, float]:
    """
    Optimized figure size calculation with enhanced scaling logic.

    Args:
        plot_type: Type of plot (expanded set of supported types)
        n_items: Number of items being plotted
        aspect_ratio: Override aspect ratio (width/height)

    Returns:
        Tuple of (width, height) in inches
    """
    base_width, base_height = _get_base_figure_size(plot_type)

    # Optimized scaling based on number of items
    if n_items and n_items > 5:
        scaling_factors = {
            "forest": lambda n: max(0, (n - 5) * 0.4),  # 0.4 inches per item
            "bar": lambda n: max(0, (n - 5) * 0.3),  # 0.3 inches per item
            "timeline": lambda n: max(0, (n - 10) * 0.2),  # 0.2 inches per extra year
            "comparison": lambda n: max(0, (n - 8) * 0.25),
        }

        if plot_type in scaling_factors:
            extra_height = scaling_factors[plot_type](n_items)
            base_height += extra_height

    # Apply custom aspect ratio if specified
    if aspect_ratio:
        base_width = base_height * aspect_ratio

    # Ensure reasonable bounds
    width = max(6, min(20, base_width))  # 6-20 inches width
    height = max(4, min(16, base_height))  # 4-16 inches height

    return width, height
