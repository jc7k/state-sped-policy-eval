"""
Base Visualization Class

Optimized base functionality and configuration for all visualization components.
Features efficient data loading with smart caching, vectorized validation, and
memory-optimized plotting infrastructure for high-performance econometric visualization.

Key Optimizations:
- Smart data caching with memory management
- Vectorized data validation and processing
- Lazy loading of configuration objects
- Efficient file pattern matching with regex
- Memory pooling for large datasets
- Optimized matplotlib figure management

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

import logging
import re
import weakref
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT_CONFIG, PlotConfig
from .utils import (
    ensure_directory,
    format_outcome_label,
    save_figure,
    validate_data_frame,
)


class BaseVisualizer(ABC):
    """
    Optimized abstract base class for all visualization components.

    Features smart caching, memory management, and efficient data processing
    for high-performance econometric visualization.
    """

    # Class-level cache for shared resources
    _shared_configs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    _figure_pool: list[plt.Figure] = []

    def __init__(
        self,
        results_dir: str | Path = "output/tables",
        figures_dir: str | Path = "output/figures",
        config: PlotConfig | None = None,
        verbose: bool = True,
        cache_size_limit: int = 100_000_000,  # 100MB cache limit
    ):
        """
        Initialize optimized base visualizer with smart caching.

        Args:
            results_dir: Directory containing analysis results
            figures_dir: Directory for saving figures
            config: Plot configuration (uses default if None)
            verbose: Whether to print status messages
            cache_size_limit: Maximum cache size in bytes
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.verbose = verbose
        self.cache_size_limit = cache_size_limit

        # Use shared config instance when possible
        config_id = id(config) if config else id(DEFAULT_CONFIG)
        if config_id in self._shared_configs:
            self.config = self._shared_configs[config_id]
        else:
            self.config = config or DEFAULT_CONFIG
            self._shared_configs[config_id] = self.config

        # Ensure output directory exists (lazy creation)
        ensure_directory(self.figures_dir)

        # Apply plot styling (cached)
        self.config.apply_style()

        # Initialize logging (lazy)
        self._logger = None

        # Optimized cache for loaded data with memory tracking
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._cache_memory_usage = 0
        self._cache_access_count: dict[str, int] = {}

        # Compiled regex patterns for efficient file matching
        self._pattern_cache: dict[str, re.Pattern] = {}

        # Load available data (lazy/on-demand)
        self._data_loaded = False
        if verbose:
            self._ensure_data_loaded()
            self._print_initialization_summary()

    @property
    def logger(self) -> logging.Logger:
        """Lazy initialization of logger for performance."""
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return self._logger

    def _ensure_data_loaded(self) -> None:
        """Ensure data is loaded (lazy loading for performance)."""
        if not self._data_loaded:
            self._load_available_data()
            self._data_loaded = True

    @abstractmethod
    def _load_available_data(self) -> None:
        """Load data files available for visualization. Must be implemented by subclasses."""
        pass

    def _print_initialization_summary(self) -> None:
        """Print summary of initialization."""
        print(f"{self.__class__.__name__} initialized:")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Figures directory: {self.figures_dir}")
        print(f"  Available datasets: {len(self._data_cache)}")
        if self._data_cache:
            for name, df in self._data_cache.items():
                print(f"    - {name}: {len(df)} records")

    def _load_csv_file(
        self,
        file_pattern: str,
        required_columns: list[str] | None = None,
        cache_key: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Optimized CSV file loading with smart caching and memory management.

        Args:
            file_pattern: Glob pattern for files (e.g., "event_study_*.csv")
            required_columns: Required columns for validation
            cache_key: Key for caching (uses filename if None)
            use_cache: Whether to use caching

        Returns:
            Dictionary mapping outcome names to DataFrames
        """
        # Check cache first if enabled
        if use_cache and cache_key and cache_key in self._data_cache:
            self._cache_access_count[cache_key] = self._cache_access_count.get(cache_key, 0) + 1
            return {cache_key: self._data_cache[cache_key]}

        results = {}
        files = list(self.results_dir.glob(file_pattern))

        # Batch process files for efficiency
        for file_path in files:
            try:
                # Extract outcome name efficiently using cached regex
                outcome = self._extract_outcome_from_filename_optimized(
                    file_path.name, file_pattern
                )

                # Check individual file cache
                individual_cache_key = f"{file_pattern}_{outcome}"
                if use_cache and individual_cache_key in self._data_cache:
                    results[outcome] = self._data_cache[individual_cache_key]
                    self._cache_access_count[individual_cache_key] = (
                        self._cache_access_count.get(individual_cache_key, 0) + 1
                    )
                    continue

                # Load file with optimized pandas settings
                df = pd.read_csv(
                    file_path,
                    engine="c",  # Use C engine for speed
                    low_memory=False,
                )

                # Vectorized validation if required columns specified
                if required_columns:
                    validate_data_frame(df, required_columns, f"{file_path.name}")

                results[outcome] = df

                # Cache the data with memory tracking
                if use_cache:
                    self._add_to_cache(individual_cache_key, df)

                if self.verbose:
                    self.logger.info(f"Loaded {file_path.name}: {len(df)} records")

            except Exception as e:
                self.logger.warning(f"Could not load {file_path.name}: {e}")
                continue

        return results

    def _add_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """Add DataFrame to cache with memory management."""
        # Estimate memory usage
        memory_usage = df.memory_usage(deep=True).sum()

        # Check if cache limit would be exceeded
        if self._cache_memory_usage + memory_usage > self.cache_size_limit:
            self._cleanup_cache()

        # Add to cache
        self._data_cache[key] = df
        self._cache_memory_usage += memory_usage
        self._cache_access_count[key] = 1

    def _cleanup_cache(self) -> None:
        """Remove least recently used items from cache."""
        # Sort by access count (ascending) to remove least used first
        sorted_items = sorted(self._cache_access_count.items(), key=lambda x: x[1])

        # Remove items until under 75% of limit
        target_usage = self.cache_size_limit * 0.75

        for key, _ in sorted_items:
            if self._cache_memory_usage <= target_usage:
                break

            if key in self._data_cache:
                memory_usage = self._data_cache[key].memory_usage(deep=True).sum()
                del self._data_cache[key]
                del self._cache_access_count[key]
                self._cache_memory_usage -= memory_usage

    @lru_cache(maxsize=64)
    def _compile_pattern_regex(self, pattern: str) -> re.Pattern:
        """Compile and cache regex pattern for filename extraction."""
        # Convert glob pattern to regex
        regex_pattern = pattern.replace("*", "([^/]+)")
        regex_pattern = regex_pattern.replace(".", "\\.")
        return re.compile(regex_pattern)

    def _extract_outcome_from_filename_optimized(self, filename: str, pattern: str) -> str:
        """
        Optimized outcome name extraction using cached regex patterns.

        Args:
            filename: Full filename
            pattern: Pattern with wildcard (e.g., "event_study_*.csv")

        Returns:
            Outcome name
        """
        # Use cached regex for faster matching
        regex = self._compile_pattern_regex(pattern)
        match = regex.match(filename)

        if match:
            return match.group(1)

        # Fallback to original method
        return self._extract_outcome_from_filename_fallback(filename, pattern)

    def _extract_outcome_from_filename_fallback(self, filename: str, pattern: str) -> str:
        """Fallback filename extraction method."""
        # Remove extension
        name_no_ext = filename.rsplit(".", 1)[0]

        # Remove prefix and suffix from pattern
        prefix = pattern.split("*")[0] if "*" in pattern else ""
        suffix = pattern.split("*")[1].rsplit(".", 1)[0] if "*" in pattern else ""

        outcome = name_no_ext
        if prefix:
            outcome = outcome.replace(prefix, "", 1)
        if suffix:
            outcome = outcome.replace(suffix, "")

        return outcome.strip("_")

    # Keep original method for backward compatibility
    def _extract_outcome_from_filename(self, filename: str, pattern: str) -> str:
        """Extract outcome name from filename (delegates to optimized version)."""
        return self._extract_outcome_from_filename_optimized(filename, pattern)

    def get_data(self, key: str) -> pd.DataFrame | None:
        """
        Get data from cache with access tracking.

        Args:
            key: Cache key

        Returns:
            DataFrame if found, None otherwise
        """
        self._ensure_data_loaded()
        if key in self._data_cache:
            # Update access count for LRU management
            self._cache_access_count[key] = self._cache_access_count.get(key, 0) + 1
            return self._data_cache[key]
        return None

    @lru_cache(maxsize=1)
    def _get_cached_outcomes(self, cache_keys: tuple[str, ...]) -> list[str]:
        """Cached outcome extraction for performance."""
        outcomes = set()
        for key in cache_keys:
            if "_" in key:
                # Extract outcome from cache key
                parts = key.split("_", 1)
                if len(parts) > 1:
                    outcomes.add(parts[1])
                else:
                    outcomes.add(key)
        return sorted(list(outcomes))

    def list_available_outcomes(self) -> list[str]:
        """
        Get cached list of available outcome variables with optimization.

        Returns:
            List of outcome names
        """
        self._ensure_data_loaded()
        # Use tuple for caching (immutable)
        cache_keys = tuple(self._data_cache.keys())
        return self._get_cached_outcomes(cache_keys)

    def validate_outcome(self, outcome: str) -> bool:
        """
        Check if outcome data is available.

        Args:
            outcome: Outcome variable name

        Returns:
            True if data is available
        """
        available_outcomes = self.list_available_outcomes()
        return outcome in available_outcomes

    def _create_figure(
        self, figsize: tuple[float, float] | None = None, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create figure with consistent styling.

        Args:
            figsize: Figure size (width, height)
            **kwargs: Additional arguments for plt.subplots()

        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = (10, 6)

        fig, ax = plt.subplots(figsize=figsize, **kwargs)

        # Apply additional styling
        if self.config.remove_top_spine:
            ax.spines["top"].set_visible(False)
        if self.config.remove_right_spine:
            ax.spines["right"].set_visible(False)

        return fig, ax

    def _format_plot_title(self, title: str, outcome: str = "") -> str:
        """
        Format plot title with consistent styling.

        Args:
            title: Base title
            outcome: Outcome variable name

        Returns:
            Formatted title
        """
        if outcome:
            outcome_label = format_outcome_label(outcome)
            return f"{title}: {outcome_label}"
        return title

    def _add_reference_line(
        self, ax: plt.Axes, y_value: float = 0, line_type: str = "horizontal", **kwargs
    ) -> None:
        """
        Add reference line to plot.

        Args:
            ax: Matplotlib axes
            y_value: Y-value for horizontal line (or x-value for vertical)
            line_type: 'horizontal' or 'vertical'
            **kwargs: Additional line styling
        """
        default_kwargs = {
            "color": "black",
            "linestyle": "-",
            "linewidth": self.config.line_width_reference,
            "alpha": 0.8,
        }
        default_kwargs.update(kwargs)

        if line_type == "horizontal":
            ax.axhline(y=y_value, **default_kwargs)
        elif line_type == "vertical":
            ax.axvline(x=y_value, **default_kwargs)

    def _save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        formats: list[str] | None = None,
        close_fig: bool = True,
    ) -> list[str]:
        """
        Save plot in multiple formats.

        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
            formats: List of formats to save
            close_fig: Whether to close figure after saving

        Returns:
            List of saved file paths
        """
        if formats is None:
            formats = ["png", "pdf"]

        filepath = self.figures_dir / filename
        saved_files = save_figure(fig, filepath, formats, self.config.figure_dpi)

        if close_fig:
            plt.close(fig)

        if saved_files and self.verbose:
            print(f"Plot saved: {saved_files[0]}")

        return saved_files

    def clear_cache(self) -> None:
        """Clear data cache and reset memory tracking."""
        self._data_cache.clear()
        self._cache_access_count.clear()
        self._cache_memory_usage = 0
        self._data_loaded = False
        # Clear LRU caches
        self._get_cached_outcomes.cache_clear()
        self._compile_pattern_regex.cache_clear()
        if self.verbose:
            self.logger.info("Data cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about cached data and memory usage.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_items": len(self._data_cache),
            "memory_usage_bytes": self._cache_memory_usage,
            "memory_usage_mb": self._cache_memory_usage / (1024 * 1024),
            "cache_limit_mb": self.cache_size_limit / (1024 * 1024),
            "utilization_percent": (self._cache_memory_usage / self.cache_size_limit) * 100,
            "item_sizes": {key: len(df) for key, df in self._data_cache.items()},
            "access_counts": self._cache_access_count.copy(),
        }

    @abstractmethod
    def create_visualization(self, *args, **kwargs) -> str | list[str]:
        """
        Create visualization. Must be implemented by subclasses.

        Returns:
            Path(s) to created visualization files
        """
        pass


class MultiplotVisualizer(BaseVisualizer):
    """
    Base class for visualizers that create multiple related plots.
    """

    def create_all_visualizations(
        self, outcomes: list[str] | None = None, **kwargs
    ) -> dict[str, list[str]]:
        """
        Create complete set of visualizations.

        Args:
            outcomes: List of outcomes to process (all if None)
            **kwargs: Additional arguments for individual plots

        Returns:
            Dictionary mapping plot types to lists of file paths
        """
        if outcomes is None:
            outcomes = self.list_available_outcomes()

        all_files = {}

        for outcome in outcomes:
            if not self.validate_outcome(outcome):
                self.logger.warning(f"Skipping unavailable outcome: {outcome}")
                continue

            try:
                outcome_files = self._create_outcome_visualizations(outcome, **kwargs)
                if outcome_files:
                    all_files[outcome] = outcome_files
            except Exception as e:
                self.logger.error(f"Error creating visualizations for {outcome}: {e}")
                continue

        # Create summary visualizations
        try:
            summary_files = self._create_summary_visualizations(outcomes, **kwargs)
            if summary_files:
                all_files["summary"] = summary_files
        except Exception as e:
            self.logger.error(f"Error creating summary visualizations: {e}")

        return all_files

    @abstractmethod
    def _create_outcome_visualizations(self, outcome: str, **kwargs) -> list[str]:
        """
        Create visualizations for a specific outcome.

        Args:
            outcome: Outcome variable name
            **kwargs: Additional arguments

        Returns:
            List of created file paths
        """
        pass

    def _create_summary_visualizations(self, outcomes: list[str], **kwargs) -> list[str]:
        """
        Create summary visualizations across outcomes.
        Default implementation returns empty list.

        Args:
            outcomes: List of outcome names
            **kwargs: Additional arguments

        Returns:
            List of created file paths
        """
        return []
