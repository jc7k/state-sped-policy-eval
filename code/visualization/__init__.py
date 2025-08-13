"""
Optimized Visualization Module

High-performance, publication-ready plotting and figure generation for econometric 
analysis results. Features advanced caching, vectorized operations, and memory-
optimized processing for special education policy research.

Performance Optimizations:
- Smart data caching with LRU management (100MB default limit)
- Vectorized confidence interval calculations using scipy.stats
- Cached matplotlib configuration with lazy evaluation
- Efficient regex-based file pattern matching
- Memory-conscious DataFrame operations with automatic cleanup

Core Components:
- BaseVisualizer: Optimized base class with smart caching and memory management
- EventStudyVisualizer: High-performance event study plots with vectorized operations
- TreatmentEffectsDashboard: Geographic visualization with cached coordinate data
- PlotConfig: Memory-efficient configuration with cached style application
- utils: Vectorized utility functions with LRU caching

Key Features:
- Automatic cache management with configurable size limits
- Comprehensive error handling with graceful fallbacks
- Publication-ready styling with consistent configuration
- Support for multiple output formats (PNG, PDF, EPS, SVG)
- Integrated logging and performance monitoring
- Memory usage tracking and optimization

Example Usage:
    from code.visualization import EventStudyVisualizer
    
    # Create visualizer with custom cache limit (50MB)
    visualizer = EventStudyVisualizer(
        results_dir="output/tables",
        figures_dir="output/figures", 
        cache_size_limit=50_000_000
    )
    
    # Generate all visualizations efficiently
    plots = visualizer.create_all_visualizations()
    
    # Monitor cache performance
    cache_info = visualizer.get_cache_info()
    print(f"Cache utilization: {cache_info['utilization_percent']:.1f}%")

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

# Core visualization classes
from .base import BaseVisualizer, MultiplotVisualizer
from .event_study_plots import EventStudyVisualizer
from .treatment_dashboard import TreatmentEffectsDashboard

# Configuration and utilities
from .config import (
    PlotConfig, 
    PlotTemplates, 
    DEFAULT_CONFIG,
    get_state_coordinates,
    get_us_regions,
    FIGURE_SIZES,
    COLOR_SCHEMES,
    SIGNIFICANCE_LEVELS
)

from .utils import (
    # Core data processing
    format_outcome_label,
    generate_ylabel,
    calculate_confidence_intervals,
    safe_numeric_operation,
    prepare_plot_data,
    validate_data_frame,
    
    # Visualization utilities  
    create_color_palette,
    get_optimal_figure_size,
    save_figure,
    add_significance_stars,
    format_number,
    
    # File management
    ensure_directory
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Jeff Chen, jeffreyc1@alumni.cmu.edu"

# Define public API
__all__ = [
    # Main classes
    "BaseVisualizer",
    "MultiplotVisualizer", 
    "EventStudyVisualizer",
    "TreatmentEffectsDashboard",
    
    # Configuration
    "PlotConfig",
    "PlotTemplates", 
    "DEFAULT_CONFIG",
    "get_state_coordinates",
    "get_us_regions",
    "FIGURE_SIZES",
    "COLOR_SCHEMES", 
    "SIGNIFICANCE_LEVELS",
    
    # Core utilities
    "format_outcome_label",
    "generate_ylabel", 
    "calculate_confidence_intervals",
    "safe_numeric_operation",
    "prepare_plot_data",
    "validate_data_frame",
    "create_color_palette",
    "get_optimal_figure_size",
    "save_figure",
    "add_significance_stars",
    "format_number",
    "ensure_directory",
    
    # Metadata
    "__version__",
    "__author__"
]
