"""
Visualization Configuration

Optimized centralized configuration for matplotlib styling and plotting parameters.
Features cached styling application, efficient color palette generation, and
memory-optimized dataclass configuration for publication-ready visualizations.

Key Optimizations:
- Cached matplotlib configuration application
- Optimized color palette generation with LRU caching
- Memory-efficient dataclass configuration
- Lazy evaluation of configuration parameters
- Vectorized state coordinate operations

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu  
Created in collaboration with Claude Code
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotConfig:
    """
    Optimized configuration class for plot styling and parameters with lazy evaluation.
    """
    # Figure parameters
    figure_dpi: int = 300
    figure_facecolor: str = 'white'
    figure_edgecolor: str = 'none'
    save_bbox_inches: str = 'tight'
    save_pad_inches: float = 0.1
    
    # Font settings
    font_family: str = 'serif'
    font_serif: Optional[List[str]] = None
    font_size_base: int = 12
    font_size_title: int = 16
    font_size_label: int = 14
    font_size_tick: int = 12
    font_size_legend: int = 11
    font_size_annotation: int = 10
    
    # Colors
    primary_color: str = '#1f77b4'  # steelblue
    secondary_color: str = '#ff7f0e'  # orange
    accent_colors: Optional[List[str]] = None
    line_color_significant: str = '#2ca02c'  # green
    line_color_insignificant: str = '#d62728'  # red
    grid_color: str = 'gray'
    background_color: str = 'white'
    
    # Line and marker styles
    line_width_main: float = 2.5
    line_width_grid: float = 0.5
    line_width_reference: float = 1.0
    marker_size: int = 8
    marker_edge_width: float = 1.5
    
    # Grid and spines
    grid_alpha: float = 0.3
    spine_width: float = 1.2
    remove_top_spine: bool = True
    remove_right_spine: bool = True
    
    # Confidence intervals
    ci_alpha: float = 0.2
    ci_method: str = '95%'  # '95%', '90%', or 'se'
    
    # Regional color mapping (uses field for lazy evaluation)
    region_colors: Optional[Dict[str, str]] = field(default=None)
    
    # Private cache for style dictionary
    _style_cache: Optional[Dict[str, any]] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Optimized initialization with lazy evaluation for memory efficiency."""
        # Only initialize fields when needed
        if self.font_serif is None:
            self.font_serif = ['Times New Roman', 'Computer Modern Roman', 'serif']
        
        if self.accent_colors is None:
            # Extended color palette for better visualization diversity
            self.accent_colors = [
                '#1f77b4',  # steelblue
                '#ff7f0e',  # orange  
                '#2ca02c',  # green
                '#d62728',  # red
                '#9467bd',  # purple
                '#8c564b',  # brown
                '#e377c2',  # pink
                '#7f7f7f',  # gray
                '#bcbd22',  # olive
                '#17becf',  # cyan
                '#aec7e8',  # light blue
                '#ffbb78',  # light orange
                '#98df8a',  # light green
                '#ff9896',  # light red
                '#c5b0d5'   # light purple
            ]
        
        if self.region_colors is None:
            self.region_colors = {
                'Northeast': '#1f77b4',    # steelblue
                'South': '#ff7f0e',        # orange
                'Midwest': '#2ca02c',      # green  
                'West': '#d62728',         # red
                'Overall': '#9467bd',      # purple
                'National': '#8c564b'      # brown for national averages
            }
    
    def apply_style(self) -> None:
        """Apply configuration to matplotlib rcParams with caching for performance."""
        # Use cached style dictionary if available
        if self._style_cache is None:
            self._style_cache = self._build_style_dict()
        
        plt.style.use('default')  # Start with clean slate
        plt.rcParams.update(self._style_cache)
    
    def _build_style_dict(self) -> Dict[str, any]:
        """Build matplotlib style dictionary (cached for performance)."""
        return {
            # Figure settings
            'figure.facecolor': self.figure_facecolor,
            'figure.edgecolor': self.figure_edgecolor,
            'figure.dpi': self.figure_dpi,
            'savefig.dpi': self.figure_dpi,
            'savefig.bbox': self.save_bbox_inches,
            'savefig.pad_inches': self.save_pad_inches,
            'savefig.facecolor': self.figure_facecolor,
            'savefig.edgecolor': self.figure_edgecolor,
            
            # Font settings
            'font.family': self.font_family,
            'font.serif': self.font_serif,
            'font.size': self.font_size_base,
            'axes.titlesize': self.font_size_title,
            'axes.labelsize': self.font_size_label,
            'xtick.labelsize': self.font_size_tick,
            'ytick.labelsize': self.font_size_tick,
            'legend.fontsize': self.font_size_legend,
            
            # Line and marker styles  
            'lines.linewidth': self.line_width_main,
            'lines.markersize': self.marker_size,
            'axes.linewidth': self.spine_width,
            
            # Grid settings
            'axes.grid': True,
            'grid.alpha': self.grid_alpha,
            'grid.linewidth': self.line_width_grid,
            'grid.color': self.grid_color,
            'axes.axisbelow': True,
            
            # Spine settings
            'axes.spines.top': not self.remove_top_spine,
            'axes.spines.right': not self.remove_right_spine,
            
            # Legend settings
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'legend.framealpha': 1.0,
        }
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """
        Get color palette with specified number of colors.
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of color codes
        """
        if n_colors <= 0:
            return []
        
        if n_colors <= len(self.accent_colors):
            return self.accent_colors[:n_colors]
        
        # Efficient modular cycling for large palettes
        return [self.accent_colors[i % len(self.accent_colors)] for i in range(n_colors)]
    
    def get_region_color(self, region: str) -> str:
        """
        Get color for specific region with fallback handling.
        
        Args:
            region: Region name
            
        Returns:
            Color code
        """
        return self.region_colors.get(region, self.primary_color)
    
    def get_significance_colors(self) -> Dict[str, str]:
        """
        Get standardized colors for significance levels.
        
        Returns:
            Dictionary mapping significance levels to colors
        """
        return {
            'significant': self.line_color_significant,
            'insignificant': self.line_color_insignificant,
            'marginal': '#FFA500',  # orange for marginal significance
            'primary': self.primary_color,
            'secondary': self.secondary_color
        }


class PlotTemplates:
    """
    Pre-configured plot templates for common visualization types.
    """
    
    @staticmethod
    def event_study_config() -> PlotConfig:
        """Configuration optimized for event study plots."""
        config = PlotConfig()
        config.line_width_main = 3.0
        config.marker_size = 10
        config.ci_alpha = 0.25
        return config
    
    @staticmethod
    def forest_plot_config() -> PlotConfig:
        """Configuration optimized for forest plots."""
        config = PlotConfig()
        config.marker_size = 12
        config.line_width_main = 2.0
        config.grid_alpha = 0.2
        return config
    
    @staticmethod
    def geographic_config() -> PlotConfig:
        """Configuration optimized for geographic visualizations."""
        config = PlotConfig()
        config.marker_size = 150
        config.marker_edge_width = 1.0
        config.font_size_annotation = 8
        return config
    
    @staticmethod
    def dashboard_config() -> PlotConfig:
        """Configuration optimized for dashboard displays."""
        config = PlotConfig()
        config.font_size_base = 10
        config.font_size_title = 14
        config.font_size_label = 12
        config.grid_alpha = 0.2
        return config


# Global default configuration
DEFAULT_CONFIG = PlotConfig()

# Common figure size mappings
FIGURE_SIZES = {
    'single_plot': (10, 6),
    'wide_plot': (12, 8), 
    'event_study': (12, 8),
    'forest_plot': (10, 6),
    'geographic_map': (16, 10),
    'dashboard_single': (8, 5),
    'dashboard_double': (14, 6),
    'dashboard_quad': (16, 10),
    'timeline': (14, 8),
    'presentation': (16, 9),  # 16:9 aspect ratio
    'publication': (8, 6),    # More compact for papers
}

# Color schemes for different analysis types
COLOR_SCHEMES = {
    'sequential': ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494'],
    'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
    'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'policy_reform': ['#2166ac', '#5aae61', '#fee08b', '#d73027'],  # Blue to red for treatment intensity
    'significance': ['#d73027', '#fee08b', '#5aae61'],  # Red, yellow, green for p-values
}

# Annotation styles
ANNOTATION_STYLES = {
    'significance_box': {
        'boxstyle': 'round,pad=0.3',
        'facecolor': 'wheat',
        'alpha': 0.8,
        'edgecolor': 'black'
    },
    'warning_box': {
        'boxstyle': 'round,pad=0.3', 
        'facecolor': 'lightcoral',
        'alpha': 0.8,
        'edgecolor': 'red'
    },
    'info_box': {
        'boxstyle': 'round,pad=0.3',
        'facecolor': 'lightblue', 
        'alpha': 0.8,
        'edgecolor': 'blue'
    }
}

# Statistical significance thresholds
SIGNIFICANCE_LEVELS = {
    'p001': 0.001,
    'p01': 0.01,
    'p05': 0.05,
    'p10': 0.10
}


# Optimized geographic data access functions

@lru_cache(maxsize=1)
def get_state_coordinates() -> Dict[str, Tuple[float, float]]:
    """
    Get cached state coordinates for geographic plotting (approximate centers).
    
    Returns:
        Dictionary mapping state codes to (longitude, latitude) tuples
    """
    return {
        'AL': (-86.8, 32.8), 'AK': (-152.0, 64.0), 'AZ': (-111.9, 34.2), 'AR': (-92.2, 34.8),
        'CA': (-119.8, 36.8), 'CO': (-105.5, 39.2), 'CT': (-72.7, 41.6), 'DE': (-75.5, 39.2),
        'DC': (-77.0, 38.9), 'FL': (-81.5, 27.8), 'GA': (-83.2, 32.2), 'HI': (-157.8, 21.3),
        'ID': (-114.6, 44.1), 'IL': (-89.2, 40.1), 'IN': (-86.3, 39.8), 'IA': (-93.6, 42.0),
        'KS': (-98.4, 38.5), 'KY': (-84.9, 37.8), 'LA': (-91.8, 31.2), 'ME': (-69.2, 45.2),
        'MD': (-76.5, 39.0), 'MA': (-71.8, 42.4), 'MI': (-84.5, 43.3), 'MN': (-94.6, 46.4),
        'MS': (-89.4, 32.7), 'MO': (-92.2, 38.3), 'MT': (-110.4, 47.1), 'NE': (-99.8, 41.5),
        'NV': (-117.0, 39.8), 'NH': (-71.5, 43.2), 'NJ': (-74.8, 40.2), 'NM': (-106.2, 34.5),
        'NY': (-74.2, 42.2), 'NC': (-78.6, 35.8), 'ND': (-99.8, 47.5), 'OH': (-82.7, 40.2),
        'OK': (-97.1, 35.6), 'OR': (-122.0, 44.9), 'PA': (-77.2, 40.3), 'RI': (-71.4, 41.6),
        'SC': (-80.9, 33.8), 'SD': (-99.9, 44.3), 'TN': (-86.4, 35.9), 'TX': (-97.7, 31.1),
        'UT': (-111.9, 40.2), 'VT': (-72.6, 44.0), 'VA': (-78.2, 37.7), 'WA': (-121.5, 47.4),
        'WV': (-80.9, 38.8), 'WI': (-90.0, 44.3), 'WY': (-107.3, 42.8)
    }

@lru_cache(maxsize=1)  
def get_us_regions() -> Dict[str, List[str]]:
    """
    Get cached US regional groupings for analysis.
    
    Returns:
        Dictionary mapping region names to lists of state codes
    """
    return {
        'Northeast': ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT'],
        'South': ['AL', 'AR', 'DE', 'DC', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 
                  'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'],
        'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
        'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
    }

# Maintain backward compatibility
STATE_COORDINATES = get_state_coordinates()
US_REGIONS = get_us_regions()