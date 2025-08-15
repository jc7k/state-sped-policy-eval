"""
Report Generation Module for Special Education Policy Analysis

This module provides comprehensive reporting capabilities including:
- Interactive HTML dashboards
- Publication-ready LaTeX tables
- Advanced visualizations
- Policy briefs for non-technical audiences
- Technical methodology appendices
"""

from .dashboard_generator import DashboardGenerator
from .latex_table_generator import LaTeXTableGenerator
from .policy_brief_generator import PolicyBriefGenerator
from .technical_appendix import TechnicalAppendixGenerator
from .visualization_suite import VisualizationSuite

__all__ = [
    "DashboardGenerator",
    "LaTeXTableGenerator",
    "VisualizationSuite",
    "PolicyBriefGenerator",
    "TechnicalAppendixGenerator",
]
