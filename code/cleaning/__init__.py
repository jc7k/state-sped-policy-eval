"""
Data Cleaning Module - Refactored
Standardization and integration of collected datasets using modular components.

This module provides:
- Modular data loaders for each source (NAEP, Census, EdFacts, OCR)
- Centralized configuration management
- Panel data creation and aggregation
- Consistent error handling and validation
- Main integration orchestrator
"""

from .config import IntegrationConfig, DEFAULT_CONFIG
from .data_integration import DataIntegrator
from .data_loaders import DataLoaderFactory
from .panel_creator import PanelDataCreator
from .utils import DataCleaningUtils, ErrorHandlingMixin

__all__ = [
    "DataIntegrator",
    "DataLoaderFactory", 
    "PanelDataCreator",
    "IntegrationConfig",
    "DEFAULT_CONFIG",
    "DataCleaningUtils",
    "ErrorHandlingMixin",
]
