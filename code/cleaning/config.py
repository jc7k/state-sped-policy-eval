#!/usr/bin/env python
"""
Configuration Module for Data Integration

Centralized configuration for file patterns, column mappings, and processing parameters.
Makes the data integration pipeline more configurable and maintainable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class FilePatterns:
    """File patterns for different data sources."""
    
    # NAEP file patterns
    NAEP_PATTERNS = [
        "naep_state_swd_data.csv",
        "naep_*_swd_*.csv",
        "naep_achievement_*.csv"
    ]
    
    # Census file patterns
    CENSUS_PATTERNS = [
        "census_education_finance_parsed.csv",
        "census_f33_*.csv",
        "census_finance_*.csv"
    ]
    
    # EdFacts file patterns (by year)
    EDFACTS_PATTERNS = {
        2019: [
            "bchildcountandedenvironment2019.csv",
            "bchildcountandedenvironment2019-20.csv"
        ],
        2020: [
            "bchildcountandedenvironment2020.csv", 
            "bchildcountandedenvironment2020-21.csv"
        ],
        2021: [
            "bchildcountandedenvironment2021.csv",
            "bchildcountandedenvironment2021-22.csv"
        ],
        2022: [
            "bchildcountandedenvironment2022.csv",
            "bchildcountandedenvironment2022-23.csv"
        ],
        2023: [
            "bchildcountandedenvironment2023.csv",
            "bchildcountandedenvironment2023-24.csv"
        ]
    }
    
    # OCR file patterns
    OCR_CSV_FILES = [
        ("CRDC-2009-State-Discipline.csv", 2009),
        ("CRDC-2011-State-Discipline.csv", 2011),
        ("CRDC-2013-State-Data.csv", 2013),
        ("CRDC-2015-State-Data.csv", 2015),
    ]
    
    OCR_DISCIPLINE_FILES = [
        "Suspensions.csv",
        "Expulsions.csv", 
        "Restraint and Seclusion.csv",
    ]
    
    OCR_EXTRACT_YEARS = [2017, 2020]


@dataclass
class ColumnMappings:
    """Column name mappings for standardization."""
    
    # Common mappings used across sources
    COMMON = {
        # State identifiers
        "STATE": "state",
        "State": "state",
        "State Name": "state_name", 
        "State Abbr": "state",
        "StateAbbr": "state",
        "SEA_STATE": "state",
        "LEA_STATE": "state",
        "LEA State": "state",
        
        # Year identifiers
        "YEAR": "year",
        "Year": "year",
        "School Year": "year",
        "SchoolYear": "year",
        
        # Demographics
        "RACE_ETHNICITY": "race_ethnicity",
        "SEX": "sex",
        "AGE": "age_group",
        "LEP": "english_learner",
    }
    
    # NAEP-specific mappings
    NAEP = {
        # NAEP uses fairly standard column names, minimal mapping needed
    }
    
    # Census-specific mappings
    CENSUS = {
        "Total Revenue": "total_revenue",
        "Total Expenditure": "total_expenditure",
        "Current Expenditure": "current_expenditure", 
        "Instruction Expenditure": "instruction_expenditure",
        "Per Pupil Expenditure": "per_pupil_expenditure",
        "Special Education Expenditure": "special_education_expenditure",
    }
    
    # EdFacts-specific mappings
    EDFACTS = {
        "DISABILITY": "disability_category",
        "SEA Disability Category": "disability_category",
        "IDEA_INDICATOR": "idea_indicator", 
        "EDUCATIONAL_ENVIRONMENT": "educational_environment",
        "SEA Education Environment": "educational_environment",
        "CHILD_COUNT": "child_count",
    }
    
    # OCR-specific mappings
    OCR = {
        "Total Enrollment": "total_enrollment",
        "Students with Disabilities": "swd_enrollment",
        "Out-of-School Suspensions": "suspensions",
        "Expulsions": "expulsions",
        "Restraint": "restraint_incidents", 
        "Seclusion": "seclusion_incidents",
    }


@dataclass 
class ProcessingParameters:
    """Parameters for data processing operations."""
    
    # Year range for analysis
    DEFAULT_YEAR_RANGE: Tuple[int, int] = (2009, 2023)
    
    # Minimum states required for production use
    MIN_STATES_PRODUCTION: int = 50
    MIN_STATES_TESTING: int = 2
    
    # Missing data thresholds
    HIGH_MISSING_THRESHOLD: float = 0.3  # 30%
    CRITICAL_MISSING_THRESHOLD: float = 0.7  # 70%
    
    # File processing parameters
    EDFACTS_SKIP_ROWS: int = 4  # Skip metadata rows
    CSV_LOW_MEMORY: bool = False  # For large files
    
    # Aggregation functions by data type
    AGGREGATION_FUNCTIONS = {
        "mean": ["avg_scale_score", "pct_below_basic", "pct_basic", "pct_proficient", "pct_advanced"],
        "sum": ["child_count", "total_enrollment", "swd_enrollment", "suspensions", "expulsions"],
        "first": ["state", "year"]  # These should be unique within groups
    }
    
    # Variable prefixes for datasets
    DATASET_PREFIXES = {
        "naep": "naep_",
        "edfacts": "edfacts_", 
        "ocr": "ocr_",
        "census": ""  # No prefix for census (financial variables)
    }


@dataclass
class ValidationRules:
    """Rules for data validation."""
    
    # Required columns by dataset
    REQUIRED_COLUMNS = {
        "naep": ["state", "year", "subject", "grade"],
        "census": ["state", "year"],
        "edfacts": ["state", "year"],
        "ocr": ["state", "year"]
    }
    
    # Expected value ranges
    VALUE_RANGES = {
        "year": (2000, 2030),
        "avg_scale_score": (0, 500),  # NAEP scale
        "pct_below_basic": (0, 100),
        "pct_basic": (0, 100),
        "pct_proficient": (0, 100), 
        "pct_advanced": (0, 100),
        "child_count": (0, 1000000),  # Reasonable upper bound
        "total_enrollment": (0, 10000000),
        "per_pupil_expenditure": (0, 50000),  # Reasonable range
    }
    
    # Outlier detection parameters
    OUTLIER_Z_SCORE_THRESHOLD: float = 3.5
    OUTLIER_IQR_MULTIPLIER: float = 1.5


@dataclass
class OutputSettings:
    """Settings for output files and directories."""
    
    # Default directory structure
    DEFAULT_DIRS = {
        "raw": "data/raw",
        "processed": "data/processed", 
        "final": "data/final"
    }
    
    # Output file names
    OUTPUT_FILES = {
        "naep_cleaned": "naep_cleaned.csv",
        "census_cleaned": "census_cleaned.csv",
        "edfacts_cleaned": "edfacts_cleaned.csv", 
        "ocr_cleaned": "ocr_cleaned.csv",
        "master_dataset": "master_analysis_dataset.csv"
    }
    
    # CSV output parameters
    CSV_SETTINGS = {
        "index": False,
        "float_format": "%.3f",  # 3 decimal places for floats
        "date_format": "%Y",  # Year format
    }


class IntegrationConfig:
    """Main configuration class that combines all settings."""
    
    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        year_range: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize configuration.
        
        Args:
            raw_data_dir: Override default raw data directory
            output_dir: Override default output directory
            year_range: Override default year range
        """
        self.file_patterns = FilePatterns()
        self.column_mappings = ColumnMappings()
        self.processing_params = ProcessingParameters()
        self.validation_rules = ValidationRules()
        self.output_settings = OutputSettings()
        
        # Override defaults if provided
        if raw_data_dir:
            self.raw_data_dir = Path(raw_data_dir)
        else:
            self.raw_data_dir = Path(self.output_settings.DEFAULT_DIRS["raw"])
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.output_settings.DEFAULT_DIRS["processed"])
        
        if year_range:
            self.processing_params.DEFAULT_YEAR_RANGE = year_range
    
    def get_data_source_config(self, source: str) -> Dict:
        """
        Get complete configuration for a specific data source.
        
        Args:
            source: Data source name (naep, census, edfacts, ocr)
            
        Returns:
            Dictionary with all configuration for the source
        """
        base_config = {
            "raw_data_dir": self.raw_data_dir,
            "required_columns": self.validation_rules.REQUIRED_COLUMNS.get(source, []),
            "common_mappings": self.column_mappings.COMMON,
            "prefix": self.processing_params.DATASET_PREFIXES.get(source, ""),
        }
        
        if source == "naep":
            base_config.update({
                "file_patterns": self.file_patterns.NAEP_PATTERNS,
                "specific_mappings": self.column_mappings.NAEP,
            })
        elif source == "census":
            base_config.update({
                "file_patterns": self.file_patterns.CENSUS_PATTERNS,
                "specific_mappings": self.column_mappings.CENSUS,
            })
        elif source == "edfacts":
            base_config.update({
                "file_patterns": self.file_patterns.EDFACTS_PATTERNS,
                "specific_mappings": self.column_mappings.EDFACTS,
                "skip_rows": self.processing_params.EDFACTS_SKIP_ROWS,
                "target_years": list(self.file_patterns.EDFACTS_PATTERNS.keys()),
            })
        elif source == "ocr":
            base_config.update({
                "csv_files": self.file_patterns.OCR_CSV_FILES,
                "discipline_files": self.file_patterns.OCR_DISCIPLINE_FILES,
                "extract_years": self.file_patterns.OCR_EXTRACT_YEARS,
                "specific_mappings": self.column_mappings.OCR,
            })
        
        return base_config
    
    def get_output_path(self, filename: str) -> Path:
        """
        Get full output path for a file.
        
        Args:
            filename: Name of output file
            
        Returns:
            Full path to output file
        """
        return self.output_dir / filename
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = IntegrationConfig()