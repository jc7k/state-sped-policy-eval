#!/usr/bin/env python
"""
Shared Utilities for Data Cleaning Module

Common functionality for data integration and cleaning operations.
Eliminates code duplication and provides consistent error handling patterns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..collection.common import StateUtils


class CleaningError(Exception):
    """Custom exception for data cleaning operations."""
    pass


class DataCleaningUtils:
    """Utilities for common data cleaning operations."""

    @staticmethod
    def standardize_state_codes(
        df: pd.DataFrame, 
        state_column: str = "state",
        state_name_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Standardize state codes using shared state mapping.
        
        Args:
            df: DataFrame to process
            state_column: Name of state code column
            state_name_column: Name of state name column (if exists)
            
        Returns:
            DataFrame with standardized state codes
        """
        df_clean = df.copy()
        
        # Convert state names to codes if state name column exists
        if state_name_column and state_name_column in df_clean.columns:
            df_clean[state_column] = df_clean[state_name_column].apply(
                StateUtils.name_to_abbrev
            )
        
        # Standardize existing state codes
        if state_column in df_clean.columns:
            df_clean[state_column] = df_clean[state_column].str.upper()
            # Filter to valid states only
            df_clean = df_clean[df_clean[state_column].isin(StateUtils.get_all_states())]
        
        return df_clean

    @staticmethod
    def safe_numeric_conversion(
        df: pd.DataFrame, 
        columns: List[str], 
        errors: str = "coerce"
    ) -> pd.DataFrame:
        """
        Safely convert columns to numeric types.
        
        Args:
            df: DataFrame to process
            columns: List of column names to convert
            errors: How to handle conversion errors
            
        Returns:
            DataFrame with converted columns
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors=errors)
        
        return df_clean

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, 
        required_columns: List[str],
        dataset_name: str = "dataset"
    ) -> None:
        """
        Validate that required columns exist in DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            dataset_name: Name of dataset for error messages
            
        Raises:
            CleaningError: If required columns are missing
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise CleaningError(
                f"{dataset_name} missing required columns: {missing_columns}"
            )

    @staticmethod
    def filter_by_years(
        df: pd.DataFrame, 
        year_column: str = "year",
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame by year range.
        
        Args:
            df: DataFrame to filter
            year_column: Name of year column
            min_year: Minimum year (inclusive)
            max_year: Maximum year (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        
        if year_column in df_filtered.columns:
            if min_year is not None:
                df_filtered = df_filtered[df_filtered[year_column] >= min_year]
            if max_year is not None:
                df_filtered = df_filtered[df_filtered[year_column] <= max_year]
        
        return df_filtered

    @staticmethod
    def aggregate_by_groups(
        df: pd.DataFrame,
        group_columns: List[str],
        agg_columns: Dict[str, str],
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Aggregate DataFrame by specified groups.
        
        Args:
            df: DataFrame to aggregate
            group_columns: Columns to group by
            agg_columns: Dictionary of {column: aggregation_function}
            prefix: Prefix to add to aggregated column names
            
        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df
        
        # Validate group columns exist
        missing_groups = [col for col in group_columns if col not in df.columns]
        if missing_groups:
            raise CleaningError(f"Group columns missing: {missing_groups}")
        
        # Filter aggregation columns to only those that exist
        valid_agg_columns = {
            col: func for col, func in agg_columns.items() 
            if col in df.columns
        }
        
        if not valid_agg_columns:
            # Return empty DataFrame with group columns
            return pd.DataFrame(columns=group_columns)
        
        # Perform aggregation
        aggregated = df.groupby(group_columns).agg(valid_agg_columns).reset_index()
        
        # Add prefix to aggregated columns
        if prefix:
            rename_dict = {
                col: f"{prefix}_{col}" 
                for col in aggregated.columns 
                if col not in group_columns
            }
            aggregated = aggregated.rename(columns=rename_dict)
        
        return aggregated


class ErrorHandlingMixin:
    """Mixin class providing consistent error handling patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.errors = []
        self.warnings = []
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error message and add to errors list."""
        self.logger.error(message)
        if exception:
            self.logger.error(f"Exception details: {str(exception)}")
        self.errors.append(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message and add to warnings list."""
        self.logger.warning(message)
        self.warnings.append(message)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def clear_logs(self) -> None:
        """Clear accumulated errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
    
    def has_errors(self) -> bool:
        """Check if any errors have been logged."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> Dict[str, List[str]]:
        """Get summary of all errors and warnings."""
        return {
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy()
        }


class FilePatternMatcher:
    """Utility class for matching file patterns across different data sources."""
    
    @staticmethod
    def find_files_by_patterns(
        directory: Path,
        patterns: List[str],
        recursive: bool = True
    ) -> List[Path]:
        """
        Find files matching any of the given patterns.
        
        Args:
            directory: Directory to search
            patterns: List of glob patterns
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        found_files = []
        
        if not directory.exists():
            return found_files
        
        for pattern in patterns:
            if recursive:
                matches = directory.rglob(pattern)
            else:
                matches = directory.glob(pattern)
            
            found_files.extend(matches)
        
        # Remove duplicates and sort
        unique_files = list(set(found_files))
        unique_files.sort()
        
        return unique_files

    @staticmethod
    def extract_year_from_filename(filename: str) -> Optional[int]:
        """
        Extract year from filename using common patterns.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Extracted year or None if not found
        """
        import re
        
        # Common year patterns in data files
        patterns = [
            r'(\d{4})',  # Four digit year
            r'(\d{4})-(\d{2})',  # School year format (e.g., 2019-20)
            r'(\d{4})-(\d{4})',  # Full school year format (e.g., 2019-2020)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                year = int(match.group(1))
                # Validate reasonable year range
                if 2000 <= year <= 2030:
                    return year
        
        return None


class ColumnStandardizer:
    """Utility class for standardizing column names across datasets."""
    
    # Common column mappings used across multiple data sources
    COMMON_MAPPINGS = {
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
        
        # Common demographic categories
        "RACE_ETHNICITY": "race_ethnicity",
        "SEX": "sex",
        "DISABILITY": "disability_category",
        "AGE": "age_group",
        "LEP": "english_learner",
        
        # Common count variables
        "CHILD_COUNT": "child_count",
        "Total Enrollment": "total_enrollment",
        "Students with Disabilities": "swd_enrollment",
    }
    
    @staticmethod
    def standardize_columns(
        df: pd.DataFrame, 
        custom_mappings: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Standardize column names using common and custom mappings.
        
        Args:
            df: DataFrame to process
            custom_mappings: Additional column mappings
            
        Returns:
            DataFrame with standardized column names
        """
        df_clean = df.copy()
        
        # Combine common and custom mappings
        all_mappings = ColumnStandardizer.COMMON_MAPPINGS.copy()
        if custom_mappings:
            all_mappings.update(custom_mappings)
        
        # Apply mappings
        for old_name, new_name in all_mappings.items():
            if old_name in df_clean.columns:
                df_clean = df_clean.rename(columns={old_name: new_name})
        
        return df_clean