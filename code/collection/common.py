#!/usr/bin/env python
"""
Common Utilities for Data Collection Module

Shared functionality to eliminate code duplication across data collectors.
Includes state mappings, API client patterns, validation logic, and file utilities.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class StateUtils:
    """Centralized state mapping utilities for all data collectors."""

    # Comprehensive state mapping: full name -> abbreviation
    STATE_NAME_TO_ABBREV = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
    }

    # FIPS codes for Census API
    FIPS_TO_ABBREV = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY",
    }

    # All state abbreviations (including DC)
    ALL_STATE_CODES = list(STATE_NAME_TO_ABBREV.values())

    @classmethod
    def name_to_abbrev(cls, state_name: str) -> str | None:
        """Convert state name to abbreviation."""
        if not state_name:
            return None
        return cls.STATE_NAME_TO_ABBREV.get(state_name.strip())

    @classmethod
    def fips_to_abbrev(cls, fips_code: str) -> str | None:
        """Convert FIPS code to state abbreviation."""
        if not fips_code:
            return None
        return cls.FIPS_TO_ABBREV.get(fips_code.strip())

    @classmethod
    def is_valid_state(cls, state_code: str) -> bool:
        """Check if state code is valid."""
        return state_code in cls.ALL_STATE_CODES

    @classmethod
    def get_all_states(cls) -> list[str]:
        """Get list of all valid state codes."""
        return cls.ALL_STATE_CODES.copy()


class APIClient:
    """Base HTTP client with rate limiting and error handling."""

    def __init__(self, rate_limit_delay: float = 1.0, timeout: int = 30):
        """
        Initialize API client.

        Args:
            rate_limit_delay: Seconds to wait between requests
            timeout: Request timeout in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._request_count = 0

    def get(self, url: str, params: dict | None = None, **kwargs) -> requests.Response:
        """
        Make GET request with automatic rate limiting and error handling.

        Args:
            url: Request URL
            params: Query parameters
            **kwargs: Additional requests parameters

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        self._request_count += 1
        
        # Apply rate limiting (except for first request)
        if self._request_count > 1:
            time.sleep(self.rate_limit_delay)

        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = requests.get(url, params=params, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {str(e)}")
            raise


class DataValidator:
    """Common data validation patterns for all collectors."""

    @staticmethod
    def validate_state_coverage(df: pd.DataFrame, state_column: str = "state") -> dict:
        """
        Validate state coverage in dataset.

        Args:
            df: DataFrame to validate
            state_column: Name of state column

        Returns:
            Validation results dictionary
        """
        if df.empty:
            return {
                "states_covered": 0,
                "passed": False,
                "errors": ["Dataset is empty"],
                "warnings": []
            }

        states_covered = df[state_column].nunique()
        
        validation = {
            "states_covered": states_covered,
            "passed": True,
            "errors": [],
            "warnings": []
        }

        # Check state coverage (expect 50 states + DC = 51)
        if states_covered < 50:
            if states_covered < 2:
                validation["errors"].append(
                    f"Only {states_covered} states covered, expected 50+"
                )
                validation["passed"] = False
            else:
                validation["warnings"].append(
                    f"Only {states_covered} states covered, expected 50+ for production"
                )

        return validation

    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: list[str]) -> dict:
        """
        Check for duplicate records.

        Args:
            df: DataFrame to check
            subset: Columns to check for duplicates

        Returns:
            Validation results dictionary
        """
        if df.empty:
            return {"duplicates": 0, "passed": True, "errors": []}

        duplicates = df.duplicated(subset=subset).sum()
        
        validation = {
            "duplicates": duplicates,
            "passed": duplicates == 0,
            "errors": []
        }

        if duplicates > 0:
            validation["errors"].append(
                f"{duplicates} duplicate records found for columns: {subset}"
            )

        return validation

    @staticmethod
    def check_missing_data_rate(
        df: pd.DataFrame, 
        column: str, 
        threshold: float = 0.3
    ) -> dict:
        """
        Check missing data rate for a column.

        Args:
            df: DataFrame to check
            column: Column name to check
            threshold: Warning threshold for missing rate

        Returns:
            Validation results dictionary
        """
        if df.empty or column not in df.columns:
            return {"missing_rate": 0, "warnings": []}

        missing_count = df[column].isna().sum()
        missing_rate = missing_count / len(df)
        
        validation = {
            "missing_rate": missing_rate,
            "missing_count": missing_count,
            "warnings": []
        }

        if missing_rate > threshold:
            validation["warnings"].append(
                f"High missing data rate in {column}: {missing_rate:.1%}"
            )

        return validation


class FileUtils:
    """File handling utilities for data collectors."""

    @staticmethod
    def ensure_directory(file_path: str | Path) -> Path:
        """
        Ensure directory exists for the given file path.

        Args:
            file_path: Path to file

        Returns:
            Path object for the directory
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent

    @staticmethod
    def save_dataframe(
        df: pd.DataFrame, 
        file_path: str | Path, 
        logger: logging.Logger | None = None
    ) -> bool:
        """
        Save DataFrame to CSV with directory creation.

        Args:
            df: DataFrame to save
            file_path: Output file path
            logger: Optional logger instance

        Returns:
            True if successful, False otherwise
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            FileUtils.ensure_directory(file_path)
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}: {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
            return False

    @staticmethod
    def get_default_output_path(
        filename: str, 
        env_var: str = "RAW_DATA_DIR", 
        default_dir: str = "data/raw"
    ) -> str:
        """
        Get default output path from environment or use default.

        Args:
            filename: Name of output file
            env_var: Environment variable for directory
            default_dir: Default directory if env var not set

        Returns:
            Full file path
        """
        data_dir = os.getenv(env_var, default_dir)
        return os.path.join(data_dir, filename)


class SafeTypeConverter:
    """Safe type conversion utilities for API data."""

    @staticmethod
    def safe_float(value: Any, invalid_values: list | None = None) -> float | None:
        """
        Safely convert value to float, handling API special codes.

        Args:
            value: Value to convert
            invalid_values: List of values to treat as None

        Returns:
            Float value or None if invalid/missing
        """
        if invalid_values is None:
            invalid_values = [None, "", "null", "‡", "*", "N/A", "#", "N", "X", "S", "D"]

        if value in invalid_values:
            return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_int(value: Any, invalid_values: list | None = None) -> int | None:
        """
        Safely convert value to int, handling API special codes.

        Args:
            value: Value to convert
            invalid_values: List of values to treat as None

        Returns:
            Integer value or None if invalid/missing
        """
        if invalid_values is None:
            invalid_values = [None, "", "null", "N", "X", "S", "D"]

        if value in invalid_values:
            return None

        try:
            return int(value)
        except (ValueError, TypeError):
            return None