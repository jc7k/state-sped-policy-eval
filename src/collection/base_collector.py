#!/usr/bin/env python
"""
Base Data Collector Class

Abstract base class that provides common infrastructure for all data collectors.
Includes logging, validation, file handling, and standardized interfaces.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .common import APIClient, DataValidator, FileUtils, StateUtils

# Load environment variables
load_dotenv()


class BaseDataCollector(ABC):
    """
    Abstract base class for all data collectors.

    Provides common infrastructure including:
    - Logging setup
    - Rate limiting
    - Data validation
    - File handling
    - Environment variable management
    """

    def __init__(self, rate_limit_delay: float | None = None, logger_name: str | None = None):
        """
        Initialize base data collector.

        Args:
            rate_limit_delay: Seconds between requests (if None, uses default)
            logger_name: Custom logger name (if None, uses class name)
        """
        # Set up logging
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)

        # Initialize API client with rate limiting
        self.rate_limit_delay = rate_limit_delay or self._get_default_rate_limit()
        self.api_client = APIClient(api_name=self.__class__.__name__.lower())

        # Initialize utilities
        self.state_utils = StateUtils()
        self.validator = DataValidator(self.state_utils)
        self.file_utils = FileUtils()

        # Storage for collected data
        self.results = []

    def _get_default_rate_limit(self) -> float:
        """Get default rate limit delay. Override in subclasses if needed."""
        return 1.0

    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch data from the source. Must be implemented by subclasses.

        Returns:
            DataFrame with collected data
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate collected data. Can be overridden for source-specific validation.

        Args:
            df: Collected DataFrame

        Returns:
            Validation results dictionary
        """
        validation = {"total_records": len(df), "passed": True, "errors": [], "warnings": []}

        if df.empty:
            validation["errors"].append("Dataset is empty")
            validation["passed"] = False
            return validation

        # Standard validations
        if "state" in df.columns:
            state_validation = self.validator.validate_state_coverage(df)
            validation.update(
                {
                    "states_covered": state_validation["states_covered"],
                    "errors": validation["errors"] + state_validation["errors"],
                    "warnings": validation["warnings"] + state_validation["warnings"],
                }
            )
            if not state_validation["passed"]:
                validation["passed"] = False

        return validation

    def save_data(self, df: pd.DataFrame, output_path: str | None = None, **kwargs) -> bool:
        """
        Save collected data to file.

        Args:
            df: DataFrame to save
            output_path: Output file path (if None, uses default)
            **kwargs: Additional arguments for file saving

        Returns:
            True if successful, False otherwise
        """
        if output_path is None:
            output_path = self._get_default_output_path()

        return self.file_utils.save_dataframe(df, output_path, self.logger)

    def _get_default_output_path(self) -> str:
        """
        Get default output path. Override in subclasses.

        Returns:
            Default file path for saving data
        """
        class_name = self.__class__.__name__.lower()
        filename = f"{class_name}_raw.csv"
        return self.file_utils.get_default_output_path(filename)

    def run_collection(self, save_data: bool = True, **kwargs) -> pd.DataFrame:
        """
        Run complete data collection workflow.

        Args:
            save_data: Whether to save data to file
            **kwargs: Arguments passed to fetch_data()

        Returns:
            Collected and validated DataFrame
        """
        self.logger.info(f"Starting {self.__class__.__name__} data collection")

        try:
            # Fetch data
            df = self.fetch_data(**kwargs)

            # Validate data
            validation = self.validate_data(df)

            # Log validation results
            self._log_validation_results(validation)

            # Save data if requested
            if save_data and not df.empty:
                self.save_data(df)

            self.logger.info(f"{self.__class__.__name__} collection completed: {len(df)} records")

            return df

        except Exception as e:
            self.logger.error(f"{self.__class__.__name__} collection failed: {str(e)}")
            raise

    def _log_validation_results(self, validation: dict) -> None:
        """Log validation results."""
        self.logger.info(f"Validation: {validation['total_records']} records")

        if "states_covered" in validation:
            self.logger.info(f"States covered: {validation['states_covered']}")

        for warning in validation.get("warnings", []):
            self.logger.warning(warning)

        for error in validation.get("errors", []):
            self.logger.error(error)

        if validation["passed"]:
            self.logger.info("Data validation passed")
        else:
            self.logger.error("Data validation failed")

    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []

    def get_collection_stats(self) -> dict:
        """
        Get statistics about current collection.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "total_results": len(self.results),
            "rate_limit_delay": self.rate_limit_delay,
            "api_requests_made": getattr(self.api_client, "_request_count", 0),
        }


class APIBasedCollector(BaseDataCollector):
    """
    Base class for API-based data collectors.

    Extends BaseDataCollector with API-specific functionality.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
        rate_limit_delay: float | None = None,
        **kwargs,
    ):
        """
        Initialize API-based collector.

        Args:
            base_url: Base URL for API
            api_key: API key (if None, loads from environment)
            api_key_env_var: Environment variable name for API key
            rate_limit_delay: Seconds between requests
            **kwargs: Additional arguments for BaseDataCollector
        """
        super().__init__(rate_limit_delay=rate_limit_delay, **kwargs)

        self.base_url = base_url
        self.api_key = self._load_api_key(api_key, api_key_env_var)

    def _load_api_key(self, api_key: str | None, env_var: str | None) -> str | None:
        """
        Load API key from parameter or environment.

        Args:
            api_key: Direct API key
            env_var: Environment variable name

        Returns:
            API key or None if not found
        """
        if api_key:
            return api_key

        if env_var:
            return os.getenv(env_var)

        return None

    def make_request(self, endpoint: str, params: dict | None = None, **kwargs) -> dict:
        """
        Make API request with automatic key injection.

        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            JSON response data

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Add API key to parameters if available
        if self.api_key and params is not None:
            params["key"] = self.api_key
        elif self.api_key:
            params = {"key": self.api_key}

        response = self.api_client.get(url, params=params, **kwargs)
        return response.json()


class FileBasedCollector(BaseDataCollector):
    """
    Base class for file-based data collectors.

    Extends BaseDataCollector for downloading and parsing files.
    """

    def __init__(self, **kwargs):
        """Initialize file-based collector."""
        super().__init__(**kwargs)

    def download_file(self, url: str, local_path: str | Path, **kwargs) -> bool:
        """
        Download file from URL.

        Args:
            url: URL to download from
            local_path: Local path to save file
            **kwargs: Additional request parameters

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.file_utils.ensure_directory(local_path)

            # Download file
            response = self.api_client.get(url, **kwargs)

            with open(local_path, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Downloaded {url} to {local_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {str(e)}")
            return False
