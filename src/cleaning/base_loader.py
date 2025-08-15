#!/usr/bin/env python
"""
Base Data Loader Classes

Abstract base classes and interfaces for data loading operations.
Provides consistent structure and error handling across all data loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .utils import ErrorHandlingMixin


class BaseDataLoader(ABC, ErrorHandlingMixin):
    """
    Abstract base class for all data loaders.

    Provides common interface and functionality for loading and cleaning data
    from different sources (NAEP, Census, EdFacts, OCR).
    """

    def __init__(self, raw_data_dir: Path):
        """
        Initialize base data loader.

        Args:
            raw_data_dir: Directory containing raw data files
        """
        super().__init__()
        self.raw_data_dir = Path(raw_data_dir)
        self._validate_data_directory()

    def _validate_data_directory(self) -> None:
        """Validate that data directory exists."""
        if not self.raw_data_dir.exists():
            self.log_warning(f"Data directory does not exist: {self.raw_data_dir}")

    @abstractmethod
    def get_expected_files(self) -> list[str]:
        """
        Get list of expected file names/patterns for this data source.

        Returns:
            List of expected file names or patterns
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """
        Get list of required columns for this data source.

        Returns:
            List of required column names
        """
        pass

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from files.

        Returns:
            Raw DataFrame loaded from files

        Raises:
            CleaningError: If data cannot be loaded
        """
        pass

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize loaded data.

        Args:
            df: Raw DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        pass

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load and clean data in a single operation.

        Returns:
            Cleaned DataFrame ready for integration
        """
        self.log_info(f"Loading {self.__class__.__name__} data...")

        try:
            # Load raw data
            raw_df = self.load_raw_data()

            if raw_df.empty:
                self.log_warning(f"No data loaded for {self.__class__.__name__}")
                return pd.DataFrame()

            # Clean data
            cleaned_df = self.clean_data(raw_df)

            # Validate result
            self._validate_cleaned_data(cleaned_df)

            self.log_info(
                f"Successfully loaded {self.__class__.__name__} data: "
                f"{len(cleaned_df)} records, {len(cleaned_df.columns)} columns"
            )

            return cleaned_df

        except Exception as e:
            self.log_error(f"Failed to load {self.__class__.__name__} data", e)
            return pd.DataFrame()

    def _validate_cleaned_data(self, df: pd.DataFrame) -> None:
        """
        Validate cleaned data meets basic requirements.

        Args:
            df: Cleaned DataFrame to validate
        """
        if df.empty:
            self.log_warning("Cleaned data is empty")
            return

        # Check for required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.log_warning(f"Missing required columns: {missing_cols}")

        # Check state coverage if state column exists
        if "state" in df.columns:
            unique_states = df["state"].nunique()
            if unique_states < 10:  # Arbitrary threshold for concern
                self.log_warning(f"Low state coverage: only {unique_states} states")

        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            self.log_warning(f"Completely empty columns: {empty_cols}")

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for loaded data.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {"records": 0, "columns": 0, "states": 0, "years": []}

        summary = {
            "records": len(df),
            "columns": len(df.columns),
            "states": df["state"].nunique() if "state" in df.columns else 0,
            "years": sorted(df["year"].unique().tolist()) if "year" in df.columns else [],
        }

        return summary


class FileBasedLoader(BaseDataLoader):
    """
    Base class for loaders that read from CSV files.

    Provides common file handling functionality for CSV-based data sources.
    """

    def __init__(self, raw_data_dir: Path, file_patterns: list[str]):
        """
        Initialize file-based loader.

        Args:
            raw_data_dir: Directory containing data files
            file_patterns: List of file patterns to search for
        """
        super().__init__(raw_data_dir)
        self.file_patterns = file_patterns

    def find_data_files(self) -> list[Path]:
        """
        Find data files matching the expected patterns.

        Returns:
            List of found data files
        """
        from .utils import FilePatternMatcher

        found_files = FilePatternMatcher.find_files_by_patterns(
            self.raw_data_dir, self.file_patterns, recursive=True
        )

        if not found_files:
            self.log_warning(f"No files found matching patterns: {self.file_patterns}")
        else:
            self.log_info(f"Found {len(found_files)} data files")

        return found_files

    def load_csv_file(self, file_path: Path, skip_rows: int = 0, **kwargs) -> pd.DataFrame:
        """
        Load a single CSV file with error handling.

        Args:
            file_path: Path to CSV file
            skip_rows: Number of rows to skip at beginning
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame loaded from file
        """
        try:
            self.log_info(f"Loading file: {file_path.name}")

            # Set default parameters
            csv_kwargs = {
                "low_memory": False,
                "skiprows": skip_rows,
            }
            csv_kwargs.update(kwargs)

            df = pd.read_csv(file_path, **csv_kwargs)

            # Remove completely empty columns (common in metadata)
            df = df.dropna(axis=1, how="all")

            self.log_info(f"Loaded {len(df)} records from {file_path.name}")
            return df

        except Exception as e:
            self.log_error(f"Failed to load {file_path.name}", e)
            return pd.DataFrame()

    def combine_dataframes(
        self, dataframes: list[pd.DataFrame], add_source_info: bool = False
    ) -> pd.DataFrame:
        """
        Combine multiple DataFrames into one.

        Args:
            dataframes: List of DataFrames to combine
            add_source_info: Whether to add source file information

        Returns:
            Combined DataFrame
        """
        if not dataframes:
            return pd.DataFrame()

        # Filter out empty DataFrames
        non_empty_dfs = [df for df in dataframes if not df.empty]

        if not non_empty_dfs:
            return pd.DataFrame()

        try:
            combined = pd.concat(non_empty_dfs, ignore_index=True, sort=False)
            self.log_info(f"Combined {len(non_empty_dfs)} files into {len(combined)} records")
            return combined

        except Exception as e:
            self.log_error("Failed to combine DataFrames", e)
            return pd.DataFrame()


class DirectoryBasedLoader(BaseDataLoader):
    """
    Base class for loaders that read from directory structures.

    Handles complex directory structures with multiple file types.
    """

    def __init__(self, raw_data_dir: Path, subdirectory: str):
        """
        Initialize directory-based loader.

        Args:
            raw_data_dir: Base directory containing data
            subdirectory: Specific subdirectory for this data source
        """
        super().__init__(raw_data_dir)
        self.data_dir = self.raw_data_dir / subdirectory

    def get_subdirectories(self) -> list[Path]:
        """
        Get list of subdirectories in data directory.

        Returns:
            List of subdirectory paths
        """
        if not self.data_dir.exists():
            return []

        subdirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
        return sorted(subdirs)

    def find_files_in_subdirs(self, pattern: str) -> dict[str, list[Path]]:
        """
        Find files matching pattern in all subdirectories.

        Args:
            pattern: File pattern to search for

        Returns:
            Dictionary mapping subdirectory name to list of files
        """
        results = {}

        for subdir in self.get_subdirectories():
            files = list(subdir.glob(pattern))
            if files:
                results[subdir.name] = files

        return results
