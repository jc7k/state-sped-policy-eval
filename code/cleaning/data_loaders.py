#!/usr/bin/env python
"""
Specific Data Loader Classes

Concrete implementations for loading and cleaning data from different sources:
- NAEP (National Assessment of Educational Progress)
- Census Education Finance
- EdFacts IDEA data  
- OCR Civil Rights Data
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .base_loader import FileBasedLoader, DirectoryBasedLoader
from .utils import (
    DataCleaningUtils, 
    ColumnStandardizer, 
    FilePatternMatcher,
    CleaningError
)


class NAEPDataLoader(FileBasedLoader):
    """Loader for NAEP state assessment data."""
    
    def __init__(self, raw_data_dir: Path):
        file_patterns = ["naep_state_swd_data.csv", "naep_*_swd_*.csv"]
        super().__init__(raw_data_dir, file_patterns)
    
    def get_expected_files(self) -> List[str]:
        return ["naep_state_swd_data.csv"]
    
    def get_required_columns(self) -> List[str]:
        return ["state", "year", "subject", "grade"]
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load NAEP data from CSV file."""
        data_files = self.find_data_files()
        
        if not data_files:
            self.log_warning("No NAEP data files found")
            return pd.DataFrame()
        
        # Use the first matching file (should be the main NAEP file)
        main_file = data_files[0]
        return self.load_csv_file(main_file)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize NAEP data."""
        if df.empty:
            return df
        
        # Standardize state codes
        df_clean = DataCleaningUtils.standardize_state_codes(df, "state")
        
        # Convert year to integer
        df_clean = DataCleaningUtils.safe_numeric_conversion(df_clean, ["year"])
        
        # Define key NAEP variables
        naep_vars = [
            "state", "year", "subject", "grade",
            "avg_scale_score", "pct_below_basic", "pct_basic", 
            "pct_proficient", "pct_advanced"
        ]
        
        # Keep only available key variables
        available_vars = [var for var in naep_vars if var in df_clean.columns]
        df_clean = df_clean[available_vars].copy()
        
        # Remove rows with missing key data
        df_clean = df_clean.dropna(subset=["state", "year"])
        
        return df_clean


class CensusDataLoader(FileBasedLoader):
    """Loader for Census education finance data."""
    
    def __init__(self, raw_data_dir: Path):
        file_patterns = [
            "census_education_finance_parsed.csv",
            "census_f33_*.csv", 
            "census_*.csv"
        ]
        super().__init__(raw_data_dir, file_patterns)
    
    def get_expected_files(self) -> List[str]:
        return ["census_education_finance_parsed.csv"]
    
    def get_required_columns(self) -> List[str]:
        return ["state", "year"]
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load Census finance data from CSV file."""
        data_files = self.find_data_files()
        
        if not data_files:
            self.log_warning("No Census data files found")
            return pd.DataFrame()
        
        # Use the first matching file (should be the parsed finance file)
        main_file = data_files[0]
        return self.load_csv_file(main_file)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize Census finance data."""
        if df.empty:
            return df
        
        # Standardize state codes
        df_clean = DataCleaningUtils.standardize_state_codes(df, "state")
        
        # Convert year and financial columns to numeric
        df_clean = DataCleaningUtils.safe_numeric_conversion(df_clean, ["year"])
        
        # Identify financial columns for conversion
        financial_vars = [
            "total_revenue", "total_expenditure", "current_expenditure",
            "instruction_expenditure", "per_pupil_expenditure", 
            "special_education_expenditure"
        ]
        
        existing_financial_vars = [
            var for var in financial_vars if var in df_clean.columns
        ]
        
        if existing_financial_vars:
            df_clean = DataCleaningUtils.safe_numeric_conversion(
                df_clean, existing_financial_vars
            )
        
        # Remove rows with missing key data
        df_clean = df_clean.dropna(subset=["state", "year"])
        
        return df_clean


class EdFactsDataLoader(DirectoryBasedLoader):
    """Loader for EdFacts IDEA data."""
    
    def __init__(self, raw_data_dir: Path):
        super().__init__(raw_data_dir, "edfacts")
        self.target_years = [2019, 2020, 2021, 2022, 2023]
    
    def get_expected_files(self) -> List[str]:
        return [
            "bchildcountandedenvironment2019.csv",
            "bchildcountandedenvironment2020.csv", 
            "bchildcountandedenvironment2021.csv",
            "bchildcountandedenvironment2022.csv",
            "bchildcountandedenvironment2023.csv"
        ]
    
    def get_required_columns(self) -> List[str]:
        return ["state", "year"]
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load EdFacts data from multiple year files."""
        all_dataframes = []
        
        for year in self.target_years:
            year_df = self._load_year_data(year)
            if not year_df.empty:
                all_dataframes.append(year_df)
        
        return self.combine_dataframes(all_dataframes)
    
    def _load_year_data(self, year: int) -> pd.DataFrame:
        """Load data for a specific year."""
        # Try different filename patterns
        patterns = [
            f"bchildcountandedenvironment{year}.csv",
            f"bchildcountandedenvironment{year}-{str(year + 1)[2:]}.csv",
            f"bchildcountandedenvironment{year}-24.csv",  # For 2023-24
        ]
        
        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                df = self.load_csv_file(file_path, skip_rows=4)  # Skip metadata
                if not df.empty:
                    # Add year if not present
                    if "year" not in df.columns and "YEAR" not in df.columns:
                        df["year"] = year
                    return df
        
        self.log_warning(f"No EdFacts file found for year {year}")
        return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize EdFacts data."""
        if df.empty:
            return df
        
        # Standardize column names first
        edfacts_mappings = {
            "SEA_STATE": "state",
            "DISABILITY": "disability_category",
            "SEA Disability Category": "disability_category", 
            "AGE": "age_group",
            "RACE_ETHNICITY": "race_ethnicity",
            "SEX": "sex",
            "LEP": "english_learner",
            "IDEA_INDICATOR": "idea_indicator",
            "EDUCATIONAL_ENVIRONMENT": "educational_environment",
            "SEA Education Environment": "educational_environment",
            "CHILD_COUNT": "child_count",
        }
        
        df_clean = ColumnStandardizer.standardize_columns(df, edfacts_mappings)
        
        # Standardize state codes (map names to codes if needed)
        df_clean = DataCleaningUtils.standardize_state_codes(
            df_clean, "state", "state_name"
        )
        
        # Convert numeric columns
        numeric_cols = ["year", "child_count"]
        df_clean = DataCleaningUtils.safe_numeric_conversion(df_clean, numeric_cols)
        
        # Remove rows with missing key data
        df_clean = df_clean.dropna(subset=["state", "year"])
        
        return df_clean


class OCRDataLoader(DirectoryBasedLoader):
    """Loader for OCR Civil Rights data."""
    
    def __init__(self, raw_data_dir: Path):
        super().__init__(raw_data_dir, "ocr")
        self.csv_years = [2009, 2011, 2013, 2015]
        self.zip_years = [2017, 2020]
    
    def get_expected_files(self) -> List[str]:
        return [
            "CRDC-2009-State-Discipline.csv",
            "CRDC-2011-State-Discipline.csv",
            "CRDC-2013-State-Data.csv", 
            "CRDC-2015-State-Data.csv",
            "extracted_2017/",
            "extracted_2020/"
        ]
    
    def get_required_columns(self) -> List[str]:
        return ["state", "year"]
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load OCR data from CSV files and extracted ZIP directories."""
        all_dataframes = []
        
        # Load direct CSV files
        csv_data = self._load_csv_files()
        if not csv_data.empty:
            all_dataframes.append(csv_data)
        
        # Load extracted ZIP data
        for year in self.zip_years:
            zip_data = self._load_zip_data(year)
            if not zip_data.empty:
                all_dataframes.append(zip_data)
        
        return self.combine_dataframes(all_dataframes)
    
    def _load_csv_files(self) -> pd.DataFrame:
        """Load direct CSV files for earlier years."""
        csv_files = [
            ("CRDC-2009-State-Discipline.csv", 2009),
            ("CRDC-2011-State-Discipline.csv", 2011), 
            ("CRDC-2013-State-Data.csv", 2013),
            ("CRDC-2015-State-Data.csv", 2015),
        ]
        
        dataframes = []
        for filename, year in csv_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                df = self.load_csv_file(file_path)
                if not df.empty:
                    df["year"] = year
                    dataframes.append(df)
        
        return self.combine_dataframes(dataframes)
    
    def _load_zip_data(self, year: int) -> pd.DataFrame:
        """Load data from extracted ZIP files."""
        extract_dir = self.data_dir / f"extracted_{year}"
        if not extract_dir.exists():
            self.log_warning(f"Extract directory not found: {extract_dir}")
            return pd.DataFrame()
        
        # Look for key discipline files
        discipline_files = [
            "Suspensions.csv",
            "Expulsions.csv", 
            "Restraint and Seclusion.csv",
        ]
        
        dataframes = []
        for discipline_file in discipline_files:
            # Search recursively for the file
            found_files = list(extract_dir.rglob(discipline_file))
            if found_files:
                file_path = found_files[0]
                df = self.load_csv_file(file_path)
                if not df.empty:
                    df["year"] = year
                    df["data_type"] = discipline_file.replace(".csv", "").lower()
                    dataframes.append(df)
        
        return self.combine_dataframes(dataframes)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize OCR data."""
        if df.empty:
            return df
        
        # Standardize column names
        ocr_mappings = {
            "StateAbbr": "state",
            "State Abbr": "state", 
            "LEA State": "state",
            "LEA_STATE": "state",
            "School Year": "school_year",
            "SchoolYear": "school_year",
            "Out-of-School Suspensions": "suspensions",
            "Expulsions": "expulsions",
            "Restraint": "restraint_incidents",
            "Seclusion": "seclusion_incidents",
        }
        
        df_clean = ColumnStandardizer.standardize_columns(df, ocr_mappings)
        
        # Standardize state codes  
        df_clean = DataCleaningUtils.standardize_state_codes(
            df_clean, "state", "state_name"
        )
        
        # Convert numeric columns
        numeric_cols = [
            "year", "total_enrollment", "swd_enrollment",
            "suspensions", "expulsions", "restraint_incidents", "seclusion_incidents"
        ]
        existing_numeric = [col for col in numeric_cols if col in df_clean.columns]
        df_clean = DataCleaningUtils.safe_numeric_conversion(df_clean, existing_numeric)
        
        # Remove rows with missing key data
        df_clean = df_clean.dropna(subset=["state", "year"])
        
        return df_clean


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    LOADER_CLASSES = {
        "naep": NAEPDataLoader,
        "census": CensusDataLoader, 
        "edfacts": EdFactsDataLoader,
        "ocr": OCRDataLoader,
    }
    
    @classmethod
    def create_loader(cls, data_source: str, raw_data_dir: Path) -> FileBasedLoader:
        """
        Create a data loader for the specified data source.
        
        Args:
            data_source: Name of data source (naep, census, edfacts, ocr)
            raw_data_dir: Directory containing raw data files
            
        Returns:
            Appropriate data loader instance
            
        Raises:
            ValueError: If data source is not supported
        """
        if data_source not in cls.LOADER_CLASSES:
            raise ValueError(
                f"Unsupported data source: {data_source}. "
                f"Available sources: {list(cls.LOADER_CLASSES.keys())}"
            )
        
        loader_class = cls.LOADER_CLASSES[data_source]
        return loader_class(raw_data_dir)
    
    @classmethod
    def get_available_sources(cls) -> List[str]:
        """Get list of available data sources."""
        return list(cls.LOADER_CLASSES.keys())
    
    @classmethod
    def create_all_loaders(cls, raw_data_dir: Path) -> Dict[str, FileBasedLoader]:
        """
        Create all available data loaders.
        
        Args:
            raw_data_dir: Directory containing raw data files
            
        Returns:
            Dictionary mapping source names to loader instances
        """
        loaders = {}
        for source in cls.get_available_sources():
            loaders[source] = cls.create_loader(source, raw_data_dir)
        return loaders