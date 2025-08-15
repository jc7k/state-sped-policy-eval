#!/usr/bin/env python
"""
Panel Data Creator

Handles aggregation and merging of multiple datasets into state-year panel format.
Creates derived variables and ensures consistent panel structure for econometric analysis.
"""

import numpy as np
import pandas as pd

from ..collection.common import StateUtils
from .utils import ErrorHandlingMixin


class PanelDataCreator(ErrorHandlingMixin):
    """
    Creates state-year panel dataset from multiple data sources.

    Handles aggregation of multiple observations per state-year into single rows,
    merges datasets consistently, and creates derived variables for analysis.
    """

    def __init__(self, year_range: tuple[int, int] = (2009, 2023)):
        """
        Initialize panel data creator.

        Args:
            year_range: Tuple of (min_year, max_year) for panel
        """
        super().__init__()
        self.min_year, self.max_year = year_range
        self.valid_states = StateUtils.get_all_states()

    def create_base_panel(self) -> pd.DataFrame:
        """
        Create base state-year panel structure.

        Returns:
            DataFrame with all state-year combinations
        """
        years = list(range(self.min_year, self.max_year + 1))
        states = self.valid_states

        panel_data = []
        for state in states:
            for year in years:
                panel_data.append({"state": state, "year": year})

        base_panel = pd.DataFrame(panel_data)

        self.log_info(
            f"Created base panel: {len(base_panel)} state-year observations "
            f"({len(states)} states × {len(years)} years)"
        )

        return base_panel

    def aggregate_naep_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate NAEP data to state-year level with properly named variables.

        NAEP has multiple observations per state-year (subject/grade combinations).
        We create separate variables for each subject-grade-type combination.

        Args:
            df: Raw NAEP data

        Returns:
            Aggregated NAEP data at state-year level with descriptive variable names
        """
        if df.empty:
            self.log_warning("NAEP data is empty")
            return pd.DataFrame()

        # Validate required columns
        required_cols = ["state", "year", "subject", "grade"]
        if not all(col in df.columns for col in required_cols):
            self.log_error(f"NAEP data missing required columns: {required_cols}")
            return pd.DataFrame()

        # Check for score columns
        score_cols = ['swd_mean_score', 'non_swd_mean_score', 'achievement_gap']
        available_score_cols = [col for col in score_cols if col in df.columns]
        
        if not available_score_cols:
            self.log_warning("No score columns found in NAEP data")
            return df[required_cols].drop_duplicates()

        # Create wide format with descriptive variable names
        # Transform from long format (subject/grade rows) to wide format (subject_grade columns)
        result_data = []
        
        # Get all state-year combinations
        state_years = df[['state', 'year']].drop_duplicates()
        
        for _, row in state_years.iterrows():
            state, year = row['state'], row['year']
            record = {'state': state, 'year': year}
            
            # Get data for this state-year
            state_year_data = df[(df['state'] == state) & (df['year'] == year)]
            
            # Create variables for each subject-grade combination
            for _, naep_row in state_year_data.iterrows():
                subject = naep_row['subject']
                grade = naep_row['grade']
                
                # Create variable name prefixes
                subject_prefix = subject.lower()
                grade_suffix = f"grade{grade}"
                
                # Add SWD score
                if 'swd_mean_score' in naep_row and pd.notna(naep_row['swd_mean_score']):
                    var_name = f"{subject_prefix}_{grade_suffix}_swd_score"
                    record[var_name] = naep_row['swd_mean_score']
                
                # Add non-SWD score
                if 'non_swd_mean_score' in naep_row and pd.notna(naep_row['non_swd_mean_score']):
                    var_name = f"{subject_prefix}_{grade_suffix}_non_swd_score"
                    record[var_name] = naep_row['non_swd_mean_score']
                
                # Add achievement gap
                if 'achievement_gap' in naep_row and pd.notna(naep_row['achievement_gap']):
                    var_name = f"{subject_prefix}_{grade_suffix}_gap"
                    record[var_name] = naep_row['achievement_gap']
            
            result_data.append(record)
        
        naep_agg = pd.DataFrame(result_data)

        self.log_info(
            f"Aggregated NAEP data: {len(df)} records → {len(naep_agg)} state-year observations"
        )
        
        # Log the variables created
        naep_vars = [col for col in naep_agg.columns if col not in ['state', 'year']]
        self.log_info(f"NAEP variables created: {naep_vars}")

        return naep_agg

    def aggregate_census_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Census finance data.

        Census data should already be at state-year level, so minimal processing needed.

        Args:
            df: Raw Census data

        Returns:
            Processed Census data
        """
        if df.empty:
            self.log_warning("Census data is empty")
            return pd.DataFrame()

        # Validate required columns
        required_cols = ["state", "year"]
        if not all(col in df.columns for col in required_cols):
            self.log_error(f"Census data missing required columns: {required_cols}")
            return pd.DataFrame()

        # Census data should be one record per state-year
        # Check for duplicates and handle if necessary
        duplicates = df.duplicated(subset=required_cols).sum()
        if duplicates > 0:
            self.log_warning(f"Found {duplicates} duplicate state-year records in Census data")
            # Keep first occurrence of each state-year
            df = df.drop_duplicates(subset=required_cols, keep="first")

        self.log_info(f"Processed Census data: {len(df)} state-year observations")
        return df

    def aggregate_edfacts_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate EdFacts data to state-year level.

        EdFacts has multiple records per state-year (by disability, age, etc.).
        We sum counts to get total special education enrollment and services.

        Args:
            df: Raw EdFacts data

        Returns:
            Aggregated EdFacts data at state-year level
        """
        if df.empty:
            self.log_warning("EdFacts data is empty")
            return pd.DataFrame()

        # Validate required columns
        required_cols = ["state", "year"]
        if not all(col in df.columns for col in required_cols):
            self.log_error(f"EdFacts data missing required columns: {required_cols}")
            return pd.DataFrame()

        # Find numeric columns that represent counts
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["state", "year"]
        count_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not count_cols:
            self.log_warning("No count columns found in EdFacts data for aggregation")
            return df[required_cols].drop_duplicates()

        # Aggregate by sum (appropriate for count data)
        agg_dict = dict.fromkeys(count_cols, "sum")
        edfacts_agg = df.groupby(required_cols).agg(agg_dict).reset_index()

        # Add EdFacts prefix
        edfacts_cols = {
            col: f"edfacts_{col}" for col in edfacts_agg.columns if col not in required_cols
        }
        edfacts_agg = edfacts_agg.rename(columns=edfacts_cols)

        self.log_info(
            f"Aggregated EdFacts data: {len(df)} records → {len(edfacts_agg)} state-year observations"
        )

        return edfacts_agg

    def aggregate_ocr_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate OCR data to state-year level.

        OCR may have multiple records per state-year (by data type, school level, etc.).
        We sum counts and incidents to get state totals.

        Args:
            df: Raw OCR data

        Returns:
            Aggregated OCR data at state-year level
        """
        if df.empty:
            self.log_warning("OCR data is empty")
            return pd.DataFrame()

        # Validate required columns
        required_cols = ["state", "year"]
        if not all(col in df.columns for col in required_cols):
            self.log_error(f"OCR data missing required columns: {required_cols}")
            return pd.DataFrame()

        # Find numeric columns that represent counts/incidents
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["state", "year"]
        count_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not count_cols:
            self.log_warning("No count columns found in OCR data for aggregation")
            return df[required_cols].drop_duplicates()

        # Aggregate by sum (appropriate for enrollment and incident counts)
        agg_dict = dict.fromkeys(count_cols, "sum")
        ocr_agg = df.groupby(required_cols).agg(agg_dict).reset_index()

        # Add OCR prefix
        ocr_cols = {col: f"ocr_{col}" for col in ocr_agg.columns if col not in required_cols}
        ocr_agg = ocr_agg.rename(columns=ocr_cols)

        self.log_info(
            f"Aggregated OCR data: {len(df)} records → {len(ocr_agg)} state-year observations"
        )

        return ocr_agg

    def merge_datasets(
        self, base_panel: pd.DataFrame, datasets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge all datasets into master panel.

        Args:
            base_panel: Base state-year panel structure
            datasets: Dictionary of aggregated datasets

        Returns:
            Master panel dataset
        """
        master_df = base_panel.copy()

        for dataset_name, df in datasets.items():
            if df.empty:
                self.log_warning(f"Skipping empty dataset: {dataset_name}")
                continue

            self.log_info(f"Merging {dataset_name} data...")

            # Merge on state and year
            master_df = master_df.merge(df, on=["state", "year"], how="left")

            # Log merge results
            non_missing = df.shape[0]
            self.log_info(f"Merged {dataset_name}: {non_missing} state-year observations added")

        self.log_info(
            f"Master dataset created: {len(master_df)} observations, "
            f"{len(master_df.columns)} variables"
        )

        return master_df

    def create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived variables for econometric analysis.

        Args:
            df: Master panel dataset

        Returns:
            Dataset with derived variables added
        """
        df_derived = df.copy()

        self.log_info("Creating derived variables...")

        # NAEP-derived variables
        if "naep_pct_proficient" in df_derived.columns:
            # Proficiency rate (proficient + advanced)
            df_derived["naep_proficiency_rate"] = df_derived[
                "naep_pct_proficient"
            ] + df_derived.get("naep_pct_advanced", 0)

        # Finance-derived variables
        if (
            "total_expenditure" in df_derived.columns
            and "ocr_total_enrollment" in df_derived.columns
        ):
            # Per-pupil expenditure (if not already calculated)
            df_derived["per_pupil_total_expenditure"] = (
                df_derived["total_expenditure"] / df_derived["ocr_total_enrollment"]
            )

        # Special education rates
        if (
            "edfacts_child_count" in df_derived.columns
            and "ocr_total_enrollment" in df_derived.columns
        ):
            # SWD identification rate
            df_derived["swd_identification_rate"] = (
                df_derived["edfacts_child_count"] / df_derived["ocr_total_enrollment"]
            )

        # Discipline rates for SWD
        if "ocr_suspensions" in df_derived.columns and "ocr_swd_enrollment" in df_derived.columns:
            df_derived["swd_suspension_rate"] = (
                df_derived["ocr_suspensions"] / df_derived["ocr_swd_enrollment"]
            )

        if "ocr_expulsions" in df_derived.columns and "ocr_swd_enrollment" in df_derived.columns:
            df_derived["swd_expulsion_rate"] = (
                df_derived["ocr_expulsions"] / df_derived["ocr_swd_enrollment"]
            )

        # Time indicators for econometric analysis
        df_derived["post_2015"] = (df_derived["year"] >= 2015).astype(int)
        df_derived["post_covid"] = (df_derived["year"] >= 2020).astype(int)

        # Replace infinite values with NaN (can occur in rate calculations)
        df_derived = df_derived.replace([np.inf, -np.inf], np.nan)

        added_vars = len(df_derived.columns) - len(df.columns)
        self.log_info(f"Added {added_vars} derived variables")

        return df_derived

    def create_panel_dataset(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create complete state-year panel from individual datasets.

        Args:
            datasets: Dictionary of loaded datasets (keys: naep, census, edfacts, ocr)

        Returns:
            Complete master panel dataset
        """
        self.log_info("Creating master state-year panel...")

        # Create base panel structure
        base_panel = self.create_base_panel()

        # Aggregate each dataset to state-year level
        aggregated_datasets = {}

        if "naep" in datasets and not datasets["naep"].empty:
            aggregated_datasets["naep"] = self.aggregate_naep_data(datasets["naep"])

        if "census" in datasets and not datasets["census"].empty:
            aggregated_datasets["census"] = self.aggregate_census_data(datasets["census"])

        if "edfacts" in datasets and not datasets["edfacts"].empty:
            aggregated_datasets["edfacts"] = self.aggregate_edfacts_data(datasets["edfacts"])

        if "ocr" in datasets and not datasets["ocr"].empty:
            aggregated_datasets["ocr"] = self.aggregate_ocr_data(datasets["ocr"])

        # Merge all datasets
        master_df = self.merge_datasets(base_panel, aggregated_datasets)

        # Create derived variables
        master_df = self.create_derived_variables(master_df)

        return master_df

    def get_panel_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the panel dataset.

        Args:
            df: Panel dataset

        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {"error": "Dataset is empty"}

        summary = {
            "total_observations": len(df),
            "states": df["state"].nunique(),
            "years": {
                "min": df["year"].min(),
                "max": df["year"].max(),
                "count": df["year"].nunique(),
            },
            "variables": len(df.columns),
            "data_coverage": {},
        }

        # Check coverage for key variables
        key_vars = [
            "naep_avg_scale_score",
            "total_expenditure",
            "edfacts_child_count",
            "ocr_total_enrollment",
        ]

        for var in key_vars:
            if var in df.columns:
                non_missing = df[var].notna().sum()
                coverage_pct = (non_missing / len(df)) * 100
                summary["data_coverage"][var] = {
                    "observations": non_missing,
                    "coverage_percent": coverage_pct,
                }

        return summary
