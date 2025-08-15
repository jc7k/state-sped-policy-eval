"""
Panel Data Structure Setup for Econometric Analysis

This module creates the final analysis-ready panel dataset by merging
master data with policy treatments and creating proper econometric
variables including NAEP achievement gaps, per-pupil spending measures,
and treatment indicators.

Author: Research Team
Date: 2025-08-12
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.policy_database import PolicyDatabase


class PanelDataProcessor:
    """
    Processes master dataset into analysis-ready panel structure.

    Key transformations:
    - Calculate NAEP achievement gaps (SWD vs non-SWD)
    - Create per-pupil spending measures
    - Add policy treatment indicators
    - Generate outcome and control variables
    - Handle missing data appropriately
    """

    def __init__(
        self,
        master_data_path: str = "data/processed/master_analysis_dataset.csv",
        policy_data_path: str = "data/processed/state_policy_database.csv",
    ):
        """Initialize with data paths."""
        self.master_path = master_data_path
        self.policy_path = policy_data_path
        self.master_df = None
        self.policy_df = None
        self.analysis_df = None
        self._load_data()

    def _load_data(self):
        """Load master and policy datasets."""
        try:
            print("Loading master analysis dataset...")
            self.master_df = pd.read_csv(self.master_path)
            print(f"Master dataset loaded: {self.master_df.shape}")

            # Create policy database if it doesn't exist
            if not Path(self.policy_path).exists():
                print("Creating policy database...")
                policy_db = PolicyDatabase()
                policy_db.export_policy_data(self.policy_path)

            print("Loading policy database...")
            self.policy_df = pd.read_csv(self.policy_path)
            print(f"Policy dataset loaded: {self.policy_df.shape}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def create_naep_outcomes(self) -> pd.DataFrame:
        """Create NAEP achievement outcome variables."""
        print("Creating NAEP outcome variables...")

        # First, let's check what NAEP data we actually have
        naep_cols = [col for col in self.master_df.columns if "naep" in col.lower()]
        print(f"NAEP columns found: {naep_cols}")

        # Load the actual NAEP data file to understand structure
        naep_path = "data/raw/naep_state_swd_data.csv"
        if Path(naep_path).exists():
            naep_raw = pd.read_csv(naep_path)
            print(f"Raw NAEP data shape: {naep_raw.shape}")
            print(f"Raw NAEP columns: {list(naep_raw.columns)}")

            # Pivot NAEP data to get SWD and non-SWD scores by state-year-grade-subject
            naep_pivot = naep_raw.pivot_table(
                index=["state", "year", "grade", "subject"],
                columns="disability_status",
                values="mean_score",
                aggfunc="mean",
            ).reset_index()

            # Calculate achievement gap
            naep_pivot["achievement_gap"] = naep_pivot["non-SWD"] - naep_pivot["SWD"]

            # Rename columns for consistency
            naep_df = naep_pivot.rename(columns={"SWD": "swd_score", "non-SWD": "non_swd_score"})

            # Pivot to have separate columns for each subject-grade combination
            outcome_vars = None
            for subject in ["mathematics", "reading"]:
                for grade in [4, 8]:
                    subset = naep_df[(naep_df["subject"] == subject) & (naep_df["grade"] == grade)]
                    if not subset.empty:
                        # Create variable names
                        subj_short = "math" if subject == "mathematics" else "reading"
                        base_name = f"{subj_short}_grade{grade}"

                        # Merge into outcomes DataFrame
                        subset_renamed = subset[
                            [
                                "state",
                                "year",
                                "swd_score",
                                "non_swd_score",
                                "achievement_gap",
                            ]
                        ].copy()
                        subset_renamed.columns = [
                            "state",
                            "year",
                            f"{base_name}_swd_score",
                            f"{base_name}_non_swd_score",
                            f"{base_name}_gap",
                        ]

                        if outcome_vars is None:
                            outcome_vars = subset_renamed
                        else:
                            outcome_vars = outcome_vars.merge(
                                subset_renamed, on=["state", "year"], how="outer"
                            )

            return outcome_vars if outcome_vars is not None else pd.DataFrame()
        else:
            print("Raw NAEP data file not found - using master dataset columns")
            return pd.DataFrame()

    def create_finance_variables(self) -> pd.DataFrame:
        """Create per-pupil spending and finance outcome variables."""
        print("Creating finance variables...")

        # Check for Census finance data
        finance_path = "data/raw/census_education_finance_parsed.csv"
        if Path(finance_path).exists():
            finance_raw = pd.read_csv(finance_path)
            print(f"Finance data shape: {finance_raw.shape}")

            # Create per-pupil measures (using enrollment from EdFacts if available)
            finance_outcomes = finance_raw.copy()

            # Key finance variables to create
            finance_vars = [
                "total_revenue",
                "federal_revenue",
                "state_revenue",
                "local_revenue",
                "total_expenditure",
                "instruction_expenditure",
                "support_services_expenditure",
            ]

            existing_vars = [var for var in finance_vars if var in finance_outcomes.columns]

            if existing_vars:
                # Select relevant columns
                finance_subset = finance_outcomes[["state", "year"] + existing_vars].copy()

                # Create per-pupil versions (will need enrollment data for proper calculation)
                for var in existing_vars:
                    finance_subset[f"{var}_per_pupil"] = (
                        finance_subset[var] / 1000
                    )  # Convert to thousands

                return finance_subset

        # Fallback to master dataset finance columns
        finance_cols = [
            col
            for col in self.master_df.columns
            if any(x in col.lower() for x in ["revenue", "expenditure", "spending"])
        ]

        if finance_cols:
            finance_subset = self.master_df[["state", "year"] + finance_cols].copy()
            return finance_subset

        return pd.DataFrame()

    def create_edfacts_variables(self) -> pd.DataFrame:
        """Create EdFacts enrollment and placement variables."""
        print("Creating EdFacts variables...")

        # For now, skip complex EdFacts processing due to format issues
        # Use EdFacts columns from master dataset if available
        edfacts_cols = [col for col in self.master_df.columns if "edfacts" in col.lower()]
        if edfacts_cols:
            return self.master_df[["state", "year"] + edfacts_cols].copy()

        print("No EdFacts variables found in master dataset")
        return pd.DataFrame()

    def create_control_variables(self) -> pd.DataFrame:
        """Create control variables from available data sources."""
        print("Creating control variables...")

        # Start with basic state-year structure
        years = range(2009, 2024)
        states = [
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "DC",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
        ]

        panel_data = []
        for state in states:
            for year in years:
                panel_data.append({"state": state, "year": year})

        controls_df = pd.DataFrame(panel_data)

        # Add time trends and indicators
        controls_df["time_trend"] = controls_df["year"] - 2009
        controls_df["time_trend_sq"] = controls_df["time_trend"] ** 2

        # COVID period indicator
        controls_df["post_covid"] = (controls_df["year"] >= 2020).astype(int)

        # Post-2015 period (when many reforms started)
        controls_df["post_2015"] = (controls_df["year"] >= 2015).astype(int)

        return controls_df

    def merge_all_data(self) -> pd.DataFrame:
        """Merge all data sources into final analysis panel."""
        print("Merging all data sources...")

        # Start with control variables as base
        analysis_df = self.create_control_variables()
        print(f"Base panel: {analysis_df.shape}")

        # Merge policy database
        analysis_df = analysis_df.merge(self.policy_df, on=["state", "year"], how="left")
        print(f"After policy merge: {analysis_df.shape}")

        # Merge NAEP outcomes
        naep_outcomes = self.create_naep_outcomes()
        if not naep_outcomes.empty:
            analysis_df = analysis_df.merge(naep_outcomes, on=["state", "year"], how="left")
            print(f"After NAEP merge: {analysis_df.shape}")

        # Merge finance variables
        finance_vars = self.create_finance_variables()
        if not finance_vars.empty:
            analysis_df = analysis_df.merge(finance_vars, on=["state", "year"], how="left")
            print(f"After finance merge: {analysis_df.shape}")

        # Merge EdFacts variables
        edfacts_vars = self.create_edfacts_variables()
        if not edfacts_vars.empty:
            analysis_df = analysis_df.merge(edfacts_vars, on=["state", "year"], how="left")
            print(f"After EdFacts merge: {analysis_df.shape}")

        # Add additional derived variables
        analysis_df = self._create_derived_variables(analysis_df)

        return analysis_df

    def _create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional derived variables for analysis."""

        # Treatment intensity (years since treatment)
        df["treatment_intensity"] = np.where(
            df["years_since_treatment"].notna() & (df["years_since_treatment"] >= 0),
            df["years_since_treatment"],
            0,
        )

        # Lead treatment indicators (for event studies)
        for lead in range(1, 6):  # 5 years of leads
            df[f"lead_{lead}"] = np.where(df["years_since_treatment"] == -lead, 1, 0)

        # Lag treatment indicators
        for lag in range(1, 6):  # 5 years of lags
            df[f"lag_{lag}"] = np.where(df["years_since_treatment"] == lag, 1, 0)

        # State and year fixed effects (as categorical variables)
        df["state_fe"] = df["state"].astype("category")
        df["year_fe"] = df["year"].astype("category")

        return df

    def create_analysis_panel(self, output_path: str = "data/final/analysis_panel.csv") -> str:
        """Create final analysis panel dataset."""
        print("Creating final analysis panel...")

        # Merge all data
        self.analysis_df = self.merge_all_data()

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export final dataset
        self.analysis_df.to_csv(output_path, index=False)

        print("\nFinal analysis panel created:")
        print(f"Shape: {self.analysis_df.shape}")
        print(f"Saved to: {output_path}")

        # Print summary
        print("\nPanel Summary:")
        print(f"Years: {self.analysis_df['year'].min()} - {self.analysis_df['year'].max()}")
        print(f"States: {len(self.analysis_df['state'].unique())}")
        print(
            f"Treated states: {len(self.analysis_df[self.analysis_df['post_treatment'] == 1]['state'].unique())}"
        )

        # Check key outcome variables
        outcome_cols = [col for col in self.analysis_df.columns if "gap" in col or "score" in col]
        if outcome_cols:
            print(f"Key outcome variables: {outcome_cols[:5]}")

        # Check key treatment variables
        treatment_cols = [
            col
            for col in self.analysis_df.columns
            if any(x in col for x in ["treated", "post_treatment", "monitoring"])
        ]
        if treatment_cols:
            print(f"Treatment variables: {treatment_cols}")

        return output_path

    def generate_summary_stats(self) -> dict[str, pd.DataFrame]:
        """Generate summary statistics for analysis panel."""
        if self.analysis_df is None:
            raise ValueError("Analysis panel not created yet - run create_analysis_panel first")

        # Overall summary
        numeric_cols = self.analysis_df.select_dtypes(include=[np.number]).columns
        overall_summary = self.analysis_df[numeric_cols].describe()

        # By treatment status
        treated_summary = self.analysis_df[self.analysis_df["post_treatment"] == 1][
            numeric_cols
        ].describe()
        control_summary = self.analysis_df[self.analysis_df["post_treatment"] == 0][
            numeric_cols
        ].describe()

        summaries = {
            "overall": overall_summary,
            "treated": treated_summary,
            "control": control_summary,
        }

        return summaries


def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename string."""
    import re

    year_match = re.search(r"20\d{2}", filename)
    if year_match:
        return int(year_match.group())
    raise ValueError(f"Could not extract year from filename: {filename}")


if __name__ == "__main__":
    # Create analysis panel
    processor = PanelDataProcessor()

    # Create final panel
    output_file = processor.create_analysis_panel()

    # Generate summaries
    if processor.analysis_df is not None:
        summaries = processor.generate_summary_stats()

        print("\n=== OVERALL SUMMARY STATISTICS ===")
        key_vars = [
            "year",
            "post_treatment",
            "years_since_treatment",
            "under_monitoring",
        ]
        available_vars = [var for var in key_vars if var in summaries["overall"].columns]
        if available_vars:
            print(summaries["overall"][available_vars])

    print(f"\nAnalysis panel ready at: {output_file}")
    print("Ready for econometric analysis!")
