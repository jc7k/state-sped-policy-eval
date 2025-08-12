#!/usr/bin/env python
"""
Data Integration Module
Combines NAEP, Census, EdFacts, and OCR data into master analysis dataset
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataIntegrator:
    """
    Integrate multiple data sources into analysis-ready dataset
    """

    def __init__(self, raw_data_dir: Path, output_dir: Path):
        """
        Initialize data integrator

        Args:
            raw_data_dir: Directory containing raw data files
            output_dir: Directory to save processed datasets
        """
        self.logger = logging.getLogger(__name__)
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State mapping for consistency
        self.state_mapping = {
            "state_name": {
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
        }

        # Valid state codes (including DC)
        self.valid_states = list(self.state_mapping["state_name"].values())

    def load_naep_data(self) -> pd.DataFrame:
        """Load and clean NAEP data"""
        self.logger.info("Loading NAEP data...")

        naep_file = self.raw_data_dir / "naep_state_swd_data.csv"
        if not naep_file.exists():
            self.logger.warning(f"NAEP file not found: {naep_file}")
            return pd.DataFrame()

        df = pd.read_csv(naep_file)

        # Standardize column names and state codes
        df["state"] = df["state"].str.upper()
        df = df[df["state"].isin(self.valid_states)]

        # Convert year to integer
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Key variables for analysis
        key_vars = [
            "state",
            "year",
            "subject",
            "grade",
            "avg_scale_score",
            "pct_below_basic",
            "pct_basic",
            "pct_proficient",
            "pct_advanced",
        ]

        available_vars = [var for var in key_vars if var in df.columns]
        df = df[available_vars].copy()

        self.logger.info(
            f"Loaded NAEP data: {len(df)} records, {len(df['state'].unique())} states"
        )
        return df

    def load_census_data(self) -> pd.DataFrame:
        """Load and clean Census finance data"""
        self.logger.info("Loading Census finance data...")

        census_file = self.raw_data_dir / "census_education_finance_parsed.csv"
        if not census_file.exists():
            self.logger.warning(f"Census file not found: {census_file}")
            return pd.DataFrame()

        df = pd.read_csv(census_file)

        # Standardize state codes
        df["state"] = df["state"].str.upper()
        df = df[df["state"].isin(self.valid_states)]

        # Convert year and numeric columns
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Key financial variables
        financial_vars = [
            "total_revenue",
            "total_expenditure",
            "current_expenditure",
            "instruction_expenditure",
            "per_pupil_expenditure",
            "special_education_expenditure",
        ]

        for var in financial_vars:
            if var in df.columns:
                df[var] = pd.to_numeric(df[var], errors="coerce")

        self.logger.info(
            f"Loaded Census data: {len(df)} records, {len(df['state'].unique())} states"
        )
        return df

    def load_edfacts_data(self) -> pd.DataFrame:
        """Load and clean EdFacts IDEA data"""
        self.logger.info("Loading EdFacts data...")

        edfacts_dir = self.raw_data_dir / "edfacts"
        if not edfacts_dir.exists():
            self.logger.warning(f"EdFacts directory not found: {edfacts_dir}")
            return pd.DataFrame()

        all_files = []
        for year in [2019, 2020, 2021, 2022, 2023]:
            # Try different filename patterns
            patterns = [
                f"bchildcountandedenvironment{year}.csv",
                f"bchildcountandedenvironment{year}-{str(year + 1)[2:]}.csv",
                f"bchildcountandedenvironment{year}-24.csv",  # For 2023-24
            ]

            for pattern in patterns:
                file_path = edfacts_dir / pattern
                if file_path.exists():
                    self.logger.info(f"Loading EdFacts file: {file_path.name}")
                    try:
                        # EdFacts files can be large, read with low_memory=False
                        # Skip metadata rows (first 4 rows are usually metadata)
                        df_year = pd.read_csv(file_path, low_memory=False, skiprows=4)

                        # Remove empty columns (metadata has many empty columns)
                        df_year = df_year.dropna(axis=1, how="all")

                        # Add year column if not present
                        if (
                            "YEAR" not in df_year.columns
                            and "Year" not in df_year.columns
                        ):
                            df_year["year"] = year

                        all_files.append(df_year)
                        break
                    except Exception as e:
                        self.logger.warning(f"Error reading {file_path}: {e}")

        if not all_files:
            self.logger.warning("No EdFacts files successfully loaded")
            return pd.DataFrame()

        # Combine all years
        df = pd.concat(all_files, ignore_index=True)

        # Standardize column names (EdFacts uses various naming conventions)
        column_mapping = {
            "STATE": "state",
            "State": "state",
            "State Name": "state_name",
            "SEA_STATE": "state",
            "YEAR": "year",
            "Year": "year",
            "School Year": "year",
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

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Map state names to codes if needed
        if "state_name" in df.columns and "state" not in df.columns:
            df["state"] = df["state_name"].map(self.state_mapping["state_name"])

        # Standardize state codes
        if "state" in df.columns:
            df["state"] = df["state"].str.upper()
            df = df[df["state"].isin(self.valid_states)]
        elif "state_name" in df.columns:
            # If we still don't have state codes, filter by state names
            valid_state_names = list(self.state_mapping["state_name"].keys()) + [
                "District of Columbia"
            ]
            df = df[df["state_name"].isin(valid_state_names)]

        # Convert numeric columns
        numeric_cols = ["year", "child_count"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.logger.info(
            f"Loaded EdFacts data: {len(df)} records, {len(df['state'].unique())} states"
        )
        return df

    def load_ocr_data(self) -> pd.DataFrame:
        """Load and clean OCR civil rights data"""
        self.logger.info("Loading OCR data...")

        ocr_dir = self.raw_data_dir / "ocr"
        if not ocr_dir.exists():
            self.logger.warning(f"OCR directory not found: {ocr_dir}")
            return pd.DataFrame()

        all_data = []

        # Load state-level CSV files (2009-2015)
        csv_files = [
            ("CRDC-2009-State-Discipline.csv", 2009),
            ("CRDC-2011-State-Discipline.csv", 2011),
            ("CRDC-2013-State-Data.csv", 2013),
            ("CRDC-2015-State-Data.csv", 2015),
        ]

        for filename, year in csv_files:
            file_path = ocr_dir / filename
            if file_path.exists():
                try:
                    df_year = pd.read_csv(file_path)
                    df_year["year"] = year
                    all_data.append(df_year)
                    self.logger.info(
                        f"Loaded OCR CSV: {filename} ({len(df_year)} records)"
                    )
                except Exception as e:
                    self.logger.warning(f"Error reading {filename}: {e}")

        # Load extracted ZIP data (2017, 2020)
        zip_years = [2017, 2020]
        for year in zip_years:
            extract_dir = ocr_dir / f"extracted_{year}"
            if extract_dir.exists():
                # Look for key discipline files
                discipline_files = [
                    "Suspensions.csv",
                    "Expulsions.csv",
                    "Restraint and Seclusion.csv",
                ]

                for discipline_file in discipline_files:
                    # Search for file in subdirectories
                    found_files = list(extract_dir.rglob(discipline_file))
                    if found_files:
                        file_path = found_files[0]
                        try:
                            df_disc = pd.read_csv(file_path)
                            df_disc["year"] = year
                            df_disc["data_type"] = discipline_file.replace(
                                ".csv", ""
                            ).lower()
                            all_data.append(df_disc)
                            self.logger.info(
                                f"Loaded OCR {year} {discipline_file}: {len(df_disc)} records"
                            )
                        except Exception as e:
                            self.logger.warning(f"Error reading {file_path}: {e}")

        if not all_data:
            self.logger.warning("No OCR files successfully loaded")
            return pd.DataFrame()

        # Combine all data
        df = pd.concat(all_data, ignore_index=True, sort=False)

        # Standardize column names
        column_mapping = {
            "State Name": "state_name",
            "State": "state",
            "StateAbbr": "state",
            "State Abbr": "state",
            "LEA State": "state",
            "LEA_STATE": "state",
            "School Year": "school_year",
            "SchoolYear": "school_year",
            "Total Enrollment": "total_enrollment",
            "Students with Disabilities": "swd_enrollment",
            "Out-of-School Suspensions": "suspensions",
            "Expulsions": "expulsions",
            "Restraint": "restraint_incidents",
            "Seclusion": "seclusion_incidents",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Map state names to codes if needed
        if "state_name" in df.columns and "state" not in df.columns:
            df["state"] = df["state_name"].map(self.state_mapping["state_name"])

        # Standardize state codes
        if "state" in df.columns:
            df["state"] = df["state"].str.upper()
            df = df[df["state"].isin(self.valid_states)]

        # Convert numeric columns
        numeric_cols = [
            "year",
            "total_enrollment",
            "swd_enrollment",
            "suspensions",
            "expulsions",
            "restraint_incidents",
            "seclusion_incidents",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.logger.info(
            f"Loaded OCR data: {len(df)} records, {len(df['state'].unique())} states"
        )
        return df

    def create_state_year_panel(
        self, datasets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create master state-year panel dataset

        Args:
            datasets: Dictionary of loaded datasets

        Returns:
            Master panel dataset
        """
        self.logger.info("Creating master state-year panel...")

        # Define the panel structure (all state-year combinations)
        years = range(2009, 2024)  # 2009-2023
        states = self.valid_states

        # Create base panel
        panel_data = []
        for state in states:
            for year in years:
                panel_data.append({"state": state, "year": year})

        master_df = pd.DataFrame(panel_data)
        self.logger.info(
            f"Created base panel: {len(master_df)} state-year observations"
        )

        # Merge each dataset
        for dataset_name, df in datasets.items():
            if df.empty:
                continue

            self.logger.info(f"Merging {dataset_name} data...")

            if dataset_name == "naep":
                # NAEP: Multiple observations per state-year (subject/grade combinations)
                # Check what columns are available and aggregate accordingly
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if "state" in numeric_cols:
                    numeric_cols.remove("state")
                if "year" in numeric_cols:
                    numeric_cols.remove("year")

                if (
                    len(numeric_cols) > 0
                    and "state" in df.columns
                    and "year" in df.columns
                ):
                    # Aggregate numeric columns by mean
                    agg_dict = dict.fromkeys(numeric_cols, "mean")
                    naep_agg = df.groupby(["state", "year"]).agg(agg_dict).reset_index()

                    # Add NAEP prefix to avoid column conflicts
                    naep_cols = {
                        col: f"naep_{col}"
                        for col in naep_agg.columns
                        if col not in ["state", "year"]
                    }
                    naep_agg = naep_agg.rename(columns=naep_cols)

                    master_df = master_df.merge(
                        naep_agg, on=["state", "year"], how="left"
                    )
                    self.logger.info(
                        f"Merged NAEP data: {len(naep_agg)} state-year observations"
                    )

            elif dataset_name == "census":
                # Census: Should be one observation per state-year
                census_vars = [
                    col for col in df.columns if col not in ["state", "year"]
                ]
                if census_vars:
                    census_subset = df[["state", "year"] + census_vars].copy()
                    master_df = master_df.merge(
                        census_subset, on=["state", "year"], how="left"
                    )

            elif dataset_name == "edfacts":
                # EdFacts: Aggregate by state-year (multiple disability categories, etc.)
                # Find numeric columns that represent counts
                if "state" in df.columns and "year" in df.columns:
                    numeric_cols = df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    # Remove non-count columns
                    exclude_cols = ["state", "year"]
                    count_cols = [
                        col for col in numeric_cols if col not in exclude_cols
                    ]

                    if count_cols:
                        # Aggregate numeric columns by sum (they represent counts)
                        agg_dict = dict.fromkeys(count_cols, "sum")
                        edfacts_agg = (
                            df.groupby(["state", "year"]).agg(agg_dict).reset_index()
                        )

                        # Add EdFacts prefix
                        edfacts_cols = {
                            col: f"edfacts_{col}"
                            for col in edfacts_agg.columns
                            if col not in ["state", "year"]
                        }
                        edfacts_agg = edfacts_agg.rename(columns=edfacts_cols)

                        master_df = master_df.merge(
                            edfacts_agg, on=["state", "year"], how="left"
                        )
                        self.logger.info(
                            f"Merged EdFacts data: {len(edfacts_agg)} state-year observations"
                        )

            elif dataset_name == "ocr":
                # OCR: Aggregate by state-year
                if "state" in df.columns and "year" in df.columns:
                    numeric_cols = df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    # Remove non-count columns
                    exclude_cols = ["state", "year"]
                    count_cols = [
                        col for col in numeric_cols if col not in exclude_cols
                    ]

                    if count_cols:
                        # Aggregate numeric columns by sum
                        agg_dict = dict.fromkeys(count_cols, "sum")
                        ocr_agg = (
                            df.groupby(["state", "year"]).agg(agg_dict).reset_index()
                        )

                        # Add OCR prefix
                        ocr_cols = {
                            col: f"ocr_{col}"
                            for col in ocr_agg.columns
                            if col not in ["state", "year"]
                        }
                        ocr_agg = ocr_agg.rename(columns=ocr_cols)

                        master_df = master_df.merge(
                            ocr_agg, on=["state", "year"], how="left"
                        )
                        self.logger.info(
                            f"Merged OCR data: {len(ocr_agg)} state-year observations"
                        )

        self.logger.info(
            f"Master dataset created: {len(master_df)} observations, {len(master_df.columns)} variables"
        )
        return master_df

    def add_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived variables for analysis"""
        self.logger.info("Creating derived variables...")

        # Achievement gaps and rates
        if "naep_pct_proficient" in df.columns:
            df["naep_proficiency_rate"] = df["naep_pct_proficient"] + df.get(
                "naep_pct_advanced", 0
            )

        # Per-pupil metrics
        if "total_expenditure" in df.columns and "ocr_total_enrollment" in df.columns:
            df["per_pupil_total_expenditure"] = (
                df["total_expenditure"] / df["ocr_total_enrollment"]
            )

        # Special education rates
        if "edfacts_total_swd" in df.columns and "ocr_total_enrollment" in df.columns:
            df["swd_identification_rate"] = (
                df["edfacts_total_swd"] / df["ocr_total_enrollment"]
            )

        # Discipline rates for SWD
        if "ocr_suspensions" in df.columns and "ocr_swd_enrollment" in df.columns:
            df["swd_suspension_rate"] = df["ocr_suspensions"] / df["ocr_swd_enrollment"]

        if "ocr_expulsions" in df.columns and "ocr_swd_enrollment" in df.columns:
            df["swd_expulsion_rate"] = df["ocr_expulsions"] / df["ocr_swd_enrollment"]

        # Time variables for econometric analysis
        df["post_2015"] = (df["year"] >= 2015).astype(int)
        df["post_covid"] = (df["year"] >= 2020).astype(int)

        self.logger.info(
            f"Added derived variables. Final dataset: {len(df.columns)} variables"
        )
        return df

    def run_integration(self) -> pd.DataFrame:
        """Run complete data integration pipeline"""
        self.logger.info("Starting data integration pipeline...")

        # Load all datasets
        datasets = {
            "naep": self.load_naep_data(),
            "census": self.load_census_data(),
            "edfacts": self.load_edfacts_data(),
            "ocr": self.load_ocr_data(),
        }

        # Create master panel
        master_df = self.create_state_year_panel(datasets)

        # Add derived variables
        master_df = self.add_derived_variables(master_df)

        # Save datasets
        output_files = {}

        # Save individual cleaned datasets
        for name, df in datasets.items():
            if not df.empty:
                output_file = self.output_dir / f"{name}_cleaned.csv"
                df.to_csv(output_file, index=False)
                output_files[f"{name}_cleaned"] = output_file
                self.logger.info(f"Saved {name} cleaned data: {output_file}")

        # Save master dataset
        master_file = self.output_dir / "master_analysis_dataset.csv"
        master_df.to_csv(master_file, index=False)
        output_files["master"] = master_file

        self.logger.info(f"Saved master analysis dataset: {master_file}")
        self.logger.info(
            f"Integration complete. Output files: {list(output_files.keys())}"
        )

        return master_df


def main():
    """Main function for running data integration"""
    import argparse

    parser = argparse.ArgumentParser(description="Integrate multiple data sources")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default="data/raw",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/processed",
        help="Directory to save processed files",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run integration
    integrator = DataIntegrator(args.raw_data_dir, args.output_dir)
    master_df = integrator.run_integration()

    # Summary statistics
    print("\n=== Integration Summary ===")
    print(f"Master dataset shape: {master_df.shape}")
    print(f"States: {master_df['state'].nunique()}")
    print(f"Years: {master_df['year'].min()}-{master_df['year'].max()}")
    print("Non-missing observations by key variables:")

    key_vars = [
        "naep_avg_scale_score",
        "total_expenditure",
        "edfacts_total_swd",
        "ocr_total_enrollment",
    ]
    for var in key_vars:
        if var in master_df.columns:
            non_missing = master_df[var].notna().sum()
            print(f"  {var}: {non_missing} ({non_missing / len(master_df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
