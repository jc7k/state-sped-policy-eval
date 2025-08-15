#!/usr/bin/env python
"""
Census F-33 Education Finance Data Parser
Parses downloaded Census Excel files and extracts special education spending data
"""

import logging
from pathlib import Path

import pandas as pd

from .common import StateUtils


class CensusDataParser:
    """Parse Census F-33 education finance Excel files"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_utils = StateUtils()

        # Key financial variables to extract
        self.key_variables = [
            "total_revenue",
            "federal_revenue",
            "state_revenue",
            "local_revenue",
            "total_expenditure",
            "instruction_expenditure",
            "support_services_expenditure",
            "special_education_expenditure",
            "total_enrollment",
            "special_education_enrollment",
            "per_pupil_expenditure",
            "per_pupil_special_education",
        ]

    def parse_f33_file(self, file_path: Path, year: int) -> pd.DataFrame:
        """
        Parse a single Census F-33 Excel file

        Args:
            file_path: Path to Excel file
            year: Year of the data

        Returns:
            DataFrame with parsed education finance data
        """
        self.logger.info(f"Parsing Census F-33 file for {year}: {file_path}")

        try:
            # Read Excel file - F-33 files typically have multiple sheets
            xl_file = pd.ExcelFile(file_path)

            # The main summary table is usually in sheet "1"
            # (sheet names are numbers in F-33 files)
            data_sheet = "1" if "1" in xl_file.sheet_names else xl_file.sheet_names[1]

            self.logger.info(f"Reading sheet: {data_sheet}")

            # Read the data sheet - skip the header rows
            # F-33 files have metadata in first few rows
            df_raw = pd.read_excel(file_path, sheet_name=data_sheet, header=None)

            # Find the header row (contains "Total", "Federal", etc.)
            header_row = None
            for i in range(min(10, len(df_raw))):
                row_str = " ".join(str(v) for v in df_raw.iloc[i].values if pd.notna(v))
                if "revenue" in row_str.lower() or "total" in row_str.lower():
                    header_row = i
                    break

            if header_row is None:
                header_row = 2  # Default to row 2 if not found

            # Re-read with correct header
            df = pd.read_excel(file_path, sheet_name=data_sheet, header=header_row)

            # Clean and standardize the data
            df_clean = self._clean_f33_data(df, year)

            return df_clean

        except Exception as e:
            self.logger.error(f"Error parsing F-33 file {file_path}: {e}")
            raise

    def _clean_f33_data(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Clean and standardize F-33 data

        Args:
            df: Raw dataframe from Excel
            year: Year of the data

        Returns:
            Cleaned and standardized dataframe
        """
        # First, identify and name the columns based on F-33 structure
        # Sheet 1 typically has: State, blank, Total Revenue, Federal, State, Local, Total Expenditure, etc.
        if len(df.columns) >= 8:
            # Standard F-33 Table 1 column mapping
            column_mapping = {
                df.columns[0]: "state_name",
                df.columns[2]: "total_revenue",
                df.columns[3]: "federal_revenue",
                df.columns[4]: "state_revenue",
                df.columns[5]: "local_revenue",
                df.columns[6]: "total_expenditure",
                df.columns[7]: "instruction_expenditure",
            }

            # Additional columns if present
            if len(df.columns) > 8:
                column_mapping[df.columns[8]] = "support_services"
            if len(df.columns) > 9:
                column_mapping[df.columns[9]] = "other_expenditure"
            if len(df.columns) > 10:
                column_mapping[df.columns[10]] = "capital_outlay"

            df = df.rename(columns=column_mapping)

        # Remove rows that don't contain state data
        # States are rows where first column contains state names
        state_rows = []
        for idx, row in df.iterrows():
            first_col = str(row.iloc[0]).strip()
            # Skip total/summary rows
            if first_col.lower() in [
                "",
                "nan",
                "total",
                "united states",
                "us",
                "geographic area",
            ]:
                continue
            # Check if it looks like a state name (contains letters, not just numbers/symbols)
            if any(c.isalpha() for c in first_col):
                # Clean state name (remove dots and extra characters)
                clean_name = first_col.replace(".", "").strip()
                if clean_name:
                    state_rows.append(idx)

        df = df.loc[state_rows].reset_index(drop=True)

        # Initialize cleaned dataframe
        clean_data = []

        # Process each row
        for _idx, row in df.iterrows():
            # Get state name from first column
            if "state_name" in df.columns:
                state_name = str(row["state_name"]).strip()
            else:
                state_name = str(row.iloc[0]).strip()

            # Clean state name (remove dots and extra characters)
            state_name = state_name.replace(".", "").strip()

            # Skip invalid rows
            if not state_name or state_name.lower() in [
                "",
                "nan",
                "total",
                "united states",
                "us",
            ]:
                continue

            # Get state abbreviation
            state_abbrev = self.state_utils.name_to_abbrev(state_name)
            if not state_abbrev:
                continue

            # Extract financial data based on renamed columns
            row_data = {"year": year, "state": state_abbrev, "state_name": state_name}

            # Map the renamed columns to our data structure
            if "total_revenue" in df.columns:
                row_data["total_revenue"] = self._parse_numeric(row.get("total_revenue"))

            if "federal_revenue" in df.columns:
                row_data["federal_revenue"] = self._parse_numeric(row.get("federal_revenue"))

            if "state_revenue" in df.columns:
                row_data["state_revenue"] = self._parse_numeric(row.get("state_revenue"))

            if "local_revenue" in df.columns:
                row_data["local_revenue"] = self._parse_numeric(row.get("local_revenue"))

            if "total_expenditure" in df.columns:
                row_data["total_expenditure"] = self._parse_numeric(row.get("total_expenditure"))

            if "instruction_expenditure" in df.columns:
                row_data["instruction_expenditure"] = self._parse_numeric(
                    row.get("instruction_expenditure")
                )

            if "support_services" in df.columns:
                row_data["support_services_expenditure"] = self._parse_numeric(
                    row.get("support_services")
                )

            # For enrollment data, we need to look at a different sheet (usually sheet 5 or similar)
            # For now, we'll skip enrollment calculations

            # Calculate per-pupil metrics if we have total expenditure
            # Note: We'd need enrollment data from another sheet for accurate per-pupil calculations

            clean_data.append(row_data)

        # Create dataframe from cleaned data
        df_clean = pd.DataFrame(clean_data)

        # Ensure we have data
        if len(df_clean) == 0:
            self.logger.warning(f"No valid state data found in file for year {year}")
            return pd.DataFrame()

        # Sort by state
        df_clean = df_clean.sort_values("state").reset_index(drop=True)

        self.logger.info(f"Parsed {len(df_clean)} states for year {year}")

        return df_clean

    def _parse_numeric(self, value) -> float | None:
        """Parse numeric value from various formats"""
        if pd.isna(value):
            return None

        # Convert to string and clean
        val_str = str(value).strip()

        # Remove common formatting
        val_str = val_str.replace(",", "")
        val_str = val_str.replace("$", "")
        val_str = val_str.replace("%", "")

        # Handle parentheses for negatives
        if val_str.startswith("(") and val_str.endswith(")"):
            val_str = "-" + val_str[1:-1]

        # Handle special values
        if val_str in ["-", "--", "N/A", "NA", "nan", ""]:
            return None

        try:
            # Convert to float
            value = float(val_str)

            # F-33 data is typically in thousands, so multiply by 1000
            # unless it's a per-pupil or percentage value
            if abs(value) < 1000:  # Likely already in full dollars or a ratio
                return value
            else:  # Likely in thousands
                return value * 1000
        except ValueError:
            return None

    def parse_all_files(self, data_dir: Path) -> pd.DataFrame:
        """
        Parse all Census F-33 files in directory

        Args:
            data_dir: Directory containing Census Excel files

        Returns:
            Combined dataframe with all years
        """
        all_data = []

        # Find all Census Excel files
        census_files = list(data_dir.glob("census_f33_*.xls"))
        census_files.extend(list(data_dir.glob("census_f33_*.xlsx")))

        for file_path in sorted(census_files):
            # Extract year from filename
            year_str = file_path.stem.split("_")[-1]
            try:
                year = int(year_str)
            except ValueError:
                self.logger.warning(f"Could not extract year from filename: {file_path}")
                continue

            # Parse file
            try:
                df_year = self.parse_f33_file(file_path, year)
                if not df_year.empty:
                    all_data.append(df_year)
                    self.logger.info(f"Successfully parsed {len(df_year)} records for {year}")
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {e}")
                continue

        if not all_data:
            self.logger.warning("No data successfully parsed from Census files")
            return pd.DataFrame()

        # Combine all years
        df_combined = pd.concat(all_data, ignore_index=True)

        # Sort by year and state
        df_combined = df_combined.sort_values(["year", "state"]).reset_index(drop=True)

        self.logger.info(
            f"Combined Census data: {len(df_combined)} total records across {df_combined['year'].nunique()} years"
        )

        return df_combined

    def validate_parsed_data(self, df: pd.DataFrame) -> dict:
        """
        Validate parsed Census data

        Args:
            df: Parsed Census dataframe

        Returns:
            Validation results dictionary
        """
        validation = {
            "total_records": len(df),
            "years": sorted(df["year"].unique().tolist()),
            "states": sorted(df["state"].unique().tolist()),
            "missing_states_by_year": {},
            "data_quality": {},
            "summary_stats": {},
        }

        # Check for missing states by year
        expected_states = set(self.state_utils.get_all_states())
        for year in df["year"].unique():
            year_data = df[df["year"] == year]
            found_states = set(year_data["state"].unique())
            missing = expected_states - found_states
            if missing:
                validation["missing_states_by_year"][int(year)] = sorted(missing)

        # Data quality checks
        for col in ["total_revenue", "total_expenditure", "total_enrollment"]:
            if col in df.columns:
                validation["data_quality"][col] = {
                    "non_null_count": df[col].notna().sum(),
                    "null_percentage": (df[col].isna().sum() / len(df)) * 100,
                }

        # Summary statistics for key metrics
        if "per_pupil_expenditure" in df.columns:
            ppe = df["per_pupil_expenditure"].dropna()
            if len(ppe) > 0:
                validation["summary_stats"]["per_pupil_expenditure"] = {
                    "mean": float(ppe.mean()),
                    "median": float(ppe.median()),
                    "min": float(ppe.min()),
                    "max": float(ppe.max()),
                    "std": float(ppe.std()),
                }

        return validation


def main():
    """Main function for testing the parser"""
    import argparse

    from src.config import get_config

    parser = argparse.ArgumentParser(description="Parse Census F-33 education finance data")
    parser.add_argument("--data-dir", type=Path, help="Directory containing Census Excel files")
    parser.add_argument("--output", type=Path, help="Output CSV file path")
    parser.add_argument("--validate", action="store_true", help="Run validation on parsed data")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        config = get_config()
        data_dir = config.raw_data_dir

    # Parse files
    parser = CensusDataParser()
    df = parser.parse_all_files(data_dir)

    if df.empty:
        print("No data parsed")
        return

    print(f"\nParsed {len(df)} records from Census F-33 files")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"States: {len(df['state'].unique())}")

    # Save to CSV
    output_path = args.output or data_dir / "census_education_finance_parsed.csv"

    df.to_csv(output_path, index=False)
    print(f"\nSaved parsed data to {output_path}")

    # Run validation if requested
    if args.validate:
        validation = parser.validate_parsed_data(df)
        print("\n=== VALIDATION RESULTS ===")
        print(f"Total records: {validation['total_records']}")
        print(f"Years: {validation['years']}")
        print(f"States found: {len(validation['states'])}")

        if validation["missing_states_by_year"]:
            print("\nMissing states by year:")
            for year, missing in validation["missing_states_by_year"].items():
                print(f"  {year}: {missing}")

        if validation["summary_stats"]:
            print("\nSummary statistics:")
            for metric, stats in validation["summary_stats"].items():
                print(f"  {metric}:")
                for stat_name, value in stats.items():
                    print(f"    {stat_name}: {value:,.2f}")


if __name__ == "__main__":
    main()
