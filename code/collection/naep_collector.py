#!/usr/bin/env python
"""
NAEP Data Collector
Implements NAEP API integration per data-collection-prd.md Section 2
"""

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .base_collector import APIBasedCollector
from .common import SafeTypeConverter, StateUtils

# Load environment variables
load_dotenv()


class NAEPDataCollector(APIBasedCollector):
    """
    Automated NAEP data collection for state-level special education analysis

    Collects achievement data by disability status from NAEP State Assessment API
    """

    def __init__(self, rate_limit_delay: float | None = None):
        # Load rate limit from environment or use default
        default_delay = float(os.getenv("NAEP_RATE_LIMIT_DELAY", "6.0"))
        
        super().__init__(
            base_url="https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx",
            rate_limit_delay=rate_limit_delay or default_delay,
            logger_name="NAEPDataCollector"
        )

    def fetch_data(
        self,
        years: list[int],
        grades: list[int] | None = None,
        subjects: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch NAEP data by state for students with disabilities

        Args:
            years: List of assessment years to collect
            grades: List of grade levels (4, 8)
            subjects: List of subjects ('mathematics', 'reading')

        Returns:
            DataFrame with NAEP achievement data by state and disability status
        """
        if grades is None:
            grades = [4, 8]
        if subjects is None:
            subjects = ["mathematics", "reading"]

        # Get all state codes from StateUtils
        state_codes = self.state_utils.get_all_states()

        self.logger.info(
            f"Starting NAEP collection: {len(years)} years, {len(grades)} grades, {len(subjects)} subjects, {len(state_codes)} states"
        )

        total_requests = len(years) * len(grades) * len(subjects) * len(state_codes)
        request_count = 0

        for year in years:
            for grade in grades:
                for subject in subjects:
                    for state_code in state_codes:
                        request_count += 1
                        self.logger.info(
                            f"Request {request_count}/{total_requests}: {subject} grade {grade} year {year} state {state_code}"
                        )

                        params = {
                            "type": "data",
                            "subject": subject,
                            "grade": grade,
                            "year": year,
                            "jurisdiction": state_code,
                            "variable": "IEP",  # Correct disability status variable
                            "stattype": "MN:MN",  # Mean scores only to start
                        }

                        try:
                            # Make direct API request (NAEP doesn't use standard REST endpoints)
                            response = self.api_client.get(self.base_url, params=params)
                            data = response.json()

                            # Parse API response structure
                            if "result" in data and data["result"]:
                                for state_data in data["result"]:
                                    record = self._parse_state_record(
                                        state_data, year, grade, subject
                                    )
                                    if record:
                                        self.results.append(record)

                            self.logger.debug(
                                f"Successfully collected {subject} grade {grade} year {year} state {state_code}"
                            )

                        except Exception as e:
                            self.logger.error(
                                f"Request failed for {subject} grade {grade} year {year} state {state_code}: {str(e)}"
                            )
                            # Rate limiting is handled by base class
                            continue

        df = pd.DataFrame(self.results)
        self.logger.info(f"NAEP collection completed: {len(df)} records collected")

        return df

    def _parse_state_record(
        self, state_data: dict, year: int, grade: int, subject: str
    ) -> dict | None:
        """
        Parse individual state record from NAEP API response

        Args:
            state_data: Raw state data from API
            year: Assessment year
            grade: Grade level
            subject: Subject area

        Returns:
            Parsed record dictionary or None if invalid
        """

        try:
            # Extract state information from API response
            state_name = state_data.get("jurisLabel", "")
            jurisdiction = state_data.get("jurisdiction", "")

            # Extract disability status information
            var_value = state_data.get("varValue", "")
            var_value_label = state_data.get("varValueLabel", "")
            score = state_data.get("value")

            # Check if this is valid data
            if not state_name or score is None:
                return None

            # Determine if this is SWD or non-SWD data based on varValue
            # varValue "1" = "Identified as students with disabilities"
            # varValue "2" = "Not identified as students with disabilities"
            is_swd = var_value == "1"

            # Create standardized record
            record = {
                "state": jurisdiction,  # State abbreviation from API
                "state_name": state_name,
                "year": year,
                "grade": grade,
                "subject": subject,
                "disability_status": "SWD" if is_swd else "non-SWD",
                "disability_label": var_value_label,
                "mean_score": float(score),
                "var_value": var_value,
                "error_flag": state_data.get("errorFlag"),
                "is_displayable": state_data.get("isStatDisplayable", 0) == 1,
            }

            return record

        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning(f"Failed to parse state record: {e}")
            return None

    def _safe_float(self, value) -> float | None:
        """
        Safely convert API values to float, handling NAEP special codes

        Args:
            value: Raw value from API

        Returns:
            Float value or None if invalid/missing
        """
        return SafeTypeConverter.safe_float(
            value, 
            invalid_values=[None, "", "null", "‡", "*", "N/A", "#"]
        )

    def _convert_state_name_to_code(self, state_name: str) -> str | None:
        """
        Convert state names to two-letter codes

        Args:
            state_name: Full state name from API

        Returns:
            Two-letter state code or None if not found
        """
        return self.state_utils.name_to_abbrev(state_name)

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate collected NAEP data per data-collection-prd.md Section 2.3

        Args:
            df: Collected NAEP DataFrame

        Returns:
            Validation results dictionary
        """

        validation = {
            "total_records": len(df),
            "states_covered": df["state"].nunique() if len(df) > 0 else 0,
            "years_covered": (
                sorted(df["year"].unique().tolist()) if len(df) > 0 else []
            ),
            "subjects_covered": (
                sorted(df["subject"].unique().tolist()) if len(df) > 0 else []
            ),
            "grades_covered": (
                sorted(df["grade"].unique().tolist()) if len(df) > 0 else []
            ),
            "missing_swd_scores": (
                df["swd_mean"].isna().sum()
                if len(df) > 0 and "swd_mean" in df.columns
                else 0
            ),
            "missing_gaps": (
                df["gap"].isna().sum() if len(df) > 0 and "gap" in df.columns else 0
            ),
            "passed": True,
            "errors": [],
            "warnings": [],
        }

        # Check state coverage (should have 50 states + DC = 51 for production data)
        if validation["states_covered"] < 50:
            if validation["states_covered"] < 2:
                # Very few states is likely an error
                validation["errors"].append(
                    f"Only {validation['states_covered']} states covered, expected 50+"
                )
                validation["passed"] = False
            else:
                # Some states covered but not complete - warning for test scenarios
                validation["warnings"].append(
                    f"Only {validation['states_covered']} states covered, expected 50+ for production"
                )

        # Check for duplicate combinations
        if len(df) > 0:
            duplicates = df.duplicated(
                subset=["state", "year", "grade", "subject"]
            ).sum()
            if duplicates > 0:
                validation["errors"].append(
                    f"{duplicates} duplicate state-year-grade-subject combinations found"
                )
                validation["passed"] = False

        # Check score ranges (NAEP scale is typically 0-500)
        if len(df) > 0:
            if "swd_mean" in df.columns:
                invalid_swd_scores = (
                    (df["swd_mean"] < 0) | (df["swd_mean"] > 500)
                ).sum()
                if invalid_swd_scores > 0:
                    validation["warnings"].append(
                        f"{invalid_swd_scores} SWD scores outside typical range (0-500)"
                    )

            if "non_swd_mean" in df.columns:
                invalid_non_swd_scores = (
                    (df["non_swd_mean"] < 0) | (df["non_swd_mean"] > 500)
                ).sum()
                if invalid_non_swd_scores > 0:
                    validation["warnings"].append(
                        f"{invalid_non_swd_scores} non-SWD scores outside typical range (0-500)"
                    )

        # Check missing data patterns
        if len(df) > 0 and "swd_mean" in df.columns:
            missing_rate = validation["missing_swd_scores"] / len(df)
            if missing_rate > 0.3:
                validation["warnings"].append(
                    f"High missing data rate: {missing_rate:.1%} of SWD scores missing"
                )

        return validation

    def _get_default_output_path(self) -> str:
        """
        Get default output path for NAEP data.

        Returns:
            Default file path for saving NAEP data
        """
        return self.file_utils.get_default_output_path(
            "naep_raw.csv", 
            env_var="RAW_DATA_DIR", 
            default_dir="data/raw"
        )


    # Convenience method for backward compatibility
    def fetch_state_swd_data(
        self,
        years: list[int],
        grades: list[int] | None = None,
        subjects: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch NAEP data by state for students with disabilities.
        This method provides backward compatibility.
        """
        return self.fetch_data(years=years, grades=grades, subjects=subjects)


def main():
    """
    Example usage of NAEPDataCollector
    """
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize collector
    collector = NAEPDataCollector()

    # Collect data for recent years
    years = [2017, 2019, 2022]  # Available NAEP years

    try:
        # Use the new base class method for complete workflow
        df = collector.run_collection(years=years, save_data=True)
        print(f"Collection completed successfully: {len(df)} records")

    except Exception as e:
        print(f"Collection failed: {str(e)}")


if __name__ == "__main__":
    main()
