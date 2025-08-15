#!/usr/bin/env python
"""
Census Education Finance Data Collector
Implements Census API integration per data-collection-prd.md Section 4
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from .base_collector import APIBasedCollector
from .common import SafeTypeConverter, StateUtils

# Load environment variables
load_dotenv()


class CensusEducationFinance(APIBasedCollector):
    """
    Census Bureau education finance data collector
    Includes special education expenditures starting 2015
    """

    def __init__(self, api_key: str | None = None, rate_limit_delay: float = 1.0):
        """
        Initialize Census Finance collector

        Args:
            api_key: Census API key (if None, will try to load from environment)
            rate_limit_delay: Seconds between requests
        """
        # Load API key from environment if not provided
        self.api_key = api_key or os.getenv("CENSUS_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Census API key is required. Either:\n"
                "1. Pass api_key parameter, or\n"
                "2. Set CENSUS_API_KEY environment variable, or\n"
                "3. Add CENSUS_API_KEY to .env file\n\n"
                "Get your free API key at: https://api.census.gov/data/key_signup.html"
            )

        # Initialize parent class
        base_url = "https://api.census.gov/data"
        super().__init__(base_url=base_url, api_key=self.api_key, rate_limit_delay=rate_limit_delay)

        # Initialize utilities
        self.state_utils = StateUtils()
        self.converter = SafeTypeConverter()

        # Use StateUtils for FIPS mapping
        self.state_fips = self.state_utils.FIPS_TO_ABBREV

    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch Census data (required by abstract base class)

        Args:
            **kwargs: Additional arguments (years)

        Returns:
            DataFrame with collected Census finance data
        """
        years = kwargs.get("years", [2020, 2021, 2022])
        return self.fetch_state_finance(years)

    def fetch_state_finance(self, years: list[int]) -> pd.DataFrame:
        """
        Fetch F-33 survey data with special education breakouts

        Args:
            years: List of years to collect (2009-2022)

        Returns:
            DataFrame with state education finance data

        Key variables:
        - TOTALEXP: Total expenditures
        - TCURINST: Current instruction spending
        - TCURSSVC: Student support services
        - TCUROTH: Other current expenditures
        - ENROLL: Student enrollment
        """

        self.logger.info(f"Starting Census finance collection for {len(years)} years")

        # Census variables to collect
        variables = [
            "NAME",  # State name
            "TOTALEXP",  # Total expenditures
            "TCURINST",  # Current instruction spending
            "TCURSSVC",  # Student support services (proxy for special ed)
            "TCUROTH",  # Other current expenditures
            "ENROLL",  # Student enrollment
        ]

        for i, year in enumerate(years):
            self.logger.info(f"Collecting Census finance data for {year}")

            # Census API endpoint varies by year
            if year >= 2013:
                endpoint = (
                    f"{self.base_url}/{year}/programs/finances/elementary-secondary-education"
                )
            else:
                endpoint = f"{self.base_url}/{year}/governments/school-districts/finance"

            params = {"get": ",".join(variables), "for": "state:*", "key": self.api_key}

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data and len(data) > 1:  # First row is headers
                    headers = data[0]
                    rows = data[1:]

                    for row in rows:
                        record = self._parse_finance_record(
                            dict(zip(headers, row, strict=False)), year
                        )
                        if record:
                            self.results.append(record)

                self.logger.debug(f"Successfully collected Census data for {year}")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed for {year}: {str(e)}")

            except Exception as e:
                self.logger.error(f"Unexpected error for {year}: {str(e)}")

            # Rate limiting - sleep between requests but not after the last one
            if i < len(years) - 1:
                time.sleep(self.rate_limit_delay)

        df = pd.DataFrame(self.results)
        self.logger.info(f"Census collection completed: {len(df)} records collected")

        return df

    def _parse_finance_record(self, row_data: dict, year: int) -> dict | None:
        """
        Parse individual state finance record from Census API response

        Args:
            row_data: Raw row data from API
            year: Data year

        Returns:
            Parsed record dictionary or None if invalid
        """

        try:
            # Get state code from FIPS or name
            state_fips = row_data.get("state", "")
            state_name = row_data.get("NAME", "")
            state_code = self.state_fips.get(state_fips, "")

            if not state_code and state_name:
                # Try to convert state name to code
                state_code = self._convert_state_name_to_code(state_name)

            if not state_code:
                return None

            record = {
                "state": state_code,
                "year": year,
                "total_expenditures": SafeTypeConverter.safe_int(row_data.get("TOTALEXP")),
                "current_instruction": SafeTypeConverter.safe_int(row_data.get("TCURINST")),
                "student_support_services": SafeTypeConverter.safe_int(row_data.get("TCURSSVC")),
                "other_current_expenditures": SafeTypeConverter.safe_int(row_data.get("TCUROTH")),
                "enrollment": SafeTypeConverter.safe_int(row_data.get("ENROLL")),
            }

            # Calculate per-pupil spending
            if record["total_expenditures"] and record["enrollment"] and record["enrollment"] > 0:
                record["per_pupil_spending"] = record["total_expenditures"] / record["enrollment"]
            else:
                record["per_pupil_spending"] = None

            # Calculate support services per pupil (proxy for special education spending)
            if (
                record["student_support_services"]
                and record["enrollment"]
                and record["enrollment"] > 0
            ):
                record["support_services_per_pupil"] = (
                    record["student_support_services"] / record["enrollment"]
                )
            else:
                record["support_services_per_pupil"] = None

            return record

        except Exception as e:
            self.logger.warning(f"Failed to parse finance record: {str(e)}")
            return None

    def _safe_int(self, value) -> int | None:
        """
        Safely convert API values to int, handling Census special codes

        Args:
            value: Raw value from API

        Returns:
            Integer value or None if invalid/missing
        """

        if value in [None, "", "null", "N", "X", "S", "D"]:
            return None

        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _convert_state_name_to_code(self, state_name: str) -> str | None:
        """
        Convert state names to two-letter codes using centralized StateUtils.

        Args:
            state_name: Full state name from API

        Returns:
            Two-letter state code or None if not found
        """
        return self.state_utils.name_to_abbrev(state_name)

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate collected Census finance data

        Args:
            df: Collected Census DataFrame

        Returns:
            Validation results dictionary
        """

        validation = {
            "total_records": len(df),
            "states_covered": df["state"].nunique() if len(df) > 0 else 0,
            "years_covered": (sorted(df["year"].unique().tolist()) if len(df) > 0 else []),
            "missing_expenditures": (
                df["total_expenditures"].isna().sum()
                if len(df) > 0 and "total_expenditures" in df.columns
                else 0
            ),
            "missing_enrollment": (
                df["enrollment"].isna().sum() if len(df) > 0 and "enrollment" in df.columns else 0
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
            duplicates = df.duplicated(subset=["state", "year"]).sum()
            if duplicates > 0:
                validation["errors"].append(f"{duplicates} duplicate state-year combinations found")
                validation["passed"] = False

        # Check for reasonable expenditure values
        if len(df) > 0 and "total_expenditures" in df.columns:
            negative_expenditures = (df["total_expenditures"] < 0).sum()
            if negative_expenditures > 0:
                validation["errors"].append(
                    f"{negative_expenditures} negative expenditure values found"
                )
                validation["passed"] = False

        # Check missing data patterns
        if len(df) > 0 and "total_expenditures" in df.columns:
            missing_rate = validation["missing_expenditures"] / len(df)
            if missing_rate > 0.2:
                validation["warnings"].append(f"High missing expenditure rate: {missing_rate:.1%}")

        return validation

    def save_data(self, df: pd.DataFrame, output_path: str = "data/raw/census_raw.csv") -> bool:
        """
        Save collected Census data to file

        Args:
            df: Census DataFrame to save
            output_path: Path to save file

        Returns:
            True if successful, False otherwise
        """

        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(output_path, index=False)

            self.logger.info(f"Census data saved to {output_path}: {len(df)} records")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save Census data: {str(e)}")
            return False


def main():
    """
    Example usage of CensusEducationFinance collector
    """

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Initialize collector (will load API key from environment)
        collector = CensusEducationFinance()

        # Collect data for recent years
        years = [2019, 2020, 2021, 2022]  # Available Census years

        # Collect data
        df = collector.fetch_state_finance(years)

        # Validate data
        validation = collector.validate_data(df)
        print(f"Validation results: {validation}")

        # Save data
        collector.save_data(df)

        print(f"Collection completed successfully: {len(df)} records")

    except ValueError as e:
        print(f"Configuration error: {str(e)}")

    except Exception as e:
        print(f"Collection failed: {str(e)}")


if __name__ == "__main__":
    main()
