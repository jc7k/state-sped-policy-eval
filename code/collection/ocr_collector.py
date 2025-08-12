#!/usr/bin/env python
"""
OCR Civil Rights Data Collection (CRDC) Collector
Collects civil rights data from the U.S. Department of Education's Office for Civil Rights
"""

import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .base_collector import FileBasedCollector
from .common import StateUtils


class OCRCollector(FileBasedCollector):
    """
    Collect OCR Civil Rights Data Collection including:
    - Discipline data by disability status
    - Restraint and seclusion incidents
    - Access to advanced coursework
    - Educational programs and services
    """

    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize OCR collector

        Args:
            rate_limit_delay: Seconds to wait between requests
        """
        super().__init__(rate_limit_delay=rate_limit_delay)

        # Base URL for OCR data downloads
        self.base_url = "https://ocrdata.ed.gov/assets"

        # Known data files by year
        self.data_files = {
            2009: {
                "url": f"{self.base_url}/downloads/CRDC-2009-State-Discipline.csv",
                "type": "csv",
                "name": "CRDC-2009-State-Discipline.csv",
            },
            2011: {
                "url": f"{self.base_url}/downloads/CRDC-2011-State-Discipline.csv",
                "type": "csv",
                "name": "CRDC-2011-State-Discipline.csv",
            },
            2013: {
                "url": f"{self.base_url}/downloads/CRDC-2013-State-Data.csv",
                "type": "csv",
                "name": "CRDC-2013-State-Data.csv",
            },
            2015: {
                "url": f"{self.base_url}/downloads/CRDC-2015-State-Data.csv",
                "type": "csv",
                "name": "CRDC-2015-State-Data.csv",
            },
            2017: {
                "url": f"{self.base_url}/ocr/docs/2017-18-crdc-data.zip",
                "type": "zip",
                "name": "2017-18-crdc-data.zip",
            },
            2020: {
                # 2020-21 data - need to find correct URL pattern
                "url": f"{self.base_url}/ocr/docs/2020-21-crdc-data.zip",
                "type": "zip",
                "name": "2020-21-crdc-data.zip",
            },
        }

        # State utilities for consistent state handling
        self.state_utils = StateUtils()
        self.state_codes = self.state_utils.get_all_states()

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from URL

        Args:
            url: URL of the file
            output_path: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Downloading: {url}")

            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()

            # Save the file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress logging for large files
                        if (
                            total_size > 0 and downloaded % (1024 * 1024) == 0
                        ):  # Every MB
                            progress = (downloaded / total_size) * 100
                            self.logger.info(f"Download progress: {progress:.1f}%")

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Saved to: {output_path} ({file_size_mb:.2f} MB)")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return True

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to download {url}: {e}")
            return False

    def extract_zip_file(self, zip_path: Path, extract_dir: Path) -> list[Path]:
        """
        Extract ZIP file and return list of extracted files

        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract to

        Returns:
            List of extracted file paths
        """
        extracted_files = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # List contents
                file_list = zip_ref.namelist()
                self.logger.info(f"ZIP contains {len(file_list)} files")

                # Extract all files
                zip_ref.extractall(extract_dir)

                # Return paths to extracted files
                for filename in file_list:
                    extracted_path = extract_dir / filename
                    if extracted_path.exists():
                        extracted_files.append(extracted_path)
                        self.logger.info(f"Extracted: {extracted_path}")

        except zipfile.BadZipFile as e:
            self.logger.error(f"Error extracting ZIP file {zip_path}: {e}")

        return extracted_files

    def try_download_year(self, year: int, output_dir: Path) -> Path | None:
        """
        Try to download OCR data for a specific year

        Args:
            year: Year of data
            output_dir: Directory to save files

        Returns:
            Path to downloaded file if successful, None otherwise
        """
        if year not in self.data_files:
            self.logger.warning(f"No known data file for year {year}")
            return None

        file_info = self.data_files[year]
        output_path = output_dir / file_info["name"]

        # If file already exists, skip download
        if output_path.exists():
            self.logger.info(f"File already exists: {output_path}")
            return output_path

        # Try multiple URL patterns
        urls_to_try = [
            file_info["url"],
            # Alternative patterns for missing files
            f"{self.base_url}/downloads/CRDC-{year}-State-Data.csv",
            f"{self.base_url}/downloads/CRDC-{year}-{year + 1}-State-Data.csv",
            f"{self.base_url}/ocr/docs/{year}-{str(year + 1)[2:]}-crdc-data.zip",
        ]

        for url in urls_to_try:
            if self.download_file(url, output_path):
                return output_path

        return None

    def collect_discipline_data(
        self, years: list[int], output_dir: Path
    ) -> dict[int, Path]:
        """
        Collect discipline and civil rights data

        Args:
            years: List of years to collect
            output_dir: Directory to save files

        Returns:
            Dictionary mapping years to file paths
        """
        self.logger.info(f"Collecting OCR civil rights data for years: {years}")

        results = {}
        for year in years:
            file_path = self.try_download_year(year, output_dir)
            if file_path:
                results[year] = file_path

                # If it's a ZIP file, extract it
                if file_path.suffix.lower() == ".zip":
                    extract_dir = output_dir / f"extracted_{year}"
                    extracted_files = self.extract_zip_file(file_path, extract_dir)

                    # Find the main data file
                    for extracted_file in extracted_files:
                        if (
                            extracted_file.suffix.lower() == ".csv"
                            and "state" in extracted_file.name.lower()
                        ):
                            results[year] = extracted_file
                            break
            else:
                self.logger.warning(f"Could not download OCR data for {year}")

        return results

    def parse_crdc_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Parse a CRDC CSV file

        Args:
            file_path: Path to CSV file

        Returns:
            Parsed dataframe
        """
        try:
            # CRDC files can have various encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # If all encodings fail, try with error handling
                df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")

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
                "Year": "year",
                "Total Enrollment": "total_enrollment",
                "Students with Disabilities": "swd_enrollment",
                "Out-of-School Suspensions": "suspensions",
                "Expulsions": "expulsions",
                "Restraint": "restraint_incidents",
                "Seclusion": "seclusion_incidents",
            }

            # Rename columns that match our mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})

            # Filter to only include states (not territories or totals)
            if "state" in df.columns:
                df = df[df["state"].isin(self.state_codes)]

            self.logger.info(f"Parsed {len(df)} records from {file_path}")

            return df

        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()

    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch OCR data (required by abstract base class)
        
        Args:
            **kwargs: Additional arguments (years, output_dir)
            
        Returns:
            Combined DataFrame with all collected data
        """
        years = kwargs.get('years', [2015, 2017, 2020])
        output_dir = kwargs.get('output_dir', Path('.'))
        
        results = self.collect_all_data(years, output_dir)
        # Return empty DataFrame for now - parsing would be implemented separately
        return pd.DataFrame()
    
    def collect_all_data(self, years: list[int], output_dir: Path) -> dict[int, Path]:
        """
        Collect all OCR civil rights data

        Args:
            years: List of years to collect
            output_dir: Directory to save files

        Returns:
            Dictionary mapping years to file paths
        """
        self.logger.info(f"Starting comprehensive OCR data collection for {years}")

        results = self.collect_discipline_data(years, output_dir)

        # Summary
        total_files = len(results)
        self.logger.info(f"Collection complete: {total_files} files downloaded")

        return results


def main():
    """Main function for testing the collector"""
    import argparse
    from code.config import get_config

    parser = argparse.ArgumentParser(description="Collect OCR Civil Rights Data")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2015, 2017, 2020],
        help="Years to collect data for",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        config = get_config()
        output_dir = config.raw_data_dir / "ocr"

    # Create collector
    collector = OCRCollector()

    # Collect data
    results = collector.collect_all_data(args.years, output_dir)

    # Report results
    print(f"\nData collection complete. Files saved to: {output_dir}")
    print("\nOCR Civil Rights Data:")
    for year, path in results.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {year}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  {year}: Not downloaded")


if __name__ == "__main__":
    main()
