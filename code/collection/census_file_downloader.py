#!/usr/bin/env python
"""
Census Education Finance File Downloader
Alternative data collection method when API access is unavailable
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class CensusFileDownloader:
    """
    Downloads Census education finance data from direct file URLs
    Used as fallback when API key is unavailable or invalid
    """

    def __init__(self, rate_limit_delay: float = 2.0):
        """
        Initialize Census file downloader

        Args:
            rate_limit_delay: Seconds between download requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)

        # Known Census education finance data URLs (update as needed)
        self.data_urls = {
            2019: {
                "base_url": "https://www.census.gov/data/tables/2019/econ/school-finances/",
                "files": [
                    "secondary-education-finance.html",
                    "elementary-secondary-education-finance-by-enrollment-size.html",
                ],
            },
            2020: {
                "base_url": "https://www.census.gov/data/tables/2020/econ/school-finances/",
                "files": [
                    "secondary-education-finance.html",
                    "elementary-secondary-education-finance-by-enrollment-size.html",
                ],
            },
            2021: {
                "base_url": "https://www.census.gov/data/tables/2021/econ/school-finances/",
                "files": [
                    "secondary-education-finance.html",
                    "elementary-secondary-education-finance-by-enrollment-size.html",
                ],
            },
        }

    def download_education_finance_data(
        self, years: list[int], output_dir: Path
    ) -> dict:
        """
        Download education finance data files for specified years

        Args:
            years: List of years to download
            output_dir: Directory to save downloaded files

        Returns:
            Dictionary with download results and file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "successful_downloads": [],
            "failed_downloads": [],
            "downloaded_files": [],
        }

        self.logger.info(f"Starting Census file downloads for {len(years)} years")

        for i, year in enumerate(years):
            if year not in self.data_urls:
                self.logger.warning(f"No known data URLs for year {year}")
                results["failed_downloads"].append(
                    {"year": year, "reason": "No URLs available"}
                )
                continue

            year_data = self.data_urls[year]
            base_url = year_data["base_url"]

            for file_name in year_data["files"]:
                file_url = base_url + file_name
                output_file = (
                    output_dir
                    / f"census_education_finance_{year}_{file_name.replace('.html', '.html')}"
                )

                try:
                    self.logger.info(f"Downloading {file_url}")

                    headers = {
                        "User-Agent": "Mozilla/5.0 (compatible; Education Research Data Collector)"
                    }

                    response = requests.get(file_url, headers=headers, timeout=30)
                    response.raise_for_status()

                    # Save the file
                    with open(output_file, "wb") as f:
                        f.write(response.content)

                    results["successful_downloads"].append(
                        {
                            "year": year,
                            "file": file_name,
                            "url": file_url,
                            "local_file": str(output_file),
                            "size": len(response.content),
                        }
                    )
                    results["downloaded_files"].append(str(output_file))

                    self.logger.info(
                        f"Successfully downloaded {file_name} for {year} ({len(response.content)} bytes)"
                    )

                except requests.exceptions.RequestException as e:
                    error_msg = f"Download failed for {file_url}: {str(e)}"
                    self.logger.error(error_msg)
                    results["failed_downloads"].append(
                        {
                            "year": year,
                            "file": file_name,
                            "url": file_url,
                            "reason": str(e),
                        }
                    )

                except Exception as e:
                    error_msg = f"Unexpected error downloading {file_url}: {str(e)}"
                    self.logger.error(error_msg)
                    results["failed_downloads"].append(
                        {
                            "year": year,
                            "file": file_name,
                            "url": file_url,
                            "reason": str(e),
                        }
                    )

                # Rate limiting between downloads
                if i < len(years) - 1 or file_name != year_data["files"][-1]:
                    time.sleep(self.rate_limit_delay)

        self.logger.info(
            f"Download completed: {len(results['successful_downloads'])} successful, "
            f"{len(results['failed_downloads'])} failed"
        )

        return results

    def extract_csv_links_from_html(self, html_file: Path) -> list[str]:
        """
        Extract CSV download links from Census HTML pages

        Args:
            html_file: Path to downloaded HTML file

        Returns:
            List of CSV download URLs
        """
        try:
            with open(html_file, encoding="utf-8") as f:
                content = f.read()

            import re

            # Look for CSV download links
            csv_pattern = r'href="([^"]*\.csv[^"]*)"'
            csv_links = re.findall(csv_pattern, content, re.IGNORECASE)

            # Convert relative URLs to absolute
            full_urls = []
            for link in csv_links:
                if link.startswith("http"):
                    full_urls.append(link)
                elif link.startswith("/"):
                    full_urls.append("https://www.census.gov" + link)
                else:
                    # Relative path
                    base_url = "https://www.census.gov/data/tables/"
                    full_urls.append(base_url + link)

            return full_urls

        except Exception as e:
            self.logger.error(f"Failed to extract CSV links from {html_file}: {e}")
            return []

    def download_csv_files(self, csv_urls: list[str], output_dir: Path) -> dict:
        """
        Download CSV files from extracted URLs

        Args:
            csv_urls: List of CSV file URLs to download
            output_dir: Directory to save CSV files

        Returns:
            Dictionary with download results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"successful_downloads": [], "failed_downloads": [], "csv_files": []}

        for i, url in enumerate(csv_urls):
            try:
                filename = url.split("/")[-1]
                if not filename.endswith(".csv"):
                    filename += ".csv"

                output_file = output_dir / filename

                self.logger.info(f"Downloading CSV: {url}")

                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; Education Research Data Collector)"
                }

                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                with open(output_file, "wb") as f:
                    f.write(response.content)

                # Verify it's actually a CSV by trying to read it
                try:
                    pd.read_csv(output_file, nrows=5)
                    results["successful_downloads"].append(
                        {
                            "url": url,
                            "local_file": str(output_file),
                            "size": len(response.content),
                        }
                    )
                    results["csv_files"].append(str(output_file))
                    self.logger.info(f"Successfully downloaded CSV: {filename}")
                except pd.errors.ParserError:
                    self.logger.warning(
                        f"Downloaded file {filename} is not a valid CSV"
                    )
                    output_file.unlink()  # Delete invalid file

            except Exception as e:
                error_msg = f"Failed to download CSV {url}: {str(e)}"
                self.logger.error(error_msg)
                results["failed_downloads"].append({"url": url, "reason": str(e)})

            # Rate limiting between downloads
            if i < len(csv_urls) - 1:
                time.sleep(self.rate_limit_delay)

        return results


if __name__ == "__main__":
    # Test the downloader
    import tempfile

    logging.basicConfig(level=logging.INFO)

    downloader = CensusFileDownloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test download for one year
        results = downloader.download_education_finance_data([2021], temp_path)
        print(f"Download results: {results}")

        # Test CSV extraction if we got HTML files
        for file_path in results["downloaded_files"]:
            if file_path.endswith(".html"):
                csv_links = downloader.extract_csv_links_from_html(Path(file_path))
                print(f"Found {len(csv_links)} CSV links in {file_path}")
                if csv_links:
                    print(f"Sample CSV links: {csv_links[:3]}")
