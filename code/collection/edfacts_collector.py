#!/usr/bin/env python
"""
EdFacts IDEA Part B Data Collector
Collects special education data from the U.S. Department of Education's EdFacts system
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
import pandas as pd
from io import StringIO


class EdFactsCollector:
    """
    Collect EdFacts IDEA Part B data including:
    - Child count by disability category
    - Educational environments (inclusion rates)
    - Exit data (graduation, dropout)
    - Personnel data
    """
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize EdFacts collector
        
        Args:
            rate_limit_delay: Seconds to wait between requests
        """
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = rate_limit_delay
        
        # Base URLs for data files
        self.osep_base_url = "https://www2.ed.gov/programs/osepidea/618-data/state-level-data-files/part-b-data"
        self.data_ed_base_url = "https://data.ed.gov/dataset"
        
        # Known file patterns and resource IDs for different data types
        self.file_patterns = {
            'child_count': {
                'path': 'child-count-and-educational-environments',
                'prefix': 'bchildcountandedenvironment',
                'years': {
                    2019: {
                        'filename': 'bchildcountandedenvironment2019.csv',
                        'resource_id': 'c49009eb-a269-4131-9bbe-7d8a3f67f649'
                    },
                    2020: {
                        'filename': 'bchildcountandedenvironment2020.csv',
                        'resource_id': 'c515f168-be9c-4505-a6d7-d52a47b9b2b7'
                    },
                    2021: {
                        'filename': 'bchildcountandedenvironment2021.csv',
                        'resource_id': '22294e78-ff8b-48cf-8f5e-5a84f183ec22'
                    },
                    2022: {
                        'filename': 'bchildcountandedenvironment2022.csv',
                        'resource_id': '91dfcdc0-7e9b-4945-8319-684f9ffd2a24'
                    },
                    2023: {
                        'filename': 'bchildcountandedenvironment2023-24.csv',
                        'resource_id': 'aa572553-f494-49a6-a01e-99c52f0cf948'
                    }
                }
            },
            'exiting': {
                'path': 'exiting',
                'prefix': 'bexiting',
                'years': {
                    2019: 'bexiting2019-20.csv',
                    2020: 'bexiting2020-21.csv',
                    2021: 'bexiting2021-22.csv',
                    2022: 'bexiting2022-23.csv',
                    2023: 'bexiting2023-24.csv'
                }
            },
            'discipline': {
                'path': 'discipline',
                'prefix': 'bdiscipline',
                'years': {
                    2019: 'bdiscipline2019-20.csv',
                    2020: 'bdiscipline2020-21.csv',
                    2021: 'bdiscipline2021-22.csv',
                    2022: 'bdiscipline2022-23.csv'
                }
            },
            'personnel': {
                'path': 'personnel',
                'prefix': 'bpersonnel',
                'years': {
                    2019: 'bpersonnel2019.csv',
                    2020: 'bpersonnel2020.csv',
                    2021: 'bpersonnel2021.csv',
                    2022: 'bpersonnel2022.csv'
                }
            }
        }
        
        # State abbreviations for filtering
        self.state_codes = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
            'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
            'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
            'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
            'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]
    
    def download_csv_file(self, url: str, output_path: Path) -> bool:
        """
        Download a CSV file from URL
        
        Args:
            url: URL of the CSV file
            output_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Downloading: {url}")
            
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            # Save the file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Saved to: {output_path}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to download {url}: {e}")
            return False
    
    def try_download_patterns(self, data_type: str, year: int, output_dir: Path) -> Optional[Path]:
        """
        Try different URL patterns to download EdFacts data
        
        Args:
            data_type: Type of data (child_count, exiting, etc.)
            year: Year of data
            output_dir: Directory to save files
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        if data_type not in self.file_patterns:
            self.logger.error(f"Unknown data type: {data_type}")
            return None
        
        pattern = self.file_patterns[data_type]
        
        # Check if we have a known filename for this year
        if year not in pattern['years']:
            self.logger.warning(f"No known file for {data_type} year {year}")
            return None
        
        year_info = pattern['years'][year]
        
        # Handle both old and new structure
        if isinstance(year_info, dict):
            filename = year_info['filename']
            resource_id = year_info.get('resource_id')
        else:
            filename = year_info
            resource_id = None
        
        output_path = output_dir / filename
        
        # If file already exists, skip download
        if output_path.exists():
            self.logger.info(f"File already exists: {output_path}")
            return output_path
        
        # Try different URL patterns
        urls_to_try = []
        
        # Primary pattern: data.ed.gov with resource ID (if available)
        if resource_id:
            urls_to_try.append(
                f"https://data.ed.gov/dataset/71ca7d0c-a161-4abe-9e2b-4e68ffb1061a/resource/{resource_id}/download/{filename}"
            )
        
        # Fallback patterns
        urls_to_try.extend([
            # OSEP website pattern
            f"{self.osep_base_url}/{pattern['path']}/{filename}",
            
            # Alternative OSEP pattern (some years use different structure)
            f"https://www2.ed.gov/programs/osepidea/618-data/state-level-data-files/part-b-data/{pattern['path']}/{pattern['prefix']}{year}-{str(year+1)[2:]}.csv",
            
            # Generic data.ed.gov pattern
            f"https://data.ed.gov/dataset/71ca7d0c-a161-4abe-9e2b-4e68ffb1061a/resource/download/{filename}"
        ])
        
        for url in urls_to_try:
            if self.download_csv_file(url, output_path):
                return output_path
        
        return None
    
    def collect_child_count_data(self, years: List[int], output_dir: Path) -> Dict[int, Path]:
        """
        Collect child count and educational environment data
        
        Args:
            years: List of years to collect
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping years to file paths
        """
        self.logger.info(f"Collecting child count data for years: {years}")
        
        results = {}
        for year in years:
            file_path = self.try_download_patterns('child_count', year, output_dir)
            if file_path:
                results[year] = file_path
            else:
                self.logger.warning(f"Could not download child count data for {year}")
        
        return results
    
    def collect_exiting_data(self, years: List[int], output_dir: Path) -> Dict[int, Path]:
        """
        Collect exiting (graduation/dropout) data
        
        Args:
            years: List of years to collect
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping years to file paths
        """
        self.logger.info(f"Collecting exiting data for years: {years}")
        
        results = {}
        for year in years:
            file_path = self.try_download_patterns('exiting', year, output_dir)
            if file_path:
                results[year] = file_path
            else:
                self.logger.warning(f"Could not download exiting data for {year}")
        
        return results
    
    def collect_discipline_data(self, years: List[int], output_dir: Path) -> Dict[int, Path]:
        """
        Collect discipline data
        
        Args:
            years: List of years to collect
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping years to file paths
        """
        self.logger.info(f"Collecting discipline data for years: {years}")
        
        results = {}
        for year in years:
            file_path = self.try_download_patterns('discipline', year, output_dir)
            if file_path:
                results[year] = file_path
            else:
                self.logger.warning(f"Could not download discipline data for {year}")
        
        return results
    
    def parse_child_count_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Parse a child count CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Parsed dataframe
        """
        try:
            # Read CSV - EdFacts files can have various formats
            df = pd.read_csv(file_path, encoding='latin-1')
            
            # Standardize column names (EdFacts uses various naming conventions)
            column_mapping = {
                'State Name': 'state_name',
                'State': 'state',
                'StateAbbr': 'state',
                'StateCode': 'state',
                'Year': 'year',
                'SchoolYear': 'year',
                'Age Group': 'age_group',
                'AgeGroup': 'age_group',
                'Disability Category': 'disability_category',
                'DisabilityCategory': 'disability_category',
                'Educational Environment': 'educational_environment',
                'EducationalEnvironment': 'educational_environment',
                'Student Count': 'student_count',
                'StudentCount': 'student_count',
                'ChildCount': 'student_count'
            }
            
            # Rename columns that match our mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Filter to only include states (not territories or national totals)
            if 'state' in df.columns:
                df = df[df['state'].isin(self.state_codes + ['DC'])]
            
            self.logger.info(f"Parsed {len(df)} records from {file_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
    
    def collect_all_data(self, years: List[int], output_dir: Path) -> Dict[str, Dict[int, Path]]:
        """
        Collect all types of EdFacts data
        
        Args:
            years: List of years to collect
            output_dir: Directory to save files
            
        Returns:
            Dictionary with data type as key and year->path mapping as value
        """
        self.logger.info(f"Starting comprehensive EdFacts data collection for {years}")
        
        results = {
            'child_count': self.collect_child_count_data(years, output_dir),
            'exiting': self.collect_exiting_data(years, output_dir),
            'discipline': self.collect_discipline_data(years, output_dir)
        }
        
        # Summary
        total_files = sum(len(files) for files in results.values())
        self.logger.info(f"Collection complete: {total_files} files downloaded")
        
        for data_type, files in results.items():
            self.logger.info(f"  {data_type}: {len(files)} files")
        
        return results


def main():
    """Main function for testing the collector"""
    import argparse
    from code.config import get_config
    
    parser = argparse.ArgumentParser(description='Collect EdFacts IDEA Part B data')
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021],
                       help='Years to collect data for')
    parser.add_argument('--data-type', choices=['child_count', 'exiting', 'discipline', 'all'],
                       default='all', help='Type of data to collect')
    parser.add_argument('--output-dir', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        config = get_config()
        output_dir = config.raw_data_dir / 'edfacts'
    
    # Create collector
    collector = EdFactsCollector()
    
    # Collect data based on type
    if args.data_type == 'all':
        results = collector.collect_all_data(args.years, output_dir)
    elif args.data_type == 'child_count':
        results = {'child_count': collector.collect_child_count_data(args.years, output_dir)}
    elif args.data_type == 'exiting':
        results = {'exiting': collector.collect_exiting_data(args.years, output_dir)}
    elif args.data_type == 'discipline':
        results = {'discipline': collector.collect_discipline_data(args.years, output_dir)}
    
    # Report results
    print(f"\nData collection complete. Files saved to: {output_dir}")
    for data_type, files in results.items():
        print(f"\n{data_type.replace('_', ' ').title()}:")
        for year, path in files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {year}: {path.name} ({size_mb:.2f} MB)")
            else:
                print(f"  {year}: Not downloaded")


if __name__ == "__main__":
    main()