#!/usr/bin/env python
"""
NAEP Data Collector
Implements NAEP API integration per data-collection-prd.md Section 2
"""

import pandas as pd
import requests
import time
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NAEPDataCollector:
    """
    Automated NAEP data collection for state-level special education analysis
    
    Collects achievement data by disability status from NAEP State Assessment API
    """
    
    def __init__(self, rate_limit_delay: Optional[float] = None):
        self.base_url = "https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx"
        # Load rate limit from environment or use default
        self.rate_limit_delay = rate_limit_delay or float(os.getenv('NAEP_RATE_LIMIT_DELAY', '6.0'))
        self.results = []
        self.logger = logging.getLogger(__name__)
        
    def fetch_state_swd_data(self, 
                           years: List[int],
                           grades: List[int] = [4, 8],
                           subjects: List[str] = ['mathematics', 'reading']) -> pd.DataFrame:
        """
        Fetch NAEP data by state for students with disabilities
        
        Args:
            years: List of assessment years to collect
            grades: List of grade levels (4, 8)
            subjects: List of subjects ('mathematics', 'reading')
            
        Returns:
            DataFrame with NAEP achievement data by state and disability status
        """
        
        self.logger.info(f"Starting NAEP collection: {len(years)} years, {len(grades)} grades, {len(subjects)} subjects")
        
        total_requests = len(years) * len(grades) * len(subjects)
        request_count = 0
        
        for year in years:
            for grade in grades:
                for subject in subjects:
                    request_count += 1
                    self.logger.info(f"Request {request_count}/{total_requests}: {subject} grade {grade} year {year}")
                    
                    params = {
                        'type': 'data',
                        'subject': subject,
                        'grade': grade,
                        'year': year,
                        'jurisdiction': 'states',
                        'variable': 'SDRACEM',  # Students with disabilities variable
                        'stattype': 'MN:MN,RP:RP',  # Mean scores and percentiles
                    }
                    
                    try:
                        response = requests.get(self.base_url, params=params, timeout=30)
                        response.raise_for_status()
                        
                        data = response.json()
                        
                        # Parse API response structure
                        if 'result' in data and data['result']:
                            for state_data in data['result']:
                                record = self._parse_state_record(state_data, year, grade, subject)
                                if record:
                                    self.results.append(record)
                                    
                        self.logger.debug(f"Successfully collected {subject} grade {grade} year {year}")
                        
                    except requests.exceptions.RequestException as e:
                        self.logger.error(f"Request failed for {subject} grade {grade} year {year}: {str(e)}")
                        
                    except Exception as e:
                        self.logger.error(f"Unexpected error for {subject} grade {grade} year {year}: {str(e)}")
                        
                    # Rate limiting - respect API limits
                    if request_count < total_requests:
                        time.sleep(self.rate_limit_delay)
        
        df = pd.DataFrame(self.results)
        self.logger.info(f"NAEP collection completed: {len(df)} records collected")
        
        return df
    
    def _parse_state_record(self, state_data: Dict, year: int, grade: int, subject: str) -> Optional[Dict]:
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
            # Extract state identifier
            state_name = state_data.get('name', '')
            state_code = self._convert_state_name_to_code(state_name)
            
            if not state_code:
                return None
                
            # Extract disability status data
            # NAEP API returns data by different demographic categories
            swd_data = None
            non_swd_data = None
            
            # Look for IEP/disability status in the data structure
            if 'datavalue' in state_data:
                for item in state_data['datavalue']:
                    category = item.get('categoryname', '')
                    if 'IEP' in category or 'disability' in category.lower():
                        if 'Yes' in category or 'with' in category.lower():
                            swd_data = item
                        elif 'No' in category or 'without' in category.lower():
                            non_swd_data = item
            
            # Create record with available data
            record = {
                'state': state_code,
                'year': year,
                'grade': grade,
                'subject': subject,
                'swd_mean': self._safe_float(swd_data.get('value') if swd_data else None),
                'swd_se': self._safe_float(swd_data.get('errorFlag') if swd_data else None),
                'non_swd_mean': self._safe_float(non_swd_data.get('value') if non_swd_data else None),
                'non_swd_se': self._safe_float(non_swd_data.get('errorFlag') if non_swd_data else None),
            }
            
            # Calculate achievement gap if both values available
            if record['non_swd_mean'] is not None and record['swd_mean'] is not None:
                record['gap'] = record['non_swd_mean'] - record['swd_mean']
            else:
                record['gap'] = None
            
            # Calculate gap standard error if both SEs available
            if record['non_swd_se'] is not None and record['swd_se'] is not None:
                record['gap_se'] = (record['non_swd_se']**2 + record['swd_se']**2)**0.5
            else:
                record['gap_se'] = None
                
            return record
            
        except Exception as e:
            self.logger.warning(f"Failed to parse state record: {str(e)}")
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """
        Safely convert API values to float, handling NAEP special codes
        
        Args:
            value: Raw value from API
            
        Returns:
            Float value or None if invalid/missing
        """
        
        if value in [None, '', 'null', '‡', '*', 'N/A', '#']:
            return None
            
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _convert_state_name_to_code(self, state_name: str) -> Optional[str]:
        """
        Convert state names to two-letter codes
        
        Args:
            state_name: Full state name from API
            
        Returns:
            Two-letter state code or None if not found
        """
        
        state_mapping = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
            'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
            'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
            'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
            'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
            'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
            'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
            'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
            'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
            'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
        }
        
        return state_mapping.get(state_name.strip())
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate collected NAEP data per data-collection-prd.md Section 2.3
        
        Args:
            df: Collected NAEP DataFrame
            
        Returns:
            Validation results dictionary
        """
        
        validation = {
            'total_records': len(df),
            'states_covered': df['state'].nunique() if len(df) > 0 else 0,
            'years_covered': sorted(df['year'].unique().tolist()) if len(df) > 0 else [],
            'subjects_covered': sorted(df['subject'].unique().tolist()) if len(df) > 0 else [],
            'grades_covered': sorted(df['grade'].unique().tolist()) if len(df) > 0 else [],
            'missing_swd_scores': df['swd_mean'].isna().sum() if len(df) > 0 else 0,
            'missing_gaps': df['gap'].isna().sum() if len(df) > 0 else 0,
            'passed': True,
            'errors': [],
            'warnings': []
        }
        
        # Check state coverage (should have 50 states + DC = 51)
        if validation['states_covered'] < 50:
            validation['errors'].append(f"Only {validation['states_covered']} states covered, expected 50+")
            validation['passed'] = False
        
        # Check for duplicate combinations
        if len(df) > 0:
            duplicates = df.duplicated(subset=['state', 'year', 'grade', 'subject']).sum()
            if duplicates > 0:
                validation['errors'].append(f"{duplicates} duplicate state-year-grade-subject combinations found")
                validation['passed'] = False
        
        # Check score ranges (NAEP scale is typically 0-500)
        if len(df) > 0:
            invalid_swd_scores = ((df['swd_mean'] < 0) | (df['swd_mean'] > 500)).sum()
            if invalid_swd_scores > 0:
                validation['warnings'].append(f"{invalid_swd_scores} SWD scores outside typical range (0-500)")
            
            invalid_non_swd_scores = ((df['non_swd_mean'] < 0) | (df['non_swd_mean'] > 500)).sum()
            if invalid_non_swd_scores > 0:
                validation['warnings'].append(f"{invalid_non_swd_scores} non-SWD scores outside typical range (0-500)")
        
        # Check missing data patterns
        if len(df) > 0:
            missing_rate = validation['missing_swd_scores'] / len(df)
            if missing_rate > 0.3:
                validation['warnings'].append(f"High missing data rate: {missing_rate:.1%} of SWD scores missing")
        
        return validation
    
    def save_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> bool:
        """
        Save collected NAEP data to file
        
        Args:
            df: NAEP DataFrame to save
            output_path: Path to save file (if None, uses environment variable or default)
            
        Returns:
            True if successful, False otherwise
        """
        
        # Use provided path or load from environment
        if output_path is None:
            raw_data_dir = os.getenv('RAW_DATA_DIR', 'data/raw/')
            output_path = os.path.join(raw_data_dir, 'naep_raw.csv')
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"NAEP data saved to {output_path}: {len(df)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save NAEP data: {str(e)}")
            return False

def main():
    """
    Example usage of NAEPDataCollector
    """
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize collector
    collector = NAEPDataCollector()
    
    # Collect data for recent years
    years = [2017, 2019, 2022]  # Available NAEP years
    
    try:
        # Collect data
        df = collector.fetch_state_swd_data(years)
        
        # Validate data
        validation = collector.validate_data(df)
        print(f"Validation results: {validation}")
        
        # Save data
        collector.save_data(df)
        
        print(f"Collection completed successfully: {len(df)} records")
        
    except Exception as e:
        print(f"Collection failed: {str(e)}")

if __name__ == "__main__":
    main()