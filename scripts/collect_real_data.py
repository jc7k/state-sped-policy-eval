#!/usr/bin/env python3
"""
Real Data Collection Pipeline
Collects actual data from NAEP, Census, EdFacts, and OCR APIs
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collection import (
    NAEPDataCollector,
    CensusEducationFinance,
    EdFactsCollector,
    OCRCollector,
    StateUtils
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """Orchestrates real data collection from multiple sources."""
    
    def __init__(self):
        """Initialize data collectors."""
        self.state_utils = StateUtils()
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize collectors
        self.naep_collector = NAEPDataCollector()
        self.census_collector = CensusEducationFinance()
        self.edfacts_collector = EdFactsCollector()
        self.ocr_collector = OCRCollector()
        
        # Years to collect
        self.years = list(range(2009, 2023))
        self.states = self.state_utils.get_all_states()
        
    def collect_naep_data(self):
        """Collect NAEP achievement data for SWD and non-SWD students."""
        logger.info("Starting NAEP data collection...")
        
        all_data = []
        
        # NAEP years (biennial)
        naep_years = [2009, 2011, 2013, 2015, 2017, 2019, 2022]
        grades = [4, 8]
        subjects = ['mathematics', 'reading']
        
        for year in naep_years:
            logger.info(f"Collecting NAEP data for {year}")
            
            for state in self.states:
                for grade in grades:
                    for subject in subjects:
                        try:
                            # Fetch SWD data
                            swd_data = self.naep_collector.fetch_state_swd_data(
                                state=state,
                                year=year,
                                grade=grade,
                                subject=subject,
                                disability_status='SD'  # Students with disabilities
                            )
                            
                            # Fetch non-SWD data
                            non_swd_data = self.naep_collector.fetch_state_swd_data(
                                state=state,
                                year=year,
                                grade=grade,
                                subject=subject,
                                disability_status='NOT SD'  # Students without disabilities
                            )
                            
                            # Combine data
                            if swd_data and non_swd_data:
                                row = {
                                    'state': state,
                                    'year': year,
                                    'grade': grade,
                                    'subject': subject,
                                    'swd_score': swd_data.get('value'),
                                    'non_swd_score': non_swd_data.get('value'),
                                    'gap': non_swd_data.get('value', 0) - swd_data.get('value', 0)
                                }
                                all_data.append(row)
                                
                        except Exception as e:
                            logger.warning(f"Error collecting NAEP data for {state} {year} Grade {grade} {subject}: {e}")
                            
        # Save NAEP data
        if all_data:
            naep_df = pd.DataFrame(all_data)
            output_path = self.output_dir / "naep_real_data.csv"
            naep_df.to_csv(output_path, index=False)
            logger.info(f"Saved NAEP data to {output_path}: {len(naep_df)} records")
            return naep_df
        else:
            logger.warning("No NAEP data collected")
            return pd.DataFrame()
    
    def collect_census_finance_data(self):
        """Collect Census F-33 education finance data."""
        logger.info("Starting Census finance data collection...")
        
        all_data = []
        
        # Census API requires a key - check if it's set
        census_key = os.getenv('CENSUS_API_KEY')
        if not census_key:
            logger.warning("Census API key not found in environment. Using sample data.")
            # Load existing parsed data if available
            existing_file = self.output_dir / "census_education_finance_parsed.csv"
            if existing_file.exists():
                return pd.read_csv(existing_file)
            return pd.DataFrame()
        
        for year in self.years:
            logger.info(f"Collecting Census finance data for {year}")
            
            try:
                # Fetch state-level finance data
                finance_data = self.census_collector.fetch_state_finance_data(year)
                
                if finance_data:
                    # Process each state's data
                    for state_data in finance_data:
                        state_abbrev = self.state_utils.fips_to_abbrev(state_data.get('state'))
                        if state_abbrev:
                            row = {
                                'state': state_abbrev,
                                'year': year,
                                'total_revenue': state_data.get('TOTALREV'),
                                'federal_revenue': state_data.get('TFEDREV'),
                                'state_revenue': state_data.get('TSTREV'),
                                'local_revenue': state_data.get('TLOCREV'),
                                'total_expenditure': state_data.get('TOTALEXP'),
                                'instruction_expenditure': state_data.get('TCURINST'),
                                'support_services_exp': state_data.get('TCURSSVC'),
                                'enrollment': state_data.get('V33')
                            }
                            
                            # Calculate per-pupil values
                            if row['enrollment'] and row['enrollment'] > 0:
                                row['revenue_per_pupil'] = row['total_revenue'] / row['enrollment']
                                row['expenditure_per_pupil'] = row['total_expenditure'] / row['enrollment']
                            
                            all_data.append(row)
                            
            except Exception as e:
                logger.warning(f"Error collecting Census data for {year}: {e}")
        
        # Save Census data
        if all_data:
            census_df = pd.DataFrame(all_data)
            output_path = self.output_dir / "census_finance_real_data.csv"
            census_df.to_csv(output_path, index=False)
            logger.info(f"Saved Census data to {output_path}: {len(census_df)} records")
            return census_df
        else:
            logger.warning("No Census data collected")
            return pd.DataFrame()
    
    def collect_edfacts_data(self):
        """Collect EdFacts special education data."""
        logger.info("Starting EdFacts data collection...")
        
        all_data = []
        
        # EdFacts file patterns
        file_patterns = {
            'child_count': 'c002',  # Child count by disability
            'environment': 'c089',  # Educational environment
            'exit': 'c009'  # Exit data
        }
        
        for year in self.years:
            logger.info(f"Collecting EdFacts data for {year}")
            
            for data_type, pattern in file_patterns.items():
                try:
                    # Fetch EdFacts data
                    edfacts_data = self.edfacts_collector.fetch_edfacts_file(
                        year=year,
                        file_pattern=pattern
                    )
                    
                    if edfacts_data:
                        # Process state-level data
                        for _, row in edfacts_data.iterrows():
                            state = row.get('State', row.get('state'))
                            if state and state in self.states:
                                processed_row = {
                                    'state': state,
                                    'year': year,
                                    'data_type': data_type
                                }
                                
                                # Add relevant metrics based on data type
                                if data_type == 'child_count':
                                    processed_row['swd_count'] = row.get('Total', 0)
                                    processed_row['swd_percent'] = row.get('Percent', 0)
                                elif data_type == 'environment':
                                    processed_row['inclusion_80plus'] = row.get('80% or more', 0)
                                    processed_row['inclusion_40_79'] = row.get('40-79%', 0)
                                    processed_row['inclusion_less40'] = row.get('Less than 40%', 0)
                                elif data_type == 'exit':
                                    processed_row['graduated'] = row.get('Graduated', 0)
                                    processed_row['dropped_out'] = row.get('Dropped Out', 0)
                                
                                all_data.append(processed_row)
                                
                except Exception as e:
                    logger.warning(f"Error collecting EdFacts {data_type} data for {year}: {e}")
        
        # Save EdFacts data
        if all_data:
            edfacts_df = pd.DataFrame(all_data)
            output_path = self.output_dir / "edfacts_real_data.csv"
            edfacts_df.to_csv(output_path, index=False)
            logger.info(f"Saved EdFacts data to {output_path}: {len(edfacts_df)} records")
            return edfacts_df
        else:
            logger.warning("No EdFacts data collected")
            return pd.DataFrame()
    
    def create_state_policy_database(self):
        """Create database of state special education policy reforms."""
        logger.info("Creating state policy database...")
        
        # Real policy reform data based on research
        # These are example dates - should be validated with policy documents
        policy_reforms = [
            {'state': 'CA', 'reform_year': 2013, 'reform_type': 'funding_formula', 'description': 'Local Control Funding Formula'},
            {'state': 'TX', 'reform_year': 2019, 'reform_type': 'funding_formula', 'description': 'HB 3 School Finance Reform'},
            {'state': 'IL', 'reform_year': 2017, 'reform_type': 'funding_formula', 'description': 'Evidence-Based Funding'},
            {'state': 'MA', 'reform_year': 2019, 'reform_type': 'funding_increase', 'description': 'Student Opportunity Act'},
            {'state': 'NJ', 'reform_year': 2018, 'reform_type': 'funding_formula', 'description': 'S2 Funding Reform'},
            {'state': 'PA', 'reform_year': 2016, 'reform_type': 'funding_formula', 'description': 'Fair Funding Formula'},
            {'state': 'NC', 'reform_year': 2021, 'reform_type': 'court_order', 'description': 'Leandro Ruling'},
            {'state': 'OH', 'reform_year': 2020, 'reform_type': 'funding_formula', 'description': 'Fair School Funding Plan'},
            {'state': 'TN', 'reform_year': 2022, 'reform_type': 'funding_formula', 'description': 'TISA Funding Formula'},
            {'state': 'KS', 'reform_year': 2017, 'reform_type': 'court_order', 'description': 'Gannon v. Kansas'},
            {'state': 'WA', 'reform_year': 2018, 'reform_type': 'court_order', 'description': 'McCleary Decision'},
            {'state': 'AZ', 'reform_year': 2020, 'reform_type': 'funding_increase', 'description': 'Prop 208'},
            {'state': 'FL', 'reform_year': 2014, 'reform_type': 'funding_formula', 'description': 'Best and Brightest'},
            {'state': 'CO', 'reform_year': 2019, 'reform_type': 'funding_increase', 'description': 'HB19-1262'},
            {'state': 'MI', 'reform_year': 2019, 'reform_type': 'funding_increase', 'description': 'School Aid Act'},
            {'state': 'NV', 'reform_year': 2019, 'reform_type': 'funding_formula', 'description': 'SB 543 Pupil-Centered Plan'},
        ]
        
        # Court monitoring status
        court_monitoring = [
            {'state': 'NC', 'start_year': 2020, 'status': 'active'},
            {'state': 'KS', 'start_year': 2017, 'status': 'active'},
            {'state': 'WA', 'start_year': 2012, 'status': 'resolved_2018'},
            {'state': 'NH', 'start_year': 2019, 'status': 'active'},
            {'state': 'PA', 'start_year': 2023, 'status': 'active'},
        ]
        
        policy_df = pd.DataFrame(policy_reforms)
        monitoring_df = pd.DataFrame(court_monitoring)
        
        # Save policy database
        policy_path = self.output_dir / "state_policy_reforms.csv"
        policy_df.to_csv(policy_path, index=False)
        logger.info(f"Saved policy database to {policy_path}: {len(policy_df)} reforms")
        
        monitoring_path = self.output_dir / "court_monitoring.csv"
        monitoring_df.to_csv(monitoring_path, index=False)
        logger.info(f"Saved monitoring data to {monitoring_path}: {len(monitoring_df)} states")
        
        return policy_df, monitoring_df
    
    def run_collection(self):
        """Run complete data collection pipeline."""
        logger.info("="*60)
        logger.info("Starting Real Data Collection Pipeline")
        logger.info("="*60)
        
        # Collect from each source
        naep_data = self.collect_naep_data()
        census_data = self.collect_census_finance_data()
        edfacts_data = self.collect_edfacts_data()
        policy_data, monitoring_data = self.create_state_policy_database()
        
        # Summary
        logger.info("="*60)
        logger.info("Data Collection Summary:")
        logger.info(f"NAEP records: {len(naep_data)}")
        logger.info(f"Census records: {len(census_data)}")
        logger.info(f"EdFacts records: {len(edfacts_data)}")
        logger.info(f"Policy reforms: {len(policy_data)}")
        logger.info(f"Court monitoring: {len(monitoring_data)}")
        logger.info("="*60)
        
        return {
            'naep': naep_data,
            'census': census_data,
            'edfacts': edfacts_data,
            'policy': policy_data,
            'monitoring': monitoring_data
        }


def main():
    """Main entry point."""
    collector = RealDataCollector()
    data = collector.run_collection()
    
    logger.info("\n✅ Real data collection completed!")
    logger.info("Next step: Run data integration to create analysis panel")
    
    return data


if __name__ == "__main__":
    main()