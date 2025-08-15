#!/usr/bin/env python3
"""
Real Data Integration Pipeline
Combines collected data sources into analysis-ready panel dataset
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collection import StateUtils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIntegrator:
    """Integrates multiple data sources into analysis panel."""
    
    def __init__(self):
        """Initialize data integrator."""
        self.state_utils = StateUtils()
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.final_dir = Path("data/final")
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
    def load_naep_data(self):
        """Load and process NAEP achievement data."""
        logger.info("Loading NAEP data...")
        
        # Try real data first, fall back to existing
        real_file = self.raw_dir / "naep_real_data.csv"
        existing_file = self.raw_dir / "naep_state_swd_data.csv"
        
        if real_file.exists():
            naep_df = pd.read_csv(real_file)
        elif existing_file.exists():
            naep_df = pd.read_csv(existing_file)
        else:
            logger.warning("No NAEP data found")
            return pd.DataFrame()
        
        # Reshape to wide format
        if not naep_df.empty:
            # Pivot to get columns for each grade/subject combination
            naep_wide = naep_df.pivot_table(
                index=['state', 'year'],
                columns=['grade', 'subject'],
                values=['swd_score', 'non_swd_score', 'gap'],
                aggfunc='mean'
            )
            
            # Flatten column names
            naep_wide.columns = ['_'.join(map(str, col)).strip() 
                                 for col in naep_wide.columns.values]
            naep_wide = naep_wide.reset_index()
            
            # Rename columns to match analysis format
            rename_map = {
                'swd_score_4_mathematics': 'math_grade4_swd_score',
                'non_swd_score_4_mathematics': 'math_grade4_non_swd_score',
                'gap_4_mathematics': 'math_grade4_gap',
                'swd_score_8_mathematics': 'math_grade8_swd_score',
                'non_swd_score_8_mathematics': 'math_grade8_non_swd_score',
                'gap_8_mathematics': 'math_grade8_gap',
                'swd_score_4_reading': 'reading_grade4_swd_score',
                'non_swd_score_4_reading': 'reading_grade4_non_swd_score',
                'gap_4_reading': 'reading_grade4_gap',
                'swd_score_8_reading': 'reading_grade8_swd_score',
                'non_swd_score_8_reading': 'reading_grade8_non_swd_score',
                'gap_8_reading': 'reading_grade8_gap'
            }
            
            for old_name, new_name in rename_map.items():
                if old_name in naep_wide.columns:
                    naep_wide.rename(columns={old_name: new_name}, inplace=True)
            
            logger.info(f"Processed NAEP data: {len(naep_wide)} state-years")
            return naep_wide
        
        return pd.DataFrame()
    
    def load_census_data(self):
        """Load and process Census finance data."""
        logger.info("Loading Census finance data...")
        
        # Try real data first, fall back to existing
        real_file = self.raw_dir / "census_finance_real_data.csv"
        existing_file = self.raw_dir / "census_education_finance_parsed.csv"
        
        if real_file.exists():
            census_df = pd.read_csv(real_file)
        elif existing_file.exists():
            census_df = pd.read_csv(existing_file)
        else:
            logger.warning("No Census data found")
            return pd.DataFrame()
        
        # Calculate per-pupil values if not already present
        if not census_df.empty and 'enrollment' in census_df.columns:
            finance_cols = ['total_revenue', 'federal_revenue', 'state_revenue', 
                          'local_revenue', 'total_expenditure', 'instruction_expenditure']
            
            for col in finance_cols:
                if col in census_df.columns:
                    per_pupil_col = f"{col}_per_pupil"
                    if per_pupil_col not in census_df.columns:
                        census_df[per_pupil_col] = (
                            census_df[col] / census_df['enrollment']
                        ).replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Processed Census data: {len(census_df)} state-years")
        return census_df
    
    def load_edfacts_data(self):
        """Load and process EdFacts special education data."""
        logger.info("Loading EdFacts data...")
        
        # Try real data first
        real_file = self.raw_dir / "edfacts_real_data.csv"
        
        if real_file.exists():
            edfacts_df = pd.read_csv(real_file)
            
            # Reshape to wide format
            if not edfacts_df.empty:
                # Pivot by data type
                edfacts_wide = edfacts_df.pivot_table(
                    index=['state', 'year'],
                    columns='data_type',
                    values=[col for col in edfacts_df.columns 
                           if col not in ['state', 'year', 'data_type']],
                    aggfunc='first'
                )
                
                # Flatten column names
                edfacts_wide.columns = ['_'.join(col).strip() 
                                       for col in edfacts_wide.columns.values]
                edfacts_wide = edfacts_wide.reset_index()
                
                # Calculate inclusion rate
                if 'inclusion_80plus_environment' in edfacts_wide.columns:
                    edfacts_wide['inclusion_rate'] = edfacts_wide['inclusion_80plus_environment']
                
                logger.info(f"Processed EdFacts data: {len(edfacts_wide)} state-years")
                return edfacts_wide
        
        logger.warning("No EdFacts data found")
        return pd.DataFrame()
    
    def load_policy_data(self):
        """Load state policy reform data."""
        logger.info("Loading policy reform data...")
        
        policy_file = self.raw_dir / "state_policy_reforms.csv"
        monitoring_file = self.raw_dir / "court_monitoring.csv"
        
        policy_df = pd.DataFrame()
        monitoring_df = pd.DataFrame()
        
        if policy_file.exists():
            policy_df = pd.read_csv(policy_file)
            logger.info(f"Loaded {len(policy_df)} policy reforms")
        
        if monitoring_file.exists():
            monitoring_df = pd.read_csv(monitoring_file)
            logger.info(f"Loaded {len(monitoring_df)} court monitoring records")
        
        return policy_df, monitoring_df
    
    def create_panel_structure(self):
        """Create balanced panel structure."""
        logger.info("Creating panel structure...")
        
        # All states and years
        states = self.state_utils.get_all_states()
        years = list(range(2009, 2023))
        
        # Create all state-year combinations
        panel_index = []
        for state in states:
            for year in years:
                panel_index.append({'state': state, 'year': year})
        
        panel_df = pd.DataFrame(panel_index)
        
        # Add time variables
        panel_df['time_trend'] = panel_df['year'] - 2009
        panel_df['time_trend_sq'] = panel_df['time_trend'] ** 2
        panel_df['post_covid'] = (panel_df['year'] >= 2020).astype(int)
        panel_df['post_2015'] = (panel_df['year'] >= 2015).astype(int)
        
        # Add regional and political variables
        region_map = {
            'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
            'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
            'PA': 'Northeast',
            'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
            'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
            'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
            'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South',
            'NC': 'South', 'SC': 'South', 'VA': 'South', 'DC': 'South',
            'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South',
            'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
            'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
            'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
            'HI': 'West', 'OR': 'West', 'WA': 'West'
        }
        
        panel_df['region'] = panel_df['state'].map(region_map)
        
        logger.info(f"Created panel structure: {len(panel_df)} observations")
        return panel_df
    
    def add_treatment_variables(self, panel_df, policy_df, monitoring_df):
        """Add treatment and policy variables."""
        logger.info("Adding treatment variables...")
        
        # Initialize treatment variables
        panel_df['treated'] = 0
        panel_df['post_treatment'] = 0
        panel_df['years_since_treatment'] = np.nan
        panel_df['reform_year'] = np.nan
        panel_df['under_monitoring'] = 0
        panel_df['court_ordered'] = 0.0
        
        # Add policy reforms
        if not policy_df.empty:
            for _, reform in policy_df.iterrows():
                state = reform['state']
                reform_year = reform['reform_year']
                
                # Mark treatment status
                mask = panel_df['state'] == state
                panel_df.loc[mask, 'treated'] = 1
                panel_df.loc[mask, 'reform_year'] = reform_year
                
                # Post-treatment indicator
                post_mask = mask & (panel_df['year'] >= reform_year)
                panel_df.loc[post_mask, 'post_treatment'] = 1
                
                # Years since treatment
                panel_df.loc[post_mask, 'years_since_treatment'] = (
                    panel_df.loc[post_mask, 'year'] - reform_year
                )
                
                # Court-ordered reforms
                if reform.get('reform_type') == 'court_order':
                    panel_df.loc[mask, 'court_ordered'] = 1.0
        
        # Add court monitoring
        if not monitoring_df.empty:
            for _, monitoring in monitoring_df.iterrows():
                state = monitoring['state']
                start_year = monitoring['start_year']
                
                if monitoring['status'] == 'active':
                    mask = (panel_df['state'] == state) & (panel_df['year'] >= start_year)
                    panel_df.loc[mask, 'under_monitoring'] = 1
                elif 'resolved' in monitoring['status']:
                    end_year = int(monitoring['status'].split('_')[1])
                    mask = (panel_df['state'] == state) & \
                           (panel_df['year'] >= start_year) & \
                           (panel_df['year'] <= end_year)
                    panel_df.loc[mask, 'under_monitoring'] = 1
        
        # Create lead and lag indicators
        for i in range(1, 6):
            panel_df[f'lead_{i}'] = 0
            panel_df[f'lag_{i}'] = 0
            
            # Leads (pre-treatment)
            lead_mask = panel_df['years_since_treatment'] == -i
            panel_df.loc[lead_mask, f'lead_{i}'] = 1
            
            # Lags (post-treatment)
            lag_mask = panel_df['years_since_treatment'] == i
            panel_df.loc[lag_mask, f'lag_{i}'] = 1
        
        # Treatment intensity (continuous measure)
        panel_df['treatment_intensity'] = panel_df['post_treatment'] * \
                                         panel_df['years_since_treatment'].fillna(0)
        
        logger.info(f"Added treatment variables: {panel_df['treated'].sum()} treated state-years")
        return panel_df
    
    def merge_all_data(self, panel_df, naep_df, census_df, edfacts_df):
        """Merge all data sources into final panel."""
        logger.info("Merging all data sources...")
        
        # Merge NAEP data
        if not naep_df.empty:
            panel_df = panel_df.merge(naep_df, on=['state', 'year'], how='left')
            logger.info(f"Merged NAEP data: {naep_df.columns.tolist()}")
        
        # Merge Census data
        if not census_df.empty:
            census_cols = [col for col in census_df.columns if col not in panel_df.columns or col in ['state', 'year']]
            panel_df = panel_df.merge(census_df[census_cols], on=['state', 'year'], how='left')
            logger.info(f"Merged Census data: {census_cols}")
        
        # Merge EdFacts data
        if not edfacts_df.empty:
            edfacts_cols = [col for col in edfacts_df.columns if col not in panel_df.columns or col in ['state', 'year']]
            panel_df = panel_df.merge(edfacts_df[edfacts_cols], on=['state', 'year'], how='left')
            logger.info(f"Merged EdFacts data: {edfacts_cols}")
        
        # Add fixed effects indicators
        panel_df['state_fe'] = panel_df['state']
        panel_df['year_fe'] = panel_df['year'].astype(str)
        
        # Add political lean (simplified - would need real data)
        blue_states = ['CA', 'NY', 'IL', 'MA', 'CT', 'NJ', 'WA', 'OR', 'VT', 'HI', 'MD', 'DE', 'RI']
        panel_df['political_lean'] = panel_df['state'].apply(
            lambda x: 'Democrat' if x in blue_states else 'Republican'
        )
        
        # Add unionization (simplified - would need real data)
        high_union = ['NY', 'CA', 'IL', 'PA', 'OH', 'MI', 'NJ', 'WA', 'OR', 'CT', 'MA']
        panel_df['unionized'] = panel_df['state'].apply(
            lambda x: 'High' if x in high_union else 'Low'
        )
        
        logger.info(f"Final panel shape: {panel_df.shape}")
        return panel_df
    
    def validate_panel(self, panel_df):
        """Validate the final panel dataset."""
        logger.info("Validating panel dataset...")
        
        # Check for duplicates
        duplicates = panel_df.duplicated(subset=['state', 'year']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate state-year observations")
        
        # Check completeness
        expected_obs = len(self.state_utils.get_all_states()) * len(range(2009, 2023))
        actual_obs = len(panel_df)
        logger.info(f"Panel completeness: {actual_obs}/{expected_obs} observations")
        
        # Check treatment assignment
        treated_states = panel_df[panel_df['treated'] == 1]['state'].nunique()
        logger.info(f"Treated states: {treated_states}")
        
        # Check outcome coverage
        outcome_cols = ['math_grade4_gap', 'math_grade8_gap', 'reading_grade4_gap', 'reading_grade8_gap']
        for col in outcome_cols:
            if col in panel_df.columns:
                coverage = panel_df[col].notna().sum() / len(panel_df) * 100
                logger.info(f"{col} coverage: {coverage:.1f}%")
        
        return panel_df
    
    def run_integration(self):
        """Run complete data integration pipeline."""
        logger.info("="*60)
        logger.info("Starting Real Data Integration Pipeline")
        logger.info("="*60)
        
        # Load all data sources
        naep_df = self.load_naep_data()
        census_df = self.load_census_data()
        edfacts_df = self.load_edfacts_data()
        policy_df, monitoring_df = self.load_policy_data()
        
        # Create panel structure
        panel_df = self.create_panel_structure()
        
        # Add treatment variables
        panel_df = self.add_treatment_variables(panel_df, policy_df, monitoring_df)
        
        # Merge all data
        panel_df = self.merge_all_data(panel_df, naep_df, census_df, edfacts_df)
        
        # Validate
        panel_df = self.validate_panel(panel_df)
        
        # Save final panel
        output_path = self.final_dir / "analysis_panel_real.csv"
        panel_df.to_csv(output_path, index=False)
        logger.info(f"Saved integrated panel to {output_path}")
        
        # Save backup of old panel
        old_panel = self.final_dir / "analysis_panel.csv"
        if old_panel.exists():
            backup_path = self.final_dir / "analysis_panel_mock.csv"
            old_panel.rename(backup_path)
            logger.info(f"Backed up old panel to {backup_path}")
        
        # Rename new panel to standard name
        output_path.rename(self.final_dir / "analysis_panel.csv")
        
        logger.info("="*60)
        logger.info("Integration Summary:")
        logger.info(f"Total observations: {len(panel_df)}")
        logger.info(f"States: {panel_df['state'].nunique()}")
        logger.info(f"Years: {panel_df['year'].min()}-{panel_df['year'].max()}")
        logger.info(f"Treated states: {panel_df[panel_df['treated']==1]['state'].nunique()}")
        logger.info(f"Columns: {len(panel_df.columns)}")
        logger.info("="*60)
        
        return panel_df


def main():
    """Main entry point."""
    integrator = DataIntegrator()
    panel_df = integrator.run_integration()
    
    logger.info("\n✅ Real data integration completed!")
    logger.info("Next step: Run analysis pipeline with real data")
    
    return panel_df


if __name__ == "__main__":
    main()