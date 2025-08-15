#!/usr/bin/env python3
"""
Use Existing Real Data
Process and integrate the real data that's already been collected
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


def process_existing_naep_data():
    """Process the existing NAEP data into analysis format."""
    logger.info("Processing existing NAEP data...")
    
    naep_file = Path("data/raw/naep_state_swd_data.csv")
    if not naep_file.exists():
        logger.warning("No NAEP data found")
        return pd.DataFrame()
    
    naep_df = pd.read_csv(naep_file)
    logger.info(f"Loaded NAEP data: {len(naep_df)} records")
    
    # Reshape data to wide format
    naep_wide = []
    
    for state in naep_df['state'].unique():
        for year in naep_df['year'].unique():
            for grade in naep_df['grade'].unique():
                for subject in naep_df['subject'].unique():
                    
                    # Get SWD and non-SWD scores
                    swd_data = naep_df[
                        (naep_df['state'] == state) & 
                        (naep_df['year'] == year) & 
                        (naep_df['grade'] == grade) & 
                        (naep_df['subject'] == subject) & 
                        (naep_df['disability_status'] == 'SWD')
                    ]
                    
                    non_swd_data = naep_df[
                        (naep_df['state'] == state) & 
                        (naep_df['year'] == year) & 
                        (naep_df['grade'] == grade) & 
                        (naep_df['subject'] == subject) & 
                        (naep_df['disability_status'] == 'non-SWD')
                    ]
                    
                    if not swd_data.empty and not non_swd_data.empty:
                        swd_score = swd_data['mean_score'].values[0]
                        non_swd_score = non_swd_data['mean_score'].values[0]
                        gap = non_swd_score - swd_score
                        
                        # Create column names
                        prefix = f"{subject}_grade{grade}"
                        
                        row = {
                            'state': state,
                            'year': year,
                            f'{prefix}_swd_score': swd_score,
                            f'{prefix}_non_swd_score': non_swd_score,
                            f'{prefix}_gap': gap
                        }
                        naep_wide.append(row)
    
    if naep_wide:
        naep_df_wide = pd.DataFrame(naep_wide)
        # Group by state/year and aggregate (in case of duplicates)
        naep_final = naep_df_wide.groupby(['state', 'year']).mean().reset_index()
        logger.info(f"Processed NAEP data: {len(naep_final)} state-years")
        return naep_final
    
    return pd.DataFrame()


def process_existing_census_data():
    """Process the existing Census finance data."""
    logger.info("Processing existing Census data...")
    
    census_file = Path("data/raw/census_education_finance_parsed.csv")
    if not census_file.exists():
        logger.warning("No Census data found")
        return pd.DataFrame()
    
    census_df = pd.read_csv(census_file)
    logger.info(f"Loaded Census data: {len(census_df)} records")
    
    # Clean and standardize column names
    if 'State' in census_df.columns:
        census_df['state'] = census_df['State']
    if 'Year' in census_df.columns:
        census_df['year'] = census_df['Year']
    
    # Keep only relevant columns
    finance_cols = ['state', 'year']
    for col in census_df.columns:
        if any(keyword in col.lower() for keyword in ['revenue', 'expenditure', 'pupil']):
            finance_cols.append(col)
    
    census_final = census_df[finance_cols]
    logger.info(f"Processed Census data: {len(census_final)} state-years")
    return census_final


def create_policy_database():
    """Create real policy reform database."""
    logger.info("Creating policy reform database...")
    
    # Real policy reforms based on research
    policy_reforms = [
        # Major funding formula reforms
        {'state': 'CA', 'reform_year': 2013, 'reform_type': 'funding_formula', 
         'description': 'Local Control Funding Formula (LCFF)', 'major_change': True},
        {'state': 'TX', 'reform_year': 2019, 'reform_type': 'funding_formula', 
         'description': 'HB 3 School Finance Reform', 'major_change': True},
        {'state': 'IL', 'reform_year': 2017, 'reform_type': 'funding_formula', 
         'description': 'Evidence-Based Funding for Student Success Act', 'major_change': True},
        {'state': 'MA', 'reform_year': 2019, 'reform_type': 'funding_increase', 
         'description': 'Student Opportunity Act', 'major_change': True},
        {'state': 'NJ', 'reform_year': 2018, 'reform_type': 'funding_formula', 
         'description': 'S2 School Funding Reform Act', 'major_change': True},
        {'state': 'PA', 'reform_year': 2016, 'reform_type': 'funding_formula', 
         'description': 'Fair Funding Formula', 'major_change': True},
        {'state': 'NC', 'reform_year': 2021, 'reform_type': 'court_order', 
         'description': 'Leandro v. State comprehensive remedial plan', 'major_change': True},
        {'state': 'OH', 'reform_year': 2021, 'reform_type': 'funding_formula', 
         'description': 'Fair School Funding Plan', 'major_change': True},
        {'state': 'TN', 'reform_year': 2022, 'reform_type': 'funding_formula', 
         'description': 'Tennessee Investment in Student Achievement (TISA)', 'major_change': True},
        {'state': 'KS', 'reform_year': 2017, 'reform_type': 'court_order', 
         'description': 'Gannon v. Kansas funding increases', 'major_change': True},
        {'state': 'WA', 'reform_year': 2018, 'reform_type': 'court_order', 
         'description': 'McCleary v. State full funding', 'major_change': True},
        {'state': 'AZ', 'reform_year': 2020, 'reform_type': 'funding_increase', 
         'description': 'Proposition 208 education tax', 'major_change': True},
        {'state': 'FL', 'reform_year': 2014, 'reform_type': 'accountability', 
         'description': 'Best and Brightest Teacher Program', 'major_change': False},
        {'state': 'CO', 'reform_year': 2019, 'reform_type': 'funding_increase', 
         'description': 'HB19-1262 school finance modernization', 'major_change': True},
        {'state': 'MI', 'reform_year': 2019, 'reform_type': 'funding_increase', 
         'description': 'Per-pupil funding increases', 'major_change': False},
        {'state': 'NV', 'reform_year': 2019, 'reform_type': 'funding_formula', 
         'description': 'SB 543 Pupil-Centered Funding Plan', 'major_change': True},
    ]
    
    # Court monitoring cases
    court_monitoring = [
        {'state': 'NC', 'start_year': 2020, 'status': 'active', 'case': 'Leandro v. State'},
        {'state': 'KS', 'start_year': 2014, 'status': 'resolved_2019', 'case': 'Gannon v. Kansas'},
        {'state': 'WA', 'start_year': 2012, 'status': 'resolved_2018', 'case': 'McCleary v. State'},
        {'state': 'NH', 'start_year': 2019, 'status': 'active', 'case': 'ConVal v. State'},
        {'state': 'PA', 'start_year': 2023, 'status': 'active', 'case': 'William Penn v. Pennsylvania'},
    ]
    
    policy_df = pd.DataFrame(policy_reforms)
    monitoring_df = pd.DataFrame(court_monitoring)
    
    logger.info(f"Created policy database: {len(policy_df)} reforms, {len(monitoring_df)} court cases")
    return policy_df, monitoring_df


def create_real_analysis_panel():
    """Create analysis panel from real data."""
    logger.info("Creating analysis panel from real data...")
    
    # Load processed data
    naep_df = process_existing_naep_data()
    census_df = process_existing_census_data()
    policy_df, monitoring_df = create_policy_database()
    
    # Create base panel structure
    state_utils = StateUtils()
    states = state_utils.get_all_states()
    years = list(range(2009, 2023))
    
    panel_data = []
    for state in states:
        for year in years:
            panel_data.append({'state': state, 'year': year})
    
    panel_df = pd.DataFrame(panel_data)
    logger.info(f"Created base panel: {len(panel_df)} observations")
    
    # Add time variables
    panel_df['time_trend'] = panel_df['year'] - 2009
    panel_df['time_trend_sq'] = panel_df['time_trend'] ** 2
    panel_df['post_covid'] = (panel_df['year'] >= 2020).astype(int)
    panel_df['post_2015'] = (panel_df['year'] >= 2015).astype(int)
    
    # Add regional variables
    region_map = {
        'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
        'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
        'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest', 
        'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest', 
        'ND': 'Midwest', 'SD': 'Midwest',
        'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South', 'SC': 'South', 
        'VA': 'South', 'DC': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South', 
        'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
        'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West', 
        'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West'
    }
    panel_df['region'] = panel_df['state'].map(region_map)
    
    # Add political lean (simplified)
    blue_states = ['CA', 'NY', 'IL', 'MA', 'CT', 'NJ', 'WA', 'OR', 'VT', 'HI', 'MD', 'DE', 'RI', 'DC']
    panel_df['political_lean'] = panel_df['state'].apply(
        lambda x: 'Democrat' if x in blue_states else 'Republican'
    )
    
    # Add unionization (simplified)
    high_union = ['NY', 'CA', 'IL', 'PA', 'OH', 'MI', 'NJ', 'WA', 'OR', 'CT', 'MA', 'WI', 'MN']
    panel_df['unionized'] = panel_df['state'].apply(
        lambda x: 'High' if x in high_union else 'Low'
    )
    
    # Add treatment variables
    panel_df['treated'] = 0
    panel_df['post_treatment'] = 0
    panel_df['years_since_treatment'] = np.nan
    panel_df['reform_year'] = np.nan
    panel_df['under_monitoring'] = 0
    panel_df['court_ordered'] = 0.0
    
    # Process policy reforms
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
    
    # Add lead and lag indicators for event study
    for i in range(1, 6):
        panel_df[f'lead_{i}'] = 0
        panel_df[f'lag_{i}'] = 0
        
        # Leads (pre-treatment)
        lead_mask = panel_df['years_since_treatment'] == -i
        panel_df.loc[lead_mask, f'lead_{i}'] = 1
        
        # Lags (post-treatment)
        lag_mask = panel_df['years_since_treatment'] == i
        panel_df.loc[lag_mask, f'lag_{i}'] = 1
    
    # Treatment intensity
    panel_df['treatment_intensity'] = panel_df['post_treatment'] * \
                                     panel_df['years_since_treatment'].fillna(0)
    
    # Merge NAEP data
    if not naep_df.empty:
        panel_df = panel_df.merge(naep_df, on=['state', 'year'], how='left')
        logger.info("Merged NAEP data")
    
    # Merge Census data
    if not census_df.empty:
        # Clean column names for merge
        census_clean = census_df.copy()
        for col in census_clean.columns:
            if col not in ['state', 'year']:
                # Clean up column names
                new_col = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
                census_clean.rename(columns={col: new_col}, inplace=True)
        
        panel_df = panel_df.merge(census_clean, on=['state', 'year'], how='left')
        logger.info("Merged Census data")
    
    # Add fixed effects
    panel_df['state_fe'] = panel_df['state']
    panel_df['year_fe'] = panel_df['year'].astype(str)
    
    # Clean up and validate
    panel_df = panel_df.drop_duplicates(subset=['state', 'year'])
    
    logger.info(f"Final panel shape: {panel_df.shape}")
    logger.info(f"Treated states: {panel_df[panel_df['treated']==1]['state'].nunique()}")
    
    # Check outcome coverage
    outcome_cols = [col for col in panel_df.columns if 'gap' in col or 'score' in col]
    for col in outcome_cols:
        if col in panel_df.columns:
            coverage = panel_df[col].notna().sum() / len(panel_df) * 100
            logger.info(f"{col} coverage: {coverage:.1f}%")
    
    return panel_df


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Creating Analysis Panel from Existing Real Data")
    logger.info("="*60)
    
    # Create panel from real data
    panel_df = create_real_analysis_panel()
    
    # Save files
    final_dir = Path("data/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup existing mock panel
    old_panel = final_dir / "analysis_panel.csv"
    if old_panel.exists():
        backup_path = final_dir / "analysis_panel_mock.csv"
        old_panel.rename(backup_path)
        logger.info(f"Backed up mock panel to {backup_path}")
    
    # Save new real panel
    output_path = final_dir / "analysis_panel.csv"
    panel_df.to_csv(output_path, index=False)
    logger.info(f"Saved real data panel to {output_path}")
    
    # Summary
    logger.info("="*60)
    logger.info("Real Data Panel Summary:")
    logger.info(f"Total observations: {len(panel_df)}")
    logger.info(f"States: {panel_df['state'].nunique()}")
    logger.info(f"Years: {panel_df['year'].min()}-{panel_df['year'].max()}")
    logger.info(f"Treated states: {panel_df[panel_df['treated']==1]['state'].nunique()}")
    logger.info(f"Reform years: {sorted(panel_df[panel_df['treated']==1]['reform_year'].dropna().unique())}")
    logger.info(f"Columns: {len(panel_df.columns)}")
    
    # Show some key outcome coverage
    key_outcomes = ['mathematics_grade4_gap', 'mathematics_grade8_gap', 'reading_grade4_gap', 'reading_grade8_gap']
    for outcome in key_outcomes:
        if outcome in panel_df.columns:
            coverage = panel_df[outcome].notna().sum()
            logger.info(f"{outcome}: {coverage} observations")
    
    logger.info("="*60)
    logger.info("✅ Real data integration completed!")
    logger.info("Next: Run analysis pipeline with real data")
    
    return panel_df


if __name__ == "__main__":
    main()