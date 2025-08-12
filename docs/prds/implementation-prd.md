# Implementation PRD: Complete Development and Automation Framework

## Document Purpose
**Audience**: Software developers, project managers, research engineers  
**Scope**: Complete technical implementation guide and automation pipeline  
**Status**: ✅ COMPLETED  
**Related Documents**: All other PRDs serve as input specifications

---

## 1. Project Architecture Overview

### 1.1 System Requirements

**Environment**:
- Python >=3.12 (confirmed working dependency setup)
- Operating System: Linux/macOS/Windows with WSL
- Memory: 8GB minimum, 16GB recommended
- Storage: 5GB for complete dataset and outputs
- Internet: Stable connection for API data collection

**Key Dependencies** (per CLAUDE.md):
```python
# Core packages
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Econometric analysis  
statsmodels>=0.14.0
linearmodels>=5.0.0
econtools>=0.3.0

# Manual DiD implementation (instead of did package)
# COVID and policy analysis tools

# Data collection
requests>=2.28.0
beautifulsoup4>=4.11.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
```

### 1.2 Directory Structure

```
state-sped-policy-eval/
├── main.py                    # Entry point (current)
├── run_analysis.py           # Master pipeline script (to be created)
├── requirements.txt          # Python dependencies (managed via pyproject.toml)
├── pyproject.toml           # Project configuration (exists, updated)
├── CLAUDE.md                # Project instructions (exists, updated)
├── README.md                # Project overview (to be updated)
├── docs/
│   ├── prds/               # Product Requirements Documents (created)
│   ├── codebook.md         # Variable definitions (to be created)
│   └── methods.md          # Detailed methods (to be created)
├── code/
│   ├── collection/         # Data collection scripts
│   ├── cleaning/          # Data standardization
│   ├── analysis/          # Econometric models
│   └── visualization/     # Plotting functions
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Cleaned individual files
│   └── final/           # Merged analysis datasets
└── output/
    ├── tables/          # LaTeX regression tables
    ├── figures/         # Event studies and plots
    └── reports/         # Final reports and briefs
```

---

## 2. Implementation Phases

### 2.1 Phase 1: Environment Setup and Data Collection (Month 1)

#### Task 1.1: Project Infrastructure
```bash
# Create complete directory structure
mkdir -p code/{collection,cleaning,analysis,visualization}
mkdir -p data/{raw,processed,final}
mkdir -p output/{tables,figures,reports}
mkdir -p docs

# Set up Python environment (already completed via uv sync)
# Dependencies installed and working per CLAUDE.md
```

#### Task 1.2: Data Collection Implementation

**File**: `code/collection/naep_collector.py`
```python
#!/usr/bin/env python
"""
NAEP Data Collection Module
Implements specifications from data-collection-prd.md
"""

import pandas as pd
import requests
import time
from typing import List, Dict
import logging

class NAEPDataCollector:
    """Automated NAEP data collection for state-level special education analysis"""
    
    def __init__(self):
        self.base_url = "https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx"
        self.results = []
        self.rate_limit_delay = 6  # seconds between requests
        
    def fetch_state_swd_data(self, years: List[int], grades: List[int] = [4, 8],
                             subjects: List[str] = ['mathematics', 'reading']) -> pd.DataFrame:
        """
        Fetch NAEP data by state for students with disabilities
        Implementation per data-collection-prd.md Section 2.2
        """
        
        logging.info(f"Collecting NAEP data for {len(years)} years, {len(grades)} grades, {len(subjects)} subjects")
        
        for year in years:
            for grade in grades:
                for subject in subjects:
                    params = {
                        'type': 'data',
                        'subject': subject,
                        'grade': grade,
                        'year': year,
                        'jurisdiction': 'states',
                        'variable': 'SDRACEM',  # Students with disabilities
                        'stattype': 'MN:MN,RP:RP',  # Mean scores and percentiles
                    }
                    
                    try:
                        response = requests.get(self.base_url, params=params, timeout=30)
                        response.raise_for_status()
                        
                        data = response.json()
                        
                        # Parse nested JSON structure per API documentation
                        for state_data in data.get('results', []):
                            record = {
                                'state': state_data.get('jurisdiction'),
                                'year': year,
                                'grade': grade,
                                'subject': subject,
                                'swd_mean': self._safe_float(state_data.get('IEP', {}).get('value')),
                                'swd_se': self._safe_float(state_data.get('IEP', {}).get('errorFlag')),
                                'non_swd_mean': self._safe_float(state_data.get('NotIEP', {}).get('value')),
                                'non_swd_se': self._safe_float(state_data.get('NotIEP', {}).get('errorFlag')),
                            }
                            
                            # Calculate achievement gap
                            if record['non_swd_mean'] and record['swd_mean']:
                                record['gap'] = record['non_swd_mean'] - record['swd_mean']
                            else:
                                record['gap'] = None
                                
                            self.results.append(record)
                            
                    except Exception as e:
                        logging.error(f"Failed to collect {subject} grade {grade} year {year}: {str(e)}")
                        
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
        
        df = pd.DataFrame(self.results)
        logging.info(f"Collected {len(df)} NAEP records")
        return df
    
    def _safe_float(self, value):
        """Safely convert API values to float"""
        try:
            return float(value) if value not in [None, '', 'null', '‡', '*'] else None
        except (ValueError, TypeError):
            return None
            
    def validate_naep_data(self, df: pd.DataFrame) -> Dict:
        """Validation per data-collection-prd.md Section 2.3"""
        
        validation = {
            'total_records': len(df),
            'states_covered': df['state'].nunique(),
            'years_covered': sorted(df['year'].unique().tolist()),
            'missing_swd_scores': df['swd_mean'].isna().sum(),
            'missing_gaps': df['gap'].isna().sum(),
            'passed': True,
            'errors': []
        }
        
        # Check for required coverage
        if validation['states_covered'] < 50:
            validation['errors'].append(f"Only {validation['states_covered']} states covered, expected 50+")
            validation['passed'] = False
            
        # Check score ranges
        invalid_scores = df[(df['swd_mean'] < 0) | (df['swd_mean'] > 500)]['swd_mean'].count()
        if invalid_scores > 0:
            validation['errors'].append(f"{invalid_scores} scores outside valid range (0-500)")
            validation['passed'] = False
            
        return validation
```

**Similar collectors required** (per data-collection-prd.md):
- `code/collection/edfacts_collector.py`
- `code/collection/census_collector.py` 
- `code/collection/ocr_collector.py`
- `code/collection/master_pipeline.py`

#### Task 1.3: Policy Database Construction

**File**: `code/collection/policy_builder.py`
```python
#!/usr/bin/env python
"""
Policy Database Builder
Implements specifications from policy-database-prd.md
"""

import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List

class PolicyDatabaseBuilder:
    """Systematic approach to coding state special education policies"""
    
    def __init__(self):
        self.states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]
        self.years = list(range(2009, 2024))
        
    def create_base_panel(self) -> pd.DataFrame:
        """Create state-year panel structure per policy-database-prd.md Section 2.1"""
        
        panel = []
        for state in self.states:
            for year in self.years:
                panel.append({
                    'state': state,
                    'year': year,
                    'state_year': f"{state}_{year}"
                })
        
        return pd.DataFrame(panel)
    
    def code_funding_formulas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Code funding formula reforms per policy-database-prd.md Section 3.2"""
        
        # Known major reforms from policy-database-prd.md Table
        reforms = {
            'CA': {'year': 2013, 'type': 'census_plus', 'description': 'Local Control Funding Formula'},
            'MA': {'year': 2019, 'type': 'circuit_breaker_enhanced', 'description': 'Increased reimbursement rate'},
            'PA': {'year': 2014, 'type': 'court_ordered_study', 'description': 'Special education funding commission'},
            'TX': {'year': 2019, 'type': 'weighted_increase', 'description': 'HB3 increased special ed weights'},
            'IL': {'year': 2017, 'type': 'evidence_based', 'description': 'Evidence-based funding formula'},
            'TN': {'year': 2016, 'type': 'weighted_student', 'description': 'Tennessee Education Finance Act'},
            'WA': {'year': 2018, 'type': 'court_mandated', 'description': 'McCleary decision implementation'},
            'KS': {'year': 2017, 'type': 'court_mandated', 'description': 'Gannon v. Kansas settlement'},
            'CT': {'year': 2018, 'type': 'excess_cost_reform', 'description': 'Special education grant reform'},
            'NJ': {'year': 2018, 'type': 'census_plus', 'description': 'S2 school funding reform'},
            'VT': {'year': 2019, 'type': 'census_to_weighted', 'description': 'Act 173 special education funding'},
            'NV': {'year': 2019, 'type': 'weighted_funding', 'description': 'SB543 pupil-centered funding'},
            'MD': {'year': 2020, 'type': 'blueprint_implementation', 'description': 'Kirwan Commission recommendations'},
            'NH': {'year': 2020, 'type': 'court_settlement', 'description': 'ConVal adequate education decision'},
            'MI': {'year': 2019, 'type': 'weighted_formula', 'description': 'Foundation formula enhancement'}
        }
        
        # Initialize reform variables
        df['reform_status'] = 0
        df['reform_year'] = 0
        df['reform_type'] = None
        df['reform_description'] = None
        
        # Apply reforms to panel
        for state, reform in reforms.items():
            # Mark reform year
            mask_reform_year = (df['state'] == state) & (df['year'] == reform['year'])
            df.loc[mask_reform_year, 'reform_year'] = 1
            
            # Mark post-reform period
            mask_post_reform = (df['state'] == state) & (df['year'] >= reform['year'])
            df.loc[mask_post_reform, 'reform_status'] = 1
            df.loc[mask_post_reform, 'reform_type'] = reform['type']
            df.loc[mask_post_reform, 'reform_description'] = reform['description']
            
        return df
    
    def code_court_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Code court orders per policy-database-prd.md Section 4.2"""
        
        court_orders = {
            'PA': {'start': 2014, 'end': 2023, 'case': 'Gaskin v. Pennsylvania'},
            'WA': {'start': 2012, 'end': 2018, 'case': 'McCleary v. State'},
            'KS': {'start': 2014, 'end': 2019, 'case': 'Gannon v. Kansas'},
            'CT': {'start': 2016, 'end': 2023, 'case': 'CCJEF v. Rell'},
            'NJ': {'start': 2009, 'end': 2011, 'case': 'Abbott v. Burke XXI'},
            'NY': {'start': 2007, 'end': 2014, 'case': 'CFE v. State'},
            'NH': {'start': 2019, 'end': 2023, 'case': 'ConVal v. State'},
            'NM': {'start': 2018, 'end': 2023, 'case': 'Yazzie/Martinez'},
        }
        
        df['court_order_active'] = 0
        df['court_case_name'] = None
        
        for state, order in court_orders.items():
            mask = (df['state'] == state) & \
                   (df['year'] >= order['start']) & \
                   (df['year'] <= order['end'])
            df.loc[mask, 'court_order_active'] = 1
            df.loc[mask, 'court_case_name'] = order['case']
            
        return df
    
    def create_final_database(self) -> pd.DataFrame:
        """Create complete policy database per policy-database-prd.md Section 9"""
        
        logging.info("Creating policy database...")
        
        # Start with base panel
        df = self.create_base_panel()
        
        # Add policy coding
        df = self.code_funding_formulas(df)
        df = self.code_court_orders(df)
        # Additional coding methods would be added here
        
        # Create treatment summary
        df['any_treatment'] = (
            (df['reform_status'] == 1) | 
            (df['court_order_active'] == 1)
        ).astype(int)
        
        # Validation
        validation = self.validate_coding(df)
        logging.info(f"Policy database validation: {validation}")
        
        return df
    
    def validate_coding(self, df: pd.DataFrame) -> Dict:
        """Validate policy coding per policy-database-prd.md Section 8"""
        
        return {
            'total_observations': len(df),
            'states_with_reforms': df[df['reform_status'] == 1]['state'].nunique(),
            'court_orders': df[df['court_order_active'] == 1]['state'].nunique(),
            'years_covered': df['year'].nunique(),
            'missing_data': df.isnull().sum().sum()
        }
```

### 2.2 Phase 2: Data Cleaning and Integration (Month 2)

#### Task 2.1: Data Standardization

**File**: `code/cleaning/standardize.py`
```python
#!/usr/bin/env python
"""
Data Standardization Module
Converts raw datasets to consistent format for analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

def standardize_naep_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize NAEP data to analysis format"""
    
    # Standardize state codes
    state_code_map = {
        'District of Columbia': 'DC',
        'DoDEA': None  # Exclude Department of Defense schools
    }
    
    df['state'] = df['state'].replace(state_code_map)
    df = df.dropna(subset=['state'])  # Remove non-state jurisdictions
    
    # Create standardized achievement measure
    df['achievement_raw'] = df['swd_mean']
    
    # Standardize within subject-grade-year
    df['achievement_std'] = df.groupby(['subject', 'grade', 'year'])['swd_mean'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Create gap measures
    df['gap_raw'] = df['gap']  
    df['gap_std'] = df.groupby(['subject', 'grade', 'year'])['gap'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return df

def standardize_edfacts_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize EdFacts data to analysis format"""
    
    # Calculate inclusion rate
    if 'regular_class_80_percent_or_more' in df.columns:
        df['inclusion_rate'] = (
            df['regular_class_80_percent_or_more'] / df['total_students']
        ) * 100
    
    # Calculate restrictive placement rate  
    if 'separate_schools_facilities' in df.columns:
        df['restrictive_rate'] = (
            df['separate_schools_facilities'] / df['total_students'] 
        ) * 100
    
    # Standardize graduation rate
    if 'regular_diploma' in df.columns and 'total_exiters' in df.columns:
        df['graduation_rate'] = (
            df['regular_diploma'] / df['total_exiters']
        ) * 100
        
    return df

def merge_all_datasets() -> pd.DataFrame:
    """Create master analysis dataset by merging all sources"""
    
    # Load standardized datasets
    datasets = {
        'naep': pd.read_csv('data/processed/naep_standardized.csv'),
        'edfacts': pd.read_csv('data/processed/edfacts_standardized.csv'),
        'census': pd.read_csv('data/processed/census_standardized.csv'),
        'policy': pd.read_csv('data/processed/policy_database.csv')
    }
    
    # Start with policy database as backbone (complete state-year panel)
    master = datasets['policy'].copy()
    
    # Merge outcome data
    master = master.merge(
        datasets['naep'][['state', 'year', 'subject', 'grade', 
                         'achievement_std', 'gap_std', 'swd_mean']],
        on=['state', 'year'],
        how='left'
    )
    
    # Merge mechanism data
    master = master.merge(
        datasets['edfacts'][['state', 'year', 'inclusion_rate', 
                            'graduation_rate', 'child_count']],
        on=['state', 'year'],
        how='left'
    )
    
    # Merge financial data
    master = master.merge(
        datasets['census'][['state', 'year', 'per_pupil_total', 
                           'total_expenditure', 'enrollment']],
        on=['state', 'year'],
        how='left'
    )
    
    # Quality checks
    logging.info(f"Master dataset: {len(master)} observations")
    logging.info(f"States: {master['state'].nunique()}")
    logging.info(f"Years: {master['year'].min()}-{master['year'].max()}")
    logging.info(f"Missing achievement: {master['achievement_std'].isna().sum()}")
    
    return master
```

### 2.3 Phase 3: Analysis Implementation (Months 3-4)

#### Task 3.1: Main Econometric Models

**File**: `code/analysis/main_models.py`
```python
#!/usr/bin/env python
"""
Main Econometric Analysis
Implements staggered DiD using statsmodels and linearmodels per CLAUDE.md
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
import logging

class MainAnalysis:
    """Main econometric specifications for policy evaluation"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.prepare_analysis_variables()
        
    def prepare_analysis_variables(self):
        """Prepare variables for analysis"""
        
        # Event time variables for event study
        self.data['years_since_reform'] = (
            self.data['year'] - 
            self.data.groupby('state')['year'].transform(
                lambda x: x[self.data.loc[x.index, 'reform_year'] == 1].min()
                if (self.data.loc[x.index, 'reform_year'] == 1).any() 
                else np.inf
            )
        )
        
        # Replace infinite values (never-treated states) with missing
        self.data['years_since_reform'] = self.data['years_since_reform'].replace([np.inf, -np.inf], np.nan)
        
        # Create event time dummies
        for t in range(-5, 6):
            if t != -1:  # Omit t=-1 as reference
                self.data[f'event_t{t}'] = (
                    self.data['years_since_reform'] == t
                ).astype(int)
                
        # COVID period indicators per covid-analysis-prd.md
        self.data['covid_period'] = self.data['year'].isin([2020, 2021]).astype(int)
        self.data['post_covid'] = (self.data['year'] >= 2022).astype(int)
    
    def run_basic_twfe(self) -> Dict:
        """Basic two-way fixed effects specification"""
        
        model = smf.ols(
            'achievement_std ~ reform_status + C(state) + C(year)',
            data=self.data.dropna(subset=['achievement_std'])
        ).fit(cov_type='cluster', cov_kwds={'groups': self.data['state']})
        
        return {
            'coefficient': model.params['reform_status'],
            'se': model.bse['reform_status'],
            'p_value': model.pvalues['reform_status'],
            'n_obs': int(model.nobs),
            'model': model
        }
    
    def run_event_study(self) -> Dict:
        """Event study specification per research-methodology-prd.md Section 5.1"""
        
        # Build event study formula
        event_terms = []
        for t in range(-5, 6):
            if t != -1:  # Omit reference period
                event_terms.append(f'event_t{t}')
                
        formula = f"achievement_std ~ {' + '.join(event_terms)} + C(state) + C(year)"
        
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['state']}
        )
        
        # Extract event study coefficients
        event_coefs = {}
        event_ses = {}
        for t in range(-5, 6):
            if t == -1:
                event_coefs[t] = 0.0  # Reference period
                event_ses[t] = 0.0
            else:
                event_coefs[t] = model.params[f'event_t{t}']
                event_ses[t] = model.bse[f'event_t{t}']
                
        return {
            'coefficients': event_coefs,
            'standard_errors': event_ses,
            'model': model
        }
    
    def run_iv_specification(self) -> Dict:
        """IV specification using court orders as instrument"""
        
        # Manual 2SLS implementation since we're using statsmodels
        
        # First stage
        first_stage = smf.ols(
            'per_pupil_total ~ court_order_active + C(state) + C(year)',
            data=self.data
        ).fit()
        
        # Predicted values
        self.data['per_pupil_predicted'] = first_stage.predict()
        
        # Second stage
        second_stage = smf.ols(
            'achievement_std ~ per_pupil_predicted + C(state) + C(year)', 
            data=self.data
        ).fit(cov_type='cluster', cov_kwds={'groups': self.data['state']})
        
        return {
            'first_stage_f': first_stage.fvalue,
            'iv_coefficient': second_stage.params['per_pupil_predicted'],
            'iv_se': second_stage.bse['per_pupil_predicted'], 
            'iv_p_value': second_stage.pvalues['per_pupil_predicted'],
            'first_stage': first_stage,
            'second_stage': second_stage
        }
    
    def run_covid_analysis(self) -> Dict:
        """COVID triple-difference per covid-analysis-prd.md Section 3"""
        
        # Create student group indicators (assuming we have SWD vs non-SWD data)
        # This would need to be adapted based on actual data structure
        
        model = smf.ols("""
            achievement_std ~ reform_status * covid_period + 
                             reform_status + covid_period +
                             C(state) + C(year)
        """, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['state']}
        )
        
        return {
            'covid_interaction': model.params['reform_status:covid_period'],
            'covid_interaction_se': model.bse['reform_status:covid_period'],
            'covid_interaction_p': model.pvalues['reform_status:covid_period'],
            'model': model
        }

def run_all_specifications(data: pd.DataFrame) -> Dict:
    """Run complete analysis suite"""
    
    analysis = MainAnalysis(data)
    
    results = {
        'twfe': analysis.run_basic_twfe(),
        'event_study': analysis.run_event_study(),
        'iv': analysis.run_iv_specification(),
        'covid': analysis.run_covid_analysis()
    }
    
    return results
```

### 2.4 Phase 4: Automation and Pipeline (Months 4-5)

#### Task 4.1: Master Execution Script

**File**: `run_analysis.py`
```python
#!/usr/bin/env python
"""
Master Analysis Pipeline
Complete automation per implementation-prd.md Section 6
"""

import argparse
import sys
import time
import logging
from pathlib import Path
import pandas as pd

# Local imports
from code.collection.master_pipeline import run_full_data_collection
from code.cleaning.standardize import standardize_all_data, merge_all_datasets
from code.analysis.main_models import run_all_specifications
from code.visualization.create_plots import generate_all_figures
from code.validation.validate_results import comprehensive_validation

def setup_logging():
    """Configure logging for pipeline execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline_execution.log'),
            logging.StreamHandler()
        ]
    )

def setup_directories():
    """Create required directory structure"""
    
    required_dirs = [
        'data/raw', 'data/processed', 'data/final',
        'code/collection', 'code/cleaning', 'code/analysis', 'code/visualization',
        'output/tables', 'output/figures', 'output/reports',
        'docs'
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created directory structure: {len(required_dirs)} directories")

def run_collection_stage():
    """Execute data collection phase"""
    
    logging.info("=" * 50)
    logging.info("STAGE 1: DATA COLLECTION")
    logging.info("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run data collection
        collection_results = run_full_data_collection()
        
        # Log results
        total_records = sum(r.get('records', 0) for r in collection_results.values() 
                          if isinstance(r, dict) and 'records' in r)
        
        logging.info(f"✓ Data collection completed: {total_records:,} total records")
        logging.info(f"✓ Sources collected: {list(collection_results.keys())}")
        
        elapsed = time.time() - start_time
        logging.info(f"✓ Collection stage completed in {elapsed:.1f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Collection stage failed: {str(e)}")
        return False

def run_cleaning_stage():
    """Execute data cleaning and integration phase"""
    
    logging.info("=" * 50)
    logging.info("STAGE 2: DATA CLEANING")
    logging.info("=" * 50)
    
    start_time = time.time()
    
    try:
        # Standardize individual datasets
        standardized_datasets = standardize_all_data()
        logging.info(f"✓ Standardized {len(standardized_datasets)} datasets")
        
        # Merge into analysis dataset
        master_dataset = merge_all_datasets()
        
        # Save final dataset
        master_dataset.to_csv('data/final/analysis_dataset.csv', index=False)
        master_dataset.to_pickle('data/final/analysis_dataset.pkl')
        
        logging.info(f"✓ Master dataset: {len(master_dataset):,} observations")
        logging.info(f"✓ Coverage: {master_dataset['state'].nunique()} states, {master_dataset['year'].nunique()} years")
        
        elapsed = time.time() - start_time
        logging.info(f"✓ Cleaning stage completed in {elapsed:.1f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Cleaning stage failed: {str(e)}")
        return False

def run_analysis_stage():
    """Execute econometric analysis phase"""
    
    logging.info("=" * 50)
    logging.info("STAGE 3: ECONOMETRIC ANALYSIS")
    logging.info("=" * 50)
    
    start_time = time.time()
    
    try:
        # Load analysis dataset
        data = pd.read_pickle('data/final/analysis_dataset.pkl')
        
        # Run all specifications
        results = run_all_specifications(data)
        
        # Save results
        results_summary = pd.DataFrame({
            'specification': ['TWFE', 'IV', 'COVID_Interaction'],
            'coefficient': [
                results['twfe']['coefficient'],
                results['iv']['iv_coefficient'], 
                results['covid']['covid_interaction']
            ],
            'se': [
                results['twfe']['se'],
                results['iv']['iv_se'],
                results['covid']['covid_interaction_se'] 
            ],
            'p_value': [
                results['twfe']['p_value'],
                results['iv']['iv_p_value'],
                results['covid']['covid_interaction_p']
            ]
        })
        
        results_summary.to_csv('output/tables/main_results.csv', index=False)
        results_summary.to_latex('output/tables/main_results.tex', index=False)
        
        # Generate figures
        generate_all_figures(data, results)
        
        logging.info(f"✓ Analysis completed: {len(results)} specifications")
        logging.info(f"✓ Main effect: {results['twfe']['coefficient']:.3f} ({results['twfe']['se']:.3f})")
        
        elapsed = time.time() - start_time
        logging.info(f"✓ Analysis stage completed in {elapsed:.1f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Analysis stage failed: {str(e)}")
        return False

def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Run special education policy analysis pipeline')
    parser.add_argument('--stage', 
                       choices=['all', 'collect', 'clean', 'analyze'],
                       default='all',
                       help='Which analysis stage to run')
    parser.add_argument('--validate', 
                       action='store_true',
                       help='Run comprehensive validation')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    setup_directories()
    
    logging.info("Starting special education policy analysis pipeline")
    logging.info(f"Stage: {args.stage}")
    
    # Execute stages
    success = True
    
    if args.stage in ['all', 'collect']:
        success = success and run_collection_stage()
        
    if success and args.stage in ['all', 'clean']:
        success = success and run_cleaning_stage()
        
    if success and args.stage in ['all', 'analyze']:
        success = success and run_analysis_stage()
    
    # Validation
    if success and args.validate:
        logging.info("Running comprehensive validation...")
        validation_results = comprehensive_validation()
        if not validation_results['overall_pass']:
            logging.error("Validation failed - check results before publication")
            success = False
        else:
            logging.info("✓ All validation checks passed")
    
    # Final status
    logging.info("=" * 50)
    if success:
        logging.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("Check output/ directory for results")
    else:
        logging.error("✗ PIPELINE FAILED")
        logging.error("Check logs for error details")
        
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
```

---

## 3. Quality Assurance Framework

### 3.1 Automated Testing Framework

**Testing Philosophy**: Comprehensive test coverage with pytest framework ensuring code quality, regression prevention, and reliable data collection pipeline.

#### 3.1.1 Testing Standards and Coverage

**Framework**: pytest with comprehensive plugins  
**Minimum Coverage Requirements**:
- Overall project: 80% line coverage  
- Data collectors: 90%+ coverage
- Critical calculations: 95%+ coverage
- Integration points: 85%+ coverage

**Quality Gates**:
- All tests pass before code integration
- Coverage reports generated for every build
- Performance benchmarks maintained
- No regression in test execution time

#### 3.1.2 Test Suite Architecture

**File**: `tests/conftest.py` - Central test configuration
```python
#!/usr/bin/env python
"""
Pytest configuration and shared fixtures for the project test suite
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
import json

@pytest.fixture
def sample_states():
    """Fixture providing standard state list for testing"""
    return ['CA', 'TX', 'NY', 'FL', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']

@pytest.fixture  
def naep_mock_response():
    """Fixture providing realistic NAEP API response structure"""
    with open('tests/fixtures/naep_sample_response.json', 'r') as f:
        return json.load(f)

@pytest.fixture
def synthetic_analysis_data():
    """Fixture providing synthetic data for econometric testing"""
    np.random.seed(42)
    
    data = []
    states = ['CA', 'TX', 'NY', 'FL', 'PA']
    years = list(range(2017, 2023))
    
    for state in states:
        for year in years:
            # Treatment assignment
            treated = state in ['CA', 'TX']
            post_treatment = year >= 2020
            
            # Synthetic outcome with treatment effect
            base_achievement = np.random.normal(250, 30)
            treatment_effect = 15 * treated * post_treatment
            achievement = base_achievement + treatment_effect
            
            data.append({
                'state': state,
                'year': year,
                'achievement_std': (achievement - 250) / 30,
                'reform_status': int(treated and post_treatment),
                'reform_year': int(treated and year == 2020),
                'per_pupil_total': np.random.normal(12000, 2000),
                'court_order_active': np.random.choice([0, 1], p=[0.85, 0.15])
            })
    
    return pd.DataFrame(data)

@pytest.fixture(autouse=True)
def setup_test_directories(tmp_path):
    """Auto-fixture to create temporary directories for testing"""
    test_data_dir = tmp_path / "data"
    test_output_dir = tmp_path / "output"
    
    for subdir in ['raw', 'processed', 'final']:
        (test_data_dir / subdir).mkdir(parents=True)
    
    for subdir in ['tables', 'figures', 'reports']:
        (test_output_dir / subdir).mkdir(parents=True)
    
    return {
        'data': test_data_dir,
        'output': test_output_dir
    }
```

#### 3.1.3 NAEP Collector Test Suite

**File**: `tests/unit/collection/test_naep_collector.py`
```python
#!/usr/bin/env python
"""
Comprehensive test suite for NAEP data collector
Target Coverage: 95%+
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import httpx
from hypothesis import given, strategies as st

from code.collection.naep_collector import NAEPDataCollector

class TestNAEPDataCollector:
    """Core NAEP collector functionality tests"""
    
    def setup_method(self):
        """Set up test environment for each test method"""
        self.collector = NAEPDataCollector(rate_limit_delay=0.1)  # Fast for testing
    
    def test_collector_initialization(self):
        """Test collector initializes with correct default values"""
        collector = NAEPDataCollector()
        
        assert collector.base_url == "https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx"
        assert collector.rate_limit_delay == 6.0
        assert collector.results == []
    
    @pytest.mark.parametrize("state_name,expected_code", [
        ("California", "CA"),
        ("New York", "NY"), 
        ("District of Columbia", "DC"),
        ("InvalidState", None),
        ("", None)
    ])
    def test_state_name_conversion(self, state_name, expected_code):
        """Test state name to code conversion covers all states"""
        result = self.collector._convert_state_name_to_code(state_name)
        assert result == expected_code
    
    @pytest.mark.parametrize("api_value,expected", [
        ("250.5", 250.5),
        ("‡", None),  # NAEP suppression symbol
        ("*", None),   # NAEP suppression symbol  
        ("", None),
        (None, None),
        ("invalid", None)
    ])
    def test_safe_float_conversion(self, api_value, expected):
        """Test safe float conversion handles NAEP special codes"""
        result = self.collector._safe_float(api_value)
        assert result == expected
    
    @patch('requests.get')
    def test_successful_data_collection(self, mock_get, naep_mock_response):
        """Test successful API data collection with realistic response"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = naep_mock_response
        mock_get.return_value.raise_for_status.return_value = None
        
        result_df = self.collector.fetch_state_swd_data(
            years=[2019], 
            grades=[4], 
            subjects=['mathematics']
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert all(col in result_df.columns for col in 
                  ['state', 'year', 'grade', 'subject', 'swd_mean', 'gap'])
    
    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        """Test API error scenarios are handled gracefully"""
        mock_get.side_effect = httpx.RequestError("Network error")
        
        result_df = self.collector.fetch_state_swd_data(
            years=[2019], 
            grades=[4], 
            subjects=['mathematics']
        )
        
        # Should return empty DataFrame, not crash
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
    
    @patch('time.sleep')
    @patch('requests.get')
    def test_rate_limiting_compliance(self, mock_get, mock_sleep):
        """Test that rate limiting delays are properly implemented"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'result': []}
        mock_get.return_value.raise_for_status.return_value = None
        
        self.collector.fetch_state_swd_data(
            years=[2019, 2020], 
            grades=[4], 
            subjects=['mathematics']
        )
        
        # Should have 1 sleep call (2 requests - 1)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(0.1)  # Our test rate limit
    
    def test_data_validation_comprehensive(self, sample_states):
        """Test comprehensive data validation logic"""
        # Create test data with known issues
        test_data = pd.DataFrame({
            'state': sample_states,
            'year': [2019] * len(sample_states),
            'grade': [4] * len(sample_states), 
            'subject': ['mathematics'] * len(sample_states),
            'swd_mean': [250, 260, None, 245, 600],  # One missing, one invalid
            'non_swd_mean': [290, 300, 285, 280, 310],
            'gap': [40, 40, None, 35, -290]  # Invalid gap
        })
        
        validation = self.collector.validate_data(test_data)
        
        assert validation['total_records'] == len(sample_states)
        assert validation['states_covered'] == len(sample_states)
        assert validation['missing_swd_scores'] == 1
        assert len(validation['warnings']) > 0  # Should flag invalid scores

class TestNAEPDataValidation:
    """Dedicated tests for data validation functionality"""
    
    def setup_method(self):
        self.collector = NAEPDataCollector()
    
    def test_validation_with_perfect_data(self, sample_states):
        """Test validation passes with high-quality data"""
        perfect_data = pd.DataFrame({
            'state': sample_states,
            'year': [2019] * len(sample_states),
            'grade': [4] * len(sample_states),
            'subject': ['mathematics'] * len(sample_states), 
            'swd_mean': np.random.normal(250, 20, len(sample_states)),
            'non_swd_mean': np.random.normal(290, 20, len(sample_states)),
            'gap': np.random.normal(40, 10, len(sample_states))
        })
        
        validation = self.collector.validate_data(perfect_data)
        
        assert validation['passed'] == True
        assert len(validation['errors']) == 0
        assert validation['states_covered'] == len(sample_states)
    
    def test_validation_flags_insufficient_states(self):
        """Test validation fails with insufficient state coverage"""
        insufficient_data = pd.DataFrame({
            'state': ['CA', 'TX'],  # Only 2 states
            'year': [2019, 2019],
            'swd_mean': [250, 260]
        })
        
        validation = self.collector.validate_data(insufficient_data)
        
        assert validation['passed'] == False
        assert any('states covered' in error for error in validation['errors'])
    
    @given(st.lists(st.text(), min_size=1, max_size=100))
    def test_state_conversion_robustness(self, state_names):
        """Property-based test for state name conversion robustness"""
        for state_name in state_names:
            # Should not crash on any input
            result = self.collector._convert_state_name_to_code(state_name)
            assert result is None or len(result) == 2

class TestNAEPIntegration:
    """Integration tests for NAEP collector with mocked external dependencies"""
    
    @patch('code.collection.naep_collector.Path.mkdir')
    @patch('pandas.DataFrame.to_csv')
    def test_save_data_creates_directory(self, mock_to_csv, mock_mkdir):
        """Test save_data creates output directory if needed"""
        collector = NAEPDataCollector()
        test_df = pd.DataFrame({'col': [1, 2, 3]})
        
        result = collector.save_data(test_df, "test_output/naep_data.csv")
        
        assert result == True
        mock_mkdir.assert_called()
        mock_to_csv.assert_called_with("test_output/naep_data.csv", index=False)

@pytest.mark.integration
class TestNAEPEndToEnd:
    """End-to-end integration tests for NAEP collection pipeline"""
    
    @patch('requests.get')
    def test_full_collection_workflow(self, mock_get, naep_mock_response, tmp_path):
        """Test complete collection workflow from API to saved file"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = naep_mock_response
        mock_get.return_value.raise_for_status.return_value = None
        
        collector = NAEPDataCollector(rate_limit_delay=0)
        
        # Collect data
        df = collector.fetch_state_swd_data(years=[2019], grades=[4], subjects=['mathematics'])
        
        # Validate data
        validation = collector.validate_data(df)
        
        # Save data
        output_path = tmp_path / "naep_test.csv"
        success = collector.save_data(df, str(output_path))
        
        # Verify end-to-end success
        assert len(df) > 0
        assert validation['total_records'] > 0
        assert success == True
        assert output_path.exists()

@pytest.mark.performance  
class TestNAEPPerformance:
    """Performance tests for NAEP collector"""
    
    def test_large_dataset_handling(self):
        """Test collector can handle large synthetic datasets"""
        collector = NAEPDataCollector()
        
        # Create large synthetic dataset
        large_data = pd.DataFrame({
            'state': ['CA'] * 10000,
            'year': list(range(2000, 10000)),
            'swd_mean': np.random.normal(250, 30, 10000),
            'non_swd_mean': np.random.normal(290, 30, 10000)
        })
        
        # Should complete validation without memory issues
        validation = collector.validate_data(large_data)
        
        assert validation['total_records'] == 10000
        assert validation is not None
```

#### 3.1.4 Test Execution and Reporting

**pytest.ini Configuration**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=code
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-report=term-missing
    --cov-fail-under=80
    --strict-markers
    --maxfail=10
    -ra
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests  
    performance: marks tests as performance tests
    unit: marks tests as unit tests
```

**Test Execution Commands**:
```bash
# Run all tests with coverage
pytest

# Run only fast unit tests
pytest -m "not slow and not performance"

# Run with parallel execution
pytest -n auto

# Generate detailed coverage report
pytest --cov=code --cov-report=html

# Run performance benchmarks
pytest -m performance --benchmark-only
```

### 3.2 Validation Framework

**File**: `code/validation/validate_results.py`
```python
#!/usr/bin/env python
"""
Comprehensive validation for analysis results
Per implementation-prd.md Section 11.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

def validate_data_quality() -> Dict:
    """Validate input data quality"""
    
    validation = {
        'data_quality': {
            'checks': [],
            'passed': True
        }
    }
    
    # Load final analysis dataset
    try:
        df = pd.read_csv('data/final/analysis_dataset.csv')
    except FileNotFoundError:
        validation['data_quality']['checks'].append({
            'check': 'dataset_exists',
            'result': False,
            'message': 'Analysis dataset not found'
        })
        validation['data_quality']['passed'] = False
        return validation
    
    # Check sample size
    check_sample_size = {
        'check': 'adequate_sample_size',
        'result': len(df) >= 500,
        'value': len(df),
        'message': f'Dataset has {len(df)} observations (target: 500+)'
    }
    validation['data_quality']['checks'].append(check_sample_size)
    
    if not check_sample_size['result']:
        validation['data_quality']['passed'] = False
    
    # Check state coverage
    check_states = {
        'check': 'state_coverage',
        'result': df['state'].nunique() >= 48,
        'value': df['state'].nunique(),
        'message': f'Coverage: {df["state"].nunique()} states (target: 48+)'
    }
    validation['data_quality']['checks'].append(check_states)
    
    if not check_states['result']:
        validation['data_quality']['passed'] = False
    
    # Check temporal coverage
    year_range = df['year'].max() - df['year'].min()
    check_years = {
        'check': 'temporal_coverage',
        'result': year_range >= 10,
        'value': year_range,
        'message': f'Time span: {year_range} years (target: 10+)'
    }
    validation['data_quality']['checks'].append(check_years)
    
    if not check_years['result']:
        validation['data_quality']['passed'] = False
    
    # Check missing data
    missing_outcomes = df['achievement_std'].isna().mean()
    check_missing = {
        'check': 'missing_outcomes',
        'result': missing_outcomes < 0.3,
        'value': missing_outcomes,
        'message': f'Missing outcomes: {missing_outcomes:.1%} (target: <30%)'
    }
    validation['data_quality']['checks'].append(check_missing)
    
    if not check_missing['result']:
        validation['data_quality']['passed'] = False
    
    return validation

def validate_results() -> Dict:
    """Validate econometric results"""
    
    validation = {
        'results_validity': {
            'checks': [],
            'passed': True
        }
    }
    
    # Load results
    try:
        results = pd.read_csv('output/tables/main_results.csv')
    except FileNotFoundError:
        validation['results_validity']['checks'].append({
            'check': 'results_exist',
            'result': False,
            'message': 'Results file not found'
        })
        validation['results_validity']['passed'] = False
        return validation
    
    # Check effect sizes are reasonable
    twfe_effect = results[results['specification'] == 'TWFE']['coefficient'].iloc[0]
    check_effect_size = {
        'check': 'reasonable_effect_size',
        'result': abs(twfe_effect) < 2.0,  # Should be less than 2 standard deviations
        'value': twfe_effect,
        'message': f'Main effect: {twfe_effect:.3f}σ (should be <2σ)'
    }
    validation['results_validity']['checks'].append(check_effect_size)
    
    if not check_effect_size['result']:
        validation['results_validity']['passed'] = False
    
    # Check standard errors are reasonable
    twfe_se = results[results['specification'] == 'TWFE']['se'].iloc[0]
    check_se = {
        'check': 'reasonable_standard_errors',
        'result': 0.01 < twfe_se < 1.0,
        'value': twfe_se,
        'message': f'Standard error: {twfe_se:.3f} (should be 0.01-1.0)'
    }
    validation['results_validity']['checks'].append(check_se)
    
    if not check_se['result']:
        validation['results_validity']['passed'] = False
    
    return validation

def validate_outputs() -> Dict:
    """Validate required outputs exist"""
    
    validation = {
        'output_completeness': {
            'checks': [],
            'passed': True
        }
    }
    
    required_outputs = [
        'output/tables/main_results.csv',
        'output/tables/main_results.tex',
        'output/figures/event_study.png',
        'data/final/analysis_dataset.csv'
    ]
    
    for output_path in required_outputs:
        exists = Path(output_path).exists()
        check = {
            'check': f'output_exists_{Path(output_path).name}',
            'result': exists,
            'message': f'{output_path} {"exists" if exists else "missing"}'
        }
        validation['output_completeness']['checks'].append(check)
        
        if not exists:
            validation['output_completeness']['passed'] = False
    
    return validation

def comprehensive_validation() -> Dict:
    """Run all validation checks"""
    
    logging.info("Running comprehensive validation...")
    
    validation_results = {
        'overall_pass': True,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Run individual validation modules
    data_validation = validate_data_quality()
    results_validation = validate_results()
    output_validation = validate_outputs()
    
    # Combine results
    validation_results.update({
        'data_quality': data_validation['data_quality'],
        'results_validity': results_validation['results_validity'],
        'output_completeness': output_validation['output_completeness']
    })
    
    # Overall pass/fail
    validation_results['overall_pass'] = (
        data_validation['data_quality']['passed'] and
        results_validation['results_validity']['passed'] and
        output_validation['output_completeness']['passed']
    )
    
    # Save validation report
    with open('output/validation_report.json', 'w') as f:
        import json
        json.dump(validation_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 60)
    
    for category, results in validation_results.items():
        if isinstance(results, dict) and 'passed' in results:
            status = "✓ PASS" if results['passed'] else "✗ FAIL"
            print(f"{category.upper()}: {status}")
            
            if 'checks' in results:
                for check in results['checks']:
                    check_status = "✓" if check['result'] else "✗"
                    print(f"  {check_status} {check.get('message', check['check'])}")
    
    print("\n" + "=" * 60)
    overall_status = "✓ ALL CHECKS PASSED" if validation_results['overall_pass'] else "✗ SOME CHECKS FAILED"
    print(f"OVERALL: {overall_status}")
    print("=" * 60)
    
    return validation_results
```

---

## 4. Deployment and Usage

### 4.1 Environment Setup

```bash
# Clone/setup project
cd state-sped-policy-eval

# Install dependencies (already completed)
# uv sync (working per current setup)

# Set environment variables
export CENSUS_API_KEY="your_key_here"  # Required for data collection
export DATA_OUTPUT_DIR="data/"

# Verify setup
python -c "import pandas, statsmodels, linearmodels; print('All dependencies available')"
```

### 4.2 Execution Commands

```bash
# Full pipeline execution
python run_analysis.py --stage all --validate

# Individual stages
python run_analysis.py --stage collect    # Data collection only
python run_analysis.py --stage clean      # Cleaning/merging only
python run_analysis.py --stage analyze    # Analysis only

# Validation only
python -m code.validation.validate_results
```

### 4.3 Expected Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Setup & Data Collection | Month 1 | Raw datasets, policy database |
| Cleaning & Integration | Month 2 | Master analysis dataset |
| Core Analysis | Month 3-4 | Main econometric results |
| COVID Analysis | Month 4 | Natural experiment results |
| Robustness & Reporting | Month 5-6 | Final paper, policy brief |

---

## 5. Success Metrics

### 5.1 Technical Success
- ✅ Automated pipeline runs end-to-end without manual intervention
- ✅ All validation checks pass (data quality, results validity, output completeness)
- ✅ Results are reproducible across different computing environments
- ✅ Code is well-documented and maintainable

### 5.2 Research Success
- ✅ Statistically significant main effects with reasonable magnitudes
- ✅ Robustness across multiple specifications  
- ✅ Clear policy mechanisms identified
- ✅ COVID analysis provides novel insights

### 5.3 Policy Impact
- ✅ Results inform IDEA reauthorization debate
- ✅ State education agencies request technical assistance
- ✅ Academic publication in top field journal
- ✅ Media coverage in education policy outlets

---

**Document Control**  
- Version: 1.0  
- Last Updated: 2025-08-11  
- Implementation Status: Ready for development  
- Dependencies: All PRDs completed, Python environment configured