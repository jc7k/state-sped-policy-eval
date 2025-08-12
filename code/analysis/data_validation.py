"""
Data Validation and Summary Statistics for Master Analysis Dataset

This module provides comprehensive validation and descriptive analysis of the
merged master dataset used for econometric analysis. Includes data quality
checks, summary statistics, and preliminary balance tests.

Author: Research Team
Date: 2025-08-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path


class DataValidator:
    """
    Comprehensive data validation and summary statistics for analysis dataset.
    
    Performs:
    - Data completeness and quality checks
    - Summary statistics by treatment status
    - Balance tests for pre-treatment characteristics
    - Outcome variable distributions and trends
    """
    
    def __init__(self, data_path: str = "data/processed/master_analysis_dataset.csv"):
        """Initialize with master analysis dataset."""
        self.data_path = data_path
        self.master_df = None
        self.policy_df = None
        self.validation_results = {}
        self._load_data()
    
    def _load_data(self):
        """Load master analysis dataset and policy database."""
        try:
            self.master_df = pd.read_csv(self.data_path)
            print(f"Loaded master dataset: {self.master_df.shape}")
            
            # Load policy database
            policy_path = "data/processed/state_policy_database.csv"
            if Path(policy_path).exists():
                self.policy_df = pd.read_csv(policy_path)
                print(f"Loaded policy database: {self.policy_df.shape}")
            else:
                print("Policy database not found - run policy_database.py first")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def validate_data_completeness(self) -> Dict[str, any]:
        """Validate data completeness and identify missing patterns."""
        if self.master_df is None:
            raise ValueError("Data not loaded")
        
        # Basic completeness
        total_obs = len(self.master_df)
        missing_by_col = self.master_df.isnull().sum()
        missing_pct = (missing_by_col / total_obs * 100).round(2)
        
        # Key outcome variables
        key_outcomes = [col for col in self.master_df.columns if 'naep' in col.lower()]
        if not key_outcomes:
            # Look for likely NAEP columns based on pattern
            key_outcomes = [col for col in self.master_df.columns 
                          if any(x in col.lower() for x in ['math', 'reading', 'achievement', 'score'])]
        
        # Financial variables
        finance_vars = [col for col in self.master_df.columns 
                       if any(x in col.lower() for x in ['revenue', 'expenditure', 'spending'])]
        
        # EdFacts variables  
        edfacts_vars = [col for col in self.master_df.columns 
                       if col.startswith('edfacts_') or 'placement' in col.lower()]
        
        # OCR variables
        ocr_vars = [col for col in self.master_df.columns if col.startswith('ocr_')]
        
        results = {
            'total_observations': total_obs,
            'total_variables': len(self.master_df.columns),
            'missing_by_variable': missing_pct.to_dict(),
            'key_outcomes': key_outcomes,
            'finance_variables': finance_vars,
            'edfacts_variables': edfacts_vars[:10],  # Limit output
            'ocr_variables': ocr_vars[:10],  # Limit output
            'completely_missing_vars': missing_pct[missing_pct == 100].index.tolist(),
            'high_missingness': missing_pct[missing_pct > 50].to_dict()
        }
        
        self.validation_results['completeness'] = results
        return results
    
    def validate_panel_structure(self) -> Dict[str, any]:
        """Validate state-year panel structure."""
        if self.master_df is None:
            raise ValueError("Data not loaded")
        
        # Check state-year combinations
        state_years = self.master_df.groupby(['state', 'year']).size()
        duplicate_state_years = state_years[state_years > 1]
        
        # Year coverage
        year_range = (self.master_df['year'].min(), self.master_df['year'].max())
        years_present = sorted(self.master_df['year'].unique())
        
        # State coverage
        states_present = sorted(self.master_df['state'].unique())
        
        # Balance check
        state_year_counts = self.master_df.groupby('state')['year'].count()
        
        results = {
            'year_range': year_range,
            'years_present': years_present,
            'total_years': len(years_present),
            'states_present': states_present,
            'total_states': len(states_present),
            'duplicate_state_years': len(duplicate_state_years),
            'balanced_panel': state_year_counts.std() == 0,
            'obs_per_state': state_year_counts.describe().to_dict()
        }
        
        self.validation_results['panel_structure'] = results
        return results
    
    def generate_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive summary statistics."""
        if self.master_df is None:
            raise ValueError("Data not loaded")
        
        # Identify numeric variables
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Overall summary
        overall_stats = self.master_df[numeric_cols].describe()
        
        # By year
        year_stats = self.master_df.groupby('year')[numeric_cols].mean()
        
        # By state (sample of key variables)
        key_vars = []
        for pattern in ['revenue', 'expenditure', 'total']:
            matches = [col for col in numeric_cols if pattern in col.lower()]
            key_vars.extend(matches[:3])  # Limit to avoid too many variables
        
        if key_vars:
            state_stats = self.master_df.groupby('state')[key_vars].mean()
        else:
            state_stats = pd.DataFrame()
        
        summaries = {
            'overall': overall_stats,
            'by_year': year_stats,
            'by_state': state_stats
        }
        
        self.validation_results['summary_stats'] = summaries
        return summaries
    
    def check_balance_by_treatment(self) -> Dict[str, pd.DataFrame]:
        """Check pre-treatment balance between treated and control states."""
        if self.policy_df is None:
            print("Policy database not available - skipping balance tests")
            return {}
        
        # Merge with policy data
        analysis_df = self.master_df.merge(
            self.policy_df, on=['state', 'year'], how='left'
        )
        
        # Pre-treatment period (before any reforms)
        pre_2013 = analysis_df[analysis_df['year'] < 2013].copy()
        
        # Identify eventually treated states
        eventually_treated = self.policy_df[self.policy_df['post_treatment'] == 1]['state'].unique()
        pre_2013['eventually_treated'] = pre_2013['state'].isin(eventually_treated)
        
        # Key variables for balance test
        balance_vars = []
        for pattern in ['revenue', 'expenditure', 'enrollment']:
            matches = [col for col in pre_2013.columns if pattern in col.lower()]
            balance_vars.extend(matches[:3])
        
        balance_vars = [var for var in balance_vars if var in pre_2013.columns]
        
        if not balance_vars:
            print("No balance variables found")
            return {}
        
        # Calculate means by treatment status
        balance_stats = pre_2013.groupby('eventually_treated')[balance_vars].agg(['mean', 'std', 'count'])
        
        # Calculate differences and t-tests
        treated_means = pre_2013[pre_2013['eventually_treated'] == True][balance_vars].mean()
        control_means = pre_2013[pre_2013['eventually_treated'] == False][balance_vars].mean()
        differences = treated_means - control_means
        
        balance_results = {
            'balance_table': balance_stats,
            'differences': differences.to_frame('difference'),
            'treated_states': eventually_treated.tolist(),
            'control_states': pre_2013[~pre_2013['eventually_treated']]['state'].unique().tolist()
        }
        
        self.validation_results['balance'] = balance_results
        return balance_results
    
    def analyze_outcome_trends(self) -> Dict[str, any]:
        """Analyze trends in key outcome variables."""
        if self.master_df is None:
            raise ValueError("Data not loaded")
        
        # Find outcome variables
        outcome_patterns = ['math', 'reading', 'achievement', 'score', 'naep']
        outcome_vars = []
        for pattern in outcome_patterns:
            matches = [col for col in self.master_df.columns if pattern in col.lower()]
            outcome_vars.extend(matches)
        
        outcome_vars = list(set(outcome_vars))  # Remove duplicates
        
        if not outcome_vars:
            print("No outcome variables identified")
            return {}
        
        # Time trends
        trends = {}
        for var in outcome_vars[:5]:  # Limit to first 5 to avoid clutter
            if var in self.master_df.columns:
                yearly_mean = self.master_df.groupby('year')[var].mean()
                trends[var] = yearly_mean.to_dict()
        
        # Cross-sectional variation
        if outcome_vars and outcome_vars[0] in self.master_df.columns:
            latest_year = self.master_df['year'].max()
            latest_data = self.master_df[self.master_df['year'] == latest_year]
            cross_section = latest_data.groupby('state')[outcome_vars[0]].mean().sort_values()
        else:
            cross_section = pd.Series()
        
        results = {
            'outcome_variables': outcome_vars,
            'time_trends': trends,
            'cross_sectional_variation': cross_section.to_dict() if not cross_section.empty else {},
            'trend_analysis_year_range': (self.master_df['year'].min(), self.master_df['year'].max())
        }
        
        self.validation_results['outcomes'] = results
        return results
    
    def detect_outliers(self) -> Dict[str, any]:
        """Detect outliers in key variables."""
        if self.master_df is None:
            raise ValueError("Data not loaded")
        
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_results = {}
        
        for col in numeric_cols[:10]:  # Limit to avoid too much output
            if col in self.master_df.columns:
                data = self.master_df[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    outlier_results[col] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(data) * 100,
                        'bounds': (lower_bound, upper_bound)
                    }
        
        self.validation_results['outliers'] = outlier_results
        return outlier_results
    
    def run_full_validation(self) -> Dict[str, any]:
        """Run complete validation suite."""
        print("=== Running Full Data Validation ===")
        
        # Run all validation checks
        print("1. Checking data completeness...")
        completeness = self.validate_data_completeness()
        
        print("2. Validating panel structure...")
        panel = self.validate_panel_structure()
        
        print("3. Generating summary statistics...")
        summaries = self.generate_summary_statistics()
        
        print("4. Checking treatment balance...")
        balance = self.check_balance_by_treatment()
        
        print("5. Analyzing outcome trends...")
        outcomes = self.analyze_outcome_trends()
        
        print("6. Detecting outliers...")
        outliers = self.detect_outliers()
        
        return self.validation_results
    
    def generate_validation_report(self, output_path: str = "data/reports/validation_report.txt"):
        """Generate comprehensive validation report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("DATA VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Data Overview
            f.write("DATA OVERVIEW\n")
            f.write("-" * 20 + "\n")
            if 'completeness' in self.validation_results:
                comp = self.validation_results['completeness']
                f.write(f"Total Observations: {comp['total_observations']}\n")
                f.write(f"Total Variables: {comp['total_variables']}\n")
                f.write(f"Key Outcomes: {len(comp['key_outcomes'])}\n")
                f.write(f"Finance Variables: {len(comp['finance_variables'])}\n")
                f.write(f"Completely Missing Variables: {len(comp['completely_missing_vars'])}\n\n")
            
            # Panel Structure
            f.write("PANEL STRUCTURE\n")
            f.write("-" * 20 + "\n")
            if 'panel_structure' in self.validation_results:
                panel = self.validation_results['panel_structure']
                f.write(f"Year Range: {panel['year_range']}\n")
                f.write(f"Number of States: {panel['total_states']}\n")
                f.write(f"Number of Years: {panel['total_years']}\n")
                f.write(f"Balanced Panel: {panel['balanced_panel']}\n")
                f.write(f"Duplicate State-Years: {panel['duplicate_state_years']}\n\n")
            
            # Treatment Balance
            f.write("TREATMENT BALANCE\n")
            f.write("-" * 20 + "\n")
            if 'balance' in self.validation_results:
                balance = self.validation_results['balance']
                f.write(f"Treated States: {len(balance['treated_states'])}\n")
                f.write(f"Control States: {len(balance['control_states'])}\n\n")
            
            # Outcome Analysis
            f.write("OUTCOME VARIABLES\n")
            f.write("-" * 20 + "\n")
            if 'outcomes' in self.validation_results:
                outcomes = self.validation_results['outcomes']
                f.write(f"Outcome Variables Found: {len(outcomes['outcome_variables'])}\n")
                for var in outcomes['outcome_variables'][:5]:
                    f.write(f"  - {var}\n")
                f.write("\n")
            
            # Data Quality Issues
            f.write("DATA QUALITY ISSUES\n")
            f.write("-" * 20 + "\n")
            if 'completeness' in self.validation_results:
                comp = self.validation_results['completeness']
                if comp['high_missingness']:
                    f.write("Variables with >50% missing data:\n")
                    for var, pct in comp['high_missingness'].items():
                        f.write(f"  - {var}: {pct}%\n")
                else:
                    f.write("No variables with high missingness detected.\n")
        
        print(f"Validation report saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    # Run validation
    validator = DataValidator()
    
    # Run full validation suite
    results = validator.run_full_validation()
    
    # Generate report
    validator.generate_validation_report()
    
    # Print key findings
    print("\n=== KEY VALIDATION FINDINGS ===")
    if 'completeness' in results:
        comp = results['completeness']
        print(f"Dataset: {comp['total_observations']} obs, {comp['total_variables']} vars")
        print(f"Outcome variables: {len(comp['key_outcomes'])}")
        print(f"High missingness variables: {len(comp['high_missingness'])}")
    
    if 'panel_structure' in results:
        panel = results['panel_structure']
        print(f"Panel: {panel['total_states']} states, {panel['total_years']} years")
        print(f"Balanced: {panel['balanced_panel']}")
    
    if 'balance' in results:
        balance = results['balance']
        print(f"Treatment: {len(balance['treated_states'])} treated, {len(balance['control_states'])} control states")