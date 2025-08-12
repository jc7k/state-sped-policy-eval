#!/usr/bin/env python
"""
Data Validation Script
Validates data quality and coverage for the state special education analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging


def validate_master_dataset(file_path: Path) -> dict:
    """
    Validate the master analysis dataset
    
    Args:
        file_path: Path to master dataset CSV
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded master dataset: {df.shape}")
    
    validation_results = {
        'basic_info': {},
        'coverage': {},
        'data_quality': {},
        'key_variables': {},
        'temporal_coverage': {},
        'recommendations': []
    }
    
    # Basic information
    validation_results['basic_info'] = {
        'total_observations': len(df),
        'total_states': df['state'].nunique(),
        'year_range': f"{df['year'].min()}-{df['year'].max()}",
        'total_columns': len(df.columns),
        'missing_data_overall': df.isnull().sum().sum()
    }
    
    # Coverage by data source - use actual variable names from the dataset
    naep_vars = [col for col in df.columns if 'naep_' in col and col != 'naep_grade']
    edfacts_vars = [col for col in df.columns if 'edfacts_' in col]
    ocr_vars = [col for col in df.columns if 'ocr_' in col]
    
    key_vars = {
        'total_expenditure': 'Census Finance'
    }
    
    # Add first available variable from each source
    if naep_vars:
        key_vars[naep_vars[0]] = 'NAEP Achievement'
    if edfacts_vars:
        key_vars[edfacts_vars[0]] = 'EdFacts Special Ed'
    if ocr_vars:
        key_vars[ocr_vars[0]] = 'OCR Civil Rights'
    
    coverage = {}
    for var, source in key_vars.items():
        if var in df.columns:
            non_missing = df[var].notna().sum()
            coverage[source] = {
                'observations': non_missing,
                'percentage': round(non_missing / len(df) * 100, 1),
                'states_covered': df[df[var].notna()]['state'].nunique() if non_missing > 0 else 0,
                'years_covered': sorted(df[df[var].notna()]['year'].unique().tolist()) if non_missing > 0 else []
            }
        else:
            coverage[source] = {'observations': 0, 'percentage': 0.0, 'states_covered': 0, 'years_covered': []}
    
    validation_results['coverage'] = coverage
    
    # Data quality checks
    quality_issues = []
    
    # Check for duplicate state-year combinations
    duplicates = df.duplicated(subset=['state', 'year']).sum()
    if duplicates > 0:
        quality_issues.append(f"Found {duplicates} duplicate state-year combinations")
    
    # Check for missing state codes
    missing_states = df['state'].isnull().sum()
    if missing_states > 0:
        quality_issues.append(f"Found {missing_states} observations with missing state codes")
    
    # Check for invalid years
    invalid_years = df[(df['year'] < 2009) | (df['year'] > 2023)]['year'].nunique()
    if invalid_years > 0:
        quality_issues.append(f"Found {invalid_years} observations with invalid years")
    
    # Check for negative values in key financial variables
    financial_vars = ['total_revenue', 'total_expenditure', 'instruction_expenditure']
    for var in financial_vars:
        if var in df.columns:
            negative_count = (df[var] < 0).sum()
            if negative_count > 0:
                quality_issues.append(f"Found {negative_count} negative values in {var}")
    
    validation_results['data_quality'] = {
        'issues_found': len(quality_issues),
        'issues': quality_issues,
        'duplicate_state_years': duplicates,
        'data_integrity_score': max(0, 100 - len(quality_issues) * 10)  # Simple scoring
    }
    
    # Temporal coverage analysis
    temporal_coverage = {}
    for year in range(2009, 2024):
        year_data = df[df['year'] == year]
        temporal_coverage[year] = {
            'total_states': len(year_data),
            'states_with_data': {}
        }
        
        for var, source in key_vars.items():
            if var in df.columns:
                states_with_data = year_data[var].notna().sum()
                temporal_coverage[year]['states_with_data'][source] = states_with_data
    
    validation_results['temporal_coverage'] = temporal_coverage
    
    # Generate recommendations
    recommendations = []
    
    # Coverage recommendations
    for source, info in coverage.items():
        if info['percentage'] < 50:
            recommendations.append(f"LOW COVERAGE: {source} has only {info['percentage']}% coverage - consider alternative data sources")
        elif info['percentage'] < 80:
            recommendations.append(f"MODERATE COVERAGE: {source} has {info['percentage']}% coverage - acceptable for analysis")
    
    # Temporal recommendations
    recent_years = [2020, 2021, 2022, 2023]
    for year in recent_years:
        if year in temporal_coverage:
            total_coverage = sum(temporal_coverage[year]['states_with_data'].values())
            if total_coverage < 50:
                recommendations.append(f"RECENT DATA GAP: Limited data coverage for {year} - may affect COVID analysis")
    
    # Analysis recommendations
    naep_coverage = coverage.get('NAEP Achievement', {}).get('percentage', 0)
    finance_coverage = coverage.get('Census Finance', {}).get('percentage', 0)
    
    if naep_coverage > 30 and finance_coverage > 15:
        recommendations.append("ANALYSIS READY: Sufficient data for basic achievement-finance analysis")
    
    if coverage.get('EdFacts Special Ed', {}).get('percentage', 0) > 40:
        recommendations.append("SPECIAL ED ANALYSIS: Good coverage for special education enrollment analysis")
    
    if len(quality_issues) == 0:
        recommendations.append("DATA QUALITY: No major quality issues detected")
    else:
        recommendations.append(f"DATA QUALITY: {len(quality_issues)} quality issues need attention")
    
    validation_results['recommendations'] = recommendations
    
    return validation_results


def print_validation_report(results: dict):
    """Print a formatted validation report"""
    
    print("="*80)
    print("STATE SPECIAL EDUCATION POLICY EVALUATION")
    print("DATA VALIDATION REPORT")
    print("="*80)
    
    # Basic info
    basic = results['basic_info']
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Total Observations: {basic['total_observations']:,}")
    print(f"   States Covered: {basic['total_states']}")
    print(f"   Time Period: {basic['year_range']}")
    print(f"   Variables: {basic['total_columns']}")
    print(f"   Missing Values: {basic['missing_data_overall']:,}")
    
    # Coverage by source
    print(f"\n📈 DATA SOURCE COVERAGE")
    for source, info in results['coverage'].items():
        print(f"   {source}:")
        print(f"      Observations: {info['observations']:,} ({info['percentage']}%)")
        print(f"      States: {info['states_covered']}")
        if info['years_covered']:
            years_str = f"{min(info['years_covered'])}-{max(info['years_covered'])}"
            print(f"      Years: {years_str} ({len(info['years_covered'])} years)")
        else:
            print(f"      Years: No data")
    
    # Data quality
    quality = results['data_quality']
    print(f"\n🔍 DATA QUALITY ASSESSMENT")
    print(f"   Issues Found: {quality['issues_found']}")
    print(f"   Integrity Score: {quality['data_integrity_score']}/100")
    if quality['issues']:
        print(f"   Issues:")
        for issue in quality['issues']:
            print(f"      • {issue}")
    else:
        print(f"   ✅ No quality issues detected")
    
    # Recent years coverage (for COVID analysis)
    print(f"\n🦠 COVID-ERA COVERAGE (2020-2023)")
    temporal = results['temporal_coverage']
    for year in [2020, 2021, 2022, 2023]:
        if year in temporal:
            year_info = temporal[year]
            print(f"   {year}: {year_info['total_states']} states")
            for source, count in year_info['states_with_data'].items():
                if count > 0:
                    print(f"      {source}: {count} states")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Analysis readiness
    print(f"\n🎯 ANALYSIS READINESS")
    naep_pct = results['coverage'].get('NAEP Achievement', {}).get('percentage', 0)
    finance_pct = results['coverage'].get('Census Finance', {}).get('percentage', 0)
    edfacts_pct = results['coverage'].get('EdFacts Special Ed', {}).get('percentage', 0)
    
    if naep_pct > 30 and finance_pct > 15:
        print(f"   ✅ READY: Basic achievement-finance analysis")
    if edfacts_pct > 40:
        print(f"   ✅ READY: Special education enrollment analysis")
    if naep_pct > 25 and edfacts_pct > 30:
        print(f"   ✅ READY: SWD achievement gap analysis")
    
    covid_readiness = sum(temporal.get(2020, {}).get('states_with_data', {}).values())
    if covid_readiness > 100:
        print(f"   ✅ READY: COVID impact analysis")
    else:
        print(f"   ⚠️  LIMITED: COVID analysis may be constrained by data availability")
    
    print("="*80)


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate master analysis dataset')
    parser.add_argument('--data-file', type=Path, 
                       default='data/processed/master_analysis_dataset.csv',
                       help='Path to master dataset CSV')
    parser.add_argument('--output-dir', type=Path, default='data/processed',
                       help='Directory to save validation report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    if not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        return
    
    results = validate_master_dataset(args.data_file)
    
    # Print report
    print_validation_report(results)
    
    # Save results to JSON
    import json
    output_file = args.output_dir / 'data_validation_report.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📝 Validation report saved to: {output_file}")


if __name__ == "__main__":
    main()