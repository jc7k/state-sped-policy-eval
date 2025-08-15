#!/usr/bin/env python
"""
Realistic Test Data Generator

Replaces synthetic/mock data with realistic test scenarios that reflect
actual API response patterns, edge cases, and data quality issues found
in production educational data APIs.
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import pandas as pd

try:
    from .common import StateUtils
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from common import StateUtils


@dataclass
class NAEPTestScenario:
    """Configuration for NAEP test scenario generation."""
    year: int
    grade: int
    subject: str
    states: List[str]
    include_missing_data: bool = True
    include_suppressed_data: bool = True
    achievement_gap_range: tuple[float, float] = (25.0, 50.0)
    score_noise_level: float = 5.0


@dataclass
class CensusTestScenario:
    """Configuration for Census test scenario generation."""
    year: int
    states: List[str]
    include_missing_data: bool = True
    per_pupil_spending_range: tuple[int, int] = (8000, 25000)
    enrollment_range: tuple[int, int] = (50000, 6500000)


class RealisticTestDataGenerator:
    """
    Generate realistic test data that reflects actual API patterns.
    
    This replaces static mock data with dynamic, realistic test scenarios
    that include edge cases, missing data patterns, and data quality issues
    found in real educational data APIs.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with reproducible random seed."""
        self.state_utils = StateUtils()
        self.rng = random.Random(seed)
        
        # Realistic NAEP score distributions by state (approximate)
        self.naep_score_baselines = {
            'MA': {'swd': 250, 'non_swd': 295},  # High-performing
            'CA': {'swd': 252, 'non_swd': 292},  # Average-high
            'TX': {'swd': 248, 'non_swd': 289},  # Average
            'AL': {'swd': 245, 'non_swd': 285},  # Lower-performing
            'DC': {'swd': 238, 'non_swd': 276},  # Urban challenges
            'VT': {'swd': 255, 'non_swd': 298},  # Small state, high-performing
            'WY': {'swd': 247, 'non_swd': 288},  # Small state, average
        }
        
        # Default baseline for states not in specific mapping
        self.default_baseline = {'swd': 248, 'non_swd': 288}
        
        # Realistic Census spending patterns by state (approximate per-pupil)
        self.census_spending_baselines = {
            'NY': 25000, 'DC': 24000, 'CT': 22000, 'NJ': 21000,
            'VT': 20000, 'MA': 19000, 'RI': 18000, 'PA': 17000,
            'CA': 15000, 'TX': 12000, 'FL': 11000, 'AL': 9000,
            'ID': 8500, 'UT': 8200, 'AZ': 8800, 'NC': 9500
        }
        
    def generate_naep_response(self, scenario: NAEPTestScenario) -> Dict[str, Any]:
        """
        Generate realistic NAEP API response.
        
        Args:
            scenario: Test scenario configuration
            
        Returns:
            Realistic NAEP API response dictionary
        """
        result = []
        
        for state in scenario.states:
            state_name = self.state_utils.abbrev_to_name(state)
            if not state_name:
                continue
                
            # Get baseline scores for this state
            baselines = self.naep_score_baselines.get(state, self.default_baseline)
            
            # Add realistic noise to scores
            swd_score = baselines['swd'] + self.rng.gauss(0, scenario.score_noise_level)
            non_swd_score = baselines['non_swd'] + self.rng.gauss(0, scenario.score_noise_level)
            
            # Round to nearest 0.5 (typical NAEP reporting)
            swd_score = round(swd_score * 2) / 2
            non_swd_score = round(non_swd_score * 2) / 2
            
            # Handle missing/suppressed data scenarios
            swd_suppressed = (scenario.include_suppressed_data and 
                            self.rng.random() < 0.05)  # 5% suppression rate
            
            # Create records for both disability statuses
            for var_value, score, disability_status in [
                ("1", swd_score, "SWD"),
                ("2", non_swd_score, "non-SWD")
            ]:
                # Determine if this record should be suppressed
                is_suppressed = (var_value == "1" and swd_suppressed)
                
                record = {
                    "jurisLabel": state_name,
                    "jurisdiction": state,
                    "varValue": var_value,
                    "varValueLabel": (
                        "Identified as students with disabilities" if var_value == "1"
                        else "Not identified as students with disabilities"
                    ),
                    "isStatDisplayable": 0 if is_suppressed else 1
                }
                
                if is_suppressed:
                    # Use actual NAEP suppression symbols
                    record.update({
                        "value": "‡",
                        "errorFlag": "N/A"
                    })
                else:
                    record.update({
                        "value": str(score),
                        "errorFlag": str(round(self.rng.uniform(1.5, 4.5), 1))
                    })
                    
                result.append(record)
                
        return {"result": result}
    
    def generate_census_response(self, scenario: CensusTestScenario) -> List[List[str]]:
        """
        Generate realistic Census API response.
        
        Args:
            scenario: Test scenario configuration
            
        Returns:
            Realistic Census API response (list of lists)
        """
        headers = ["NAME", "TOTALEXP", "TCURINST", "TCURSSVC", "TCUROTH", "ENROLL", "state"]
        data_rows = []
        
        for state in scenario.states:
            state_name = self.state_utils.abbrev_to_name(state)
            state_fips = self.state_utils.abbrev_to_fips(state)
            
            if not state_name or not state_fips:
                continue
                
            # Generate realistic enrollment
            enrollment = self.rng.randint(*scenario.enrollment_range)
            
            # Get baseline per-pupil spending
            baseline_spending = self.census_spending_baselines.get(
                state, 
                self.rng.randint(*scenario.per_pupil_spending_range)
            )
            
            # Calculate total expenditures
            total_expenditures = int(baseline_spending * enrollment)
            
            # Distribute expenditures realistically
            current_instruction = int(total_expenditures * self.rng.uniform(0.55, 0.65))
            student_support = int(total_expenditures * self.rng.uniform(0.08, 0.15))
            other_current = total_expenditures - current_instruction - student_support
            
            # Handle missing data scenarios
            if scenario.include_missing_data and self.rng.random() < 0.02:  # 2% missing rate
                # Randomly make some fields missing
                missing_fields = self.rng.sample(
                    ["TOTALEXP", "TCURINST", "TCURSSVC", "TCUROTH"], 
                    k=self.rng.randint(1, 2)
                )
                for field in missing_fields:
                    if field == "TOTALEXP":
                        total_expenditures = "N"
                    elif field == "TCURINST":
                        current_instruction = "X"
                    elif field == "TCURSSVC":
                        student_support = "S"
                    elif field == "TCUROTH":
                        other_current = "D"
                        
            row = [
                state_name,
                str(total_expenditures),
                str(current_instruction),
                str(student_support),
                str(other_current),
                str(enrollment),
                state_fips
            ]
            
            data_rows.append(row)
            
        return [headers] + data_rows
    
    def generate_edge_case_scenarios(self) -> Dict[str, Any]:
        """
        Generate edge case test scenarios.
        
        Returns:
            Dictionary of edge case test data
        """
        edge_cases = {}
        
        # Empty response
        edge_cases['empty_naep'] = {"result": []}
        edge_cases['empty_census'] = [["NAME", "TOTALEXP", "state"]]
        
        # Invalid state codes
        edge_cases['invalid_states_naep'] = {
            "result": [
                {
                    "jurisLabel": "Invalid State",
                    "jurisdiction": "XX",
                    "varValue": "1",
                    "value": "250",
                    "errorFlag": "2.5"
                }
            ]
        }
        
        # Malformed data
        edge_cases['malformed_naep'] = {
            "result": [
                {
                    "jurisLabel": "California",
                    "jurisdiction": "CA",
                    "varValue": "1",
                    "value": "not_a_number",
                    "errorFlag": "invalid"
                }
            ]
        }
        
        # Extreme values
        edge_cases['extreme_values_naep'] = {
            "result": [
                {
                    "jurisLabel": "California", 
                    "jurisdiction": "CA",
                    "varValue": "1",
                    "value": "1000",  # Impossible NAEP score
                    "errorFlag": "2.5"
                }
            ]
        }
        
        return edge_cases
    
    def create_comprehensive_test_dataset(self, size: str = 'medium') -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive test dataset with multiple scenarios.
        
        Args:
            size: Dataset size ('small', 'medium', 'large')
            
        Returns:
            Dictionary of test DataFrames
        """
        size_configs = {
            'small': {'states': 5, 'years': 2},
            'medium': {'states': 15, 'years': 3}, 
            'large': {'states': 51, 'years': 5}
        }
        
        config = size_configs.get(size, size_configs['medium'])
        states = self.state_utils.get_all_states()[:config['states']]
        years = [2020, 2021, 2022][:config['years']]
        
        datasets = {}
        
        # Generate NAEP data
        naep_data = []
        for year in years:
            for grade in [4, 8]:
                for subject in ['mathematics', 'reading']:
                    scenario = NAEPTestScenario(
                        year=year,
                        grade=grade, 
                        subject=subject,
                        states=states
                    )
                    response = self.generate_naep_response(scenario)
                    
                    # Parse response using our standardized parser
                    try:
                        from .common import ResponseParserFactory
                    except ImportError:
                        from common import ResponseParserFactory
                    parser = ResponseParserFactory.create_parser('naep', self.state_utils)
                    records = parser.parse_response(
                        response, 
                        year=year, 
                        grade=grade, 
                        subject=subject
                    )
                    naep_data.extend(records)
                    
        datasets['naep'] = pd.DataFrame(naep_data)
        
        # Generate Census data
        census_data = []
        for year in years:
            scenario = CensusTestScenario(
                year=year,
                states=states
            )
            response = self.generate_census_response(scenario)
            
            # Parse response using our standardized parser
            parser = ResponseParserFactory.create_parser('census', self.state_utils)
            records = parser.parse_response(response, year=year)
            census_data.extend(records)
            
        datasets['census'] = pd.DataFrame(census_data)
        
        return datasets


def create_realistic_test_fixtures():
    """Create realistic test fixtures to replace synthetic data."""
    generator = RealisticTestDataGenerator(seed=42)
    
    # Create diverse test scenarios
    scenarios = {
        'standard': NAEPTestScenario(
            year=2022,
            grade=4,
            subject='mathematics',
            states=['CA', 'TX', 'NY', 'FL', 'PA']
        ),
        'with_suppression': NAEPTestScenario(
            year=2022,
            grade=8,
            subject='reading',
            states=['VT', 'WY', 'MT', 'ND', 'SD'],
            include_suppressed_data=True
        ),
        'large_states': NAEPTestScenario(
            year=2019,
            grade=4,
            subject='mathematics',
            states=['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
        )
    }
    
    # Generate and save realistic fixtures
    fixtures = {}
    for name, scenario in scenarios.items():
        fixtures[f'naep_{name}'] = generator.generate_naep_response(scenario)
        
    # Add edge cases
    fixtures.update(generator.generate_edge_case_scenarios())
    
    return fixtures


if __name__ == "__main__":
    # Generate comprehensive test data for validation
    generator = RealisticTestDataGenerator()
    test_datasets = generator.create_comprehensive_test_dataset('medium')
    
    print("Generated realistic test datasets:")
    for name, df in test_datasets.items():
        print(f"  {name}: {len(df)} records, {df.columns.tolist()}")
        
    print("Realistic test data generation completed!")