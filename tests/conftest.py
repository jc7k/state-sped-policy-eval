"""
Pytest configuration and shared fixtures for state-sped-policy-eval tests
"""

import pytest
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

# Set test environment variables
os.environ['CENSUS_API_KEY'] = 'test_census_key_for_testing'
os.environ['TEST_MODE'] = 'true'
os.environ['DATA_OUTPUT_DIR'] = 'test_data/'
os.environ['RAW_DATA_DIR'] = 'test_data/raw/'


@pytest.fixture
def sample_naep_api_response():
    """Sample NAEP API response for testing"""
    return {
        'result': [
            {
                'jurisLabel': 'Alabama',
                'jurisdiction': 'AL',
                'varValue': '1',
                'varValueLabel': 'Identified as students with disabilities',
                'value': '245',
                'errorFlag': '3.2'
            },
            {
                'jurisLabel': 'Alabama', 
                'jurisdiction': 'AL',
                'varValue': '2',
                'varValueLabel': 'Not identified as students with disabilities',
                'value': '285',
                'errorFlag': '2.1'
            }
        ]
    }


@pytest.fixture
def expected_naep_dataframe():
    """Expected NAEP DataFrame structure"""
    return pd.DataFrame([
        {
            'state': 'AL',
            'year': 2022,
            'grade': 4,
            'subject': 'mathematics',
            'swd_mean': 245.0,
            'swd_se': 3.2,
            'non_swd_mean': 285.0,
            'non_swd_se': 2.1,
            'gap': 40.0,
            'gap_se': 3.84
        },
        {
            'state': 'CA',
            'year': 2022,
            'grade': 4,
            'subject': 'mathematics',
            'swd_mean': 252.0,
            'swd_se': 2.8,
            'non_swd_mean': 292.0,
            'non_swd_se': 1.9,
            'gap': 40.0,
            'gap_se': 3.39
        }
    ])


@pytest.fixture
def all_states_list():
    """Complete list of US states and DC"""
    return [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    ]


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    return Mock()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data"""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "final").mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_edfacts_response():
    """Sample EdFacts API response"""
    return {
        'data': [
            {
                'state': 'Alabama',
                'year': '2022',
                'child_count_total': '98567',
                'environment_regular_80plus': '45.2',
                'environment_separate_school': '3.1',
                'graduation_rate': '67.8',
                'dropout_rate': '8.9'
            }
        ]
    }


@pytest.fixture
def sample_census_response():
    """Sample Census API response"""
    return [
        ['Alabama', '2022', '15000000', '8500000', '1200000', '850000'],
        ['California', '2022', '85000000', '52000000', '9800000', '6200000']
    ]


@pytest.fixture
def test_years():
    """Standard test years for data collection"""
    return [2019, 2022]


@pytest.fixture
def test_grades():
    """Standard test grades"""
    return [4, 8]


@pytest.fixture
def test_subjects():
    """Standard test subjects"""
    return ['mathematics', 'reading']