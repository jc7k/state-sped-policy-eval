"""
Unit tests for CensusEducationFinance collector
Target Coverage: 90%+ as specified in data-collection-prd.md Section 11
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.collection.census_collector import CensusEducationFinance


class TestCensusEducationFinanceInit:
    """Test CensusEducationFinance initialization"""

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key parameter"""
        collector = CensusEducationFinance(api_key="test_key_123")

        assert collector.api_key == "test_key_123"
        assert collector.base_url == "https://api.census.gov/data"
        assert collector.rate_limit_delay == 1.0
        assert collector.results == []
        assert collector.logger is not None

    def test_init_with_environment_api_key(self, monkeypatch):
        """Test initialization with API key from environment"""
        monkeypatch.setenv("CENSUS_API_KEY", "env_test_key")

        collector = CensusEducationFinance()

        assert collector.api_key == "env_test_key"

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test initialization fails without API key"""
        monkeypatch.delenv("CENSUS_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            CensusEducationFinance()

        assert "Census API key is required" in str(exc_info.value)
        assert "https://api.census.gov/data/key_signup.html" in str(exc_info.value)

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit"""
        collector = CensusEducationFinance(api_key="test_key", rate_limit_delay=2.5)

        assert collector.rate_limit_delay == 2.5

    def test_state_fips_mapping(self):
        """Test state FIPS code mapping is complete"""
        collector = CensusEducationFinance(api_key="test_key")

        # Should have 50 states + DC
        assert len(collector.state_fips) == 51
        assert "01" in collector.state_fips  # Alabama
        assert "06" in collector.state_fips  # California
        assert "11" in collector.state_fips  # DC
        assert collector.state_fips["01"] == "AL"
        assert collector.state_fips["06"] == "CA"
        assert collector.state_fips["11"] == "DC"


class TestFetchStateFinance:
    """Test main data collection method"""

    @patch("requests.get")
    @patch("time.sleep")
    def test_successful_single_year(self, mock_sleep, mock_get, census_sample_response):
        """Test successful data collection for single year"""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = census_sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = CensusEducationFinance(api_key="test_key")
        result = collector.fetch_state_finance([2022])

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args

        # Check endpoint construction
        assert "2022" in call_args[0][0]
        assert "programs/finances/elementary-secondary-education" in call_args[0][0]

        # Check parameters
        params = call_args[1]["params"]
        assert params["key"] == "test_key"
        assert "NAME,TOTALEXP,TCURINST,TCURSSVC,TCUROTH,ENROLL" in params["get"]
        assert params["for"] == "state:*"

        # Verify result structure
        assert isinstance(result, pd.DataFrame)

    @patch("requests.get")
    @patch("time.sleep")
    def test_multiple_years(self, mock_sleep, mock_get, census_sample_response):
        """Test data collection across multiple years"""
        mock_response = Mock()
        mock_response.json.return_value = census_sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = CensusEducationFinance(api_key="test_key")
        years = [2019, 2020, 2021]

        result = collector.fetch_state_finance(years)

        # Should make 3 requests (one per year)
        assert mock_get.call_count == 3

        # Should sleep 2 times (between requests, not after last)
        assert mock_sleep.call_count == 2

    @patch("requests.get")
    def test_request_exception_handling(self, mock_get):
        """Test handling of network request exceptions"""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        collector = CensusEducationFinance(api_key="test_key")
        result = collector.fetch_state_finance([2022])

        # Should return empty DataFrame when all requests fail
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("requests.get")
    def test_different_endpoint_for_older_years(self, mock_get):
        """Test that older years use different API endpoint"""
        mock_response = Mock()
        mock_response.json.return_value = [[], []]  # Empty data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = CensusEducationFinance(api_key="test_key")
        collector.fetch_state_finance([2012])  # Pre-2013 year

        call_args = mock_get.call_args
        endpoint = call_args[0][0]

        # Should use old endpoint format
        assert "governments/school-districts/finance" in endpoint
        assert "2012" in endpoint


class TestParseFinanceRecord:
    """Test individual record parsing"""

    def test_valid_record_parsing(self):
        """Test parsing of valid finance record"""
        collector = CensusEducationFinance(api_key="test_key")

        row_data = {
            "NAME": "Alabama",
            "state": "01",  # Alabama FIPS
            "TOTALEXP": "8500000000",
            "TCURINST": "4200000000",
            "TCURSSVC": "1200000000",
            "TCUROTH": "850000000",
            "ENROLL": "745000",
        }

        result = collector._parse_finance_record(row_data, 2022)

        assert result is not None
        assert result["state"] == "AL"
        assert result["year"] == 2022
        assert result["total_expenditures"] == 8500000000
        assert result["current_instruction"] == 4200000000
        assert result["student_support_services"] == 1200000000
        assert result["enrollment"] == 745000

        # Check calculated per-pupil amounts
        assert result["per_pupil_spending"] is not None
        assert result["per_pupil_spending"] == 8500000000 / 745000
        assert result["support_services_per_pupil"] == 1200000000 / 745000

    def test_missing_enrollment_data(self):
        """Test parsing when enrollment data is missing"""
        collector = CensusEducationFinance(api_key="test_key")

        row_data = {
            "NAME": "Alabama",
            "state": "01",
            "TOTALEXP": "8500000000",
            "ENROLL": "N",  # Missing data code
        }

        result = collector._parse_finance_record(row_data, 2022)

        assert result is not None
        assert result["enrollment"] is None
        assert result["per_pupil_spending"] is None  # Can't calculate without enrollment

    def test_invalid_state_code(self):
        """Test parsing with invalid state code"""
        collector = CensusEducationFinance(api_key="test_key")

        row_data = {
            "NAME": "Unknown Territory",
            "state": "99",  # Invalid FIPS
            "TOTALEXP": "1000000",
        }

        result = collector._parse_finance_record(row_data, 2022)

        assert result is None


class TestSafeInt:
    """Test safe integer conversion utility"""

    def test_valid_integer_conversion(self):
        """Test conversion of valid integer values"""
        collector = CensusEducationFinance(api_key="test_key")

        # Now using SafeTypeConverter through collector.converter
        assert collector.converter.safe_int("12345") == 12345
        assert collector.converter.safe_int("0") == 0
        assert collector.converter.safe_int(67890) == 67890

    def test_invalid_values(self):
        """Test handling of invalid values"""
        collector = CensusEducationFinance(api_key="test_key")

        invalid_values = [None, "", "null", "N", "X", "S", "D", "invalid"]

        for value in invalid_values:
            assert collector.converter.safe_int(value) is None


class TestStateNameConversion:
    """Test state name to code conversion"""

    def test_valid_state_names(self):
        """Test conversion of valid state names"""
        collector = CensusEducationFinance(api_key="test_key")

        state_names = {
            "Alabama": "AL",
            "California": "CA",
            "Texas": "TX",
            "New York": "NY",
            "District of Columbia": "DC",
        }

        for name, expected_code in state_names.items():
            assert collector._convert_state_name_to_code(name) == expected_code

    def test_invalid_state_name(self):
        """Test handling of invalid state names"""
        collector = CensusEducationFinance(api_key="test_key")

        assert collector._convert_state_name_to_code("Invalid State") is None
        assert collector._convert_state_name_to_code("") is None
        assert collector._convert_state_name_to_code(None) is None


class TestValidateData:
    """Test data validation functionality"""

    def test_valid_dataframe_validation(self):
        """Test validation of valid DataFrame"""
        collector = CensusEducationFinance(api_key="test_key")

        df = pd.DataFrame(
            [
                {
                    "state": "AL",
                    "year": 2022,
                    "total_expenditures": 8500000000,
                    "enrollment": 745000,
                },
                {
                    "state": "CA",
                    "year": 2022,
                    "total_expenditures": 85000000000,
                    "enrollment": 6200000,
                },
            ]
        )

        validation = collector.validate_data(df)

        assert validation["total_records"] == 2
        assert validation["states_covered"] == 2
        assert validation["years_covered"] == [2022]
        assert validation["missing_expenditures"] == 0
        assert validation["missing_enrollment"] == 0

    def test_insufficient_state_coverage(self):
        """Test validation fails with insufficient state coverage"""
        collector = CensusEducationFinance(api_key="test_key")

        # Create DataFrame with only a few states
        df = pd.DataFrame([{"state": "AL", "year": 2022, "total_expenditures": 8500000000}])

        validation = collector.validate_data(df)

        assert not validation["passed"]
        assert any("Only 1 states covered" in error for error in validation["errors"])

    def test_negative_expenditures_detected(self):
        """Test detection of negative expenditure values"""
        collector = CensusEducationFinance(api_key="test_key")

        df = pd.DataFrame(
            [
                {"state": "AL", "year": 2022, "total_expenditures": -1000000}  # Invalid
            ]
        )

        validation = collector.validate_data(df)

        assert not validation["passed"]
        assert any("negative expenditure" in error for error in validation["errors"])


class TestSaveData:
    """Test data saving functionality"""

    def test_successful_save(self, temp_data_dir):
        """Test successful data saving"""
        collector = CensusEducationFinance(api_key="test_key")

        df = pd.DataFrame([{"state": "AL", "year": 2022, "total_expenditures": 8500000000}])

        output_path = str(temp_data_dir / "raw" / "test_census.csv")
        result = collector.save_data(df, output_path)

        assert result is True
        assert Path(output_path).exists()

        # Verify saved data can be read back
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(df)


@pytest.fixture
def census_sample_response():
    """Sample Census API response for testing"""
    return [
        ["NAME", "TOTALEXP", "TCURINST", "TCURSSVC", "TCUROTH", "ENROLL", "state"],
        ["Alabama", "8500000000", "4200000000", "1200000000", "850000000", "745000", "01"],
        ["California", "85000000000", "52000000000", "9800000000", "6200000000", "6200000", "06"],
    ]
