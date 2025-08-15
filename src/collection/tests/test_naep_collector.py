"""
Comprehensive unit tests for NAEPDataCollector
Target Coverage: 95%+ as specified in data-collection-prd.md Section 11
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException, Timeout

from src.collection.naep_collector import NAEPDataCollector


class TestNAEPDataCollectorInit:
    """Test NAEPDataCollector initialization"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        collector = NAEPDataCollector()

        assert (
            collector.base_url == "https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx"
        )
        assert collector.rate_limit_delay == 6.0
        assert collector.results == []
        assert collector.logger is not None

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit"""
        collector = NAEPDataCollector(rate_limit_delay=10.0)

        assert collector.rate_limit_delay == 10.0

    def test_init_logger_setup(self):
        """Test logger is properly initialized"""
        collector = NAEPDataCollector()

        # Logger name is now just the class name due to refactoring
        assert collector.logger.name == "NAEPDataCollector"


class TestFetchStateSWDData:
    """Test main data collection method"""

    @patch("src.collection.common.requests.get")
    @patch("src.collection.common.time.sleep")
    def test_successful_single_request(self, mock_sleep, mock_get, sample_naep_api_response):
        """Test successful data collection for single request"""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = sample_naep_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = NAEPDataCollector()
        result = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        # The actual implementation makes one call per state (51 total)
        assert mock_get.call_count == 51  # 50 states + DC

        # Verify first API call has correct structure
        first_call = mock_get.call_args_list[0]
        first_call_params = (
            first_call.kwargs["params"] if first_call.kwargs else first_call[1]["params"]
        )
        assert first_call_params["type"] == "data"
        assert first_call_params["subject"] == "mathematics"
        assert first_call_params["grade"] == 4
        assert first_call_params["year"] == 2022
        assert first_call_params["variable"] == "IEP"  # Updated variable name
        assert first_call_params["stattype"] == "MN:MN"
        assert "jurisdiction" in first_call_params  # Should be a state code

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ["state", "year", "grade", "subject"])

    @patch("src.collection.common.requests.get")
    @patch("src.collection.common.time.sleep")
    def test_multiple_years_grades_subjects(self, mock_sleep, mock_get, sample_naep_api_response):
        """Test data collection across multiple years, grades, and subjects"""
        mock_response = Mock()
        mock_response.json.return_value = sample_naep_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = NAEPDataCollector()
        years = [2019, 2022]
        grades = [4, 8]
        subjects = ["mathematics", "reading"]

        result = collector.fetch_state_swd_data(years, grades, subjects)

        # Should make 408 requests (2 years × 2 grades × 2 subjects × 51 states)
        assert mock_get.call_count == 408

        # Should sleep between requests (not after last)
        assert mock_sleep.call_count >= 407

    @patch("src.collection.common.requests.get")
    def test_request_exception_handling(self, mock_get):
        """Test handling of network request exceptions"""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        collector = NAEPDataCollector()
        result = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        # Should return empty DataFrame when all requests fail
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.collection.common.requests.get")
    def test_http_error_handling(self, mock_get):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        collector = NAEPDataCollector()
        result = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.collection.common.requests.get")
    def test_timeout_handling(self, mock_get):
        """Test handling of request timeouts"""
        mock_get.side_effect = Timeout("Request timeout")

        collector = NAEPDataCollector()
        result = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.collection.common.requests.get")
    def test_malformed_json_response(self, mock_get):
        """Test handling of malformed JSON responses"""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = NAEPDataCollector()
        result = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.collection.common.requests.get")
    def test_empty_api_response(self, mock_get):
        """Test handling of empty API response"""
        mock_response = Mock()
        mock_response.json.return_value = {"result": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = NAEPDataCollector()
        result = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestParseStateRecord:
    """Test individual state record parsing"""

    def test_valid_state_record_parsing(self):
        """Test parsing of valid state record"""
        collector = NAEPDataCollector()

        # Use the format that the current implementation expects
        state_data = {
            "jurisLabel": "Alabama",
            "jurisdiction": "AL",
            "varValue": "1",  # SWD
            "varValueLabel": "Identified as students with disabilities",
            "value": "245",
            "errorFlag": "3.2",
            "isStatDisplayable": 1,
        }

        result = collector._parse_state_record(state_data, 2022, 4, "mathematics")

        assert result is not None
        assert result["state"] == "AL"
        assert result["year"] == 2022
        assert result["grade"] == 4
        assert result["subject"] == "mathematics"
        assert result["disability_status"] == "SWD"
        assert result["mean_score"] == 245.0
        assert result["error_flag"] == "3.2"

    def test_unknown_state_name(self):
        """Test handling of unknown state names"""
        collector = NAEPDataCollector()

        state_data = {"name": "Unknown State", "datavalue": []}

        result = collector._parse_state_record(state_data, 2022, 4, "mathematics")

        assert result is None

    def test_missing_swd_data(self):
        """Test parsing when required data is missing"""
        collector = NAEPDataCollector()

        # Test with missing state name
        state_data = {
            "jurisdiction": "AL",
            "varValue": "1",
            "value": "245",
            "errorFlag": "3.2",
            # Missing 'jurisLabel'
        }

        result = collector._parse_state_record(state_data, 2022, 4, "mathematics")

        assert result is None  # Should return None when required data is missing

    def test_missing_non_swd_data(self):
        """Test parsing when non-SWD data is missing"""
        collector = NAEPDataCollector()

        state_data = {
            "name": "Alabama",
            "datavalue": [
                {"categoryname": "Students with IEP - Yes", "value": "245", "errorFlag": "3.2"}
            ],
        }

        result = collector._parse_state_record(state_data, 2022, 4, "mathematics")

        assert result is not None
        assert result["swd_mean"] == 245.0
        assert result["non_swd_mean"] is None
        assert result["gap"] is None

    def test_malformed_state_data(self):
        """Test parsing of malformed state data"""
        collector = NAEPDataCollector()

        state_data = {
            "name": "Alabama"
            # Missing 'datavalue' key
        }

        result = collector._parse_state_record(state_data, 2022, 4, "mathematics")

        assert result is not None  # Should still create record with None values
        assert result["swd_mean"] is None
        assert result["non_swd_mean"] is None


class TestSafeFloat:
    """Test safe float conversion utility"""

    def test_valid_float_conversion(self):
        """Test conversion of valid float values"""
        collector = NAEPDataCollector()

        assert collector._safe_float("245.5") == 245.5
        assert collector._safe_float("0") == 0.0
        assert collector._safe_float(123) == 123.0

    def test_invalid_values(self):
        """Test handling of invalid values"""
        collector = NAEPDataCollector()

        invalid_values = [None, "", "null", "‡", "*", "N/A", "#", "invalid"]

        for value in invalid_values:
            assert collector._safe_float(value) is None

    def test_boundary_values(self):
        """Test boundary float values"""
        collector = NAEPDataCollector()

        assert collector._safe_float("0.0") == 0.0
        assert collector._safe_float("500.0") == 500.0
        assert collector._safe_float("-1") == -1.0


class TestStateNameConversion:
    """Test state name to code conversion"""

    def test_valid_state_names(self, all_states_list):
        """Test conversion of all valid state names"""
        collector = NAEPDataCollector()

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
        collector = NAEPDataCollector()

        assert collector._convert_state_name_to_code("Invalid State") is None
        assert collector._convert_state_name_to_code("") is None
        assert collector._convert_state_name_to_code(None) is None

    def test_case_sensitivity(self):
        """Test state name case handling"""
        collector = NAEPDataCollector()

        # Current implementation is case-sensitive
        assert collector._convert_state_name_to_code("alabama") is None
        assert collector._convert_state_name_to_code("ALABAMA") is None
        assert collector._convert_state_name_to_code("Alabama") == "AL"

    def test_whitespace_handling(self):
        """Test whitespace handling in state names"""
        collector = NAEPDataCollector()

        assert collector._convert_state_name_to_code(" Alabama ") == "AL"
        assert collector._convert_state_name_to_code("\tAlabama\n") == "AL"


class TestValidateData:
    """Test data validation functionality"""

    def test_valid_dataframe_validation(self, expected_naep_dataframe):
        """Test validation of valid DataFrame"""
        collector = NAEPDataCollector()

        validation = collector.validate_data(expected_naep_dataframe)

        assert validation["total_records"] == 2
        assert validation["states_covered"] == 2
        assert validation["years_covered"] == [2022]
        assert validation["subjects_covered"] == ["mathematics"]
        assert validation["grades_covered"] == [4]
        assert validation["missing_swd_scores"] == 0
        assert validation["missing_gaps"] == 0
        assert len(validation["errors"]) == 0

    def test_empty_dataframe_validation(self):
        """Test validation of empty DataFrame"""
        collector = NAEPDataCollector()
        empty_df = pd.DataFrame()

        validation = collector.validate_data(empty_df)

        assert validation["total_records"] == 0
        assert validation["states_covered"] == 0
        assert validation["years_covered"] == []

    def test_insufficient_state_coverage(self):
        """Test validation fails with insufficient state coverage"""
        collector = NAEPDataCollector()

        # Create DataFrame with only a few states
        df = pd.DataFrame(
            [
                {
                    "state": "AL",
                    "year": 2022,
                    "grade": 4,
                    "subject": "math",
                    "swd_mean": 245,
                    "non_swd_mean": 285,
                    "gap": 40,
                }
            ]
        )

        validation = collector.validate_data(df)

        assert not validation["passed"]
        assert any("Only 1 states covered" in error for error in validation["errors"])

    def test_duplicate_combinations(self):
        """Test detection of duplicate state-year-grade-subject combinations"""
        collector = NAEPDataCollector()

        df = pd.DataFrame(
            [
                {"state": "AL", "year": 2022, "grade": 4, "subject": "math", "swd_mean": 245},
                {
                    "state": "AL",
                    "year": 2022,
                    "grade": 4,
                    "subject": "math",
                    "swd_mean": 245,
                },  # Duplicate
            ]
        )

        validation = collector.validate_data(df)

        assert not validation["passed"]
        assert any("duplicate" in error for error in validation["errors"])

    def test_invalid_score_ranges(self):
        """Test detection of invalid score ranges"""
        collector = NAEPDataCollector()

        df = pd.DataFrame(
            [
                {
                    "state": "AL",
                    "year": 2022,
                    "grade": 4,
                    "subject": "math",
                    "swd_mean": 600,
                    "non_swd_mean": -50,
                }  # Invalid ranges
            ]
        )

        validation = collector.validate_data(df)

        assert len(validation["warnings"]) >= 2  # Should warn about both invalid scores

    def test_high_missing_data_rate(self):
        """Test detection of high missing data rates"""
        collector = NAEPDataCollector()

        # Create DataFrame with mostly missing SWD scores
        df = pd.DataFrame(
            [
                {"state": f"S{i}", "year": 2022, "grade": 4, "subject": "math", "swd_mean": None}
                for i in range(100)
            ]
        )

        validation = collector.validate_data(df)

        assert any("High missing data rate" in warning for warning in validation["warnings"])


class TestSaveData:
    """Test data saving functionality"""

    def test_successful_save(self, expected_naep_dataframe, temp_data_dir):
        """Test successful data saving"""
        collector = NAEPDataCollector()

        output_path = str(temp_data_dir / "raw" / "test_naep.csv")
        result = collector.save_data(expected_naep_dataframe, output_path)

        assert result is True
        assert Path(output_path).exists()

        # Verify saved data can be read back
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(expected_naep_dataframe)

    def test_save_creates_directory(self, expected_naep_dataframe, tmp_path):
        """Test that save creates output directory if it doesn't exist"""
        collector = NAEPDataCollector()

        output_path = str(tmp_path / "new_dir" / "naep_data.csv")
        result = collector.save_data(expected_naep_dataframe, output_path)

        assert result is True
        assert Path(output_path).exists()
        assert Path(output_path).parent.exists()

    @patch("pathlib.Path.mkdir")
    def test_save_permission_error(self, mock_mkdir, expected_naep_dataframe):
        """Test handling of permission errors during save"""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        collector = NAEPDataCollector()
        result = collector.save_data(expected_naep_dataframe, "/invalid/path/naep.csv")

        assert result is False


class TestIntegrationScenarios:
    """Integration-style tests within unit test scope"""

    @patch("src.collection.common.requests.get")
    @patch("src.collection.common.time.sleep")
    def test_full_collection_workflow(
        self, mock_sleep, mock_get, sample_naep_api_response, temp_data_dir
    ):
        """Test complete collection workflow"""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = sample_naep_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        collector = NAEPDataCollector()

        # Collect data
        df = collector.fetch_state_swd_data([2022], [4], ["mathematics"])

        # Validate data
        validation = collector.validate_data(df)

        # Save data
        output_path = str(temp_data_dir / "raw" / "naep_test.csv")
        save_result = collector.save_data(df, output_path)

        # Verify workflow completion
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(validation, dict)
        assert save_result is True
        assert Path(output_path).exists()

    def test_error_recovery_partial_success(self):
        """Test partial success scenario with some failed requests"""
        collector = NAEPDataCollector()

        with patch("src.collection.naep_collector.requests.get") as mock_get:
            # First request succeeds, second fails
            success_response = Mock()
            success_response.json.return_value = {"result": []}
            success_response.raise_for_status.return_value = None

            mock_get.side_effect = [success_response, RequestException("Network error")]

            with patch("src.collection.naep_collector.time.sleep"):
                df = collector.fetch_state_swd_data([2022], [4], ["mathematics", "reading"])

            # Should still return DataFrame, even with partial failures
            assert isinstance(df, pd.DataFrame)
            assert mock_get.call_count == 2


# Property-based testing using hypothesis
try:
    from hypothesis import given
    from hypothesis import strategies as st

    class TestPropertyBased:
        """Property-based tests using hypothesis"""

        @given(st.floats(min_value=0, max_value=500, allow_nan=False))
        def test_safe_float_preserves_valid_ranges(self, value):
            """Property: valid NAEP scores should be preserved"""
            collector = NAEPDataCollector()
            result = collector._safe_float(str(value))
            assert result == value

        @given(st.text())
        def test_state_conversion_returns_valid_codes_or_none(self, state_name):
            """Property: state conversion returns valid 2-letter codes or None"""
            collector = NAEPDataCollector()
            result = collector._convert_state_name_to_code(state_name)

            if result is not None:
                assert isinstance(result, str)
                assert len(result) == 2
                assert result.isupper()

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass
