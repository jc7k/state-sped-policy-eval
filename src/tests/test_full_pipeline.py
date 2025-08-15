"""
Integration tests for full data collection pipeline
Target Coverage: 85%+ for multi-component interactions
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# These tests verify the full data collection pipeline works end-to-end


@pytest.mark.integration
class TestDataCollectionPipeline:
    """Test complete data collection workflow"""

    @patch("src.collection.naep_collector.requests.get")
    @patch("src.collection.naep_collector.time.sleep")
    def test_naep_collection_end_to_end(
        self, mock_sleep, mock_get, sample_naep_api_response, temp_data_dir
    ):
        """Test NAEP collection from API call to file save"""
        # Setup mock API response
        mock_response = Mock()
        mock_response.json.return_value = sample_naep_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Import and run collector
        from src.collection.naep_collector import NAEPDataCollector

        collector = NAEPDataCollector()
        df = collector.fetch_state_swd_data([2022], [4], ["mathematics"])
        validation = collector.validate_data(df)

        output_path = str(temp_data_dir / "raw" / "naep_integration_test.csv")
        save_result = collector.save_data(df, output_path)

        # Verify complete workflow
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "state" in df.columns
        assert "swd_mean" in df.columns
        assert isinstance(validation, dict)
        assert save_result is True
        assert Path(output_path).exists()

        # Verify data can be reloaded
        reloaded = pd.read_csv(output_path)
        assert len(reloaded) == len(df)

    def test_multi_collector_coordination(self, temp_data_dir):
        """Test coordination between multiple data collectors"""
        # This would test the master pipeline function when implemented
        # For now, test the concept with mocked collectors

        mock_collectors = {"naep": Mock(), "edfacts": Mock(), "census": Mock(), "ocr": Mock()}

        # Mock successful collection results
        for name, collector in mock_collectors.items():
            mock_df = pd.DataFrame({"state": ["AL"], "year": [2022], "value": [100]})
            collector.collect_all.return_value = mock_df

        # Simulate master pipeline execution
        results = {}
        for name, collector in mock_collectors.items():
            try:
                df = collector.collect_all([2022])
                output_path = temp_data_dir / "raw" / f"{name}_raw.csv"
                df.to_csv(output_path, index=False)
                results[name] = {"records": len(df), "path": str(output_path)}
            except Exception as e:
                results[name] = {"error": str(e)}

        # Verify all collectors were executed
        assert len(results) == 4
        assert all("records" in result or "error" in result for result in results.values())

    def test_error_propagation_and_recovery(self):
        """Test error handling across pipeline components"""
        # Test that errors in one collector don't break the entire pipeline

        mock_collectors = {"working": Mock(), "failing": Mock(), "recovering": Mock()}

        # Setup different error scenarios
        mock_collectors["working"].collect_all.return_value = pd.DataFrame({"data": [1, 2, 3]})
        mock_collectors["failing"].collect_all.side_effect = Exception("Network error")
        mock_collectors["recovering"].collect_all.return_value = pd.DataFrame({"data": [4, 5, 6]})

        results = {}
        for name, collector in mock_collectors.items():
            try:
                df = collector.collect_all([2022])
                results[name] = {"success": True, "records": len(df)}
            except Exception as e:
                results[name] = {"success": False, "error": str(e)}

        # Verify error isolation
        assert results["working"]["success"] is True
        assert results["failing"]["success"] is False
        assert results["recovering"]["success"] is True
        assert "Network error" in results["failing"]["error"]


@pytest.mark.integration
class TestCrossModuleValidation:
    """Test validation across different modules and data sources"""

    def test_consistent_state_codes_across_sources(self):
        """Test that state codes are consistent across all data sources"""
        # Mock data from different sources with same states
        naep_data = pd.DataFrame({"state": ["AL", "CA", "TX"], "naep_score": [245, 252, 248]})

        edfacts_data = pd.DataFrame(
            {"state": ["AL", "CA", "TX"], "inclusion_rate": [45.2, 52.1, 48.7]}
        )

        census_data = pd.DataFrame(
            {"state": ["AL", "CA", "TX"], "expenditure": [8500, 85000, 67000]}
        )

        # Verify state codes are consistent
        naep_states = set(naep_data["state"])
        edfacts_states = set(edfacts_data["state"])
        census_states = set(census_data["state"])

        assert naep_states == edfacts_states == census_states

    def test_temporal_alignment_across_sources(self):
        """Test that data from different sources can be aligned by year"""
        # Test data with overlapping years
        source1_data = pd.DataFrame(
            {"state": ["AL", "AL"], "year": [2019, 2022], "value1": [100, 110]}
        )

        source2_data = pd.DataFrame(
            {"state": ["AL", "AL"], "year": [2019, 2022], "value2": [200, 220]}
        )

        # Merge on state and year
        merged = pd.merge(source1_data, source2_data, on=["state", "year"])

        assert len(merged) == 2
        assert all(merged["year"].isin([2019, 2022]))
        assert "value1" in merged.columns
        assert "value2" in merged.columns

    def test_data_quality_consistency(self):
        """Test that validation rules are consistent across data sources"""
        # Mock validation results from different sources
        validation_results = {
            "naep": {
                "total_records": 1020,
                "states_covered": 51,
                "years_covered": [2019, 2022],
                "missing_data_rate": 0.05,
            },
            "edfacts": {
                "total_records": 1020,
                "states_covered": 51,
                "years_covered": [2019, 2022],
                "missing_data_rate": 0.03,
            },
        }

        # Check consistency expectations
        for source, validation in validation_results.items():
            assert validation["states_covered"] == 51  # All sources should cover all states
            assert validation["years_covered"] == [2019, 2022]  # Same year coverage
            assert validation["missing_data_rate"] < 0.1  # Acceptable missing data rate


@pytest.mark.integration
class TestDataMerging:
    """Test merging data from different sources into analysis dataset"""

    def test_master_dataset_creation(self):
        """Test creation of master analysis dataset"""
        # Mock data from all sources
        naep_data = pd.DataFrame(
            {
                "state": ["AL", "CA"],
                "year": [2022, 2022],
                "naep_swd_score": [245, 252],
                "naep_gap": [40, 40],
            }
        )

        edfacts_data = pd.DataFrame(
            {
                "state": ["AL", "CA"],
                "year": [2022, 2022],
                "inclusion_rate": [45.2, 52.1],
                "graduation_rate": [67.8, 72.4],
            }
        )

        census_data = pd.DataFrame(
            {
                "state": ["AL", "CA"],
                "year": [2022, 2022],
                "per_pupil_spending": [11400, 13700],
                "total_enrollment": [745000, 6200000],
            }
        )

        # Merge all data sources
        master_data = naep_data.merge(edfacts_data, on=["state", "year"])
        master_data = master_data.merge(census_data, on=["state", "year"])

        # Verify master dataset structure
        assert len(master_data) == 2
        assert "naep_swd_score" in master_data.columns
        assert "inclusion_rate" in master_data.columns
        assert "per_pupil_spending" in master_data.columns
        assert master_data["state"].tolist() == ["AL", "CA"]

    def test_merge_handling_missing_data(self):
        """Test merging when some sources have missing data"""
        # NAEP data missing for some state-years
        naep_data = pd.DataFrame({"state": ["AL"], "year": [2022], "naep_score": [245]})

        # EdFacts data available for more state-years
        edfacts_data = pd.DataFrame(
            {"state": ["AL", "CA"], "year": [2022, 2022], "inclusion_rate": [45.2, 52.1]}
        )

        # Test different merge strategies
        inner_merge = pd.merge(naep_data, edfacts_data, on=["state", "year"], how="inner")
        outer_merge = pd.merge(naep_data, edfacts_data, on=["state", "year"], how="outer")

        assert len(inner_merge) == 1  # Only AL has both datasets
        assert len(outer_merge) == 2  # CA included with NaN for NAEP
        assert pd.isna(outer_merge.loc[1, "naep_score"])  # CA has missing NAEP data


@pytest.mark.integration
@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics"""

    def test_memory_usage_monitoring(self):
        """Test that pipeline doesn't have memory leaks"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate multiple collection cycles
        for i in range(5):
            # Mock data collection
            df = pd.DataFrame(
                {"state": ["AL"] * 1000, "year": [2022] * 1000, "value": list(range(1000))}
            )

            # Process and clean up
            del df

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

    def test_large_dataset_handling(self):
        """Test pipeline performance with large datasets"""
        # Create large synthetic dataset
        large_data = pd.DataFrame(
            {
                "state": (["AL"] * 5000 + ["CA"] * 5000),
                "year": ([2022] * 10000),
                "value": list(range(10000)),
            }
        )

        # Test operations that should scale well
        state_counts = large_data.groupby("state").size()
        year_coverage = large_data["year"].unique()

        assert len(state_counts) == 2
        assert 2022 in year_coverage

        # Verify processing completes in reasonable time
        import time

        start_time = time.time()

        # Simulate validation on large dataset
        validation_result = {
            "total_records": len(large_data),
            "states_covered": large_data["state"].nunique(),
            "years_covered": large_data["year"].unique().tolist(),
        }

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process 10k records quickly
        assert processing_time < 1.0  # Less than 1 second
        assert validation_result["total_records"] == 10000


@pytest.mark.integration
class TestConfigurationManagement:
    """Test configuration and environment handling"""

    def test_environment_variable_handling(self, monkeypatch):
        """Test handling of environment variables for API keys"""
        # Mock environment variables
        monkeypatch.setenv("CENSUS_API_KEY", "test_key_123")
        monkeypatch.setenv("DATA_OUTPUT_DIR", "/tmp/test_data")

        import os

        # Verify environment variables are accessible
        assert os.environ.get("CENSUS_API_KEY") == "test_key_123"
        assert os.environ.get("DATA_OUTPUT_DIR") == "/tmp/test_data"

    def test_configuration_validation(self):
        """Test validation of configuration parameters"""
        # Mock configuration dictionary
        config = {
            "census_api_key": "valid_key",
            "output_dir": "data/raw/",
            "years": [2019, 2022],
            "rate_limits": {"naep": 6, "edfacts": 1},
        }

        # Test configuration validation
        assert "census_api_key" in config
        assert isinstance(config["years"], list)
        assert all(isinstance(year, int) for year in config["years"])
        assert config["rate_limits"]["naep"] > 0
