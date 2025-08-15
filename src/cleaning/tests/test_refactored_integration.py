#!/usr/bin/env python
"""
Tests for Refactored Data Integration Module

Tests the new modular components to ensure functionality is preserved
and the refactoring improvements work correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.cleaning import (
    DataCleaningUtils,
    DataIntegrator,
    DataLoaderFactory,
    IntegrationConfig,
    PanelDataCreator,
)


class TestDataCleaningUtils:
    """Test shared cleaning utilities."""

    def test_standardize_state_codes(self):
        """Test state code standardization."""
        df = pd.DataFrame(
            {"state": ["ca", "ny", "tx"], "year": [2020, 2020, 2020], "value": [1, 2, 3]}
        )

        result = DataCleaningUtils.standardize_state_codes(df)

        assert result["state"].tolist() == ["CA", "NY", "TX"]
        assert len(result) == 3  # All states should be valid

    def test_safe_numeric_conversion(self):
        """Test safe numeric conversion."""
        df = pd.DataFrame({"year": ["2020", "2021", "invalid"], "count": ["100", "200", "N/A"]})

        result = DataCleaningUtils.safe_numeric_conversion(df, ["year", "count"])

        assert result["year"].iloc[0] == 2020
        assert result["year"].iloc[1] == 2021
        assert pd.isna(result["year"].iloc[2])
        assert result["count"].iloc[0] == 100
        assert pd.isna(result["count"].iloc[2])


class TestDataLoaderFactory:
    """Test data loader factory."""

    def test_create_loader(self):
        """Test creating individual data loaders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            loader = DataLoaderFactory.create_loader("naep", temp_path)
            assert loader.__class__.__name__ == "NAEPDataLoader"

            loader = DataLoaderFactory.create_loader("census", temp_path)
            assert loader.__class__.__name__ == "CensusDataLoader"

    def test_create_all_loaders(self):
        """Test creating all data loaders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            loaders = DataLoaderFactory.create_all_loaders(temp_path)

            assert "naep" in loaders
            assert "census" in loaders
            assert "edfacts" in loaders
            assert "ocr" in loaders

            assert loaders["naep"].__class__.__name__ == "NAEPDataLoader"

    def test_invalid_source(self):
        """Test error handling for invalid data source."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(ValueError, match="Unsupported data source"):
                DataLoaderFactory.create_loader("invalid", temp_path)


class TestPanelDataCreator:
    """Test panel data creation."""

    def test_create_base_panel(self):
        """Test base panel creation."""
        creator = PanelDataCreator(year_range=(2020, 2022))
        base_panel = creator.create_base_panel()

        # Should have 51 states × 3 years = 153 observations
        assert len(base_panel) == 51 * 3
        assert "state" in base_panel.columns
        assert "year" in base_panel.columns
        assert base_panel["year"].min() == 2020
        assert base_panel["year"].max() == 2022

    def test_aggregate_naep_data(self):
        """Test NAEP data aggregation."""
        creator = PanelDataCreator()

        # Create sample NAEP data with multiple observations per state-year
        naep_data = pd.DataFrame(
            {
                "state": ["CA", "CA", "NY", "NY"],
                "year": [2020, 2020, 2020, 2020],
                "subject": ["math", "reading", "math", "reading"],
                "avg_scale_score": [250, 260, 240, 250],
            }
        )

        result = creator.aggregate_naep_data(naep_data)

        assert len(result) == 2  # Should aggregate to 2 state-year observations
        assert "naep_avg_scale_score" in result.columns
        # CA average should be (250+260)/2 = 255
        ca_score = result[result["state"] == "CA"]["naep_avg_scale_score"].iloc[0]
        assert ca_score == 255

    def test_create_panel_dataset(self):
        """Test full panel dataset creation."""
        creator = PanelDataCreator(year_range=(2020, 2021))

        # Create sample datasets
        datasets = {
            "naep": pd.DataFrame(
                {"state": ["CA", "NY"], "year": [2020, 2020], "avg_scale_score": [250, 240]}
            ),
            "census": pd.DataFrame(
                {"state": ["CA", "NY"], "year": [2020, 2020], "total_expenditure": [50000, 60000]}
            ),
        }

        result = creator.create_panel_dataset(datasets)

        # Should have full panel structure
        assert len(result) == 51 * 2  # 51 states × 2 years
        assert "naep_avg_scale_score" in result.columns
        assert "total_expenditure" in result.columns
        assert "post_2015" in result.columns  # Derived variable
        assert "post_covid" in result.columns  # Derived variable


class TestIntegrationConfig:
    """Test integration configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = IntegrationConfig()

        assert config.file_patterns is not None
        assert config.column_mappings is not None
        assert config.processing_params is not None
        assert "naep" in config.processing_params.DATASET_PREFIXES

    def test_get_data_source_config(self):
        """Test data source specific configuration."""
        config = IntegrationConfig()

        naep_config = config.get_data_source_config("naep")
        assert "file_patterns" in naep_config
        assert "required_columns" in naep_config
        assert naep_config["prefix"] == "naep_"

        census_config = config.get_data_source_config("census")
        assert census_config["prefix"] == ""  # No prefix for census


class TestDataIntegrator:
    """Test main data integrator."""

    def test_initialization(self):
        """Test integrator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            integrator = DataIntegrator(raw_data_dir=temp_path, output_dir=temp_path)

            assert integrator.config is not None
            assert integrator.data_loaders is not None
            assert integrator.panel_creator is not None
            assert len(integrator.data_loaders) == 4  # naep, census, edfacts, ocr

    @patch("src.cleaning.data_loaders.NAEPDataLoader.load_and_clean")
    @patch("src.cleaning.data_loaders.CensusDataLoader.load_and_clean")
    def test_load_datasets(self, mock_census_load, mock_naep_load):
        """Test dataset loading with mocked loaders."""
        # Mock return values
        mock_naep_load.return_value = pd.DataFrame(
            {"state": ["CA", "NY"], "year": [2020, 2020], "avg_scale_score": [250, 240]}
        )
        mock_census_load.return_value = pd.DataFrame(
            {"state": ["CA", "NY"], "year": [2020, 2020], "total_expenditure": [50000, 60000]}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            integrator = DataIntegrator(raw_data_dir=temp_path, output_dir=temp_path)

            datasets = integrator.load_datasets(sources=["naep", "census"])

            assert "naep" in datasets
            assert "census" in datasets
            assert len(datasets["naep"]) == 2
            assert len(datasets["census"]) == 2
            mock_naep_load.assert_called_once()
            mock_census_load.assert_called_once()

    def test_get_integration_summary(self):
        """Test integration summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            integrator = DataIntegrator(raw_data_dir=temp_path, output_dir=temp_path)

            # Create sample datasets
            datasets = {"naep": pd.DataFrame({"state": ["CA", "NY"], "year": [2020, 2020]})}
            master_df = pd.DataFrame({"state": ["CA", "NY"], "year": [2020, 2020]})

            summary = integrator.get_integration_summary(datasets, master_df)

            assert "individual_datasets" in summary
            assert "master_dataset" in summary
            assert "errors_warnings" in summary


class TestEndToEndIntegration:
    """Test end-to-end integration functionality."""

    def test_integration_with_empty_data(self):
        """Test integration when no data files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            integrator = DataIntegrator(raw_data_dir=temp_path, output_dir=temp_path)

            # Should handle missing data gracefully
            result = integrator.run_integration()

            # Should return empty DataFrame but not crash
            assert isinstance(result, pd.DataFrame)
            # May be empty or have just the base panel structure

    def test_refactoring_preserves_interface(self):
        """Test that refactoring preserves the original interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Should be able to create integrator with original signature
            integrator = DataIntegrator(raw_data_dir=temp_path, output_dir=temp_path)

            # Should have run_integration method
            assert hasattr(integrator, "run_integration")
            assert callable(integrator.run_integration)

            # Should be able to call without arguments (using defaults)
            result = integrator.run_integration()
            assert isinstance(result, pd.DataFrame)
