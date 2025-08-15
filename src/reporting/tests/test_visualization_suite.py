"""
Tests for Visualization Suite

Tests the advanced visualization functionality for publication-quality figures.
"""

import tempfile
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
from unittest.mock import patch

from src.reporting.visualization_suite import VisualizationSuite


class TestVisualizationSuite:
    """Test VisualizationSuite functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def suite(self, temp_output_dir):
        """VisualizationSuite instance."""
        return VisualizationSuite(output_dir=temp_output_dir)

    @pytest.fixture
    def mock_results(self):
        """Mock analysis results."""
        return {
            "twfe": {"math_grade4_gap": {"coefficient": 0.15, "std_err": 0.08}},
            "bootstrap_inference": {"math_grade4_gap": {"coefficient": 0.14, "std_err": 0.09}},
            "jackknife_inference": {"math_grade4_gap": {"coefficient": 0.13, "std_err": 0.10}},
        }

    @pytest.fixture
    def mock_spec_results(self):
        """Mock specification curve results."""
        return {
            "math_grade4_gap": [
                {"coefficient": 0.15, "std_err": 0.08},
                {"coefficient": 0.12, "std_err": 0.09},
                {"coefficient": 0.18, "std_err": 0.07},
                {"coefficient": 0.11, "std_err": 0.10},
            ]
        }

    def test_init_creates_output_directory(self, temp_output_dir):
        """Test initialization creates output directory."""
        output_dir = temp_output_dir / "figures"
        VisualizationSuite(output_dir=output_dir)
        assert output_dir.exists()

    def test_create_forest_plot_grid(self, suite, mock_results, temp_output_dir):
        """Test forest plot grid generation."""
        output_path = suite.create_forest_plot_grid(mock_results, "test_forest_grid.png")

        expected_path = str(temp_output_dir / "test_forest_grid.png")
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_create_robustness_funnel_plot(self, suite, mock_spec_results, temp_output_dir):
        """Test funnel plot generation."""
        output_path = suite.create_robustness_funnel_plot(mock_spec_results, "test_funnel.png")

        expected_path = str(temp_output_dir / "test_funnel.png")
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_create_method_reliability_heatmap(self, suite, mock_results, temp_output_dir):
        """Test reliability heatmap generation."""
        output_path = suite.create_method_reliability_heatmap(mock_results, "test_heatmap.png")

        expected_path = str(temp_output_dir / "test_heatmap.png")
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_create_enhanced_specification_curve(self, suite, mock_spec_results, temp_output_dir):
        """Test enhanced specification curve generation."""
        output_path = suite.create_enhanced_specification_curve(
            mock_spec_results, "test_spec_curve.png"
        )

        expected_path = str(temp_output_dir / "test_spec_curve.png")
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_format_outcome_label(self, suite):
        """Test outcome label formatting."""
        assert suite._format_outcome_label("math_grade4_gap") == "Math G4 Gap"
        assert suite._format_outcome_label("reading_grade8_gap") == "Reading G8 Gap"

    def test_get_outcome_color(self, suite):
        """Test outcome color assignment."""
        math_color = suite._get_outcome_color("math_grade4_gap")
        reading_color = suite._get_outcome_color("reading_grade4_gap")

        assert math_color == suite.colors["primary"]
        assert reading_color == suite.colors["secondary"]

    def test_test_funnel_asymmetry(self, suite):
        """Test funnel plot asymmetry test."""
        effects = [0.1, 0.2, 0.15, 0.18, 0.12]
        precisions = [10, 8, 9, 11, 7]

        p_value = suite._test_funnel_asymmetry(effects, precisions)
        assert 0 <= p_value <= 1

    def test_empty_data_handling(self, suite, temp_output_dir):
        """Test handling of empty data."""
        empty_results = {}

        # Should not raise errors with empty data
        output_path = suite.create_forest_plot_grid(empty_results)
        assert Path(output_path).exists()

    def test_create_combined_figure(self, suite, temp_output_dir):
        """Test combined figure generation."""
        mock_all_results = {
            "causal": {"math_grade4_gap": {"coefficient": 0.15}},
            "robustness": {"bootstrap": {"math_grade4_gap": {"coefficient": 0.14}}},
            "power": {"average_power": 0.65},
            "effect_sizes": {"math_grade4_gap": {"cohens_d": 0.12}},
            "specification_curve": {"math_grade4_gap": [{"coefficient": 0.15}]},
        }

        output_path = suite.create_combined_figure(mock_all_results, "test_combined.png")

        expected_path = str(temp_output_dir / "test_combined.png")
        assert output_path == expected_path
        assert Path(output_path).exists()

    def test_plot_with_missing_outcome(self, suite):
        """Test plotting with missing outcome data."""
        incomplete_results = {"twfe": {"missing_outcome": {"coefficient": 0.15, "std_err": 0.08}}}

        # Should handle gracefully
        output_path = suite.create_forest_plot_grid(incomplete_results)
        assert Path(output_path).exists()

    def test_figure_size_customization(self, suite, mock_results, temp_output_dir):
        """Test custom figure size handling."""
        output_path = suite.create_forest_plot_grid(
            mock_results, "test_custom_size.png", figsize=(20, 16)
        )

        assert Path(output_path).exists()

    def test_matplotlib_backend_compatibility(self, suite, mock_results):
        """Test compatibility with different matplotlib backends."""
        # Should work with Agg backend (non-interactive)
        output_path = suite.create_forest_plot_grid(mock_results)
        assert Path(output_path).exists()

    @patch("matplotlib.pyplot.savefig")
    def test_high_dpi_output(self, mock_savefig, suite, mock_results):
        """Test high DPI output for publication quality."""
        suite.create_forest_plot_grid(mock_results)

        # Check that savefig was called with dpi=300
        mock_savefig.assert_called()
        args, kwargs = mock_savefig.call_args
        assert kwargs.get("dpi") == 300
