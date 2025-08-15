"""
Tests for Dashboard Generator

Tests the interactive HTML dashboard generation functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.reporting.dashboard_generator import DashboardGenerator


class TestDashboardGenerator:
    """Test DashboardGenerator functionality."""

    @pytest.fixture
    def mock_results(self):
        """Fixture providing mock analysis results."""
        return {
            "robustness_results": {
                "bootstrap_inference": {
                    "math_grade4_gap": {"coefficient": 0.15, "std_err": 0.08, "p_value": 0.05}
                },
                "jackknife_inference": {
                    "math_grade4_gap": {"coefficient": 0.14, "std_err": 0.09, "p_value": 0.12}
                },
                "wild_cluster_bootstrap": {
                    "math_grade4_gap": {"coefficient": 0.13, "std_err": 0.10, "p_value": 0.18}
                },
            },
            "causal_results": {
                "math_grade4_gap": {"coefficient": 0.16, "std_err": 0.07, "p_value": 0.03}
            },
            "enhanced_inference_results": {
                "power_analysis": {"overall_assessment": {"average_power": 0.65}},
                "effect_sizes": {
                    "math_grade4_gap": {"cohens_d": 0.12, "ci_lower": 0.02, "ci_upper": 0.22}
                },
                "multiple_testing_corrections": {
                    "original_p_values": {"math_grade4_gap": 0.03},
                    "bonferroni_corrected": {"math_grade4_gap": 0.12},
                    "fdr_corrected": {"math_grade4_gap": 0.08},
                    "romano_wolf_corrected": {"math_grade4_gap": 0.05},
                },
            },
        }

    @pytest.fixture
    def temp_output_dir(self):
        """Fixture providing temporary output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def generator(self, temp_output_dir, mock_results):
        """Fixture providing DashboardGenerator instance."""
        return DashboardGenerator(
            robustness_results=mock_results["robustness_results"],
            causal_results=mock_results["causal_results"],
            enhanced_inference_results=mock_results["enhanced_inference_results"],
        )

    def test_init_creates_output_directory(self, temp_output_dir):
        """Test that initialization creates output directory."""
        output_dir = temp_output_dir / "dashboards"
        generator = DashboardGenerator()
        generator.output_dir = output_dir

        # Directory should be created during __init__
        assert output_dir.exists()

    def test_create_summary_card(self, generator):
        """Test summary card generation."""
        summary = generator.create_summary_card()

        assert isinstance(summary, dict)
        assert "total_states" in summary
        assert "statistical_power" in summary
        assert summary["total_states"] == 50

    def test_generate_dashboard_creates_file(self, generator, temp_output_dir):
        """Test that dashboard generation creates HTML file."""
        generator.output_dir = temp_output_dir

        with patch("plotly.graph_objects.Figure.write_html") as mock_write:
            output_path = generator.generate_dashboard("test_dashboard.html")

            assert mock_write.called
            expected_path = str(temp_output_dir / "test_dashboard.html")
            assert output_path == expected_path

    def test_dashboard_with_empty_results(self, temp_output_dir):
        """Test dashboard generation with empty results."""
        generator = DashboardGenerator()
        generator.output_dir = temp_output_dir

        with patch("plotly.graph_objects.Figure.write_html"):
            # Should not raise an error with empty results
            output_path = generator.generate_dashboard()
            assert output_path is not None

    def test_dashboard_filename_handling(self, generator, temp_output_dir):
        """Test proper filename handling."""
        generator.output_dir = temp_output_dir

        with patch("plotly.graph_objects.Figure.write_html"):
            # Test default filename
            output_path = generator.generate_dashboard()
            assert output_path.endswith("analysis_dashboard.html")

            # Test custom filename
            output_path = generator.generate_dashboard("custom.html")
            assert output_path.endswith("custom.html")

    def test_treatment_effects_plot_data_handling(self, generator):
        """Test treatment effects plot handles data correctly."""
        # This tests the private method indirectly through the main function
        with patch("plotly.graph_objects.Figure") as mock_fig:
            mock_fig.return_value.write_html = Mock()
            generator.generate_dashboard()

            # Should have been called without errors
            assert mock_fig.called

    def test_method_comparison_with_missing_data(self, temp_output_dir):
        """Test method comparison handles missing data gracefully."""
        partial_results = {
            "bootstrap_inference": {"math_grade4_gap": {"coefficient": 0.15}}
            # Missing other methods
        }

        generator = DashboardGenerator(robustness_results=partial_results)
        generator.output_dir = temp_output_dir

        with patch("plotly.graph_objects.Figure.write_html"):
            # Should handle missing data without errors
            output_path = generator.generate_dashboard()
            assert output_path is not None

    def test_power_indicator_bounds(self, generator):
        """Test power indicator handles edge cases."""
        # Test with very low power
        generator.enhanced_inference_results = {
            "power_analysis": {"overall_assessment": {"average_power": 0.01}}
        }

        with patch("plotly.graph_objects.Figure.write_html"):
            output_path = generator.generate_dashboard()
            assert output_path is not None

        # Test with very high power
        generator.enhanced_inference_results = {
            "power_analysis": {"overall_assessment": {"average_power": 0.99}}
        }

        with patch("plotly.graph_objects.Figure.write_html"):
            output_path = generator.generate_dashboard()
            assert output_path is not None

    def test_effect_sizes_plot_empty_data(self, generator, temp_output_dir):
        """Test effect sizes plot with empty data."""
        generator.enhanced_inference_results = {"effect_sizes": {}}
        generator.output_dir = temp_output_dir

        with patch("plotly.graph_objects.Figure.write_html"):
            # Should handle empty effect sizes gracefully
            output_path = generator.generate_dashboard()
            assert output_path is not None
