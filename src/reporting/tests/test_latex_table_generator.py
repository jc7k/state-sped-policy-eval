"""
Tests for LaTeX Table Generator

Tests the publication-ready LaTeX table generation functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.reporting.latex_table_generator import LaTeXTableGenerator


class TestLaTeXTableGenerator:
    """Test LaTeXTableGenerator functionality."""

    @pytest.fixture
    def mock_results(self):
        """Mock analysis results for testing."""
        return {
            "twfe": {"math_grade4_gap": {"coefficient": 0.15, "std_err": 0.08, "p_value": 0.06}},
            "bootstrap_inference": {
                "math_grade4_gap": {"coefficient": 0.14, "std_err": 0.09, "p_value": 0.12}
            },
            "jackknife_inference": {
                "math_grade4_gap": {"coefficient": 0.13, "std_err": 0.10, "p_value": 0.19}
            },
            "wild_cluster_bootstrap": {
                "math_grade4_gap": {"coefficient": 0.12, "std_err": 0.11, "p_value": 0.28}
            },
        }

    @pytest.fixture
    def mock_effect_sizes(self):
        """Mock effect size data."""
        return {
            "math_grade4_gap": {
                "cohens_d": 0.12,
                "ci_lower": 0.02,
                "ci_upper": 0.22,
                "cross_method_range": [0.10, 0.16],
                "cross_method_consistency": "High",
            }
        }

    @pytest.fixture
    def mock_power_results(self):
        """Mock power analysis results."""
        return {
            "by_outcome": {
                "math_grade4_gap": {
                    "post_hoc_power": 0.65,
                    "minimum_detectable_effect": 0.25,
                    "required_sample_size": 1200,
                    "current_sample_size": 700,
                    "adequately_powered": False,
                }
            },
            "overall_assessment": {
                "average_power": 0.58,
                "median_power": 0.62,
                "min_power": 0.45,
                "proportion_adequate": 0.33,
                "recommendation": "Increase sample size for improved power",
            },
        }

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def generator(self, temp_output_dir):
        """LaTeXTableGenerator instance."""
        return LaTeXTableGenerator(output_dir=temp_output_dir)

    def test_init_creates_output_directory(self, temp_output_dir):
        """Test initialization creates output directory."""
        output_dir = temp_output_dir / "tables"
        LaTeXTableGenerator(output_dir=output_dir)
        assert output_dir.exists()

    def test_create_multi_method_comparison_table(self, generator, mock_results):
        """Test multi-method comparison table generation."""
        output_path = generator.create_multi_method_comparison_table(
            mock_results, "test_table4.tex"
        )

        assert Path(output_path).exists()

        # Check content
        with open(output_path) as f:
            content = f.read()

        assert "Multi-Method Comparison" in content
        assert "TWFE" in content
        assert "Bootstrap" in content
        assert "Jackknife" in content
        assert "Wild Bootstrap" in content
        assert "\\toprule" in content
        assert "\\bottomrule" in content

    def test_create_effect_size_table(self, generator, mock_effect_sizes):
        """Test effect size table generation."""
        output_path = generator.create_effect_size_table(mock_effect_sizes, "test_table5.tex")

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        assert "Effect Size Analysis" in content
        assert "Cohen's $d$" in content
        assert "0.120" in content  # Cohen's d value
        assert "Small" in content  # Interpretation

    def test_create_power_analysis_table(self, generator, mock_power_results):
        """Test power analysis table generation."""
        output_path = generator.create_power_analysis_table(mock_power_results, "test_table6.tex")

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        assert "Statistical Power Analysis" in content
        assert "Post-hoc" in content
        assert "0.650" in content  # Power value
        assert "Increase sample size" in content

    def test_format_coefficient_with_stars(self, generator):
        """Test coefficient formatting with significance stars."""
        # Highly significant
        result = {"coefficient": 0.15, "p_value": 0.001}
        formatted = generator._format_coefficient(result)
        assert formatted == "0.150***"

        # Moderately significant
        result = {"coefficient": 0.10, "p_value": 0.03}
        formatted = generator._format_coefficient(result)
        assert formatted == "0.100**"

        # Marginally significant
        result = {"coefficient": 0.08, "p_value": 0.07}
        formatted = generator._format_coefficient(result)
        assert formatted == "0.080*"

        # Non-significant
        result = {"coefficient": 0.05, "p_value": 0.15}
        formatted = generator._format_coefficient(result)
        assert formatted == "0.050"

    def test_format_se(self, generator):
        """Test standard error formatting."""
        result = {"std_err": 0.075}
        formatted = generator._format_se(result)
        assert formatted == "(0.075)"

        # Empty result
        formatted = generator._format_se({})
        assert formatted == ""

    def test_format_outcome_name(self, generator):
        """Test outcome name formatting."""
        assert generator._format_outcome_name("math_grade4_gap") == "Math G4 Achievement Gap"
        assert generator._format_outcome_name("reading_grade8_swd_score") == "Reading G8 SWD Score"
        assert generator._format_outcome_name("unknown_outcome") == "Unknown Outcome"

    def test_interpret_effect_size(self, generator):
        """Test effect size interpretation."""
        assert generator._interpret_effect_size(0.1) == "Small"
        assert generator._interpret_effect_size(0.3) == "Medium"
        assert generator._interpret_effect_size(0.6) == "Large"
        assert generator._interpret_effect_size(0.9) == "Very Large"
        assert generator._interpret_effect_size(-0.4) == "Medium"  # Absolute value

    def test_create_summary_statistics_table(self, generator, temp_output_dir):
        """Test summary statistics table creation."""
        # Create mock data
        data = pd.DataFrame(
            {
                "state": ["CA", "TX", "NY"] * 10,
                "year": [2020] * 30,
                "math_grade4_swd_score": [250, 260, 240] * 10,
                "math_grade4_gap": [25, 30, 20] * 10,
                "per_pupil_total": [12000, 11000, 13000] * 10,
                "post_treatment": [1, 0, 1] * 10,
            }
        )

        output_path = generator.create_summary_statistics_table(data, "test_table1.tex")

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        assert "Summary Statistics by Treatment Status" in content
        assert "Full Sample" in content
        assert "Treated" in content
        assert "Control" in content

    def test_combine_tables_for_paper(self, generator, mock_results):
        """Test combining multiple tables."""
        # Create individual tables first
        generator.create_multi_method_comparison_table(mock_results, "table4.tex")

        # Combine tables
        output_path = generator.combine_tables_for_paper(["table4.tex"], "combined_tables.tex")

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        assert "Combined Tables for Paper Submission" in content
        assert "Table 1" in content
        assert "\\clearpage" in content

    def test_empty_results_handling(self, generator):
        """Test handling of empty results."""
        empty_results = {}

        # Should not raise an error
        output_path = generator.create_multi_method_comparison_table(
            empty_results, "empty_test.tex"
        )

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        # Should have basic table structure even with empty data
        assert "\\begin{table}" in content
        assert "\\end{table}" in content

    def test_missing_data_in_results(self, generator):
        """Test handling missing data in results."""
        incomplete_results = {
            "twfe": {
                "math_grade4_gap": {"coefficient": 0.15}  # Missing std_err, p_value
            }
        }

        output_path = generator.create_multi_method_comparison_table(
            incomplete_results, "incomplete_test.tex"
        )

        assert Path(output_path).exists()

        with open(output_path) as f:
            content = f.read()

        # Should handle missing values gracefully
        assert "0.150" in content  # Coefficient should be present
        assert "—" in content or "(0.000)" in content  # Missing SE handled
