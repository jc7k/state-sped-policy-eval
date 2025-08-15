"""
Unit tests for Phase 4.1: Descriptive Analysis Module

Tests the DescriptiveAnalyzer class functionality including:
- Data loading and validation
- Summary statistics generation
- Trend plot creation
- Report generation

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

# Import the module with numeric name using importlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "descriptive_module", "src/analysis/01_descriptive.py"
)
descriptive_module = importlib.util.module_from_spec(spec)
sys.modules["descriptive_module"] = descriptive_module
spec.loader.exec_module(descriptive_module)

DescriptiveAnalyzer = descriptive_module.DescriptiveAnalyzer


@pytest.fixture
def sample_analysis_data():
    """Create sample analysis panel data for testing."""
    np.random.seed(42)

    states = ["CA", "TX", "NY", "FL", "PA"]
    years = list(range(2015, 2023))

    data = []
    for state in states:
        for year in years:
            # Create treatment effect (CA and TX treated in 2018)
            treated = 1 if state in ["CA", "TX"] and year >= 2018 else 0

            # Simulate outcomes with treatment effect
            base_achievement = 250 + np.random.normal(0, 5)
            treatment_effect = 3 if treated else 0
            achievement = base_achievement + treatment_effect + np.random.normal(0, 2)

            gap = 40 - (treatment_effect * 0.5) + np.random.normal(0, 3)
            inclusion_rate = 0.65 + (treatment_effect * 0.02) + np.random.normal(0, 0.05)
            per_pupil_spending = 12000 + (treatment_effect * 500) + np.random.normal(0, 500)

            data.append(
                {
                    "state": state,
                    "year": year,
                    "post_treatment": treated,
                    "math_grade8_score": achievement,
                    "math_grade8_gap": gap,
                    "inclusion_rate": inclusion_rate,
                    "per_pupil_total": per_pupil_spending,
                    "time_trend": year - 2015,
                    "post_covid": 1 if year >= 2020 else 0,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_analysis_data):
    """Create temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_analysis_data.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestDescriptiveAnalyzer:
    """Test cases for DescriptiveAnalyzer class."""

    def test_init(self, temp_data_file):
        """Test DescriptiveAnalyzer initialization."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)

        assert analyzer.data_path == temp_data_file
        assert analyzer.df is None
        assert analyzer.output_dir.name == "output"
        assert analyzer.tables_dir.name == "tables"
        assert analyzer.figures_dir.name == "figures"

    def test_load_data_success(self, temp_data_file):
        """Test successful data loading."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        assert analyzer.df is not None
        assert len(analyzer.df) == 40  # 5 states × 8 years
        assert "state" in analyzer.df.columns
        assert "year" in analyzer.df.columns
        assert "post_treatment" in analyzer.df.columns

    def test_load_data_missing_file(self):
        """Test data loading with missing file."""
        analyzer = DescriptiveAnalyzer(data_path="nonexistent_file.csv")

        with pytest.raises(FileNotFoundError):
            analyzer.load_data()

    def test_load_data_missing_columns(self, temp_output_dir):
        """Test data loading with missing required columns."""
        # Create data missing required columns
        bad_data = pd.DataFrame({"state": ["CA"], "year": [2020]})  # Missing post_treatment
        bad_file = Path(temp_output_dir) / "bad_data.csv"
        bad_data.to_csv(bad_file, index=False)

        analyzer = DescriptiveAnalyzer(data_path=str(bad_file))

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.load_data()

    def test_get_outcome_columns(self, temp_data_file):
        """Test outcome column identification."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        outcomes = analyzer._get_outcome_columns()

        assert "achievement" in outcomes
        assert "gaps" in outcomes
        assert "inclusion" in outcomes
        assert "finance" in outcomes

        # Check that it finds the right variables
        assert any("score" in col for col in outcomes["achievement"])
        assert any("gap" in col for col in outcomes["gaps"])
        assert any("inclusion" in col for col in outcomes["inclusion"])
        assert any("per_pupil" in col for col in outcomes["finance"])

    def test_create_summary_statistics(self, temp_data_file, temp_output_dir):
        """Test summary statistics creation."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.tables_dir.mkdir(parents=True, exist_ok=True)

        analyzer.load_data()
        summary_stats = analyzer.create_summary_statistics()

        assert isinstance(summary_stats, pd.DataFrame)
        assert not summary_stats.empty

        # Check that treated/control/difference rows exist
        assert 0 in summary_stats.index  # Control
        assert 1 in summary_stats.index  # Treated
        assert "difference" in summary_stats.index

        # Check files were created
        assert (analyzer.tables_dir / "table1_summary_stats.csv").exists()
        assert (analyzer.tables_dir / "table1_summary_stats.tex").exists()

    def test_create_trend_figures(self, temp_data_file, temp_output_dir):
        """Test trend figure creation."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.figures_dir = analyzer.output_dir / "figures"
        analyzer.figures_dir.mkdir(parents=True, exist_ok=True)

        analyzer.load_data()
        figures = analyzer.create_trend_figures()

        assert isinstance(figures, dict)
        assert "trends" in figures

        # Check files were created
        assert (analyzer.figures_dir / "figure1_trends.png").exists()
        assert (analyzer.figures_dir / "figure1_trends.pdf").exists()

    def test_plot_trends_missing_variable(self, temp_data_file):
        """Test plotting trends with missing variable."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Should handle missing variable gracefully
        analyzer._plot_trends(ax, "nonexistent_variable", "Test Y Label", "Test Title")

        # Check that title indicates missing data
        assert "Missing" in ax.get_title()
        plt.close(fig)

    def test_format_latex_table(self, temp_data_file):
        """Test LaTeX table formatting."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)

        # Create sample data for formatting
        test_df = pd.DataFrame(
            {"mean": [1.234, 2.345, 3.456], "std": [0.123, 0.234, 0.345]},
            index=[0, 1, "difference"],
        )

        latex_output = analyzer._format_latex_table(test_df, "Test Title", "tab:test")

        assert "\\begin{table}" in latex_output
        assert "Test Title" in latex_output
        assert "tab:test" in latex_output
        assert "\\end{table}" in latex_output
        assert "1.234" in latex_output

    def test_generate_descriptive_report(self, temp_data_file, temp_output_dir):
        """Test full descriptive report generation."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.figures_dir = analyzer.output_dir / "figures"
        analyzer.tables_dir.mkdir(parents=True, exist_ok=True)
        analyzer.figures_dir.mkdir(parents=True, exist_ok=True)

        report_path = analyzer.generate_descriptive_report()

        assert Path(report_path).exists()
        assert Path(report_path).suffix == ".md"

        # Check report content
        with open(report_path) as f:
            content = f.read()

        assert "Phase 4.1: Descriptive Analysis Report" in content
        assert "Jeff Chen" in content
        assert "Claude Code" in content
        assert "Data Overview" in content
        assert "Key Findings" in content

    def test_format_summary_findings_empty(self, temp_data_file):
        """Test formatting summary findings with empty data."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)

        empty_df = pd.DataFrame()
        findings = analyzer._format_summary_findings(empty_df)

        assert "No summary statistics available" in findings

    def test_format_summary_findings_with_data(self, temp_data_file):
        """Test formatting summary findings with actual data."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)

        # Create sample summary data
        test_df = pd.DataFrame(
            {"math_score_mean": [250.0, 253.0, 3.0], "gap_mean": [40.0, 37.0, -3.0]},
            index=[0, 1, "difference"],
        )

        findings = analyzer._format_summary_findings(test_df)

        assert "Math Score" in findings
        assert "3.000 units higher" in findings
        assert "Gap" in findings
        assert "3.000 units lower" in findings


class TestDescriptiveAnalyzerIntegration:
    """Integration tests for DescriptiveAnalyzer."""

    def test_full_pipeline_with_minimal_data(self, temp_output_dir):
        """Test full pipeline with minimal data."""
        # Create minimal test data
        minimal_data = pd.DataFrame(
            {
                "state": ["CA", "TX"] * 3,
                "year": [2020, 2020, 2021, 2021, 2022, 2022],
                "post_treatment": [0, 1, 0, 1, 0, 1],
                "test_outcome": [100, 105, 102, 108, 101, 107],
            }
        )

        data_file = Path(temp_output_dir) / "minimal_data.csv"
        minimal_data.to_csv(data_file, index=False)

        analyzer = DescriptiveAnalyzer(data_path=str(data_file))
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.figures_dir = analyzer.output_dir / "figures"

        # Should not raise errors even with minimal data
        report_path = analyzer.generate_descriptive_report()

        assert Path(report_path).exists()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_creation_without_saving(self, mock_savefig, temp_data_file):
        """Test plot creation without actually saving files."""
        analyzer = DescriptiveAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        figures = analyzer.create_trend_figures()

        assert isinstance(figures, dict)
        # savefig should have been called for both PNG and PDF
        assert mock_savefig.call_count == 2

    def test_main_function_success(self, temp_data_file, temp_output_dir):
        """Test main function execution."""
        import importlib.util

        # Import the module with numeric name
        spec = importlib.util.spec_from_file_location(
            "descriptive_module", "src/analysis/01_descriptive.py"
        )
        descriptive_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(descriptive_module)

        # Mock the analyzer to use our test data
        with patch.object(descriptive_module, "DescriptiveAnalyzer") as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.generate_descriptive_report.return_value = str(
                Path(temp_output_dir) / "report.md"
            )
            mock_analyzer_class.return_value = mock_analyzer

            result = descriptive_module.main()

            assert result is True
            mock_analyzer.generate_descriptive_report.assert_called_once()

    def test_main_function_failure(self):
        """Test main function with failure."""
        import importlib.util

        # Import the module with numeric name
        spec = importlib.util.spec_from_file_location(
            "descriptive_module", "src/analysis/01_descriptive.py"
        )
        descriptive_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(descriptive_module)

        # Mock the analyzer to raise an exception
        with patch.object(descriptive_module, "DescriptiveAnalyzer") as mock_analyzer_class:
            mock_analyzer_class.side_effect = Exception("Test error")

            result = descriptive_module.main()

            assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
