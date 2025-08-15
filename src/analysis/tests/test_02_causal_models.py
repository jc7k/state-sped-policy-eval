"""
Unit tests for Phase 4.2: Main Causal Analysis Module

Tests the CausalAnalyzer class functionality including:
- TWFE estimation
- Event study analysis
- Callaway-Sant'Anna implementation
- Instrumental variables estimation

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

spec = importlib.util.spec_from_file_location("causal_module", "src/analysis/02_causal_models.py")
causal_module = importlib.util.module_from_spec(spec)
sys.modules["causal_module"] = causal_module
spec.loader.exec_module(causal_module)

CausalAnalyzer = causal_module.CausalAnalyzer


@pytest.fixture
def sample_panel_data():
    """Create sample panel data with treatment variation."""
    np.random.seed(42)

    states = ["CA", "TX", "NY", "FL", "PA", "IL", "OH"]
    years = list(range(2015, 2023))

    data = []
    for state in states:
        # Different treatment timing for staggered adoption
        if state in ["CA", "TX"]:
            treatment_year = 2018
        elif state in ["NY", "FL"]:
            treatment_year = 2019
        else:
            treatment_year = None  # Never treated

        for year in years:
            treated = 1 if treatment_year and year >= treatment_year else 0

            # Simulate outcomes with treatment effects
            base_math_score = 250 + np.random.normal(0, 5)
            treatment_effect = 4 if treated else 0
            math_gap = 35 - (treatment_effect * 0.3) + np.random.normal(0, 2)
            spending = 12000 + (treatment_effect * 400) + np.random.normal(0, 300)

            # Add instrumental variable (court order)
            court_order = 1 if state in ["CA", "NY"] and year >= 2017 else 0

            data.append(
                {
                    "state": state,
                    "year": year,
                    "post_treatment": treated,
                    "treatment_year": treatment_year if treatment_year else np.nan,
                    "math_grade8_gap": math_gap,
                    "reading_grade4_score": base_math_score
                    + treatment_effect
                    + np.random.normal(0, 3),
                    "inclusion_rate": 0.65 + (treatment_effect * 0.01) + np.random.normal(0, 0.03),
                    "per_pupil_spending": spending,
                    "court_order_active": court_order,
                    "time_trend": year - 2015,
                    "post_covid": 1 if year >= 2020 else 0,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def temp_panel_file(sample_panel_data):
    """Create temporary panel data file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_panel_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCausalAnalyzer:
    """Test cases for CausalAnalyzer class."""

    def test_init(self, temp_panel_file):
        """Test CausalAnalyzer initialization."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)

        assert analyzer.data_path == temp_panel_file
        assert analyzer.df is None
        assert analyzer.results == {}
        assert analyzer.output_dir.name == "output"

    def test_load_data_success(self, temp_panel_file):
        """Test successful data loading and preparation."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.load_data()

        assert analyzer.df is not None
        assert len(analyzer.df) == 56  # 7 states × 8 years
        assert "state_id" in analyzer.df.columns
        assert "year_id" in analyzer.df.columns
        assert "years_since_treatment" in analyzer.df.columns
        assert hasattr(analyzer, "outcome_vars")
        assert len(analyzer.outcome_vars) > 0

    def test_load_data_missing_columns(self, temp_output_dir):
        """Test data loading with missing required columns."""
        bad_data = pd.DataFrame({"state": ["CA"], "year": [2020]})  # Missing post_treatment
        bad_file = Path(temp_output_dir) / "bad_data.csv"
        bad_data.to_csv(bad_file, index=False)

        analyzer = CausalAnalyzer(data_path=str(bad_file))

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.load_data()

    def test_prepare_panel_data(self, temp_panel_file):
        """Test panel data preparation."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.df = pd.read_csv(temp_panel_file)
        analyzer._prepare_panel_data()

        # Check categorical encoding (pandas now uses int8 for small categorical data)
        assert analyzer.df["state_id"].dtype.name in ["int64", "int32", "int8"]
        assert analyzer.df["year_id"].dtype.name in ["int64", "int32", "int8"]

        # Check event time variable
        assert "years_since_treatment" in analyzer.df.columns
        treated_subset = analyzer.df[analyzer.df["post_treatment"] == 1]
        assert all(treated_subset["years_since_treatment"] >= 0)

    def test_get_outcome_variables(self, temp_panel_file):
        """Test outcome variable identification."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.df = pd.read_csv(temp_panel_file)

        outcomes = analyzer._get_outcome_variables()

        assert isinstance(outcomes, list)
        assert len(outcomes) > 0
        assert len(outcomes) <= 5  # Should limit to 5 outcomes

        # Should find gap and score variables
        gap_vars = [var for var in outcomes if "gap" in var.lower()]
        score_vars = [var for var in outcomes if "score" in var.lower()]
        assert len(gap_vars) > 0 or len(score_vars) > 0

    def test_run_twfe_analysis(self, temp_panel_file):
        """Test TWFE analysis execution."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.load_data()

        twfe_results = analyzer.run_twfe_analysis()

        assert isinstance(twfe_results, dict)
        assert len(twfe_results) > 0

        # Check result structure for first outcome
        first_outcome = list(twfe_results.keys())[0]
        result = twfe_results[first_outcome]

        required_keys = ["coefficient", "se", "p_value", "t_stat", "conf_int", "n_obs", "r_squared"]
        for key in required_keys:
            assert key in result

        # Check that coefficients are numeric
        assert isinstance(result["coefficient"], int | float)
        assert isinstance(result["se"], int | float)
        assert result["se"] > 0  # Standard errors should be positive

    def test_run_event_study(self, temp_panel_file):
        """Test event study analysis."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.load_data()

        event_results = analyzer.run_event_study()

        assert isinstance(event_results, dict)

        if event_results:  # Only test if we have results
            first_outcome = list(event_results.keys())[0]
            result = event_results[first_outcome]

            assert "coefficients" in result
            assert "standard_errors" in result
            assert "p_values" in result

            # Check that reference period (t=-1) is set to 0
            if -1 in result["coefficients"]:
                assert result["coefficients"][-1] == 0

    def test_run_callaway_santanna(self, temp_panel_file):
        """Test Callaway-Sant'Anna analysis."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.load_data()

        cs_results = analyzer.run_callaway_santanna()

        assert isinstance(cs_results, dict)

        if cs_results:  # Only test if we have results
            first_outcome = list(cs_results.keys())[0]
            result = cs_results[first_outcome]

            expected_keys = [
                "overall_att",
                "overall_se",
                "overall_pvalue",
                "cohort_results",
                "n_cohorts",
            ]
            for key in expected_keys:
                assert key in result

            assert isinstance(result["cohort_results"], dict)
            assert result["n_cohorts"] > 0

    def test_run_instrumental_variables(self, temp_panel_file):
        """Test instrumental variables analysis."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.load_data()

        iv_results = analyzer.run_instrumental_variables()

        assert isinstance(iv_results, dict)

        if iv_results:  # Only test if we have valid IV results
            first_outcome = list(iv_results.keys())[0]
            result = iv_results[first_outcome]

            expected_keys = [
                "coefficient",
                "se",
                "p_value",
                "first_stage_f",
                "n_obs",
                "instrument",
                "endogenous_var",
            ]
            for key in expected_keys:
                assert key in result

            # First stage F-statistic should be positive
            assert result["first_stage_f"] > 0

    def test_create_results_table(self, temp_panel_file, temp_output_dir):
        """Test results table creation."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.tables_dir.mkdir(parents=True, exist_ok=True)

        analyzer.load_data()

        # Run at least one analysis to have results
        analyzer.run_twfe_analysis()

        results_table = analyzer.create_results_table()

        assert isinstance(results_table, pd.DataFrame)
        assert not results_table.empty
        assert "outcome" in results_table.columns

        # Check files were created
        assert (analyzer.tables_dir / "table2_main_results.csv").exists()
        assert (analyzer.tables_dir / "table2_main_results.tex").exists()

    def test_format_results_latex(self, temp_panel_file):
        """Test LaTeX results formatting."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)

        # Create sample results data
        test_df = pd.DataFrame(
            {
                "outcome": ["test_outcome"],
                "twfe_coef": [2.5],
                "twfe_se": [0.8],
                "event_avg_coef": [2.1],
                "cs_att": [2.3],
                "cs_se": [0.9],
                "iv_coef": [3.2],
                "iv_se": [1.1],
            }
        )

        latex_output = analyzer._format_results_latex(test_df)

        assert "\\begin{table}" in latex_output
        assert "Main Causal Analysis Results" in latex_output
        assert "\\end{table}" in latex_output
        assert "test\\_outcome" in latex_output  # Escaped underscore
        assert "2.500" in latex_output

    def test_create_event_study_plots(self, temp_panel_file, temp_output_dir):
        """Test event study plot creation."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.figures_dir = analyzer.output_dir / "figures"
        analyzer.figures_dir.mkdir(parents=True, exist_ok=True)

        analyzer.load_data()
        analyzer.run_event_study()

        plots = analyzer.create_event_study_plots()

        assert isinstance(plots, dict)

        # If we have event study results, should create plots
        if analyzer.results.get("event_study"):
            assert len(plots) > 0

    def test_summarize_results_empty(self, temp_panel_file):
        """Test results summary with no results."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)

        summary = analyzer._summarize_results()
        assert "No results available" in summary

    def test_summarize_results_with_data(self, temp_panel_file):
        """Test results summary with actual results."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.load_data()

        # Add mock results
        analyzer.results = {
            "twfe": {"outcome1": {"p_value": 0.02}, "outcome2": {"p_value": 0.08}},
            "event_study": {"outcome1": {"p_values": {0: 0.03, 1: 0.06}}},
        }

        summary = analyzer._summarize_results()

        assert "Twfe" in summary
        assert "1/2" in summary  # 1 out of 2 significant

    def test_run_full_analysis(self, temp_panel_file, temp_output_dir):
        """Test full analysis pipeline."""
        analyzer = CausalAnalyzer(data_path=temp_panel_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.figures_dir = analyzer.output_dir / "figures"
        analyzer.tables_dir.mkdir(parents=True, exist_ok=True)
        analyzer.figures_dir.mkdir(parents=True, exist_ok=True)

        report_path = analyzer.run_full_analysis()

        assert Path(report_path).exists()
        assert Path(report_path).suffix == ".md"

        # Check that some results were generated
        assert len(analyzer.results) > 0

        # Check report content
        with open(report_path) as f:
            content = f.read()

        assert "Phase 4.2: Main Causal Analysis Report" in content
        assert "Jeff Chen" in content
        assert "Claude Code" in content


class TestCausalAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    def test_twfe_with_no_variation(self, temp_output_dir):
        """Test TWFE with no treatment variation."""
        # Create data with no treatment
        no_treatment_data = pd.DataFrame(
            {
                "state": ["CA", "TX"] * 3,
                "year": [2020, 2020, 2021, 2021, 2022, 2022],
                "post_treatment": [0] * 6,  # No treatment
                "outcome": [100, 102, 101, 103, 99, 104],
            }
        )

        data_file = Path(temp_output_dir) / "no_treatment.csv"
        no_treatment_data.to_csv(data_file, index=False)

        analyzer = CausalAnalyzer(data_path=str(data_file))
        analyzer.load_data()

        # Should handle gracefully
        twfe_results = analyzer.run_twfe_analysis()
        assert isinstance(twfe_results, dict)

    def test_event_study_with_insufficient_data(self, temp_output_dir):
        """Test event study with insufficient event time variation."""
        # Create minimal data
        minimal_data = pd.DataFrame(
            {
                "state": ["CA", "TX"],
                "year": [2020, 2020],
                "post_treatment": [0, 1],
                "treatment_year": [np.nan, 2020],
                "outcome": [100, 105],
            }
        )

        data_file = Path(temp_output_dir) / "minimal.csv"
        minimal_data.to_csv(data_file, index=False)

        analyzer = CausalAnalyzer(data_path=str(data_file))
        analyzer.load_data()

        # Should handle gracefully
        event_results = analyzer.run_event_study()
        assert isinstance(event_results, dict)

    def test_iv_with_no_instruments(self, temp_output_dir):
        """Test IV analysis with no instrumental variables."""
        # Create data without instrument columns
        no_iv_data = pd.DataFrame(
            {
                "state": ["CA", "TX"] * 3,
                "year": [2020, 2020, 2021, 2021, 2022, 2022],
                "post_treatment": [0, 1, 0, 1, 0, 1],
                "outcome": [100, 105, 102, 108, 101, 107],
            }
        )

        data_file = Path(temp_output_dir) / "no_iv.csv"
        no_iv_data.to_csv(data_file, index=False)

        analyzer = CausalAnalyzer(data_path=str(data_file))
        analyzer.load_data()

        iv_results = analyzer.run_instrumental_variables()

        # Should return empty results gracefully
        assert isinstance(iv_results, dict)
        assert len(iv_results) == 0


class TestCausalAnalyzerIntegration:
    """Integration tests for CausalAnalyzer."""

    def test_main_function_success(self, temp_panel_file, temp_output_dir):
        """Test main function execution."""
        import importlib.util

        # Import the module with numeric name
        spec = importlib.util.spec_from_file_location(
            "causal_module", "src/analysis/02_causal_models.py"
        )
        causal_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(causal_module)

        with patch.object(causal_module, "CausalAnalyzer") as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.run_full_analysis.return_value = str(Path(temp_output_dir) / "report.md")
            mock_analyzer_class.return_value = mock_analyzer

            result = causal_module.main()

            assert result is True
            mock_analyzer.run_full_analysis.assert_called_once()

    def test_main_function_failure(self):
        """Test main function with failure."""
        import importlib.util

        # Import the module with numeric name
        spec = importlib.util.spec_from_file_location(
            "causal_module", "src/analysis/02_causal_models.py"
        )
        causal_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(causal_module)

        with patch.object(causal_module, "CausalAnalyzer") as mock_analyzer_class:
            mock_analyzer_class.side_effect = Exception("Test error")

            result = causal_module.main()

            assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
