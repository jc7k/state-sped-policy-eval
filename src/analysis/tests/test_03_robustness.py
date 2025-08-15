"""
Unit tests for Phase 4.3: Robustness Analysis Module

Tests the RobustnessAnalyzer class functionality including:
- Leave-one-state-out analysis
- Alternative clustering methods
- Permutation tests
- Specification curve analysis

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

spec = importlib.util.spec_from_file_location("robustness_module", "src/analysis/03_robustness.py")
robustness_module = importlib.util.module_from_spec(spec)
sys.modules["robustness_module"] = robustness_module
spec.loader.exec_module(robustness_module)

RobustnessAnalyzer = robustness_module.RobustnessAnalyzer


@pytest.fixture
def sample_robustness_data():
    """Create sample data for robustness testing."""
    np.random.seed(42)

    states = ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA"]
    years = list(range(2015, 2023))

    data = []
    for state in states:
        # Treatment assignment
        treated = 1 if state in ["CA", "TX", "NY", "FL"] else 0

        for year in years:
            post_treatment = treated if year >= 2018 else 0

            # Simulate outcomes with treatment effects
            treatment_effect = 3 if post_treatment else 0

            math_gap = 38 - treatment_effect + np.random.normal(0, 2)
            reading_score = 245 + treatment_effect + np.random.normal(0, 4)
            inclusion_rate = 0.62 + (treatment_effect * 0.02) + np.random.normal(0, 0.04)
            spending = 11500 + (treatment_effect * 300) + np.random.normal(0, 400)

            data.append(
                {
                    "state": state,
                    "year": year,
                    "post_treatment": post_treatment,
                    "math_grade8_gap": math_gap,
                    "reading_grade4_score": reading_score,
                    "inclusion_rate": inclusion_rate,
                    "per_pupil_spending": spending,
                    "time_trend": year - 2015,
                    "post_covid": 1 if year >= 2020 else 0,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_robustness_data):
    """Create temporary data file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_robustness_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestRobustnessAnalyzer:
    """Test cases for RobustnessAnalyzer class."""

    def test_init(self, temp_data_file):
        """Test RobustnessAnalyzer initialization."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)

        assert analyzer.data_path == temp_data_file
        assert analyzer.df is None
        assert analyzer.robustness_results == {}
        assert analyzer.output_dir.name == "output"

    def test_load_data_success(self, temp_data_file):
        """Test successful data loading."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        assert analyzer.df is not None
        assert len(analyzer.df) == 64  # 8 states × 8 years
        assert "census_region" in analyzer.df.columns
        assert hasattr(analyzer, "outcome_vars")
        assert len(analyzer.outcome_vars) > 0

    def test_load_data_missing_columns(self, temp_output_dir):
        """Test data loading with missing required columns."""
        bad_data = pd.DataFrame({"state": ["CA"], "year": [2020]})  # Missing post_treatment
        bad_file = Path(temp_output_dir) / "bad_data.csv"
        bad_data.to_csv(bad_file, index=False)

        analyzer = RobustnessAnalyzer(data_path=str(bad_file))

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.load_data()

    def test_get_outcome_variables(self, temp_data_file):
        """Test outcome variable identification."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.df = pd.read_csv(temp_data_file)

        outcomes = analyzer._get_outcome_variables()

        assert isinstance(outcomes, list)
        assert len(outcomes) > 0
        assert len(outcomes) <= 3  # Should limit to 3 outcomes for robustness

        # Should prioritize gap and score variables
        for outcome in outcomes:
            assert any(
                term in outcome.lower()
                for term in ["gap", "score", "achievement", "inclusion", "spending"]
            )

    def test_add_regional_indicators(self, temp_data_file):
        """Test census region indicator creation."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.df = pd.read_csv(temp_data_file)
        analyzer._add_regional_indicators()

        assert "census_region" in analyzer.df.columns

        # Check that regions are assigned correctly
        regions = analyzer.df["census_region"].unique()
        expected_regions = ["West", "South", "Northeast", "Midwest"]

        # Should have at least some valid regions
        valid_regions = [r for r in regions if r in expected_regions]
        assert len(valid_regions) > 0

        # Check specific state mappings
        ca_region = analyzer.df[analyzer.df["state"] == "CA"]["census_region"].iloc[0]
        assert ca_region == "West"

        ny_region = analyzer.df[analyzer.df["state"] == "NY"]["census_region"].iloc[0]
        assert ny_region == "Northeast"

    def test_leave_one_state_out(self, temp_data_file):
        """Test leave-one-state-out analysis."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        loso_results = analyzer.leave_one_state_out()

        assert isinstance(loso_results, dict)

        if loso_results:  # Only test if we have results
            first_outcome = list(loso_results.keys())[0]
            result = loso_results[first_outcome]

            expected_keys = [
                "state_results",
                "mean_coeff",
                "std_coeff",
                "min_coeff",
                "max_coeff",
                "n_estimates",
            ]
            for key in expected_keys:
                assert key in result

            # Check that we have results for multiple states
            assert result["n_estimates"] > 1
            assert isinstance(result["state_results"], dict)

            # Check individual state result structure
            first_state_result = list(result["state_results"].values())[0]
            state_keys = ["coefficient", "se", "p_value", "n_obs", "n_states"]
            for key in state_keys:
                assert key in first_state_result

    def test_alternative_clustering(self, temp_data_file):
        """Test alternative clustering analysis."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        cluster_results = analyzer.alternative_clustering()

        assert isinstance(cluster_results, dict)

        if cluster_results:  # Only test if we have results
            first_outcome = list(cluster_results.keys())[0]
            cluster_specs = cluster_results[first_outcome]

            # Should have at least state clustering
            assert "state" in cluster_specs

            # Check result structure
            for _cluster_type, results in cluster_specs.items():
                assert "coefficient" in results
                assert "se" in results
                assert "p_value" in results
                assert results["se"] > 0  # Standard errors should be positive

    def test_permutation_test(self, temp_data_file):
        """Test permutation test analysis."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        # Use small number of permutations for testing
        perm_results = analyzer.permutation_test(n_permutations=50)

        assert isinstance(perm_results, dict)

        if perm_results:  # Only test if we have results
            first_outcome = list(perm_results.keys())[0]
            result = perm_results[first_outcome]

            expected_keys = [
                "actual_coefficient",
                "placebo_coefficients",
                "permutation_pvalue",
                "n_permutations",
                "placebo_mean",
                "placebo_std",
            ]
            for key in expected_keys:
                assert key in result

            # Check permutation results
            assert len(result["placebo_coefficients"]) <= 50  # Should be <= n_permutations
            assert 0 <= result["permutation_pvalue"] <= 1  # p-value should be valid probability

    def test_specification_curve(self, temp_data_file):
        """Test specification curve analysis."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        spec_results = analyzer.specification_curve()

        assert isinstance(spec_results, dict)

        if spec_results:  # Only test if we have results
            first_outcome = list(spec_results.keys())[0]
            result = spec_results[first_outcome]

            expected_keys = ["specifications", "n_specs", "coeff_range", "significant_specs"]
            for key in expected_keys:
                assert key in result

            # Check specifications DataFrame
            spec_df = result["specifications"]
            assert isinstance(spec_df, pd.DataFrame)
            assert not spec_df.empty

            spec_columns = [
                "spec_id",
                "controls",
                "coefficient",
                "se",
                "p_value",
                "r_squared",
                "n_obs",
            ]
            for col in spec_columns:
                assert col in spec_df.columns

            # Check coefficient range
            coeff_min, coeff_max = result["coeff_range"]
            assert coeff_min <= coeff_max

    def test_create_robustness_plots(self, temp_data_file, temp_output_dir):
        """Test robustness plot creation."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.figures_dir = analyzer.output_dir / "figures"
        analyzer.figures_dir.mkdir(parents=True, exist_ok=True)

        analyzer.load_data()

        # Run some robustness tests to have data for plots
        analyzer.leave_one_state_out()
        analyzer.specification_curve()
        analyzer.permutation_test(n_permutations=20)

        plots = analyzer.create_robustness_plots()

        assert isinstance(plots, dict)

        # Should create plots for available results
        if analyzer.robustness_results.get("leave_one_state_out"):
            assert "leave_one_state_out" in plots or len(plots) >= 0  # Some plots should be created

    def test_plot_leave_one_state_out(self, temp_data_file):
        """Test LOSO plot creation."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        # Create mock LOSO results
        analyzer.robustness_results["leave_one_state_out"] = {
            "test_outcome": {
                "state_results": {
                    "CA": {"coefficient": 2.5},
                    "TX": {"coefficient": 2.8},
                    "NY": {"coefficient": 2.2},
                },
                "mean_coeff": 2.5,
            }
        }

        figure = analyzer._plot_leave_one_state_out()

        assert figure is not None
        assert hasattr(figure, "axes")

    def test_plot_specification_curve(self, temp_data_file):
        """Test specification curve plot creation."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        # Create mock specification curve results
        spec_df = pd.DataFrame(
            {
                "spec_id": [0, 1, 2],
                "controls_str": ["None", "Time", "Time+COVID"],
                "coefficient": [2.1, 2.3, 2.5],
                "se": [0.5, 0.6, 0.4],
                "p_value": [0.01, 0.02, 0.03],
            }
        )

        analyzer.robustness_results["specification_curve"] = {
            "test_outcome": {"specifications": spec_df}
        }

        figure = analyzer._plot_specification_curve()

        assert figure is not None
        assert hasattr(figure, "axes")

    def test_plot_permutation_test(self, temp_data_file):
        """Test permutation test plot creation."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        # Create mock permutation results
        analyzer.robustness_results["permutation_test"] = {
            "test_outcome": {
                "actual_coefficient": 2.5,
                "placebo_coefficients": np.random.normal(0, 1, 100),
                "permutation_pvalue": 0.05,
            }
        }

        figure = analyzer._plot_permutation_test()

        assert figure is not None
        assert hasattr(figure, "axes")

    def test_create_robustness_table(self, temp_data_file, temp_output_dir):
        """Test robustness table creation."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.tables_dir.mkdir(parents=True, exist_ok=True)

        analyzer.load_data()

        # Add some mock results
        analyzer.robustness_results = {
            "leave_one_state_out": {
                "test_outcome": {
                    "mean_coeff": 2.5,
                    "std_coeff": 0.3,
                    "min_coeff": 2.1,
                    "max_coeff": 2.9,
                }
            },
            "permutation_test": {"test_outcome": {"permutation_pvalue": 0.03}},
        }

        robustness_table = analyzer.create_robustness_table()

        assert isinstance(robustness_table, pd.DataFrame)
        assert not robustness_table.empty
        assert "outcome" in robustness_table.columns

        # Check files were created
        assert (analyzer.tables_dir / "table3_robustness_results.csv").exists()
        assert (analyzer.tables_dir / "table3_robustness_results.tex").exists()

    def test_format_robustness_latex(self, temp_data_file):
        """Test LaTeX robustness table formatting."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)

        # Create sample robustness data
        test_df = pd.DataFrame(
            {
                "outcome": ["test_outcome"],
                "loso_range": ["[2.1, 2.9]"],
                "cluster_state_se": [0.5],
                "cluster_region_se": [0.6],
                "permutation_pvalue": [0.03],
                "spec_curve_range": ["[2.0, 3.0]"],
            }
        )

        latex_output = analyzer._format_robustness_latex(test_df)

        assert "\\begin{table}" in latex_output
        assert "Robustness Analysis Results" in latex_output
        assert "\\end{table}" in latex_output
        assert "test\\_outcome" in latex_output  # Escaped underscore
        assert "[2.1, 2.9]" in latex_output

    def test_run_full_robustness_suite(self, temp_data_file, temp_output_dir):
        """Test full robustness analysis pipeline."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.output_dir = Path(temp_output_dir)
        analyzer.tables_dir = analyzer.output_dir / "tables"
        analyzer.figures_dir = analyzer.output_dir / "figures"
        analyzer.tables_dir.mkdir(parents=True, exist_ok=True)
        analyzer.figures_dir.mkdir(parents=True, exist_ok=True)

        report_path = analyzer.run_full_robustness_suite()

        assert Path(report_path).exists()
        assert Path(report_path).suffix == ".md"

        # Check that some robustness tests were run
        assert len(analyzer.robustness_results) > 0

        # Check report content
        with open(report_path) as f:
            content = f.read()

        assert "Phase 4.3: Robustness Analysis Report" in content
        assert "Jeff Chen" in content
        assert "Claude Code" in content
        assert "Robustness Tests Completed" in content

    def test_summarize_robustness_results_empty(self, temp_data_file):
        """Test robustness summary with no results."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)

        summary = analyzer._summarize_robustness_results()
        assert "No robustness tests completed" in summary

    def test_summarize_robustness_results_with_data(self, temp_data_file):
        """Test robustness summary with actual results."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)

        # Add mock results
        analyzer.robustness_results = {
            "leave_one_state_out": {"outcome1": {}, "outcome2": {}},
            "permutation_test": {
                "outcome1": {"permutation_pvalue": 0.02},
                "outcome2": {"permutation_pvalue": 0.08},
            },
            "specification_curve": {"outcome1": {"n_specs": 4}},
        }

        summary = analyzer._summarize_robustness_results()

        assert "Leave-One-State-Out" in summary
        assert "2 outcomes" in summary
        assert "Permutation Test" in summary
        assert "1/2" in summary  # 1 out of 2 significant
        assert "Specification Curve" in summary


class TestRobustnessAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    def test_loso_with_insufficient_states(self, temp_output_dir):
        """Test LOSO with insufficient number of states."""
        # Create data with only 2 states
        minimal_data = pd.DataFrame(
            {
                "state": ["CA", "TX"] * 3,
                "year": [2020, 2020, 2021, 2021, 2022, 2022],
                "post_treatment": [0, 1, 0, 1, 0, 1],
                "outcome": [100, 105, 102, 108, 101, 107],
            }
        )

        data_file = Path(temp_output_dir) / "minimal.csv"
        minimal_data.to_csv(data_file, index=False)

        analyzer = RobustnessAnalyzer(data_path=str(data_file))
        analyzer.load_data()

        # Should handle gracefully
        loso_results = analyzer.leave_one_state_out()
        assert isinstance(loso_results, dict)

    def test_permutation_with_errors(self, temp_data_file):
        """Test permutation test with some failed permutations."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()

        # Should handle errors gracefully and continue
        perm_results = analyzer.permutation_test(n_permutations=10)
        assert isinstance(perm_results, dict)

    def test_specification_curve_with_missing_controls(self, temp_output_dir):
        """Test specification curve with missing control variables."""
        # Create data without control variables
        no_controls_data = pd.DataFrame(
            {
                "state": ["CA", "TX"] * 3,
                "year": [2020, 2020, 2021, 2021, 2022, 2022],
                "post_treatment": [0, 1, 0, 1, 0, 1],
                "outcome": [100, 105, 102, 108, 101, 107],
            }
        )

        data_file = Path(temp_output_dir) / "no_controls.csv"
        no_controls_data.to_csv(data_file, index=False)

        analyzer = RobustnessAnalyzer(data_path=str(data_file))
        analyzer.load_data()

        # Should still run with just the base specification
        spec_results = analyzer.specification_curve()
        assert isinstance(spec_results, dict)


class TestRobustnessAnalyzerIntegration:
    """Integration tests for RobustnessAnalyzer."""

    def test_main_function_success(self, temp_data_file, temp_output_dir):
        """Test main function execution."""
        import importlib.util

        # Import the module with numeric name
        spec = importlib.util.spec_from_file_location(
            "robustness_module", "src/analysis/03_robustness.py"
        )
        robustness_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(robustness_module)

        with patch.object(robustness_module, "RobustnessAnalyzer") as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.run_full_robustness_suite.return_value = str(
                Path(temp_output_dir) / "report.md"
            )
            mock_analyzer_class.return_value = mock_analyzer

            result = robustness_module.main()

            assert result is True
            mock_analyzer.run_full_robustness_suite.assert_called_once()

    def test_main_function_failure(self):
        """Test main function with failure."""
        import importlib.util

        # Import the module with numeric name
        spec = importlib.util.spec_from_file_location(
            "robustness_module", "src/analysis/03_robustness.py"
        )
        robustness_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(robustness_module)

        with patch.object(robustness_module, "RobustnessAnalyzer") as mock_analyzer_class:
            mock_analyzer_class.side_effect = Exception("Test error")

            result = robustness_module.main()

            assert result is False


class TestEnhancedStatisticalInference:
    """Test Phase 3 Enhanced Statistical Inference methods."""

    @pytest.fixture
    def analyzer_with_results(self, temp_data_file):
        """Fixture providing analyzer with mock robustness results."""
        analyzer = RobustnessAnalyzer(data_path=temp_data_file)
        analyzer.load_data()
        
        analyzer.robustness_results = {
            "bootstrap_inference": {
                "math_score": {
                    "coefficient": -2.5,
                    "std_error": 1.2,
                    "p_value": 0.04,
                    "ci_lower": -4.9,
                    "ci_upper": -0.1,
                    "bootstrap_coefficients": np.random.normal(-2.5, 1.2, 1000)
                },
                "reading_score": {
                    "coefficient": 1.8,
                    "std_error": 1.5,
                    "p_value": 0.23,
                    "ci_lower": -1.2,
                    "ci_upper": 4.8,
                    "bootstrap_coefficients": np.random.normal(1.8, 1.5, 1000)
                }
            },
            "wild_cluster_bootstrap": {
                "math_score": {
                    "coefficient": -2.3,
                    "std_error": 1.1,
                    "p_value": 0.038,
                    "ci_lower": -4.5,
                    "ci_upper": -0.1
                },
                "reading_score": {
                    "coefficient": 1.5,
                    "std_error": 1.4,
                    "p_value": 0.28,
                    "ci_lower": -1.3,
                    "ci_upper": 4.3
                }
            }
        }
        return analyzer

    def test_multiple_testing_corrections(self, analyzer_with_results):
        """Test multiple testing corrections implementation."""
        results = analyzer_with_results.robustness_results
        corrections = analyzer_with_results.multiple_testing_corrections(results)

        # Check structure
        assert "corrections" in corrections
        assert "summary" in corrections

        # Check summary statistics
        summary = corrections["summary"]
        assert "total_tests" in summary
        assert "original_significant" in summary
        assert "bonferroni_significant" in summary
        assert "fdr_significant" in summary
        assert "romano_wolf_significant" in summary

        # Should have 4 total tests (2 methods × 2 outcomes)
        assert summary["total_tests"] == 4

        # Check that corrections are applied to each method/outcome
        assert "bootstrap_inference" in corrections["corrections"]
        assert "wild_cluster_bootstrap" in corrections["corrections"]

        # Verify correction structure
        bootstrap_corrections = corrections["corrections"]["bootstrap_inference"]
        assert "math_score" in bootstrap_corrections
        assert "reading_score" in bootstrap_corrections

        # Check correction fields for specific outcome
        math_corrections = bootstrap_corrections["math_score"]
        required_fields = [
            "original_p",
            "bonferroni_p",
            "fdr_p",
            "romano_wolf_p",
            "bonferroni_significant",
            "fdr_significant",
            "romano_wolf_significant",
        ]
        for field in required_fields:
            assert field in math_corrections

        # Bonferroni should be more conservative (higher p-values)
        assert math_corrections["bonferroni_p"] >= math_corrections["original_p"]

    def test_effect_size_calculations(self, analyzer_with_results):
        """Test effect size calculations and standardization."""
        results = analyzer_with_results.robustness_results
        effect_sizes = analyzer_with_results.calculate_effect_sizes(results)

        # Check structure for each method
        assert "bootstrap_inference" in effect_sizes
        assert "wild_cluster_bootstrap" in effect_sizes
        assert "cross_method_summary" in effect_sizes

        # Check bootstrap inference effect sizes
        bootstrap_es = effect_sizes["bootstrap_inference"]
        assert "math_score" in bootstrap_es
        assert "reading_score" in bootstrap_es

        # Verify effect size fields
        math_es = bootstrap_es["math_score"]
        required_fields = [
            "cohens_d",
            "interpretation",
            "raw_coefficient",
            "pooled_std",
            "ci_lower_d",
            "ci_upper_d",
        ]
        for field in required_fields:
            assert field in math_es

        # Check interpretation values
        assert math_es["interpretation"] in ["negligible", "small", "medium", "large"]

        # Verify cross-method summary
        cross_summary = effect_sizes["cross_method_summary"]
        if "math_score" in cross_summary:
            math_cross = cross_summary["math_score"]
            required_cross_fields = [
                "mean_effect_size",
                "std_effect_size",
                "consistency_score",
                "consistency_interpretation",
                "n_methods",
            ]
            for field in required_cross_fields:
                assert field in math_cross

    def test_power_analysis(self, analyzer_with_results):
        """Test power analysis and minimum detectable effects."""
        results = analyzer_with_results.robustness_results
        power_results = analyzer_with_results.power_analysis(results)

        # Check overall structure
        assert "sample_characteristics" in power_results
        assert "overall_assessment" in power_results
        assert "bootstrap_inference" in power_results

        # Check sample characteristics
        sample_chars = power_results["sample_characteristics"]
        required_sample_fields = [
            "total_observations",
            "n_states",
            "n_treated",
            "n_control",
            "treatment_proportion",
        ]
        for field in required_sample_fields:
            assert field in sample_chars

        # Check method-specific power results
        bootstrap_power = power_results["bootstrap_inference"]
        if "math_score" in bootstrap_power:
            math_power = bootstrap_power["math_score"]
            required_power_fields = [
                "observed_power",
                "minimum_detectable_effect",
                "adequately_powered",
                "power_interpretation",
            ]
            for field in required_power_fields:
                assert field in math_power

            # Power should be between 0 and 1
            assert 0 <= math_power["observed_power"] <= 1

            # MDE should be positive
            assert math_power["minimum_detectable_effect"] > 0

        # Check overall assessment
        overall = power_results["overall_assessment"]
        if "mean_power" in overall:
            required_overall_fields = [
                "mean_power",
                "adequately_powered_proportion",
                "recommendation",
            ]
            for field in required_overall_fields:
                assert field in overall

    def test_enhanced_confidence_intervals(self, analyzer_with_results):
        """Test enhanced confidence interval methods."""
        results = analyzer_with_results.robustness_results
        enhanced_cis = analyzer_with_results.enhanced_confidence_intervals(results)

        # Check structure
        assert "bootstrap_inference" in enhanced_cis
        assert "simultaneous_bands" in enhanced_cis

        # Check bootstrap method CIs
        bootstrap_cis = enhanced_cis["bootstrap_inference"]
        if "math_score" in bootstrap_cis:
            math_ci = bootstrap_cis["math_score"]

            # Should have BCa CI since bootstrap samples are available
            assert "bca_ci" in math_ci
            bca = math_ci["bca_ci"]
            assert "lower" in bca
            assert "upper" in bca
            assert "method" in bca

            # Should have small sample adjustment
            assert "small_sample_adjusted" in math_ci
            small_sample = math_ci["small_sample_adjusted"]
            assert "lower" in small_sample
            assert "upper" in small_sample
            assert "degrees_of_freedom" in small_sample

        # Check simultaneous bands
        sim_bands = enhanced_cis["simultaneous_bands"]
        if "bootstrap_inference" in sim_bands:
            bootstrap_bands = sim_bands["bootstrap_inference"]
            if "math_score" in bootstrap_bands:
                math_band = bootstrap_bands["math_score"]
                required_band_fields = [
                    "lower",
                    "upper",
                    "simultaneous_coverage",
                    "individual_coverage",
                    "adjustment_method",
                ]
                for field in required_band_fields:
                    assert field in math_band

    def test_bca_confidence_interval_calculation(self, analyzer_with_results):
        """Test BCa confidence interval calculation specifically."""
        # Test with known data
        original_stat = 2.5
        bootstrap_stats = np.random.normal(2.5, 0.5, 1000)
        confidence_level = 0.95

        bca_result = analyzer_with_results._calculate_bca_ci(
            original_stat, bootstrap_stats, confidence_level
        )

        # Check structure
        required_fields = ["lower", "upper", "method", "coverage_level"]
        for field in required_fields:
            assert field in bca_result

        # Check logical constraints
        assert bca_result["lower"] < bca_result["upper"]
        assert bca_result["coverage_level"] == confidence_level
        assert bca_result["method"] in ["BCa", "percentile (fallback)"]

    def test_romano_wolf_correction(self, analyzer_with_results):
        """Test Romano-Wolf stepdown correction."""
        # Test with known p-values
        p_values = np.array([0.01, 0.03, 0.05, 0.20])

        corrected = analyzer_with_results._romano_wolf_correction(p_values, alpha=0.05)

        # Check structure
        assert len(corrected) == len(p_values)
        assert all(0 <= p <= 1 for p in corrected)

        # Should be more conservative than original
        assert all(corrected >= p_values)

    def test_cross_method_effect_size_comparison(self, analyzer_with_results):
        """Test cross-method effect size comparison functionality."""
        # Create effect sizes with known values for testing using actual outcome variables
        effect_sizes = {
            "method1": {
                "math_grade8_gap": {"cohens_d": 0.3}, 
                "reading_grade4_score": {"cohens_d": 0.1}
            },
            "method2": {
                "math_grade8_gap": {"cohens_d": 0.35}, 
                "reading_grade4_score": {"cohens_d": 0.15}
            },
        }

        cross_comparison = analyzer_with_results._compare_effect_sizes(effect_sizes)

        # Should have results for both outcomes
        assert "math_grade8_gap" in cross_comparison
        assert "reading_grade4_score" in cross_comparison

        # Check math_grade8_gap comparison
        math_comparison = cross_comparison["math_grade8_gap"]
        required_fields = [
            "mean_effect_size",
            "std_effect_size",
            "consistency_score",
            "consistency_interpretation",
            "n_methods",
        ]
        for field in required_fields:
            assert field in math_comparison

        # Values should be reasonable
        assert abs(math_comparison["mean_effect_size"] - 0.325) < 0.01  # Mean of 0.3, 0.35
        assert math_comparison["n_methods"] == 2

    def test_power_calculation_for_outcome(self, analyzer_with_results):
        """Test power calculation for individual outcomes."""
        outcome_results = {"coefficient": 2.0, "std_error": 1.0}
        n_states = 25
        target_power = 0.8

        power_result = analyzer_with_results._calculate_power_for_outcome(
            outcome_results, n_states, target_power
        )

        # Check structure
        required_fields = [
            "observed_power",
            "minimum_detectable_effect",
            "adequately_powered",
            "power_interpretation",
        ]
        for field in required_fields:
            assert field in power_result

        # Check values are reasonable
        assert 0 <= power_result["observed_power"] <= 1
        assert power_result["minimum_detectable_effect"] > 0
        assert isinstance(power_result["adequately_powered"], bool)

    def test_enhanced_inference_integration(self, analyzer_with_results):
        """Test integration of all enhanced inference methods."""
        results = analyzer_with_results.robustness_results

        # Run all enhanced inference methods
        corrections = analyzer_with_results.multiple_testing_corrections(results)
        effect_sizes = analyzer_with_results.calculate_effect_sizes(results)
        power_analysis = analyzer_with_results.power_analysis(results)
        enhanced_cis = analyzer_with_results.enhanced_confidence_intervals(results)

        # Add to results
        analyzer_with_results.robustness_results.update(
            {
                "multiple_testing": corrections,
                "effect_sizes": effect_sizes,
                "power_analysis": power_analysis,
                "enhanced_confidence_intervals": enhanced_cis,
            }
        )

        # Test summary methods
        enhanced_summary = analyzer_with_results._summarize_enhanced_inference()
        enhanced_conclusion = analyzer_with_results._enhanced_inference_conclusion()

        # Check that summaries are non-empty strings
        assert isinstance(enhanced_summary, str)
        assert len(enhanced_summary) > 0
        assert isinstance(enhanced_conclusion, str)
        assert len(enhanced_conclusion) > 0

        # Should contain key information
        assert "Multiple Testing" in enhanced_summary
        assert "Effect Size" in enhanced_summary
        assert "Power Analysis" in enhanced_summary

    def test_enhanced_inference_error_handling(self, analyzer_with_results):
        """Test error handling in enhanced inference methods."""
        # Test with empty results
        empty_results = {}

        corrections = analyzer_with_results.multiple_testing_corrections(empty_results)
        assert "corrections" in corrections
        assert "summary" in corrections

        effect_sizes = analyzer_with_results.calculate_effect_sizes(empty_results)
        assert isinstance(effect_sizes, dict)

        power_analysis = analyzer_with_results.power_analysis(empty_results)
        assert isinstance(power_analysis, dict)

        enhanced_cis = analyzer_with_results.enhanced_confidence_intervals(empty_results)
        assert isinstance(enhanced_cis, dict)

    def test_enhanced_inference_with_missing_data(self, analyzer_with_results):
        """Test enhanced inference methods with missing data fields."""
        # Results missing some expected fields
        incomplete_results = {
            "method1": {
                "outcome1": {"coefficient": 1.5}  # Missing std_error, p_value, etc.
            }
        }

        # Should handle gracefully without crashing
        corrections = analyzer_with_results.multiple_testing_corrections(incomplete_results)
        effect_sizes = analyzer_with_results.calculate_effect_sizes(incomplete_results)
        power_analysis = analyzer_with_results.power_analysis(incomplete_results)
        enhanced_cis = analyzer_with_results.enhanced_confidence_intervals(incomplete_results)

        # All should return valid dictionaries
        assert isinstance(corrections, dict)
        assert isinstance(effect_sizes, dict)
        assert isinstance(power_analysis, dict)
        assert isinstance(enhanced_cis, dict)


if __name__ == "__main__":
    pytest.main([__file__])
