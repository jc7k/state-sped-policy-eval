"""
Phase 4.3: Robustness and Extensions Module

Comprehensive robustness checks and sensitivity analysis for special education
policy evaluation study.

Robustness tests implemented:
1. Leave-one-state-out analysis
2. Alternative clustering (regional vs state)
3. Synthetic control for key treated states
4. Permutation tests for inference
5. Specification curve analysis

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)


class RobustnessAnalyzer:
    """
    Comprehensive robustness testing for causal analysis.

    Implements various sensitivity checks to assess the
    stability of main treatment effect estimates.
    """

    def __init__(self, data_path: str = "data/final/analysis_panel.csv"):
        """
        Initialize robustness analyzer.

        Args:
            data_path: Path to the analysis panel dataset
        """
        self.data_path = data_path
        self.df = None
        self.robustness_results = {}
        self.output_dir = Path("output")
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        # Create output directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load and prepare data for robustness analysis with comprehensive validation."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded analysis panel: {self.df.shape}")

            # Validate required columns
            required_cols = ["state", "year", "post_treatment"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Clean and validate data structure
            self._validate_and_clean_data()

            # Identify outcome variables
            self.outcome_vars = self._get_outcome_variables()
            print(f"Outcome variables for robustness: {self.outcome_vars}")

            # Add regional indicators using existing region column
            self._add_regional_indicators()

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _validate_and_clean_data(self) -> None:
        """Validate and clean data structure to prevent clustering failures."""
        print("  Validating and cleaning data structure...")

        # Check for missing values in key columns
        key_cols = ["state", "year", "post_treatment"]
        for col in key_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                print(f"    Warning: {missing_count} missing values in {col}")
                if col == "post_treatment":
                    self.df[col] = self.df[col].fillna(0)

        # Ensure state column is string type for proper grouping
        self.df["state"] = self.df["state"].astype(str)

        # Clean year column
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce")

        # Remove any completely empty rows
        initial_rows = len(self.df)
        self.df = self.df.dropna(how="all")
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            print(f"    Removed {removed_rows} completely empty rows")

        # Reset index to ensure clean indexing
        self.df = self.df.reset_index(drop=True)

        # Validate state coverage
        n_states = self.df["state"].nunique()
        n_years = self.df["year"].nunique()
        print(f"    Data validation complete: {n_states} states, {n_years} years")

    def _get_outcome_variables(self) -> list[str]:
        """Identify main outcome variables for robustness analysis."""
        potential_outcomes = []

        for col in self.df.columns:
            col_lower = col.lower()
            # Prioritize achievement gaps and scores
            if any(term in col_lower for term in ["gap", "score", "achievement"]) or any(
                term in col_lower for term in ["inclusion", "placement", "spending"]
            ):
                potential_outcomes.append(col)

        # If no specific outcomes found, use numeric variables
        if not potential_outcomes:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            excluded = ["year", "post_treatment"]
            potential_outcomes = [col for col in numeric_cols if col not in excluded][:3]

        return potential_outcomes[:3]  # Focus on top 3 outcomes for robustness

    def _add_regional_indicators(self) -> None:
        """Add census region indicators for alternative clustering."""
        # First, check if we already have a region column
        if "region" in self.df.columns:
            # Use existing region column, just standardize the name
            self.df["census_region"] = self.df["region"]
            print("  Using existing region column for clustering")
        else:
            # Create region mapping from state abbreviations
            regions = {
                "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
                "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
                "South": [
                    "DE",
                    "FL",
                    "GA",
                    "MD",
                    "NC",
                    "SC",
                    "VA",
                    "DC",
                    "WV",
                    "AL",
                    "KY",
                    "MS",
                    "TN",
                    "AR",
                    "LA",
                    "OK",
                    "TX",
                ],
                "West": [
                    "AZ",
                    "CO",
                    "ID",
                    "MT",
                    "NV",
                    "NM",
                    "UT",
                    "WY",
                    "AK",
                    "CA",
                    "HI",
                    "OR",
                    "WA",
                ],
            }

            # Create reverse mapping
            state_to_region = {}
            for region, states in regions.items():
                for state in states:
                    state_to_region[state] = region

            # Add region column
            self.df["census_region"] = self.df["state"].map(state_to_region)

        # Fill missing regions
        self.df["census_region"] = self.df["census_region"].fillna("Unknown")

        # Ensure census_region is string type for clustering
        self.df["census_region"] = self.df["census_region"].astype(str)

        # Validate region distribution
        region_counts = self.df["census_region"].value_counts().to_dict()
        print(f"  Regional distribution: {region_counts}")

        # Warn about small regional clusters
        small_regions = [region for region, count in region_counts.items() if count < 20]
        if small_regions:
            print(f"  Warning: Small regional clusters may cause issues: {small_regions}")

    def leave_one_state_out(self) -> dict[str, Any]:
        """
        Leave-one-state-out robustness analysis with improved error handling.

        Returns:
            Dictionary of LOSO results
        """
        print("Running leave-one-state-out analysis...")

        loso_results = {}
        states = self.df["state"].unique()

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            print(f"  Analyzing {outcome}...")
            state_results = {}

            for state in states:
                try:
                    # Create subset excluding this state
                    subset = self.df[self.df["state"] != state].copy()

                    if subset.empty or subset["post_treatment"].sum() == 0:
                        continue

                    # Clean subset data for clustering
                    subset = subset.dropna(subset=["state", "year", outcome, "post_treatment"])
                    subset = subset.reset_index(drop=True)

                    if len(subset) < 10:  # Need minimum observations
                        continue

                    # Try clustered standard errors first
                    formula = f"{outcome} ~ post_treatment + C(state) + C(year)"

                    try:
                        # Attempt state clustering
                        unique_states = subset["state"].unique()
                        if len(unique_states) >= 5:  # Need minimum clusters
                            model = smf.ols(formula, data=subset).fit(
                                cov_type="cluster", cov_kwds={"groups": subset["state"]}
                            )
                        else:
                            # Fall back to robust standard errors
                            model = smf.ols(formula, data=subset).fit(cov_type="HC1")

                    except Exception as cluster_error:
                        # Final fallback to OLS standard errors
                        print(f"    Clustering failed for {state}, using OLS SEs: {cluster_error}")
                        model = smf.ols(formula, data=subset).fit()

                    state_results[state] = {
                        "coefficient": model.params["post_treatment"],
                        "se": model.bse["post_treatment"],
                        "p_value": model.pvalues["post_treatment"],
                        "n_obs": int(model.nobs),
                        "n_states": len(subset["state"].unique()),
                    }

                except Exception as e:
                    print(f"    Error excluding {state}: {e}")
                    continue

            if state_results:
                # Calculate summary statistics
                coeffs = [r["coefficient"] for r in state_results.values()]

                loso_results[outcome] = {
                    "state_results": state_results,
                    "mean_coeff": np.mean(coeffs),
                    "std_coeff": np.std(coeffs),
                    "min_coeff": np.min(coeffs),
                    "max_coeff": np.max(coeffs),
                    "n_estimates": len(coeffs),
                }

                print(
                    f"    {outcome}: {len(coeffs)} successful estimates, "
                    f"Mean={np.mean(coeffs):.4f}, Std={np.std(coeffs):.4f}"
                )
            else:
                print(f"    {outcome}: No successful LOSO estimates")

        self.robustness_results["leave_one_state_out"] = loso_results
        return loso_results

    def alternative_clustering(self) -> dict[str, Any]:
        """
        Test robustness to alternative clustering strategies with improved error handling.

        Returns:
            Dictionary of clustering results
        """
        print("Running alternative clustering analysis...")

        clustering_results = {}

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            print(f"  Analyzing {outcome}...")
            cluster_specs = {}

            # Clean data for this outcome
            analysis_df = self.df.dropna(subset=["state", "year", outcome, "post_treatment"]).copy()
            analysis_df = analysis_df.reset_index(drop=True)

            if len(analysis_df) < 20:
                print(f"    Insufficient data for {outcome}: {len(analysis_df)} observations")
                continue

            formula = f"{outcome} ~ post_treatment + C(state) + C(year)"

            # 1. State clustering (baseline)
            try:
                unique_states = analysis_df["state"].unique()
                if len(unique_states) >= 5:
                    # Ensure state groups align with data
                    state_groups = analysis_df["state"].values
                    model_state = smf.ols(formula, data=analysis_df).fit(
                        cov_type="cluster", cov_kwds={"groups": state_groups}
                    )

                    cluster_specs["state"] = {
                        "coefficient": model_state.params["post_treatment"],
                        "se": model_state.bse["post_treatment"],
                        "p_value": model_state.pvalues["post_treatment"],
                        "n_clusters": len(unique_states),
                    }
                else:
                    print(f"    Too few states for clustering: {len(unique_states)}")

            except Exception as e:
                print(f"    Error with state clustering: {e}")

            # 2. Regional clustering
            try:
                if "census_region" in analysis_df.columns:
                    unique_regions = analysis_df["census_region"].unique()
                    if len(unique_regions) >= 3:  # Need minimum clusters
                        region_groups = analysis_df["census_region"].values
                        model_region = smf.ols(formula, data=analysis_df).fit(
                            cov_type="cluster", cov_kwds={"groups": region_groups}
                        )

                        cluster_specs["region"] = {
                            "coefficient": model_region.params["post_treatment"],
                            "se": model_region.bse["post_treatment"],
                            "p_value": model_region.pvalues["post_treatment"],
                            "n_clusters": len(unique_regions),
                        }
                    else:
                        print(f"    Too few regions for clustering: {len(unique_regions)}")

            except Exception as e:
                print(f"    Error with regional clustering: {e}")

            # 3. Year clustering
            try:
                unique_years = analysis_df["year"].unique()
                if len(unique_years) >= 5:  # Need minimum clusters
                    year_groups = analysis_df["year"].values
                    model_year = smf.ols(formula, data=analysis_df).fit(
                        cov_type="cluster", cov_kwds={"groups": year_groups}
                    )

                    cluster_specs["year"] = {
                        "coefficient": model_year.params["post_treatment"],
                        "se": model_year.bse["post_treatment"],
                        "p_value": model_year.pvalues["post_treatment"],
                        "n_clusters": len(unique_years),
                    }
                else:
                    print(f"    Too few years for clustering: {len(unique_years)}")

            except Exception as e:
                print(f"    Error with year clustering: {e}")

            # 4. Robust standard errors (always works as fallback)
            try:
                model_robust = smf.ols(formula, data=analysis_df).fit(cov_type="HC1")

                cluster_specs["robust"] = {
                    "coefficient": model_robust.params["post_treatment"],
                    "se": model_robust.bse["post_treatment"],
                    "p_value": model_robust.pvalues["post_treatment"],
                    "n_clusters": 0,  # No clustering
                }

            except Exception as e:
                print(f"    Error with robust SEs: {e}")

            if cluster_specs:
                clustering_results[outcome] = cluster_specs

                # Print comparison
                print(f"    Clustering comparison for {outcome}:")
                for cluster_type, results in cluster_specs.items():
                    n_clust = results.get("n_clusters", 0)
                    print(
                        f"      {cluster_type}: β={results['coefficient']:.4f}, SE={results['se']:.4f} "
                        f"({n_clust} clusters)"
                        if n_clust > 0
                        else f"      {cluster_type}: β={results['coefficient']:.4f}, SE={results['se']:.4f}"
                    )
            else:
                print(f"    No successful clustering methods for {outcome}")

        self.robustness_results["alternative_clustering"] = clustering_results
        return clustering_results

    def permutation_test(self, n_permutations: int = 1000) -> dict[str, Any]:
        """
        Permutation test for treatment assignment.

        Args:
            n_permutations: Number of permutations to perform

        Returns:
            Dictionary of permutation test results
        """
        print(f"Running permutation test with {n_permutations} permutations...")

        permutation_results = {}

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            print(f"  Testing {outcome}...")

            # Get actual treatment effect
            try:
                formula = f"{outcome} ~ post_treatment + C(state) + C(year)"
                actual_model = smf.ols(formula, data=self.df).fit()
                actual_coeff = actual_model.params["post_treatment"]

            except Exception as e:
                print(f"    Error getting actual coefficient: {e}")
                continue

            # Run permutations
            placebo_coeffs = []
            random.seed(42)  # For reproducibility

            for i in range(n_permutations):
                try:
                    # Randomly permute treatment assignment
                    df_perm = self.df.copy()
                    df_perm["placebo_treatment"] = np.random.permutation(df_perm["post_treatment"])

                    # Run placebo regression
                    placebo_formula = f"{outcome} ~ placebo_treatment + C(state) + C(year)"
                    placebo_model = smf.ols(placebo_formula, data=df_perm).fit()

                    placebo_coeffs.append(placebo_model.params["placebo_treatment"])

                    if (i + 1) % 100 == 0:
                        print(f"    Completed {i + 1}/{n_permutations} permutations")

                except Exception as e:
                    if i < 10:  # Only print first few errors
                        print(f"    Error in permutation {i}: {e}")
                    continue

            if placebo_coeffs:
                # Calculate permutation p-value
                placebo_coeffs = np.array(placebo_coeffs)
                p_value_perm = np.mean(np.abs(placebo_coeffs) >= np.abs(actual_coeff))

                permutation_results[outcome] = {
                    "actual_coefficient": actual_coeff,
                    "placebo_coefficients": placebo_coeffs,
                    "permutation_pvalue": p_value_perm,
                    "n_permutations": len(placebo_coeffs),
                    "placebo_mean": np.mean(placebo_coeffs),
                    "placebo_std": np.std(placebo_coeffs),
                }

                print(f"    {outcome}: Actual={actual_coeff:.4f}, Perm p-value={p_value_perm:.3f}")

        self.robustness_results["permutation_test"] = permutation_results
        return permutation_results

    def specification_curve(self) -> dict[str, Any]:
        """
        Specification curve analysis with improved error handling.

        Returns:
            Dictionary of specification curve results
        """
        print("Running specification curve analysis...")

        spec_curve_results = {}

        # Define specification variations
        control_sets = [
            [],  # No controls
            ["time_trend"] if "time_trend" in self.df.columns else [],
            ["post_covid"] if "post_covid" in self.df.columns else [],
            (
                ["time_trend", "post_covid"]
                if all(col in self.df.columns for col in ["time_trend", "post_covid"])
                else []
            ),
        ]

        # Filter out empty control sets (but keep the empty list for no-controls spec)
        control_sets = [controls for controls in control_sets if isinstance(controls, list)]

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            print(f"  Analyzing {outcome}...")
            specifications = []

            # Clean data for this outcome
            analysis_df = self.df.dropna(subset=["state", "year", outcome, "post_treatment"]).copy()
            analysis_df = analysis_df.reset_index(drop=True)

            if len(analysis_df) < 20:
                print(f"    Insufficient data for {outcome}: {len(analysis_df)} observations")
                continue

            for i, controls in enumerate(control_sets):
                try:
                    # Build formula
                    formula = f"{outcome} ~ post_treatment"
                    if controls:
                        # Verify controls exist in data
                        available_controls = [c for c in controls if c in analysis_df.columns]
                        if available_controls:
                            formula += " + " + " + ".join(available_controls)
                        else:
                            print(f"    Specification {i}: No valid controls found")
                            continue
                    formula += " + C(state) + C(year)"

                    # Run model with appropriate standard errors
                    unique_states = analysis_df["state"].unique()
                    if len(unique_states) >= 5:
                        # Try clustered standard errors
                        try:
                            state_groups = analysis_df["state"].values
                            model = smf.ols(formula, data=analysis_df).fit(
                                cov_type="cluster", cov_kwds={"groups": state_groups}
                            )
                        except Exception:
                            # Fall back to robust standard errors
                            model = smf.ols(formula, data=analysis_df).fit(cov_type="HC1")
                    else:
                        # Use robust standard errors for small samples
                        model = smf.ols(formula, data=analysis_df).fit(cov_type="HC1")

                    specifications.append(
                        {
                            "spec_id": i,
                            "controls": controls,
                            "controls_str": ", ".join(controls) if controls else "None",
                            "coefficient": model.params["post_treatment"],
                            "se": model.bse["post_treatment"],
                            "p_value": model.pvalues["post_treatment"],
                            "r_squared": model.rsquared,
                            "n_obs": int(model.nobs),
                        }
                    )

                except Exception as e:
                    print(f"    Error with specification {i}: {e}")
                    continue

            if specifications:
                spec_df = pd.DataFrame(specifications)

                spec_curve_results[outcome] = {
                    "specifications": spec_df,
                    "n_specs": len(spec_df),
                    "coeff_range": (spec_df["coefficient"].min(), spec_df["coefficient"].max()),
                    "significant_specs": (spec_df["p_value"] < 0.05).sum(),
                }

                print(
                    f"    {outcome}: {len(spec_df)} specifications, "
                    f"{(spec_df['p_value'] < 0.05).sum()} significant"
                )
            else:
                print(f"    {outcome}: No successful specifications")

        self.robustness_results["specification_curve"] = spec_curve_results
        return spec_curve_results

    def bootstrap_inference(self, n_bootstrap: int = 1000) -> dict[str, Any]:
        """
        Implement bootstrap-based inference as alternative to clustering.

        Args:
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Dictionary with bootstrap results for each outcome
        """
        print("\n=== Bootstrap Inference Analysis ===")

        if self.df is None:
            self.load_data()

        bootstrap_results = {}

        for outcome in self.outcome_vars:
            print(f"\nBootstrap analysis for {outcome}...")

            try:
                # Create clean dataset for this outcome
                df_clean = (
                    self.df[[outcome, "post_treatment", "state", "year", "region", "treated"]]
                    .dropna()
                    .reset_index(drop=True)
                )

                if len(df_clean) < 20:
                    print(f"  Insufficient data for {outcome}: {len(df_clean)} observations")
                    bootstrap_results[outcome] = {
                        "coefficient": np.nan,
                        "bootstrap_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_bootstraps": 0,
                    }
                    continue

                # Fit base model
                formula = f"{outcome} ~ post_treatment + C(state) + C(year)"
                try:
                    base_model = smf.ols(formula, data=df_clean).fit()
                    base_coef = base_model.params["post_treatment"]
                except Exception as e:
                    print(f"  Base model failed for {outcome}: {e}")
                    bootstrap_results[outcome] = {
                        "coefficient": np.nan,
                        "bootstrap_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_bootstraps": 0,
                    }
                    continue

                # Bootstrap procedure
                bootstrap_coefs = []
                states = df_clean["state"].unique()

                for _i in range(n_bootstrap):
                    try:
                        # Sample states with replacement (cluster bootstrap)
                        bootstrap_states = np.random.choice(states, size=len(states), replace=True)

                        # Create bootstrap sample
                        bootstrap_df = pd.concat(
                            [
                                df_clean[df_clean["state"] == state].copy()
                                for state in bootstrap_states
                            ],
                            ignore_index=True,
                        )

                        # Fit model on bootstrap sample
                        bootstrap_model = smf.ols(formula, data=bootstrap_df).fit()
                        bootstrap_coefs.append(bootstrap_model.params["post_treatment"])

                    except Exception:
                        # Silent failure for individual bootstrap draws
                        continue

                if len(bootstrap_coefs) < 100:  # Need at least 100 successful draws
                    print(
                        f"  Too few successful bootstrap draws for {outcome}: {len(bootstrap_coefs)}"
                    )
                    bootstrap_results[outcome] = {
                        "coefficient": base_coef,
                        "bootstrap_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_bootstraps": len(bootstrap_coefs),
                    }
                    continue

                # Calculate bootstrap statistics
                bootstrap_coefs = np.array(bootstrap_coefs)
                bootstrap_se = np.std(bootstrap_coefs)
                ci_lower = np.percentile(bootstrap_coefs, 2.5)
                ci_upper = np.percentile(bootstrap_coefs, 97.5)

                # Two-tailed p-value (assuming null hypothesis = 0)
                p_value = 2 * min(np.mean(bootstrap_coefs <= 0), np.mean(bootstrap_coefs >= 0))

                bootstrap_results[outcome] = {
                    "coefficient": base_coef,
                    "bootstrap_se": bootstrap_se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                    "successful_bootstraps": len(bootstrap_coefs),
                }

                print(f"  Coefficient: {base_coef:.3f}")
                print(f"  Bootstrap SE: {bootstrap_se:.3f}")
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"  p-value: {p_value:.3f}")
                print(f"  Successful bootstraps: {len(bootstrap_coefs)}/{n_bootstrap}")

            except Exception as e:
                print(f"  Bootstrap failed for {outcome}: {e}")
                bootstrap_results[outcome] = {
                    "coefficient": np.nan,
                    "bootstrap_se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "successful_bootstraps": 0,
                }

        self.robustness_results["bootstrap"] = bootstrap_results
        print("\nBootstrap inference completed.")
        return bootstrap_results

    def jackknife_inference(self) -> dict[str, Any]:
        """
        Implement jackknife-based inference as alternative to clustering.

        Returns:
            Dictionary with jackknife results for each outcome
        """
        print("\n=== Jackknife Inference Analysis ===")

        if self.df is None:
            self.load_data()

        jackknife_results = {}

        for outcome in self.outcome_vars:
            print(f"\nJackknife analysis for {outcome}...")

            try:
                # Create clean dataset for this outcome
                df_clean = (
                    self.df[[outcome, "post_treatment", "state", "year", "region", "treated"]]
                    .dropna()
                    .reset_index(drop=True)
                )

                if len(df_clean) < 20:
                    print(f"  Insufficient data for {outcome}: {len(df_clean)} observations")
                    jackknife_results[outcome] = {
                        "coefficient": np.nan,
                        "jackknife_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_jackknife": 0,
                    }
                    continue

                # Fit base model
                formula = f"{outcome} ~ post_treatment + C(state) + C(year)"
                try:
                    base_model = smf.ols(formula, data=df_clean).fit()
                    base_coef = base_model.params["post_treatment"]
                except Exception as e:
                    print(f"  Base model failed for {outcome}: {e}")
                    jackknife_results[outcome] = {
                        "coefficient": np.nan,
                        "jackknife_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_jackknife": 0,
                    }
                    continue

                # Jackknife procedure (leave-one-state-out)
                jackknife_coefs = []
                states = df_clean["state"].unique()

                for state in states:
                    try:
                        # Create jackknife sample (exclude one state)
                        jackknife_df = df_clean[df_clean["state"] != state].copy()

                        if len(jackknife_df) < 10:  # Need minimum observations
                            continue

                        # Fit model on jackknife sample
                        jackknife_model = smf.ols(formula, data=jackknife_df).fit()
                        jackknife_coefs.append(jackknife_model.params["post_treatment"])

                    except Exception:
                        # Silent failure for individual jackknife samples
                        continue

                if len(jackknife_coefs) < 5:  # Need at least 5 successful samples
                    print(
                        f"  Too few successful jackknife samples for {outcome}: {len(jackknife_coefs)}"
                    )
                    jackknife_results[outcome] = {
                        "coefficient": base_coef,
                        "jackknife_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_jackknife": len(jackknife_coefs),
                    }
                    continue

                # Calculate jackknife statistics
                jackknife_coefs = np.array(jackknife_coefs)
                n_samples = len(jackknife_coefs)

                # Jackknife standard error
                jackknife_se = np.sqrt((n_samples - 1) * np.var(jackknife_coefs))

                # Confidence interval using t-distribution
                from scipy.stats import t

                t_critical = t.ppf(0.975, df=n_samples - 1)
                ci_lower = base_coef - t_critical * jackknife_se
                ci_upper = base_coef + t_critical * jackknife_se

                # p-value using t-test
                t_stat = base_coef / jackknife_se if jackknife_se > 0 else 0
                p_value = 2 * (1 - t.cdf(abs(t_stat), df=n_samples - 1))

                jackknife_results[outcome] = {
                    "coefficient": base_coef,
                    "jackknife_se": jackknife_se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                    "successful_jackknife": len(jackknife_coefs),
                }

                print(f"  Coefficient: {base_coef:.3f}")
                print(f"  Jackknife SE: {jackknife_se:.3f}")
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"  p-value: {p_value:.3f}")
                print(f"  Successful jackknife samples: {len(jackknife_coefs)}/{len(states)}")

            except Exception as e:
                print(f"  Jackknife failed for {outcome}: {e}")
                jackknife_results[outcome] = {
                    "coefficient": np.nan,
                    "jackknife_se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "successful_jackknife": 0,
                }

        self.robustness_results["jackknife"] = jackknife_results
        print("\nJackknife inference completed.")
        return jackknife_results

    def wild_cluster_bootstrap(self, n_bootstrap: int = 999) -> dict[str, Any]:
        """
        Implement wild cluster bootstrap for robust inference with small number of clusters.

        Args:
            n_bootstrap: Number of bootstrap iterations (odd number recommended)

        Returns:
            Dictionary with wild bootstrap results for each outcome
        """
        print("\n=== Wild Cluster Bootstrap Analysis ===")

        if self.df is None:
            self.load_data()

        wild_bootstrap_results = {}

        for outcome in self.outcome_vars:
            print(f"\nWild bootstrap analysis for {outcome}...")

            try:
                # Create clean dataset for this outcome
                df_clean = (
                    self.df[[outcome, "post_treatment", "state", "year", "region", "treated"]]
                    .dropna()
                    .reset_index(drop=True)
                )

                if len(df_clean) < 20:
                    print(f"  Insufficient data for {outcome}: {len(df_clean)} observations")
                    wild_bootstrap_results[outcome] = {
                        "coefficient": np.nan,
                        "wild_bootstrap_p": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "successful_bootstraps": 0,
                    }
                    continue

                # Fit base model and get residuals
                formula = f"{outcome} ~ post_treatment + C(state) + C(year)"
                try:
                    base_model = smf.ols(formula, data=df_clean).fit()
                    base_coef = base_model.params["post_treatment"]
                    base_t_stat = base_model.tvalues["post_treatment"]
                    residuals = base_model.resid
                except Exception as e:
                    print(f"  Base model failed for {outcome}: {e}")
                    wild_bootstrap_results[outcome] = {
                        "coefficient": np.nan,
                        "wild_bootstrap_p": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "successful_bootstraps": 0,
                    }
                    continue

                # Wild bootstrap procedure
                bootstrap_t_stats = []
                states = df_clean["state"].unique()

                for _i in range(n_bootstrap):
                    try:
                        # Generate wild bootstrap weights (Rademacher: +1 or -1)
                        wild_weights = np.random.choice([-1, 1], size=len(states))

                        # Create weight mapping for each observation
                        weight_map = dict(zip(states, wild_weights, strict=False))
                        obs_weights = df_clean["state"].map(weight_map)

                        # Create wild bootstrap dependent variable
                        y_star = base_model.fittedvalues + obs_weights * residuals
                        df_bootstrap = df_clean.copy()
                        df_bootstrap[outcome] = y_star

                        # Fit model on bootstrap sample
                        bootstrap_model = smf.ols(formula, data=df_bootstrap).fit()
                        bootstrap_t_stats.append(bootstrap_model.tvalues["post_treatment"])

                    except Exception:
                        # Silent failure for individual bootstrap draws
                        continue

                if len(bootstrap_t_stats) < min(
                    50, n_bootstrap * 0.5
                ):  # Need at least 50% successful draws
                    print(
                        f"  Too few successful wild bootstrap draws for {outcome}: {len(bootstrap_t_stats)}"
                    )
                    wild_bootstrap_results[outcome] = {
                        "coefficient": base_coef,
                        "wild_bootstrap_p": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "successful_bootstraps": len(bootstrap_t_stats),
                    }
                    continue

                # Calculate wild bootstrap p-value
                bootstrap_t_stats = np.array(bootstrap_t_stats)

                # Two-tailed p-value
                p_value = np.mean(np.abs(bootstrap_t_stats) >= np.abs(base_t_stat))

                # Confidence interval using percentile method
                # Transform t-statistics back to coefficients (approximate)
                ci_lower = np.percentile(
                    bootstrap_t_stats * base_model.bse["post_treatment"] + base_coef, 2.5
                )
                ci_upper = np.percentile(
                    bootstrap_t_stats * base_model.bse["post_treatment"] + base_coef, 97.5
                )

                wild_bootstrap_results[outcome] = {
                    "coefficient": base_coef,
                    "wild_bootstrap_p": p_value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "successful_bootstraps": len(bootstrap_t_stats),
                }

                print(f"  Coefficient: {base_coef:.3f}")
                print(f"  Wild bootstrap p-value: {p_value:.3f}")
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"  Successful bootstraps: {len(bootstrap_t_stats)}/{n_bootstrap}")

            except Exception as e:
                print(f"  Wild bootstrap failed for {outcome}: {e}")
                wild_bootstrap_results[outcome] = {
                    "coefficient": np.nan,
                    "wild_bootstrap_p": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "successful_bootstraps": 0,
                }

        self.robustness_results["wild_bootstrap"] = wild_bootstrap_results
        print("\nWild cluster bootstrap completed.")
        return wild_bootstrap_results

    def multiple_testing_corrections(
        self, results: dict[str, Any], alpha: float = 0.05
    ) -> dict[str, Any]:
        """
        Apply multiple testing corrections to robustness results.

        Args:
            results: Dictionary of results from various robustness methods
            alpha: Significance level for corrections

        Returns:
            Dictionary with corrected p-values and significance indicators
        """
        import numpy as np
        from scipy import stats

        corrections_results = {}
        p_values = []
        method_outcome_pairs = []

        # Extract p-values from all methods and outcomes
        for method_name, method_results in results.items():
            if isinstance(method_results, dict):
                for outcome, outcome_results in method_results.items():
                    if isinstance(outcome_results, dict) and "p_value" in outcome_results:
                        p_values.append(outcome_results["p_value"])
                        method_outcome_pairs.append((method_name, outcome))

        if not p_values:
            print("No p-values found for multiple testing correction")
            return {"corrections": {}, "summary": {}}

        p_values = np.array(p_values)

        # Bonferroni correction
        bonferroni_corrected = np.minimum(p_values * len(p_values), 1.0)

        # Benjamini-Hochberg FDR control
        try:
            # Use statsmodels multipletests function instead
            from statsmodels.stats.multitest import multipletests
            reject, fdr_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        except Exception:
            # Fallback to manual BH procedure
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]
            n_tests = len(p_values)
            
            fdr_corrected = np.zeros_like(p_values)
            for i in range(n_tests):
                fdr_corrected[sorted_indices[i]] = min(1.0, 
                    sorted_p_values[i] * n_tests / (i + 1))

        # Romano-Wolf stepdown (simplified implementation)
        romano_wolf_corrected = self._romano_wolf_correction(p_values, alpha)

        # Organize results by method and outcome
        corrections = {}
        for i, (method_name, outcome) in enumerate(method_outcome_pairs):
            if method_name not in corrections:
                corrections[method_name] = {}
            corrections[method_name][outcome] = {
                "original_p": p_values[i],
                "bonferroni_p": bonferroni_corrected[i],
                "fdr_p": fdr_corrected[i],
                "romano_wolf_p": romano_wolf_corrected[i],
                "bonferroni_significant": bonferroni_corrected[i] < alpha,
                "fdr_significant": fdr_corrected[i] < alpha,
                "romano_wolf_significant": romano_wolf_corrected[i] < alpha,
            }

        # Summary statistics
        summary = {
            "total_tests": len(p_values),
            "original_significant": np.sum(p_values < alpha),
            "bonferroni_significant": np.sum(bonferroni_corrected < alpha),
            "fdr_significant": np.sum(fdr_corrected < alpha),
            "romano_wolf_significant": np.sum(romano_wolf_corrected < alpha),
            "alpha_level": alpha,
        }

        corrections_results = {"corrections": corrections, "summary": summary}

        print(
            f"Multiple testing corrections applied: "
            f"Original: {summary['original_significant']}/{summary['total_tests']}, "
            f"Bonferroni: {summary['bonferroni_significant']}/{summary['total_tests']}, "
            f"FDR: {summary['fdr_significant']}/{summary['total_tests']}, "
            f"Romano-Wolf: {summary['romano_wolf_significant']}/{summary['total_tests']}"
        )

        return corrections_results

    def _romano_wolf_correction(
        self, p_values: np.ndarray, alpha: float = 0.05, n_bootstrap: int = 1000
    ) -> np.ndarray:
        """
        Romano-Wolf stepdown correction using bootstrap resampling.

        Args:
            p_values: Array of original p-values
            alpha: Significance level
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Array of Romano-Wolf corrected p-values
        """
        import numpy as np
        from scipy.stats import norm

        n_tests = len(p_values)
        if n_tests == 1:
            return p_values

        # Convert p-values to test statistics (assuming two-tailed z-tests)
        test_stats = np.abs(norm.ppf(p_values / 2))

        # Sort test statistics in descending order
        sorted_indices = np.argsort(test_stats)[::-1]
        sorted_stats = test_stats[sorted_indices]

        # Bootstrap resampling for dependent test statistics
        corrected_p = np.zeros(n_tests)

        try:
            for i in range(n_tests):
                # For each test, count how often bootstrap exceeds observed statistic
                bootstrap_maxes = []
                for _ in range(n_bootstrap):
                    # Generate correlated bootstrap sample
                    bootstrap_stats = np.abs(
                        np.random.multivariate_normal(
                            np.zeros(n_tests - i),
                            np.eye(n_tests - i) * 0.8 + np.ones((n_tests - i, n_tests - i)) * 0.2,
                        )
                    )
                    bootstrap_maxes.append(np.max(bootstrap_stats))

                # Calculate adjusted p-value
                corrected_p[sorted_indices[i]] = np.mean(
                    np.array(bootstrap_maxes) >= sorted_stats[i]
                )

            # Apply stepdown procedure
            for i in range(1, n_tests):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                corrected_p[idx] = max(corrected_p[idx], corrected_p[prev_idx])

        except Exception as e:
            print(f"Romano-Wolf correction failed: {e}, using Bonferroni")
            corrected_p = np.minimum(p_values * n_tests, 1.0)

        return np.minimum(corrected_p, 1.0)

    def calculate_effect_sizes(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate standardized effect sizes for robustness results.

        Args:
            results: Dictionary of results from various robustness methods

        Returns:
            Dictionary with Cohen's d and standardized effect sizes
        """
        effect_sizes = {}

        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return effect_sizes

        outcome_vars = self._get_outcome_variables()

        # Calculate pooled standard deviations for standardization
        pooled_stds = {}
        for outcome in outcome_vars:
            if outcome in self.df.columns:
                # Calculate pooled standard deviation across treatment groups
                treated = self.df[self.df["post_treatment"] == 1][outcome]
                control = self.df[self.df["post_treatment"] == 0][outcome]

                if len(treated) > 1 and len(control) > 1:
                    pooled_var = (
                        (len(treated) - 1) * treated.var() + (len(control) - 1) * control.var()
                    ) / (len(treated) + len(control) - 2)
                    pooled_stds[outcome] = np.sqrt(pooled_var)
                else:
                    pooled_stds[outcome] = self.df[outcome].std()

        # Calculate effect sizes for each method and outcome
        for method_name, method_results in results.items():
            if isinstance(method_results, dict):
                effect_sizes[method_name] = {}

                for outcome, outcome_results in method_results.items():
                    if isinstance(outcome_results, dict) and "coefficient" in outcome_results:
                        coef = outcome_results["coefficient"]

                        # Cohen's d (standardized mean difference)
                        if outcome in pooled_stds and pooled_stds[outcome] > 0:
                            cohens_d = coef / pooled_stds[outcome]
                        else:
                            cohens_d = np.nan

                        # Effect size interpretation
                        abs_d = abs(cohens_d) if not np.isnan(cohens_d) else 0
                        if abs_d < 0.2:
                            interpretation = "negligible"
                        elif abs_d < 0.5:
                            interpretation = "small"
                        elif abs_d < 0.8:
                            interpretation = "medium"
                        else:
                            interpretation = "large"

                        # Confidence interval for effect size (if available)
                        ci_lower = ci_upper = np.nan
                        if ("ci_lower" in outcome_results and "ci_upper" in outcome_results and
                            outcome in pooled_stds and pooled_stds[outcome] > 0):
                            ci_lower = outcome_results["ci_lower"] / pooled_stds[outcome]
                            ci_upper = outcome_results["ci_upper"] / pooled_stds[outcome]

                        effect_sizes[method_name][outcome] = {
                            "cohens_d": cohens_d,
                            "interpretation": interpretation,
                            "raw_coefficient": coef,
                            "pooled_std": pooled_stds.get(outcome, np.nan),
                            "ci_lower_d": ci_lower,
                            "ci_upper_d": ci_upper,
                        }

        # Cross-method comparison
        effect_sizes["cross_method_summary"] = self._compare_effect_sizes(effect_sizes)

        return effect_sizes

    def _compare_effect_sizes(self, effect_sizes: dict[str, Any]) -> dict[str, Any]:
        """
        Compare effect sizes across different robustness methods.

        Args:
            effect_sizes: Dictionary of effect sizes by method and outcome

        Returns:
            Summary of cross-method effect size consistency
        """
        cross_method = {}
        outcome_vars = self._get_outcome_variables()

        for outcome in outcome_vars:
            outcome_effects = []
            method_names = []

            # Collect effect sizes for this outcome across methods
            for method_name, method_effects in effect_sizes.items():
                if method_name != "cross_method_summary" and isinstance(method_effects, dict):
                    if outcome in method_effects and not np.isnan(
                        method_effects[outcome]["cohens_d"]
                    ):
                        outcome_effects.append(method_effects[outcome]["cohens_d"])
                        method_names.append(method_name)

            if len(outcome_effects) > 1:
                outcome_effects = np.array(outcome_effects)

                cross_method[outcome] = {
                    "mean_effect_size": np.mean(outcome_effects),
                    "std_effect_size": np.std(outcome_effects),
                    "min_effect_size": np.min(outcome_effects),
                    "max_effect_size": np.max(outcome_effects),
                    "range": np.max(outcome_effects) - np.min(outcome_effects),
                    "consistency_score": 1
                    - (np.std(outcome_effects) / (np.abs(np.mean(outcome_effects)) + 1e-6)),
                    "n_methods": len(outcome_effects),
                    "methods": method_names,
                }

                # Consistency interpretation
                consistency = cross_method[outcome]["consistency_score"]
                if consistency > 0.9:
                    cross_method[outcome]["consistency_interpretation"] = "very high"
                elif consistency > 0.7:
                    cross_method[outcome]["consistency_interpretation"] = "high"
                elif consistency > 0.5:
                    cross_method[outcome]["consistency_interpretation"] = "moderate"
                else:
                    cross_method[outcome]["consistency_interpretation"] = "low"

        return cross_method

    def power_analysis(self, results: dict[str, Any], target_power: float = 0.8) -> dict[str, Any]:
        """
        Conduct power analysis and minimum detectable effects calculation.

        Args:
            results: Dictionary of results from various robustness methods
            target_power: Target power level for calculations

        Returns:
            Dictionary with power analysis results
        """
        
        power_results = {}

        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return power_results

        # Get sample characteristics
        total_n = len(self.df)
        n_states = self.df["state"].nunique()
        n_treated = len(self.df[self.df["post_treatment"] == 1])
        n_control = len(self.df[self.df["post_treatment"] == 0])

        power_results["sample_characteristics"] = {
            "total_observations": total_n,
            "n_states": n_states,
            "n_treated": n_treated,
            "n_control": n_control,
            "treatment_proportion": n_treated / total_n,
        }

        # Calculate power for each method and outcome
        for method_name, method_results in results.items():
            if isinstance(method_results, dict):
                power_results[method_name] = {}

                for outcome, outcome_results in method_results.items():
                    if isinstance(outcome_results, dict):
                        power_analysis_result = self._calculate_power_for_outcome(
                            outcome_results, n_states, target_power
                        )
                        power_results[method_name][outcome] = power_analysis_result

        # Overall assessment
        power_results["overall_assessment"] = self._assess_overall_power(power_results)

        return power_results

    def _calculate_power_for_outcome(
        self, outcome_results: dict[str, Any], n_states: int, target_power: float
    ) -> dict[str, Any]:
        """
        Calculate power analysis for a specific outcome.

        Args:
            outcome_results: Results for a specific outcome
            n_states: Number of states in sample
            target_power: Target power level

        Returns:
            Dictionary with power analysis for this outcome
        """
        from scipy.stats import norm, t

        power_result = {}

        if "coefficient" in outcome_results and "std_error" in outcome_results:
            coef = outcome_results["coefficient"]
            se = outcome_results["std_error"]

            # Degrees of freedom (conservative estimate)
            df = max(n_states - 2, 1)

            # Critical value for two-tailed test
            alpha = 0.05
            t_crit = t.ppf(1 - alpha / 2, df)

            # Calculate observed power (post-hoc)
            if se > 0:
                t_stat = abs(coef / se)
                # Power = P(|T| > t_crit | effect exists)
                observed_power = 1 - t.cdf(t_crit - t_stat, df) + t.cdf(-t_crit - t_stat, df)

                # Minimum detectable effect (MDE) for target power
                z_power = norm.ppf(target_power)
                z_alpha = norm.ppf(1 - alpha / 2)
                mde = (z_alpha + z_power) * se

                power_result = {
                    "observed_power": max(0, min(1, observed_power)),
                    "minimum_detectable_effect": mde,
                    "mde_as_percent_of_observed": abs(mde / coef) * 100 if coef != 0 else np.inf,
                    "target_power": target_power,
                    "adequately_powered": bool(observed_power >= target_power),  # Convert to Python bool
                    "effect_size_needed_for_target_power": mde,
                    "current_effect_size": abs(coef),
                    "degrees_of_freedom": df,
                }

                # Power interpretation
                if observed_power >= 0.8:
                    power_result["power_interpretation"] = "adequate"
                elif observed_power >= 0.6:
                    power_result["power_interpretation"] = "moderate"
                elif observed_power >= 0.4:
                    power_result["power_interpretation"] = "low"
                else:
                    power_result["power_interpretation"] = "very low"

        return power_result

    def _assess_overall_power(self, power_results: dict[str, Any]) -> dict[str, Any]:
        """
        Assess overall power characteristics across methods and outcomes.

        Args:
            power_results: Power analysis results

        Returns:
            Overall power assessment summary
        """
        all_powers = []
        adequately_powered_count = 0
        total_tests = 0

        for method_name, method_results in power_results.items():
            if method_name in ["sample_characteristics", "overall_assessment"]:
                continue

            if isinstance(method_results, dict):
                for _outcome, power_result in method_results.items():
                    if isinstance(power_result, dict) and "observed_power" in power_result:
                        all_powers.append(power_result["observed_power"])
                        total_tests += 1
                        if power_result["adequately_powered"]:
                            adequately_powered_count += 1

        if all_powers:
            overall = {
                "mean_power": np.mean(all_powers),
                "median_power": np.median(all_powers),
                "min_power": np.min(all_powers),
                "max_power": np.max(all_powers),
                "adequately_powered_proportion": adequately_powered_count / total_tests,
                "total_tests": total_tests,
                "adequately_powered_count": adequately_powered_count,
            }

            # Overall recommendation
            if overall["adequately_powered_proportion"] >= 0.8:
                overall["recommendation"] = "study is well-powered for detecting meaningful effects"
            elif overall["adequately_powered_proportion"] >= 0.5:
                overall["recommendation"] = (
                    "study has moderate power; consider larger sample or effect sizes"
                )
            else:
                overall["recommendation"] = (
                    "study is underpowered; results should be interpreted cautiously"
                )

            return overall

        return {"message": "No power calculations available"}

    def enhanced_confidence_intervals(
        self, results: dict[str, Any], confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """
        Calculate enhanced confidence intervals using advanced methods.

        Args:
            results: Dictionary of results from various robustness methods
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with enhanced confidence intervals
        """
        enhanced_cis = {}

        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return enhanced_cis

        # For each method that has bootstrap results, calculate BCa intervals
        for method_name, method_results in results.items():
            if isinstance(method_results, dict):
                enhanced_cis[method_name] = {}

                for outcome, outcome_results in method_results.items():
                    if isinstance(outcome_results, dict):
                        enhanced_ci = self._calculate_enhanced_ci_for_outcome(
                            outcome_results, confidence_level
                        )
                        if enhanced_ci:
                            enhanced_cis[method_name][outcome] = enhanced_ci

        # Calculate simultaneous confidence bands
        enhanced_cis["simultaneous_bands"] = self._calculate_simultaneous_bands(
            results, confidence_level
        )

        return enhanced_cis

    def _calculate_enhanced_ci_for_outcome(
        self, outcome_results: dict[str, Any], confidence_level: float
    ) -> dict[str, Any]:
        """
        Calculate enhanced confidence interval for a specific outcome.

        Args:
            outcome_results: Results for specific outcome
            confidence_level: Confidence level

        Returns:
            Enhanced confidence interval results
        """
        enhanced_ci = {}

        if "coefficient" in outcome_results:
            coef = outcome_results["coefficient"]

            # Standard confidence interval (if available)
            if "ci_lower" in outcome_results and "ci_upper" in outcome_results:
                enhanced_ci["standard_ci"] = {
                    "lower": outcome_results["ci_lower"],
                    "upper": outcome_results["ci_upper"],
                    "method": "asymptotic normal",
                }

            # If bootstrap samples are available, calculate BCa
            if "bootstrap_coefficients" in outcome_results:
                bootstrap_coefs = outcome_results["bootstrap_coefficients"]
                bca_ci = self._calculate_bca_ci(coef, bootstrap_coefs, confidence_level)
                enhanced_ci["bca_ci"] = bca_ci

            # Small-sample adjustment (if standard error available)
            if "std_error" in outcome_results:
                se = outcome_results["std_error"]
                # Use t-distribution for small samples
                from scipy.stats import t

                n_states = self.df["state"].nunique()
                df = max(n_states - 2, 1)
                t_crit = t.ppf((1 + confidence_level) / 2, df)

                enhanced_ci["small_sample_adjusted"] = {
                    "lower": coef - t_crit * se,
                    "upper": coef + t_crit * se,
                    "method": f"t-distribution (df={df})",
                    "degrees_of_freedom": df,
                }

        return enhanced_ci

    def _calculate_bca_ci(
        self, original_stat: float, bootstrap_stats: np.ndarray, confidence_level: float
    ) -> dict[str, Any]:
        """
        Calculate bias-corrected and accelerated (BCa) confidence interval.

        Args:
            original_stat: Original statistic
            bootstrap_stats: Bootstrap replications
            confidence_level: Confidence level

        Returns:
            BCa confidence interval
        """
        import numpy as np
        from scipy.stats import norm

        try:
            alpha = 1 - confidence_level
            n_boot = len(bootstrap_stats)

            # Bias correction
            n_below = np.sum(bootstrap_stats < original_stat)
            bias_correction = norm.ppf(n_below / n_boot) if n_boot > 0 else 0

            # Acceleration (simplified jackknife approach)
            # For more accurate acceleration, would need jackknife samples
            acceleration = 0  # Conservative approach

            # BCa percentiles
            z_alpha_2 = norm.ppf(alpha / 2)
            z_1_alpha_2 = norm.ppf(1 - alpha / 2)

            lower_percentile = norm.cdf(
                bias_correction
                + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2))
            )
            upper_percentile = norm.cdf(
                bias_correction
                + (bias_correction + z_1_alpha_2)
                / (1 - acceleration * (bias_correction + z_1_alpha_2))
            )

            # Ensure percentiles are within bounds
            lower_percentile = max(0.001, min(0.999, lower_percentile))
            upper_percentile = max(0.001, min(0.999, upper_percentile))

            # Calculate confidence interval
            ci_lower = np.percentile(bootstrap_stats, lower_percentile * 100)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile * 100)

            return {
                "lower": ci_lower,
                "upper": ci_upper,
                "method": "BCa",
                "bias_correction": bias_correction,
                "acceleration": acceleration,
                "n_bootstrap": n_boot,
                "coverage_level": confidence_level,
            }

        except Exception as e:
            print(f"BCa calculation failed: {e}")
            # Fall back to percentile method
            alpha = 1 - confidence_level
            return {
                "lower": np.percentile(bootstrap_stats, (alpha / 2) * 100),
                "upper": np.percentile(bootstrap_stats, (1 - alpha / 2) * 100),
                "method": "percentile (fallback)",
                "coverage_level": confidence_level,
            }

    def _calculate_simultaneous_bands(
        self, results: dict[str, Any], confidence_level: float
    ) -> dict[str, Any]:
        """
        Calculate simultaneous confidence bands for multiple outcomes.

        Args:
            results: Results from various methods
            confidence_level: Confidence level

        Returns:
            Simultaneous confidence bands
        """
        from scipy.stats import norm

        simultaneous_bands = {}
        outcome_vars = self._get_outcome_variables()

        # Bonferroni adjustment for simultaneous coverage
        n_outcomes = len(outcome_vars)
        bonferroni_level = confidence_level + (1 - confidence_level) / n_outcomes

        for method_name, method_results in results.items():
            if isinstance(method_results, dict):
                method_bands = {}

                for outcome in outcome_vars:
                    if outcome in method_results:
                        outcome_results = method_results[outcome]
                        if isinstance(outcome_results, dict) and "coefficient" in outcome_results:
                            coef = outcome_results["coefficient"]
                            se = outcome_results.get("std_error", 0)

                            if se > 0:
                                # Bonferroni-adjusted interval
                                z_crit = norm.ppf((1 + bonferroni_level) / 2)

                                method_bands[outcome] = {
                                    "lower": coef - z_crit * se,
                                    "upper": coef + z_crit * se,
                                    "simultaneous_coverage": confidence_level,
                                    "individual_coverage": bonferroni_level,
                                    "adjustment_method": "Bonferroni",
                                }

                if method_bands:
                    simultaneous_bands[method_name] = method_bands

        return simultaneous_bands

    def create_robustness_plots(self) -> dict[str, plt.Figure]:
        """Create visualization plots for robustness results including Phase 3 enhanced inference."""
        figures = {}

        # 1. Leave-one-state-out plot
        if "leave_one_state_out" in self.robustness_results:
            fig_loso = self._plot_leave_one_state_out()
            if fig_loso:
                figures["leave_one_state_out"] = fig_loso

        # 2. Specification curve plot
        if "specification_curve" in self.robustness_results:
            fig_spec = self._plot_specification_curve()
            if fig_spec:
                figures["specification_curve"] = fig_spec

        # 3. Permutation test plot
        if "permutation_test" in self.robustness_results:
            fig_perm = self._plot_permutation_test()
            if fig_perm:
                figures["permutation_test"] = fig_perm

        # Phase 3: Enhanced inference visualizations
        # 4. Effect sizes forest plot
        if "effect_sizes" in self.robustness_results:
            fig_forest = self._plot_effect_sizes_forest()
            if fig_forest:
                figures["effect_sizes_forest"] = fig_forest

        # 5. Multiple testing corrections comparison
        if "multiple_testing" in self.robustness_results:
            fig_mt = self._plot_multiple_testing_corrections()
            if fig_mt:
                figures["multiple_testing_corrections"] = fig_mt

        # 6. Power analysis dashboard
        if "power_analysis" in self.robustness_results:
            fig_power = self._plot_power_analysis_dashboard()
            if fig_power:
                figures["power_analysis_dashboard"] = fig_power

        # 7. Enhanced confidence intervals comparison
        if "enhanced_confidence_intervals" in self.robustness_results:
            fig_ci = self._plot_enhanced_confidence_intervals()
            if fig_ci:
                figures["enhanced_confidence_intervals"] = fig_ci

        # 8. Comprehensive Phase 3 dashboard
        fig_dashboard = self._plot_phase3_dashboard()
        if fig_dashboard:
            figures["phase3_comprehensive_dashboard"] = fig_dashboard

        return figures

    def _plot_leave_one_state_out(self) -> plt.Figure | None:
        """Plot leave-one-state-out results."""
        loso_results = self.robustness_results.get("leave_one_state_out", {})
        if not loso_results:
            return None

        n_outcomes = len(loso_results)
        fig, axes = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 6))
        if n_outcomes == 1:
            axes = [axes]

        for i, (outcome, results) in enumerate(loso_results.items()):
            ax = axes[i]

            # Extract coefficients and states
            state_results = results["state_results"]
            states = list(state_results.keys())
            coeffs = [state_results[state]["coefficient"] for state in states]

            # Create plot
            ax.scatter(range(len(coeffs)), coeffs, alpha=0.7)
            ax.axhline(
                y=results["mean_coeff"],
                color="red",
                linestyle="--",
                label=f"Mean: {results['mean_coeff']:.3f}",
            )
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_xlabel("Excluded State (ordered)")
            ax.set_ylabel("Treatment Coefficient")
            ax.set_title(f"LOSO: {outcome.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.figures_dir / "robustness_loso.png", dpi=300, bbox_inches="tight")
        return fig

    def _plot_specification_curve(self) -> plt.Figure | None:
        """Plot specification curve results."""
        spec_results = self.robustness_results.get("specification_curve", {})
        if not spec_results:
            return None

        # Create subplots for each outcome
        n_outcomes = len(spec_results)
        fig, axes = plt.subplots(
            2, n_outcomes, figsize=(5 * n_outcomes, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        if n_outcomes == 1:
            axes = axes.reshape(-1, 1)

        for i, (outcome, results) in enumerate(spec_results.items()):
            spec_df = results["specifications"]

            # Sort by coefficient
            spec_df_sorted = spec_df.sort_values("coefficient").reset_index(drop=True)
            spec_df_sorted["spec_rank"] = range(len(spec_df_sorted))

            # Top panel: Coefficients
            ax_coeff = axes[0, i]
            ax_coeff.scatter(
                spec_df_sorted["spec_rank"],
                spec_df_sorted["coefficient"],
                c=["red" if p < 0.05 else "blue" for p in spec_df_sorted["p_value"]],
            )

            # Add confidence intervals
            ci_lower = spec_df_sorted["coefficient"] - 1.96 * spec_df_sorted["se"]
            ci_upper = spec_df_sorted["coefficient"] + 1.96 * spec_df_sorted["se"]
            ax_coeff.fill_between(spec_df_sorted["spec_rank"], ci_lower, ci_upper, alpha=0.2)

            ax_coeff.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax_coeff.set_ylabel("Treatment Effect")
            ax_coeff.set_title(f"{outcome.replace('_', ' ').title()}")
            ax_coeff.grid(True, alpha=0.3)

            # Bottom panel: Specification details
            ax_spec = axes[1, i]

            # Show which controls are included (simplified)
            for j, row in spec_df_sorted.iterrows():
                controls_text = (
                    row["controls_str"][:10] + "..."
                    if len(row["controls_str"]) > 10
                    else row["controls_str"]
                )
                ax_spec.text(j, 0, controls_text, rotation=45, ha="right", va="bottom", fontsize=8)

            ax_spec.set_xlabel("Specification (sorted by coefficient)")
            ax_spec.set_ylabel("Controls")
            ax_spec.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        fig.savefig(
            self.figures_dir / "robustness_specification_curve.png", dpi=300, bbox_inches="tight"
        )
        return fig

    def _plot_permutation_test(self) -> plt.Figure | None:
        """Plot permutation test results."""
        perm_results = self.robustness_results.get("permutation_test", {})
        if not perm_results:
            return None

        n_outcomes = len(perm_results)
        fig, axes = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 6))
        if n_outcomes == 1:
            axes = [axes]

        for i, (outcome, results) in enumerate(perm_results.items()):
            ax = axes[i]

            # Plot histogram of placebo coefficients
            placebo_coeffs = results["placebo_coefficients"]
            actual_coeff = results["actual_coefficient"]

            ax.hist(
                placebo_coeffs,
                bins=50,
                alpha=0.7,
                density=True,
                label=f"Placebo (n={len(placebo_coeffs)})",
            )
            ax.axvline(
                x=actual_coeff,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Actual: {actual_coeff:.3f}",
            )
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

            ax.set_xlabel("Treatment Coefficient")
            ax.set_ylabel("Density")
            ax.set_title(
                f"Permutation Test: {outcome.replace('_', ' ').title()}\\n"
                f"p-value = {results['permutation_pvalue']:.3f}"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.figures_dir / "robustness_permutation.png", dpi=300, bbox_inches="tight")
        return fig

    def _plot_effect_sizes_forest(self) -> plt.Figure:
        """Create forest plot of effect sizes across methods and outcomes."""
        try:
            if "effect_sizes" not in self.robustness_results:
                return None

            effect_sizes = self.robustness_results["effect_sizes"]
            outcome_vars = self._get_outcome_variables()

            fig, axes = plt.subplots(len(outcome_vars), 1, figsize=(12, 4 * len(outcome_vars)))
            if len(outcome_vars) == 1:
                axes = [axes]

            colors = plt.cm.Set3(
                np.linspace(0, 1, len(effect_sizes) - 1)
            )  # -1 for cross_method_summary

            for i, outcome in enumerate(outcome_vars):
                ax = axes[i]

                # Collect effect sizes for this outcome
                methods = []
                cohens_ds = []
                ci_lowers = []
                ci_uppers = []

                for method_name, method_effects in effect_sizes.items():
                    if method_name == "cross_method_summary":
                        continue

                    if isinstance(method_effects, dict) and outcome in method_effects:
                        outcome_effect = method_effects[outcome]
                        if not np.isnan(outcome_effect.get("cohens_d", np.nan)):
                            methods.append(method_name.replace("_", " ").title())
                            cohens_ds.append(outcome_effect["cohens_d"])
                            ci_lowers.append(outcome_effect.get("ci_lower_d", np.nan))
                            ci_uppers.append(outcome_effect.get("ci_upper_d", np.nan))

                if not methods:
                    ax.text(
                        0.5,
                        0.5,
                        f"No effect sizes available for {outcome}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Create forest plot
                y_positions = range(len(methods))

                # Plot confidence intervals
                for j, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers, strict=False)):
                    if not (np.isnan(lower) or np.isnan(upper)):
                        ax.plot([lower, upper], [j, j], "k-", alpha=0.6, linewidth=2)
                        ax.plot([lower, lower], [j - 0.1, j + 0.1], "k-", alpha=0.6, linewidth=2)
                        ax.plot([upper, upper], [j - 0.1, j + 0.1], "k-", alpha=0.6, linewidth=2)

                # Plot point estimates
                ax.scatter(
                    cohens_ds, y_positions, c=colors[: len(methods)], s=100, alpha=0.8, zorder=5
                )

                # Reference lines
                ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
                ax.axvline(x=0.2, color="gray", linestyle=":", alpha=0.5, label="Small effect")
                ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Medium effect")
                ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5, label="Large effect")
                ax.axvline(x=-0.2, color="gray", linestyle=":", alpha=0.5)
                ax.axvline(x=-0.5, color="gray", linestyle=":", alpha=0.5)
                ax.axvline(x=-0.8, color="gray", linestyle=":", alpha=0.5)

                # Formatting
                ax.set_yticks(y_positions)
                ax.set_yticklabels(methods)
                ax.set_xlabel("Cohen's d (Effect Size)")
                ax.set_title(f"Effect Sizes: {outcome.replace('_', ' ').title()}")
                ax.grid(True, alpha=0.3)

                # Add interpretation legend
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / "effect_sizes_forest_plot.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            return fig

        except Exception as e:
            print(f"Error creating effect sizes forest plot: {e}")
            return None

    def _plot_multiple_testing_corrections(self) -> plt.Figure:
        """Create visualization of multiple testing corrections impact."""
        try:
            if "multiple_testing" not in self.robustness_results:
                return None

            mt_results = self.robustness_results["multiple_testing"]
            if "summary" not in mt_results:
                return None

            summary = mt_results["summary"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Left panel: Summary bar chart
            correction_methods = ["Original", "Bonferroni", "FDR (B-H)", "Romano-Wolf"]
            significant_counts = [
                summary.get("original_significant", 0),
                summary.get("bonferroni_significant", 0),
                summary.get("fdr_significant", 0),
                summary.get("romano_wolf_significant", 0),
            ]
            total_tests = summary.get("total_tests", 1)

            bars = ax1.bar(
                correction_methods,
                significant_counts,
                color=["skyblue", "lightcoral", "lightgreen", "gold"],
            )

            # Add value labels on bars
            for bar, count in zip(bars, significant_counts, strict=False):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{count}/{total_tests}",
                    ha="center",
                    va="bottom",
                )

            ax1.set_ylabel("Number of Significant Results")
            ax1.set_title("Impact of Multiple Testing Corrections")
            ax1.set_ylim(0, total_tests + 1)
            ax1.grid(True, alpha=0.3)

            # Right panel: P-value comparison heatmap
            corrections = mt_results.get("corrections", {})
            if corrections:
                # Collect p-values for heatmap
                methods = list(corrections.keys())
                outcomes = []
                for method_data in corrections.values():
                    outcomes.extend(method_data.keys())
                outcomes = list(set(outcomes))

                p_value_types = ["original_p", "bonferroni_p", "fdr_p", "romano_wolf_p"]
                p_value_matrix = np.full((len(methods) * len(outcomes), len(p_value_types)), np.nan)
                labels = []

                idx = 0
                for method in methods:
                    for outcome in outcomes:
                        if outcome in corrections[method]:
                            outcome_data = corrections[method][outcome]
                            for j, p_type in enumerate(p_value_types):
                                if p_type in outcome_data:
                                    p_value_matrix[idx, j] = outcome_data[p_type]
                            labels.append(f"{method}\n{outcome}")
                        idx += 1

                # Remove empty rows
                non_empty_rows = ~np.all(np.isnan(p_value_matrix), axis=1)
                p_value_matrix = p_value_matrix[non_empty_rows]
                labels = [label for i, label in enumerate(labels) if non_empty_rows[i]]

                if p_value_matrix.shape[0] > 0:
                    im = ax2.imshow(p_value_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax2)
                    cbar.set_label("P-value", rotation=270, labelpad=15)

                    # Set labels
                    ax2.set_xticks(range(len(p_value_types)))
                    ax2.set_xticklabels(
                        [p.replace("_", " ").title() for p in p_value_types],
                        rotation=45,
                        ha="right",
                    )
                    ax2.set_yticks(range(len(labels)))
                    ax2.set_yticklabels(labels, fontsize=8)
                    ax2.set_title("P-value Corrections Heatmap")

                    # Add significance threshold line
                    alpha = summary.get("alpha_level", 0.05)
                    for i in range(p_value_matrix.shape[0]):
                        for j in range(p_value_matrix.shape[1]):
                            if not np.isnan(p_value_matrix[i, j]):
                                if p_value_matrix[i, j] < alpha:
                                    ax2.text(
                                        j,
                                        i,
                                        "*",
                                        ha="center",
                                        va="center",
                                        color="white",
                                        fontweight="bold",
                                        fontsize=12,
                                    )

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / "multiple_testing_corrections.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            return fig

        except Exception as e:
            print(f"Error creating multiple testing corrections plot: {e}")
            return None

    def _plot_power_analysis_dashboard(self) -> plt.Figure:
        """Create comprehensive power analysis dashboard."""
        try:
            if "power_analysis" not in self.robustness_results:
                return None

            power_results = self.robustness_results["power_analysis"]

            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # Sample characteristics (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            if "sample_characteristics" in power_results:
                sample_chars = power_results["sample_characteristics"]
                chars_text = f"""Sample Characteristics:
• Total observations: {sample_chars.get("total_observations", 0):,}
• Number of states: {sample_chars.get("n_states", 0)}
• Treated observations: {sample_chars.get("n_treated", 0):,}
• Control observations: {sample_chars.get("n_control", 0):,}
• Treatment proportion: {sample_chars.get("treatment_proportion", 0):.1%}"""

                ax1.text(
                    0.05,
                    0.95,
                    chars_text,
                    transform=ax1.transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    fontfamily="monospace",
                )
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.axis("off")
                ax1.set_title("Sample Characteristics")

            # Power distribution (top middle and right)
            ax2 = fig.add_subplot(gs[0, 1:])
            all_powers = []
            method_labels = []
            outcome_labels = []

            for method_name, method_results in power_results.items():
                if method_name in ["sample_characteristics", "overall_assessment"]:
                    continue

                if isinstance(method_results, dict):
                    for outcome, power_result in method_results.items():
                        if isinstance(power_result, dict) and "observed_power" in power_result:
                            all_powers.append(power_result["observed_power"])
                            method_labels.append(method_name.replace("_", " ").title())
                            outcome_labels.append(outcome.replace("_", " ").title())

            if all_powers:
                # Power histogram
                ax2.hist(all_powers, bins=10, alpha=0.7, color="skyblue", edgecolor="black")
                ax2.axvline(x=0.8, color="red", linestyle="--", label="Adequate Power (0.8)")
                ax2.set_xlabel("Observed Power")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Statistical Power")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Method-outcome power heatmap (middle row)
            ax3 = fig.add_subplot(gs[1, :])
            if all_powers:
                unique_methods = list(set(method_labels))
                unique_outcomes = list(set(outcome_labels))

                power_matrix = np.full((len(unique_methods), len(unique_outcomes)), np.nan)

                for i, power in enumerate(all_powers):
                    method_idx = unique_methods.index(method_labels[i])
                    outcome_idx = unique_outcomes.index(outcome_labels[i])
                    power_matrix[method_idx, outcome_idx] = power

                im = ax3.imshow(power_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label("Statistical Power", rotation=270, labelpad=15)

                # Labels and annotations
                ax3.set_xticks(range(len(unique_outcomes)))
                ax3.set_xticklabels(unique_outcomes)
                ax3.set_yticks(range(len(unique_methods)))
                ax3.set_yticklabels(unique_methods)
                ax3.set_title("Statistical Power by Method and Outcome")

                # Add power values as text
                for i in range(len(unique_methods)):
                    for j in range(len(unique_outcomes)):
                        if not np.isnan(power_matrix[i, j]):
                            text_color = "white" if power_matrix[i, j] < 0.5 else "black"
                            ax3.text(
                                j,
                                i,
                                f"{power_matrix[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontweight="bold",
                            )

            # Overall assessment (bottom row)
            ax4 = fig.add_subplot(gs[2, :])
            if "overall_assessment" in power_results:
                overall = power_results["overall_assessment"]
                if "mean_power" in overall:
                    assessment_text = f"""Overall Power Assessment:
• Mean power across all tests: {overall.get("mean_power", 0):.3f}
• Proportion adequately powered (≥0.8): {overall.get("adequately_powered_proportion", 0):.1%}
• Tests adequately powered: {overall.get("adequately_powered_count", 0)}/{overall.get("total_tests", 0)}
• Recommendation: {overall.get("recommendation", "No recommendation available")}"""

                    ax4.text(
                        0.05,
                        0.95,
                        assessment_text,
                        transform=ax4.transAxes,
                        verticalalignment="top",
                        fontsize=11,
                        fontfamily="monospace",
                    )
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis("off")
            ax4.set_title("Overall Assessment")

            plt.suptitle("Statistical Power Analysis Dashboard", fontsize=16, y=0.98)
            plt.savefig(
                self.figures_dir / "power_analysis_dashboard.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            return fig

        except Exception as e:
            print(f"Error creating power analysis dashboard: {e}")
            return None

    def _plot_enhanced_confidence_intervals(self) -> plt.Figure:
        """Create comparison plot of different confidence interval methods."""
        try:
            if "enhanced_confidence_intervals" not in self.robustness_results:
                return None

            enhanced_cis = self.robustness_results["enhanced_confidence_intervals"]
            outcome_vars = self._get_outcome_variables()

            fig, axes = plt.subplots(len(outcome_vars), 1, figsize=(14, 5 * len(outcome_vars)))
            if len(outcome_vars) == 1:
                axes = [axes]

            for i, outcome in enumerate(outcome_vars):
                ax = axes[i]

                # Collect CI data for this outcome
                ci_data = []

                for method_name, method_cis in enhanced_cis.items():
                    if method_name == "simultaneous_bands":
                        continue

                    if isinstance(method_cis, dict) and outcome in method_cis:
                        outcome_cis = method_cis[outcome]

                        # Standard CI
                        if "standard_ci" in outcome_cis:
                            std_ci = outcome_cis["standard_ci"]
                            ci_data.append(
                                {
                                    "method": f"{method_name} (Standard)",
                                    "lower": std_ci["lower"],
                                    "upper": std_ci["upper"],
                                    "point": (std_ci["lower"] + std_ci["upper"]) / 2,
                                    "ci_method": std_ci.get("method", "standard"),
                                }
                            )

                        # BCa CI
                        if "bca_ci" in outcome_cis:
                            bca_ci = outcome_cis["bca_ci"]
                            ci_data.append(
                                {
                                    "method": f"{method_name} (BCa)",
                                    "lower": bca_ci["lower"],
                                    "upper": bca_ci["upper"],
                                    "point": (bca_ci["lower"] + bca_ci["upper"]) / 2,
                                    "ci_method": bca_ci.get("method", "BCa"),
                                }
                            )

                        # Small sample adjusted
                        if "small_sample_adjusted" in outcome_cis:
                            ss_ci = outcome_cis["small_sample_adjusted"]
                            ci_data.append(
                                {
                                    "method": f"{method_name} (t-adjusted)",
                                    "lower": ss_ci["lower"],
                                    "upper": ss_ci["upper"],
                                    "point": (ss_ci["lower"] + ss_ci["upper"]) / 2,
                                    "ci_method": ss_ci.get("method", "t-adjusted"),
                                }
                            )

                # Add simultaneous bands
                if "simultaneous_bands" in enhanced_cis:
                    sim_bands = enhanced_cis["simultaneous_bands"]
                    for method_name, method_bands in sim_bands.items():
                        if outcome in method_bands:
                            band_data = method_bands[outcome]
                            ci_data.append(
                                {
                                    "method": f"{method_name} (Simultaneous)",
                                    "lower": band_data["lower"],
                                    "upper": band_data["upper"],
                                    "point": (band_data["lower"] + band_data["upper"]) / 2,
                                    "ci_method": "Simultaneous",
                                }
                            )

                if not ci_data:
                    ax.text(
                        0.5,
                        0.5,
                        f"No CI data available for {outcome}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Sort by method name for consistent ordering
                ci_data.sort(key=lambda x: x["method"])

                # Plot confidence intervals
                y_positions = range(len(ci_data))
                colors = plt.cm.Set2(np.linspace(0, 1, len(ci_data)))

                for j, ci_info in enumerate(ci_data):
                    # CI line
                    ax.plot(
                        [ci_info["lower"], ci_info["upper"]],
                        [j, j],
                        color=colors[j],
                        linewidth=3,
                        alpha=0.7,
                    )

                    # CI bounds
                    ax.plot(
                        [ci_info["lower"], ci_info["lower"]],
                        [j - 0.1, j + 0.1],
                        color=colors[j],
                        linewidth=2,
                    )
                    ax.plot(
                        [ci_info["upper"], ci_info["upper"]],
                        [j - 0.1, j + 0.1],
                        color=colors[j],
                        linewidth=2,
                    )

                    # Point estimate
                    ax.scatter(
                        [ci_info["point"]],
                        [j],
                        color=colors[j],
                        s=60,
                        zorder=5,
                        edgecolor="black",
                        linewidth=1,
                    )

                # Reference line at zero
                ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

                # Formatting
                ax.set_yticks(y_positions)
                ax.set_yticklabels([ci["method"] for ci in ci_data], fontsize=9)
                ax.set_xlabel("Effect Size")
                ax.set_title(
                    f"Confidence Intervals Comparison: {outcome.replace('_', ' ').title()}"
                )
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / "enhanced_confidence_intervals.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            return fig

        except Exception as e:
            print(f"Error creating enhanced confidence intervals plot: {e}")
            return None

    def _plot_phase3_dashboard(self) -> plt.Figure:
        """Create comprehensive Phase 3 enhanced inference dashboard."""
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

            # Multiple testing summary (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            if "multiple_testing" in self.robustness_results:
                mt_summary = self.robustness_results["multiple_testing"].get("summary", {})
                if mt_summary:
                    methods = ["Original", "Bonferroni", "FDR", "Romano-Wolf"]
                    counts = [
                        mt_summary.get("original_significant", 0),
                        mt_summary.get("bonferroni_significant", 0),
                        mt_summary.get("fdr_significant", 0),
                        mt_summary.get("romano_wolf_significant", 0),
                    ]
                    bars = ax1.bar(
                        methods, counts, color=["skyblue", "lightcoral", "lightgreen", "gold"]
                    )
                    ax1.set_ylabel("Significant Results")
                    ax1.set_title("Multiple Testing Impact")
                    ax1.tick_params(axis="x", rotation=45)

                    # Add value labels
                    for bar, count in zip(bars, counts, strict=False):
                        height = bar.get_height()
                        ax1.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02,
                            f"{count}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            # Effect size distribution (top middle-right)
            ax2 = fig.add_subplot(gs[0, 1:3])
            if "effect_sizes" in self.robustness_results:
                effect_sizes = self.robustness_results["effect_sizes"]
                all_cohens_d = []

                for method_name, method_effects in effect_sizes.items():
                    if method_name == "cross_method_summary":
                        continue
                    if isinstance(method_effects, dict):
                        for _outcome, outcome_data in method_effects.items():
                            if isinstance(outcome_data, dict):
                                cohens_d = outcome_data.get("cohens_d", np.nan)
                                if not np.isnan(cohens_d):
                                    all_cohens_d.append(cohens_d)

                if all_cohens_d:
                    ax2.hist(all_cohens_d, bins=15, alpha=0.7, color="lightblue", edgecolor="black")
                    ax2.axvline(x=0, color="black", linestyle="-", alpha=0.5)
                    ax2.axvline(x=0.2, color="orange", linestyle="--", alpha=0.7, label="Small")
                    ax2.axvline(x=0.5, color="orange", linestyle="--", alpha=0.7, label="Medium")
                    ax2.axvline(x=0.8, color="red", linestyle="--", alpha=0.7, label="Large")
                    ax2.axvline(x=-0.2, color="orange", linestyle="--", alpha=0.7)
                    ax2.axvline(x=-0.5, color="orange", linestyle="--", alpha=0.7)
                    ax2.axvline(x=-0.8, color="red", linestyle="--", alpha=0.7)
                    ax2.set_xlabel("Cohen's d")
                    ax2.set_ylabel("Frequency")
                    ax2.set_title("Effect Size Distribution")
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.3)

            # Power analysis summary (top right)
            ax3 = fig.add_subplot(gs[0, 3])
            if "power_analysis" in self.robustness_results:
                power_results = self.robustness_results["power_analysis"]
                if "overall_assessment" in power_results:
                    overall = power_results["overall_assessment"]
                    mean_power = overall.get("mean_power", 0)
                    adequately_powered_prop = overall.get("adequately_powered_proportion", 0)

                    # Power gauge-style visualization
                    theta = np.linspace(0, np.pi, 100)
                    r = np.ones_like(theta)

                    ax3_polar = plt.subplot(gs[0, 3], projection="polar")
                    ax3_polar.plot(theta, r, "k-", linewidth=2)
                    ax3_polar.fill_between(theta, 0, r, alpha=0.2, color="lightgray")

                    # Power indicator
                    power_angle = mean_power * np.pi
                    ax3_polar.plot([power_angle, power_angle], [0, 1], "r-", linewidth=4)
                    ax3_polar.plot(
                        [0.8 * np.pi, 0.8 * np.pi], [0, 1], "g--", linewidth=2, alpha=0.7
                    )

                    ax3_polar.set_ylim(0, 1.1)
                    ax3_polar.set_theta_zero_location("W")
                    ax3_polar.set_theta_direction(1)
                    ax3_polar.set_thetagrids(
                        [0, 45, 90, 135, 180], ["1.0", "0.75", "0.5", "0.25", "0"]
                    )
                    ax3_polar.set_title(
                        f"Mean Power: {mean_power:.2f}\n{adequately_powered_prop:.0%} Adequate",
                        pad=20,
                    )
                else:
                    ax3.text(
                        0.5,
                        0.5,
                        "Power analysis\nnot available",
                        ha="center",
                        va="center",
                        transform=ax3.transAxes,
                    )
                    ax3.axis("off")

            # Method comparison heatmap (second row)
            ax4 = fig.add_subplot(gs[1, :])
            self._add_method_comparison_heatmap(ax4)

            # Confidence intervals comparison (third row)
            ax5 = fig.add_subplot(gs[2, :])
            self._add_confidence_intervals_comparison(ax5)

            # Summary statistics table (bottom row)
            ax6 = fig.add_subplot(gs[3, :])
            self._add_summary_statistics_table(ax6)

            plt.suptitle("Phase 3: Enhanced Statistical Inference Dashboard", fontsize=20, y=0.98)
            plt.savefig(
                self.figures_dir / "phase3_comprehensive_dashboard.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return fig

        except Exception as e:
            print(f"Error creating Phase 3 dashboard: {e}")
            return None

    def _add_method_comparison_heatmap(self, ax):
        """Add method comparison heatmap to existing axis."""
        try:
            # Collect all coefficients and p-values
            methods = []
            outcomes = []
            coefficients = []
            p_values = []

            for method_name, method_results in self.robustness_results.items():
                if method_name in [
                    "multiple_testing",
                    "effect_sizes",
                    "power_analysis",
                    "enhanced_confidence_intervals",
                ]:
                    continue

                if isinstance(method_results, dict):
                    for outcome, outcome_results in method_results.items():
                        if isinstance(outcome_results, dict):
                            if "coefficient" in outcome_results and "p_value" in outcome_results:
                                methods.append(method_name.replace("_", " ").title())
                                outcomes.append(outcome.replace("_", " ").title())
                                coefficients.append(outcome_results["coefficient"])
                                p_values.append(outcome_results["p_value"])

            if not methods:
                ax.text(
                    0.5,
                    0.5,
                    "No method comparison data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

            # Create matrices for heatmap
            unique_methods = list(set(methods))
            unique_outcomes = list(set(outcomes))

            coef_matrix = np.full((len(unique_methods), len(unique_outcomes)), np.nan)
            p_matrix = np.full((len(unique_methods), len(unique_outcomes)), np.nan)

            for i, (method, outcome, coef, p_val) in enumerate(
                zip(methods, outcomes, coefficients, p_values, strict=False)
            ):
                method_idx = unique_methods.index(method)
                outcome_idx = unique_outcomes.index(outcome)
                coef_matrix[method_idx, outcome_idx] = coef
                p_matrix[method_idx, outcome_idx] = p_val

            # Create heatmap
            im = ax.imshow(coef_matrix, cmap="RdBu_r", aspect="auto")

            # Add significance stars
            for i in range(len(unique_methods)):
                for j in range(len(unique_outcomes)):
                    if not np.isnan(p_matrix[i, j]):
                        if p_matrix[i, j] < 0.001:
                            stars = "***"
                        elif p_matrix[i, j] < 0.01:
                            stars = "**"
                        elif p_matrix[i, j] < 0.05:
                            stars = "*"
                        else:
                            stars = ""

                        if stars:
                            ax.text(
                                j,
                                i,
                                stars,
                                ha="center",
                                va="center",
                                color="white",
                                fontweight="bold",
                                fontsize=12,
                            )

            # Labels
            ax.set_xticks(range(len(unique_outcomes)))
            ax.set_xticklabels(unique_outcomes, rotation=45, ha="right")
            ax.set_yticks(range(len(unique_methods)))
            ax.set_yticklabels(unique_methods)
            ax.set_title(
                "Method Comparison: Coefficients with Significance (* p<0.05, ** p<0.01, *** p<0.001)"
            )

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Coefficient Value", rotation=270, labelpad=15)

        except Exception as e:
            print(f"Error adding method comparison heatmap: {e}")
            ax.text(
                0.5,
                0.5,
                "Error creating method comparison",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    def _add_confidence_intervals_comparison(self, ax):
        """Add confidence intervals comparison to existing axis."""
        try:
            if "enhanced_confidence_intervals" not in self.robustness_results:
                ax.text(
                    0.5,
                    0.5,
                    "Enhanced confidence intervals not available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

            # Just show a summary comparison for the first available outcome
            enhanced_cis = self.robustness_results["enhanced_confidence_intervals"]

            # Find first available method and outcome
            sample_data = None
            for method_name, method_cis in enhanced_cis.items():
                if method_name == "simultaneous_bands":
                    continue
                if isinstance(method_cis, dict):
                    for outcome, outcome_cis in method_cis.items():
                        if isinstance(outcome_cis, dict):
                            sample_data = (method_name, outcome, outcome_cis)
                            break
                    if sample_data:
                        break

            if not sample_data:
                ax.text(
                    0.5,
                    0.5,
                    "No CI data available for visualization",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

            method_name, outcome, outcome_cis = sample_data

            # Extract different CI types
            ci_types = []
            ci_data = []

            if "standard_ci" in outcome_cis:
                ci_types.append("Standard")
                std_ci = outcome_cis["standard_ci"]
                ci_data.append((std_ci["lower"], std_ci["upper"]))

            if "bca_ci" in outcome_cis:
                ci_types.append("BCa Bootstrap")
                bca_ci = outcome_cis["bca_ci"]
                ci_data.append((bca_ci["lower"], bca_ci["upper"]))

            if "small_sample_adjusted" in outcome_cis:
                ci_types.append("t-adjusted")
                ss_ci = outcome_cis["small_sample_adjusted"]
                ci_data.append((ss_ci["lower"], ss_ci["upper"]))

            if not ci_types:
                ax.text(
                    0.5,
                    0.5,
                    "No CI types available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

            # Plot CIs
            colors = ["blue", "red", "green"]
            for i, (ci_type, (lower, upper)) in enumerate(zip(ci_types, ci_data, strict=False)):
                y_pos = i
                ax.plot(
                    [lower, upper],
                    [y_pos, y_pos],
                    color=colors[i % len(colors)],
                    linewidth=4,
                    alpha=0.7,
                    label=ci_type,
                )
                ax.plot(
                    [lower, lower],
                    [y_pos - 0.1, y_pos + 0.1],
                    color=colors[i % len(colors)],
                    linewidth=2,
                )
                ax.plot(
                    [upper, upper],
                    [y_pos - 0.1, y_pos + 0.1],
                    color=colors[i % len(colors)],
                    linewidth=2,
                )

                # Point estimate (midpoint)
                midpoint = (lower + upper) / 2
                ax.scatter(
                    [midpoint],
                    [y_pos],
                    color=colors[i % len(colors)],
                    s=60,
                    zorder=5,
                    edgecolor="black",
                )

            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            ax.set_yticks(range(len(ci_types)))
            ax.set_yticklabels(ci_types)
            ax.set_xlabel("Effect Size")
            ax.set_title(f"Confidence Intervals Comparison: {outcome.replace('_', ' ').title()}")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        except Exception as e:
            print(f"Error adding confidence intervals comparison: {e}")
            ax.text(
                0.5,
                0.5,
                "Error creating CI comparison",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    def _add_summary_statistics_table(self, ax):
        """Add summary statistics table to existing axis."""
        try:
            # Create summary data
            summary_data = []

            # Multiple testing summary
            if "multiple_testing" in self.robustness_results:
                mt_summary = self.robustness_results["multiple_testing"].get("summary", {})
                summary_data.append(
                    [
                        "Multiple Testing",
                        f"{mt_summary.get('total_tests', 0)} total tests",
                        f"{mt_summary.get('original_significant', 0)} originally significant",
                        f"{mt_summary.get('bonferroni_significant', 0)} Bonferroni significant",
                    ]
                )

            # Effect sizes summary
            if "effect_sizes" in self.robustness_results:
                effect_sizes = self.robustness_results["effect_sizes"]
                if "cross_method_summary" in effect_sizes:
                    cross_summary = effect_sizes["cross_method_summary"]
                    n_outcomes = len(cross_summary)
                    mean_consistency = (
                        np.mean(
                            [
                                data.get("consistency_score", 0)
                                for data in cross_summary.values()
                                if isinstance(data, dict)
                            ]
                        )
                        if cross_summary
                        else 0
                    )

                    summary_data.append(
                        [
                            "Effect Sizes",
                            f"{n_outcomes} outcomes analyzed",
                            f"Mean consistency: {mean_consistency:.2f}",
                            "Cross-method comparison completed",
                        ]
                    )

            # Power analysis summary
            if "power_analysis" in self.robustness_results:
                power_results = self.robustness_results["power_analysis"]
                if "overall_assessment" in power_results:
                    overall = power_results["overall_assessment"]
                    summary_data.append(
                        [
                            "Power Analysis",
                            f"Mean power: {overall.get('mean_power', 0):.3f}",
                            f"{overall.get('adequately_powered_proportion', 0):.0%} adequately powered",
                            overall.get("recommendation", "No recommendation")[:30] + "...",
                        ]
                    )

            # Enhanced CIs summary
            if "enhanced_confidence_intervals" in self.robustness_results:
                enhanced_cis = self.robustness_results["enhanced_confidence_intervals"]
                n_methods = len([k for k in enhanced_cis if k != "simultaneous_bands"])
                has_simultaneous = "simultaneous_bands" in enhanced_cis

                summary_data.append(
                    [
                        "Enhanced CIs",
                        f"{n_methods} methods analyzed",
                        f"Simultaneous bands: {'Yes' if has_simultaneous else 'No'}",
                        "BCa and t-adjusted CIs computed",
                    ]
                )

            if not summary_data:
                ax.text(
                    0.5,
                    0.5,
                    "No summary data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

            # Create table
            headers = ["Analysis Type", "Statistic 1", "Statistic 2", "Notes"]

            # Create table plot
            table = ax.table(
                cellText=summary_data,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                colWidths=[0.2, 0.25, 0.25, 0.3],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#4CAF50")
                table[(0, i)].set_text_props(weight="bold", color="white")

            for i in range(1, len(summary_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor("#f2f2f2")
                    else:
                        table[(i, j)].set_facecolor("white")

            ax.axis("off")
            ax.set_title("Phase 3 Enhanced Inference Summary", pad=20, fontweight="bold")

        except Exception as e:
            print(f"Error adding summary statistics table: {e}")
            ax.text(
                0.5,
                0.5,
                "Error creating summary table",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    def create_robustness_table(self) -> pd.DataFrame:
        """Create comprehensive robustness results table."""
        print("Creating robustness results table...")

        table_data = []

        for outcome in self.outcome_vars:
            row = {"outcome": outcome}

            # LOSO results
            if (
                "leave_one_state_out" in self.robustness_results
                and outcome in self.robustness_results["leave_one_state_out"]
            ):
                loso = self.robustness_results["leave_one_state_out"][outcome]
                row["loso_mean"] = loso["mean_coeff"]
                row["loso_std"] = loso["std_coeff"]
                row["loso_range"] = f"[{loso['min_coeff']:.3f}, {loso['max_coeff']:.3f}]"

            # Clustering results
            if (
                "alternative_clustering" in self.robustness_results
                and outcome in self.robustness_results["alternative_clustering"]
            ):
                cluster = self.robustness_results["alternative_clustering"][outcome]

                if "state" in cluster:
                    row["cluster_state_se"] = cluster["state"]["se"]
                if "region" in cluster:
                    row["cluster_region_se"] = cluster["region"]["se"]
                if "robust" in cluster:
                    row["cluster_robust_se"] = cluster["robust"]["se"]

            # Permutation test
            if (
                "permutation_test" in self.robustness_results
                and outcome in self.robustness_results["permutation_test"]
            ):
                perm = self.robustness_results["permutation_test"][outcome]
                row["permutation_pvalue"] = perm["permutation_pvalue"]

            # Specification curve
            if (
                "specification_curve" in self.robustness_results
                and outcome in self.robustness_results["specification_curve"]
            ):
                spec = self.robustness_results["specification_curve"][outcome]
                row["spec_curve_n"] = spec["n_specs"]
                row["spec_curve_significant"] = spec["significant_specs"]
                coeff_min, coeff_max = spec["coeff_range"]
                row["spec_curve_range"] = f"[{coeff_min:.3f}, {coeff_max:.3f}]"

            table_data.append(row)

        robustness_df = pd.DataFrame(table_data)

        # Save table
        robustness_df.to_csv(self.tables_dir / "table3_robustness_results.csv", index=False)

        # Create LaTeX version
        latex_table = self._format_robustness_latex(robustness_df)
        with open(self.tables_dir / "table3_robustness_results.tex", "w") as f:
            f.write(latex_table)

        print(f"Robustness table saved to {self.tables_dir}")
        return robustness_df

    def _format_robustness_latex(self, df: pd.DataFrame) -> str:
        """Format robustness table as LaTeX."""
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Robustness Analysis Results}
\\label{tab:robustness}
\\begin{tabular}{lcccc}
\\toprule
Outcome & LOSO Range & Alt. Clustering & Permutation p & Spec. Curve \\\\
\\midrule
"""

        for _, row in df.iterrows():
            outcome_name = row["outcome"].replace("_", "\\_")

            # LOSO column
            loso_str = row.get("loso_range", "--")

            # Clustering column (show range of SEs)
            cluster_ses = []
            for col in ["cluster_state_se", "cluster_region_se", "cluster_robust_se"]:
                if pd.notna(row.get(col)):
                    cluster_ses.append(row[col])

            if cluster_ses:
                cluster_str = f"[{min(cluster_ses):.3f}, {max(cluster_ses):.3f}]"
            else:
                cluster_str = "--"

            # Permutation p-value
            perm_str = (
                f"{row.get('permutation_pvalue', np.nan):.3f}"
                if pd.notna(row.get("permutation_pvalue"))
                else "--"
            )

            # Specification curve
            spec_str = row.get("spec_curve_range", "--")

            latex += f"{outcome_name} & {loso_str} & {cluster_str} & {perm_str} & {spec_str} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\footnotesize
\\textit{Notes:} LOSO = Leave-One-State-Out coefficient range.
Alt. Clustering shows range of standard errors across clustering methods.
Permutation p shows p-value from permutation test.
Spec. Curve shows coefficient range across specifications.
\\end{table}"""

        return latex

    def run_full_robustness_suite(self) -> str:
        """
        Run complete robustness analysis pipeline with enhanced Phase 3 methods.

        Returns:
            Path to robustness report
        """
        print("Phase 4.3: Running Enhanced Robustness Analysis")
        print("=" * 50)

        # Load data
        if self.df is None:
            self.load_data()

        # Run traditional robustness tests
        print("Running traditional robustness tests...")
        self.leave_one_state_out()
        self.alternative_clustering()
        self.permutation_test(n_permutations=500)  # Reduced for speed
        self.specification_curve()

        # Run alternative robust methods (Phase 2)
        print("\nRunning alternative robust methods...")
        self.bootstrap_inference(n_bootstrap=1000)
        self.jackknife_inference()
        self.wild_cluster_bootstrap(n_bootstrap=999)

        # Run Phase 3: Enhanced Statistical Inference
        print("\nRunning Phase 3: Enhanced Statistical Inference...")

        # Apply multiple testing corrections
        corrections = self.multiple_testing_corrections(self.robustness_results)
        self.robustness_results["multiple_testing"] = corrections

        # Calculate effect sizes
        effect_sizes = self.calculate_effect_sizes(self.robustness_results)
        self.robustness_results["effect_sizes"] = effect_sizes

        # Conduct power analysis
        power_analysis = self.power_analysis(self.robustness_results)
        self.robustness_results["power_analysis"] = power_analysis

        # Calculate enhanced confidence intervals
        enhanced_cis = self.enhanced_confidence_intervals(self.robustness_results)
        self.robustness_results["enhanced_confidence_intervals"] = enhanced_cis

        # Create outputs
        self.create_robustness_table()
        self.create_robustness_plots()

        # Generate enhanced report with Phase 3 results
        report_path = self.reports_dir / "enhanced_robustness_analysis_report.md"

        with open(report_path, "w") as f:
            f.write(f"""# Phase 4.3: Enhanced Robustness Analysis Report with Phase 3 Statistical Inference

Generated by: Jeff Chen (jeffreyc1@alumni.cmu.edu)
Created in collaboration with Claude Code
Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Robustness Tests Completed

### Traditional Methods
1. **Leave-One-State-Out Analysis**: Tests sensitivity to individual states
2. **Alternative Clustering**: Compares standard errors across clustering methods
3. **Permutation Test**: Tests significance under random treatment assignment
4. **Specification Curve**: Tests sensitivity to model specification choices

### Alternative Robust Methods (Phase 2)
5. **Bootstrap Inference**: Cluster bootstrap for robust standard errors
6. **Jackknife Inference**: Leave-one-state-out inference method
7. **Wild Cluster Bootstrap**: Robust inference for small cluster counts

### Phase 3: Enhanced Statistical Inference ✨ NEW
8. **Multiple Testing Corrections**: Bonferroni, Benjamini-Hochberg FDR, Romano-Wolf
9. **Effect Size Analysis**: Cohen's d and cross-method comparisons
10. **Power Analysis**: Post-hoc power and minimum detectable effects
11. **Enhanced Confidence Intervals**: BCa bootstrap and simultaneous bands

## Key Findings

{self._summarize_robustness_results()}

## Alternative Method Results

{self._summarize_alternative_methods()}

## Phase 3: Enhanced Statistical Inference Results

{self._summarize_enhanced_inference()}

## Files Generated

1. **Tables**:
   - `table3_robustness_results.csv` - Robustness results in CSV format
   - `table3_robustness_results.tex` - LaTeX formatted table
   - `enhanced_inference_summary.json` - Phase 3 detailed results

2. **Figures**:
   - `robustness_loso.png` - Leave-one-state-out plots
   - `robustness_specification_curve.png` - Specification curve analysis
   - `robustness_permutation.png` - Permutation test results
   - `enhanced_inference_dashboard.png` - Phase 3 comprehensive visualization

## Conclusion

{self._robustness_conclusion()}

### Phase 3 Statistical Inference Conclusion

{self._enhanced_inference_conclusion()}
""")

        print("\n✅ Enhanced robustness analysis with Phase 3 methods completed successfully!")
        print(f"📊 Report available at: {report_path}")

        return str(report_path)

    def _summarize_robustness_results(self) -> str:
        """Summarize key robustness findings."""
        if not self.robustness_results:
            return "No robustness results available."

        summary = []

        # LOSO summary
        if "leave_one_state_out" in self.robustness_results:
            loso = self.robustness_results["leave_one_state_out"]
            n_outcomes = len(loso)
            summary.append(
                f"- **Leave-One-State-Out**: Tested {n_outcomes} outcomes across individual state exclusions"
            )

        # Permutation test summary
        if "permutation_test" in self.robustness_results:
            perm = self.robustness_results["permutation_test"]
            sig_count = sum(1 for r in perm.values() if r["permutation_pvalue"] < 0.05)
            summary.append(
                f"- **Permutation Test**: {sig_count}/{len(perm)} outcomes significant at 5% level"
            )

        # Specification curve summary
        if "specification_curve" in self.robustness_results:
            spec = self.robustness_results["specification_curve"]
            total_specs = sum(r["n_specs"] for r in spec.values())
            summary.append(
                f"- **Specification Curve**: {total_specs} total specifications tested across outcomes"
            )

        return "\n".join(summary) if summary else "No robustness tests completed."

    def _summarize_alternative_methods(self) -> str:
        """Generate summary of alternative robust method results."""
        summary_lines = []

        # Bootstrap results
        if "bootstrap" in self.robustness_results:
            bootstrap_results = self.robustness_results["bootstrap"]
            successful_outcomes = sum(
                1
                for outcome, results in bootstrap_results.items()
                if results.get("successful_bootstraps", 0) > 100
            )
            summary_lines.append(
                f"- **Bootstrap Inference**: {successful_outcomes}/{len(bootstrap_results)} outcomes with robust results"
            )

            for outcome, results in bootstrap_results.items():
                if results.get("successful_bootstraps", 0) > 100:
                    coef = results.get("coefficient", np.nan)
                    se = results.get("bootstrap_se", np.nan)
                    p_val = results.get("p_value", np.nan)
                    summary_lines.append(f"  - {outcome}: β={coef:.3f}, SE={se:.3f}, p={p_val:.3f}")

        # Jackknife results
        if "jackknife" in self.robustness_results:
            jackknife_results = self.robustness_results["jackknife"]
            successful_outcomes = sum(
                1
                for outcome, results in jackknife_results.items()
                if results.get("successful_jackknife", 0) > 5
            )
            summary_lines.append(
                f"- **Jackknife Inference**: {successful_outcomes}/{len(jackknife_results)} outcomes with robust results"
            )

            for outcome, results in jackknife_results.items():
                if results.get("successful_jackknife", 0) > 5:
                    coef = results.get("coefficient", np.nan)
                    se = results.get("jackknife_se", np.nan)
                    p_val = results.get("p_value", np.nan)
                    summary_lines.append(f"  - {outcome}: β={coef:.3f}, SE={se:.3f}, p={p_val:.3f}")

        # Wild bootstrap results
        if "wild_bootstrap" in self.robustness_results:
            wild_results = self.robustness_results["wild_bootstrap"]
            successful_outcomes = sum(
                1
                for outcome, results in wild_results.items()
                if results.get("successful_bootstraps", 0) > 100
            )
            summary_lines.append(
                f"- **Wild Cluster Bootstrap**: {successful_outcomes}/{len(wild_results)} outcomes with robust results"
            )

            for outcome, results in wild_results.items():
                if results.get("successful_bootstraps", 0) > 100:
                    coef = results.get("coefficient", np.nan)
                    p_val = results.get("wild_bootstrap_p", np.nan)
                    summary_lines.append(f"  - {outcome}: β={coef:.3f}, wild p={p_val:.3f}")

        if not summary_lines:
            return "No alternative method results available."

        return "\n".join(summary_lines)

    def _robustness_conclusion(self) -> str:
        """Provide overall robustness conclusion."""
        return """The robustness analysis provides evidence on the stability of main treatment effects. 
Results should be interpreted alongside the main causal analysis to assess the 
overall credibility of the special education policy evaluation findings."""

    def _summarize_enhanced_inference(self) -> str:
        """
        Summarize Phase 3 enhanced statistical inference results.

        Returns:
            Formatted summary string
        """
        summary = []

        # Multiple testing corrections summary
        if "multiple_testing" in self.robustness_results:
            mt_results = self.robustness_results["multiple_testing"]
            if "summary" in mt_results:
                summary.append("### Multiple Testing Corrections")
                mt_summary = mt_results["summary"]
                summary.append(f"- Total tests conducted: {mt_summary.get('total_tests', 0)}")
                summary.append(
                    f"- Originally significant (α=0.05): {mt_summary.get('original_significant', 0)}"
                )
                summary.append(
                    f"- Bonferroni-corrected significant: {mt_summary.get('bonferroni_significant', 0)}"
                )
                summary.append(
                    f"- FDR-corrected significant: {mt_summary.get('fdr_significant', 0)}"
                )
                summary.append(
                    f"- Romano-Wolf corrected significant: {mt_summary.get('romano_wolf_significant', 0)}"
                )
                summary.append("")

        # Effect sizes summary
        if "effect_sizes" in self.robustness_results:
            es_results = self.robustness_results["effect_sizes"]
            if "cross_method_summary" in es_results:
                summary.append("### Effect Size Analysis")
                cross_summary = es_results["cross_method_summary"]
                for outcome, outcome_summary in cross_summary.items():
                    if isinstance(outcome_summary, dict):
                        mean_es = outcome_summary.get("mean_effect_size", 0)
                        consistency = outcome_summary.get("consistency_interpretation", "unknown")
                        summary.append(
                            f"- {outcome}: Mean Cohen's d = {mean_es:.3f} ({consistency} consistency)"
                        )
                summary.append("")

        # Power analysis summary
        if "power_analysis" in self.robustness_results:
            power_results = self.robustness_results["power_analysis"]
            if "overall_assessment" in power_results:
                summary.append("### Power Analysis")
                overall = power_results["overall_assessment"]
                if "mean_power" in overall:
                    mean_power = overall.get("mean_power", 0)
                    adequately_powered_prop = overall.get("adequately_powered_proportion", 0)
                    recommendation = overall.get("recommendation", "No recommendation available")
                    summary.append(f"- Mean statistical power: {mean_power:.3f}")
                    summary.append(
                        f"- Proportion adequately powered (≥0.8): {adequately_powered_prop:.1%}"
                    )
                    summary.append(f"- Recommendation: {recommendation}")
                summary.append("")

        # Enhanced confidence intervals summary
        if "enhanced_confidence_intervals" in self.robustness_results:
            summary.append("### Enhanced Confidence Intervals")
            summary.append("- BCa bootstrap intervals calculated where bootstrap samples available")
            summary.append("- Small-sample t-distribution adjustments applied")
            summary.append("- Simultaneous confidence bands with Bonferroni adjustment")
            summary.append("")

        return "\n".join(summary) if summary else "Enhanced inference analysis completed."

    def _enhanced_inference_conclusion(self) -> str:
        """
        Generate conclusion for Phase 3 enhanced statistical inference.

        Returns:
            Conclusion string
        """
        conclusions = []

        # Check multiple testing impact
        if "multiple_testing" in self.robustness_results:
            mt_results = self.robustness_results["multiple_testing"]
            if "summary" in mt_results:
                mt_summary = mt_results["summary"]
                original_sig = mt_summary.get("original_significant", 0)
                bonferroni_sig = mt_summary.get("bonferroni_significant", 0)
                fdr_sig = mt_summary.get("fdr_significant", 0)

                if original_sig > 0:
                    if bonferroni_sig == 0 and fdr_sig == 0:
                        conclusions.append(
                            "**Multiple Testing Impact**: All originally significant results "
                            "become non-significant after multiple testing corrections, suggesting "
                            "findings may be due to multiple comparisons rather than true effects."
                        )
                    elif bonferroni_sig < original_sig:
                        conclusions.append(
                            f"**Multiple Testing Impact**: Significance reduced from {original_sig} "
                            f"to {bonferroni_sig} (Bonferroni) and {fdr_sig} (FDR) after corrections."
                        )
                else:
                    conclusions.append(
                        "**Multiple Testing Impact**: No originally significant results to correct."
                    )

        # Effect size interpretation
        if "effect_sizes" in self.robustness_results:
            es_results = self.robustness_results["effect_sizes"]
            if "cross_method_summary" in es_results:
                cross_summary = es_results["cross_method_summary"]
                effect_sizes_found = any(
                    isinstance(outcome_data, dict)
                    and abs(outcome_data.get("mean_effect_size", 0)) > 0.2
                    for outcome_data in cross_summary.values()
                )

                if effect_sizes_found:
                    conclusions.append(
                        "**Effect Sizes**: Some meaningful effect sizes (Cohen's d > 0.2) detected, "
                        "suggesting policy reforms may have practical significance even if not "
                        "statistically significant."
                    )
                else:
                    conclusions.append(
                        "**Effect Sizes**: All effect sizes are small (Cohen's d < 0.2), indicating "
                        "limited practical significance of detected policy effects."
                    )

        # Power analysis conclusion
        if "power_analysis" in self.robustness_results:
            power_results = self.robustness_results["power_analysis"]
            if "overall_assessment" in power_results:
                overall = power_results["overall_assessment"]
                adequately_powered_prop = overall.get("adequately_powered_proportion", 0)

                if adequately_powered_prop < 0.5:
                    conclusions.append(
                        "**Statistical Power**: Study appears underpowered for detecting "
                        "policy effects. Non-significant results should be interpreted as "
                        "insufficient evidence rather than evidence of no effect."
                    )
                elif adequately_powered_prop >= 0.8:
                    conclusions.append(
                        "**Statistical Power**: Study is adequately powered. Non-significant "
                        "results provide stronger evidence against meaningful policy effects."
                    )

        # Overall Phase 3 conclusion
        conclusions.append(
            "\n**Overall Phase 3 Assessment**: The enhanced statistical inference methods provide "
            "important context for interpreting robustness results. Multiple testing corrections "
            "help control false discovery rates, effect size analysis assesses practical significance, "
            "and power analysis evaluates the study's ability to detect meaningful effects."
        )

        return "\n\n".join(conclusions)


def main():
    """Run robustness analysis pipeline."""
    analyzer = RobustnessAnalyzer()

    try:
        analyzer.run_full_robustness_suite()
        return True

    except Exception as e:
        print(f"\n❌ Error in robustness analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
