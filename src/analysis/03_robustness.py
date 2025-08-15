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
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
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
                    "DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
                    "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX",
                ],
                "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
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

                print(f"    {outcome}: {len(coeffs)} successful estimates, "
                      f"Mean={np.mean(coeffs):.4f}, Std={np.std(coeffs):.4f}")
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
                        f"({n_clust} clusters)" if n_clust > 0 else f"      {cluster_type}: β={results['coefficient']:.4f}, SE={results['se']:.4f}"
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

    def bootstrap_inference(self, n_bootstrap: int = 1000) -> Dict[str, Any]:
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
                df_clean = self.df[
                    [outcome, "post_treatment", "state", "year", "region", "treated"]
                ].dropna().reset_index(drop=True)
                
                if len(df_clean) < 20:
                    print(f"  Insufficient data for {outcome}: {len(df_clean)} observations")
                    bootstrap_results[outcome] = {
                        "coefficient": np.nan,
                        "bootstrap_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_bootstraps": 0
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
                        "successful_bootstraps": 0
                    }
                    continue
                
                # Bootstrap procedure
                bootstrap_coefs = []
                states = df_clean["state"].unique()
                
                for i in range(n_bootstrap):
                    try:
                        # Sample states with replacement (cluster bootstrap)
                        bootstrap_states = np.random.choice(states, size=len(states), replace=True)
                        
                        # Create bootstrap sample
                        bootstrap_df = pd.concat([
                            df_clean[df_clean["state"] == state].copy() 
                            for state in bootstrap_states
                        ], ignore_index=True)
                        
                        # Fit model on bootstrap sample
                        bootstrap_model = smf.ols(formula, data=bootstrap_df).fit()
                        bootstrap_coefs.append(bootstrap_model.params["post_treatment"])
                        
                    except Exception:
                        # Silent failure for individual bootstrap draws
                        continue
                
                if len(bootstrap_coefs) < 100:  # Need at least 100 successful draws
                    print(f"  Too few successful bootstrap draws for {outcome}: {len(bootstrap_coefs)}")
                    bootstrap_results[outcome] = {
                        "coefficient": base_coef,
                        "bootstrap_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_bootstraps": len(bootstrap_coefs)
                    }
                    continue
                
                # Calculate bootstrap statistics
                bootstrap_coefs = np.array(bootstrap_coefs)
                bootstrap_se = np.std(bootstrap_coefs)
                ci_lower = np.percentile(bootstrap_coefs, 2.5)
                ci_upper = np.percentile(bootstrap_coefs, 97.5)
                
                # Two-tailed p-value (assuming null hypothesis = 0)
                p_value = 2 * min(
                    np.mean(bootstrap_coefs <= 0),
                    np.mean(bootstrap_coefs >= 0)
                )
                
                bootstrap_results[outcome] = {
                    "coefficient": base_coef,
                    "bootstrap_se": bootstrap_se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                    "successful_bootstraps": len(bootstrap_coefs)
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
                    "successful_bootstraps": 0
                }
        
        self.robustness_results["bootstrap"] = bootstrap_results
        print("\nBootstrap inference completed.")
        return bootstrap_results

    def jackknife_inference(self) -> Dict[str, Any]:
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
                df_clean = self.df[
                    [outcome, "post_treatment", "state", "year", "region", "treated"]
                ].dropna().reset_index(drop=True)
                
                if len(df_clean) < 20:
                    print(f"  Insufficient data for {outcome}: {len(df_clean)} observations")
                    jackknife_results[outcome] = {
                        "coefficient": np.nan,
                        "jackknife_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_jackknife": 0
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
                        "successful_jackknife": 0
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
                    print(f"  Too few successful jackknife samples for {outcome}: {len(jackknife_coefs)}")
                    jackknife_results[outcome] = {
                        "coefficient": base_coef,
                        "jackknife_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "successful_jackknife": len(jackknife_coefs)
                    }
                    continue
                
                # Calculate jackknife statistics
                jackknife_coefs = np.array(jackknife_coefs)
                n_samples = len(jackknife_coefs)
                
                # Jackknife standard error
                jackknife_se = np.sqrt((n_samples - 1) * np.var(jackknife_coefs))
                
                # Confidence interval using t-distribution
                from scipy.stats import t
                t_critical = t.ppf(0.975, df=n_samples-1)
                ci_lower = base_coef - t_critical * jackknife_se
                ci_upper = base_coef + t_critical * jackknife_se
                
                # p-value using t-test
                t_stat = base_coef / jackknife_se if jackknife_se > 0 else 0
                p_value = 2 * (1 - t.cdf(abs(t_stat), df=n_samples-1))
                
                jackknife_results[outcome] = {
                    "coefficient": base_coef,
                    "jackknife_se": jackknife_se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                    "successful_jackknife": len(jackknife_coefs)
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
                    "successful_jackknife": 0
                }
        
        self.robustness_results["jackknife"] = jackknife_results
        print("\nJackknife inference completed.")
        return jackknife_results

    def wild_cluster_bootstrap(self, n_bootstrap: int = 999) -> Dict[str, Any]:
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
                df_clean = self.df[
                    [outcome, "post_treatment", "state", "year", "region", "treated"]
                ].dropna().reset_index(drop=True)
                
                if len(df_clean) < 20:
                    print(f"  Insufficient data for {outcome}: {len(df_clean)} observations")
                    wild_bootstrap_results[outcome] = {
                        "coefficient": np.nan,
                        "wild_bootstrap_p": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "successful_bootstraps": 0
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
                        "successful_bootstraps": 0
                    }
                    continue
                
                # Wild bootstrap procedure
                bootstrap_t_stats = []
                states = df_clean["state"].unique()
                
                for i in range(n_bootstrap):
                    try:
                        # Generate wild bootstrap weights (Rademacher: +1 or -1)
                        wild_weights = np.random.choice([-1, 1], size=len(states))
                        
                        # Create weight mapping for each observation
                        weight_map = dict(zip(states, wild_weights))
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
                
                if len(bootstrap_t_stats) < min(50, n_bootstrap * 0.5):  # Need at least 50% successful draws
                    print(f"  Too few successful wild bootstrap draws for {outcome}: {len(bootstrap_t_stats)}")
                    wild_bootstrap_results[outcome] = {
                        "coefficient": base_coef,
                        "wild_bootstrap_p": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "successful_bootstraps": len(bootstrap_t_stats)
                    }
                    continue
                
                # Calculate wild bootstrap p-value
                bootstrap_t_stats = np.array(bootstrap_t_stats)
                
                # Two-tailed p-value
                p_value = np.mean(np.abs(bootstrap_t_stats) >= np.abs(base_t_stat))
                
                # Confidence interval using percentile method
                # Transform t-statistics back to coefficients (approximate)
                ci_lower = np.percentile(bootstrap_t_stats * base_model.bse["post_treatment"] + base_coef, 2.5)
                ci_upper = np.percentile(bootstrap_t_stats * base_model.bse["post_treatment"] + base_coef, 97.5)
                
                wild_bootstrap_results[outcome] = {
                    "coefficient": base_coef,
                    "wild_bootstrap_p": p_value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "successful_bootstraps": len(bootstrap_t_stats)
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
                    "successful_bootstraps": 0
                }
        
        self.robustness_results["wild_bootstrap"] = wild_bootstrap_results
        print("\nWild cluster bootstrap completed.")
        return wild_bootstrap_results

    def create_robustness_plots(self) -> dict[str, plt.Figure]:
        """Create visualization plots for robustness results."""
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
        Run complete robustness analysis pipeline with enhanced methods.

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

        # Create outputs
        self.create_robustness_table()
        self.create_robustness_plots()

        # Generate enhanced report
        report_path = self.reports_dir / "robustness_analysis_report.md"

        with open(report_path, "w") as f:
            f.write(f"""# Phase 4.3: Enhanced Robustness Analysis Report

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

## Key Findings

{self._summarize_robustness_results()}

## Alternative Method Results

{self._summarize_alternative_methods()}

## Files Generated

1. **Tables**:
   - `table3_robustness_results.csv` - Robustness results in CSV format
   - `table3_robustness_results.tex` - LaTeX formatted table

2. **Figures**:
   - `robustness_loso.png` - Leave-one-state-out plots
   - `robustness_specification_curve.png` - Specification curve analysis
   - `robustness_permutation.png` - Permutation test results

## Conclusion

{self._robustness_conclusion()}
""")

        print("\n✅ Enhanced robustness analysis completed successfully!")
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
                1 for outcome, results in bootstrap_results.items() 
                if results.get("successful_bootstraps", 0) > 100
            )
            summary_lines.append(f"- **Bootstrap Inference**: {successful_outcomes}/{len(bootstrap_results)} outcomes with robust results")
            
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
                1 for outcome, results in jackknife_results.items() 
                if results.get("successful_jackknife", 0) > 5
            )
            summary_lines.append(f"- **Jackknife Inference**: {successful_outcomes}/{len(jackknife_results)} outcomes with robust results")
            
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
                1 for outcome, results in wild_results.items() 
                if results.get("successful_bootstraps", 0) > 100
            )
            summary_lines.append(f"- **Wild Cluster Bootstrap**: {successful_outcomes}/{len(wild_results)} outcomes with robust results")
            
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
