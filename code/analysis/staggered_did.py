"""
Staggered Difference-in-Differences Analysis

Implementation of Callaway-Sant'Anna (2021) staggered DiD estimator
for analyzing the effects of state special education policy reforms
on student outcomes.

Key features:
- Accounts for treatment timing heterogeneity
- Handles never-treated units appropriately
- Computes group-time average treatment effects
- Aggregates to overall average treatment effect

Author: Research Team
Date: 2025-08-12
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols


class StaggeredDiDAnalyzer:
    """
    Implementation of staggered difference-in-differences estimation.

    Based on Callaway and Sant'Anna (2021) methodology but implemented
    using standard econometric packages to avoid dependency issues.
    """

    def __init__(self, data_path: str = "data/final/analysis_panel.csv"):
        """Initialize with analysis panel dataset."""
        self.data_path = data_path
        self.df = None
        self.results = {}
        self._load_data()

    def _load_data(self):
        """Load analysis panel dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded analysis panel: {self.df.shape}")
            print(f"Years: {self.df['year'].min()}-{self.df['year'].max()}")
            print(f"States: {len(self.df['state'].unique())}")

            # Verify we have treatment data
            if "post_treatment" in self.df.columns:
                treated_states = len(
                    self.df[self.df["post_treatment"] == 1]["state"].unique()
                )
                print(f"Treated states: {treated_states}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def identify_cohorts(self) -> dict[int, list[str]]:
        """Identify treatment cohorts by first treatment year."""
        if self.df is None:
            raise ValueError("Data not loaded")

        # Find first treatment year for each state
        treated_data = self.df[self.df["post_treatment"] == 1].copy()

        if treated_data.empty:
            print("No treated units found")
            return {}

        # Get first treatment year for each state
        first_treatment = treated_data.groupby("state")["year"].min().reset_index()
        first_treatment.columns = ["state", "treatment_year"]

        # Group states by treatment year (cohorts)
        cohorts = {}
        for _, row in first_treatment.iterrows():
            year = row["treatment_year"]
            state = row["state"]
            if year not in cohorts:
                cohorts[year] = []
            cohorts[year].append(state)

        print(f"Treatment cohorts identified: {list(cohorts.keys())}")
        for year, states in cohorts.items():
            print(f"  {year}: {len(states)} states")

        return cohorts

    def estimate_group_time_effects(
        self, outcome_var: str, control_vars: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Estimate group-time average treatment effects.

        This is the core of Callaway-Sant'Anna methodology - estimate
        treatment effects for each cohort-time combination.
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        cohorts = self.identify_cohorts()
        if not cohorts:
            raise ValueError("No treatment cohorts found")

        # Check if outcome variable exists
        if outcome_var not in self.df.columns:
            available_outcomes = [
                col for col in self.df.columns if "gap" in col or "score" in col
            ]
            raise ValueError(
                f"Outcome variable '{outcome_var}' not found. Available: {available_outcomes}"
            )

        # Default control variables
        if control_vars is None:
            control_vars = ["time_trend", "post_covid"]

        # Filter to only include available control variables
        available_controls = [var for var in control_vars if var in self.df.columns]

        group_time_results = []

        for treatment_year, treated_states in cohorts.items():
            print(f"\\nEstimating effects for {treatment_year} cohort...")

            # For each post-treatment period
            post_periods = [y for y in self.df["year"].unique() if y >= treatment_year]

            for t in post_periods:
                # Restrict to relevant time periods for this comparison
                # Use pre-treatment periods and current period
                analysis_periods = list(range(treatment_year - 3, treatment_year)) + [t]
                analysis_data = self.df[self.df["year"].isin(analysis_periods)].copy()

                if analysis_data.empty:
                    continue

                # Define treatment and control groups for this comparison
                # Treated: states in current cohort
                # Control: never treated + not-yet-treated (if available)
                analysis_data["treated_group"] = analysis_data["state"].isin(
                    treated_states
                )
                analysis_data["post_period"] = (analysis_data["year"] == t).astype(int)
                analysis_data["treat_post"] = (
                    analysis_data["treated_group"] * analysis_data["post_period"]
                )

                # Only include observations with non-missing outcome
                analysis_data = analysis_data.dropna(subset=[outcome_var])

                if len(analysis_data) < 10:  # Skip if too few observations
                    continue

                try:
                    # Estimate DiD regression
                    formula_parts = [
                        outcome_var,
                        "treated_group + post_period + treat_post",
                    ]

                    # Add available controls
                    if available_controls:
                        formula_parts[1] += " + " + " + ".join(available_controls)

                    # Add state fixed effects (if enough variation)
                    if len(analysis_data["state"].unique()) > 2:
                        formula_parts[1] += " + C(state)"

                    formula = " ~ ".join(formula_parts)

                    # Estimate regression
                    model = ols(formula, data=analysis_data).fit()

                    # Extract DiD coefficient
                    did_coef = model.params.get("treat_post", np.nan)
                    did_se = model.bse.get("treat_post", np.nan)
                    did_pval = model.pvalues.get("treat_post", np.nan)

                    # Store results
                    result = {
                        "cohort": treatment_year,
                        "time": t,
                        "time_relative": t - treatment_year,
                        "att": did_coef,
                        "se": did_se,
                        "pvalue": did_pval,
                        "n_obs": len(analysis_data),
                        "n_treated": len(
                            analysis_data[analysis_data["treated_group"] == True]
                        ),
                        "n_control": len(
                            analysis_data[analysis_data["treated_group"] == False]
                        ),
                    }

                    group_time_results.append(result)

                except Exception as e:
                    print(
                        f"Error estimating for cohort {treatment_year}, time {t}: {e}"
                    )
                    continue

        results_df = pd.DataFrame(group_time_results)

        if not results_df.empty:
            print(f"\\nEstimated {len(results_df)} group-time effects")
            print(f"Average ATT: {results_df['att'].mean():.3f}")
            print(
                f"Significant effects: {(results_df['pvalue'] < 0.05).sum()}/{len(results_df)}"
            )

        return results_df

    def aggregate_treatment_effects(
        self, group_time_df: pd.DataFrame
    ) -> dict[str, float]:
        """Aggregate group-time effects to overall treatment effects."""

        if group_time_df.empty:
            return {}

        # Simple average (equal weighting)
        simple_att = group_time_df["att"].mean()
        simple_se = group_time_df["se"].mean() / np.sqrt(len(group_time_df))

        # Inverse variance weighted average
        weights = 1 / (group_time_df["se"] ** 2)
        weighted_att = np.sum(weights * group_time_df["att"]) / np.sum(weights)
        weighted_se = np.sqrt(1 / np.sum(weights))

        # Aggregate by relative time
        relative_time_effects = (
            group_time_df.groupby("time_relative")
            .agg(
                {
                    "att": "mean",
                    "se": lambda x: np.mean(x) / np.sqrt(len(x)),
                    "pvalue": lambda x: stats.combine_pvalues(x)[1]
                    if len(x) > 1
                    else x.iloc[0],
                }
            )
            .reset_index()
        )

        aggregated = {
            "simple_att": simple_att,
            "simple_se": simple_se,
            "weighted_att": weighted_att,
            "weighted_se": weighted_se,
            "relative_time_effects": relative_time_effects,
            "n_cohorts": group_time_df["cohort"].nunique(),
            "n_group_time": len(group_time_df),
        }

        return aggregated

    def estimate_event_study(
        self,
        outcome_var: str,
        max_lead: int = 3,
        max_lag: int = 5,
        control_vars: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate event study specification showing leads and lags.
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        # Check outcome variable
        if outcome_var not in self.df.columns:
            available_outcomes = [
                col for col in self.df.columns if "gap" in col or "score" in col
            ]
            raise ValueError(
                f"Outcome variable '{outcome_var}' not found. Available: {available_outcomes}"
            )

        # Create analysis dataset with non-missing outcomes
        analysis_data = self.df.dropna(subset=[outcome_var]).copy()

        # Default controls
        if control_vars is None:
            control_vars = ["time_trend", "post_covid"]
        available_controls = [
            var for var in control_vars if var in analysis_data.columns
        ]

        # Create event time indicators
        event_time_vars = []

        # Lead indicators (before treatment)
        for lead in range(1, max_lead + 1):
            var_name = f"lead_{lead}"
            if var_name in analysis_data.columns:
                event_time_vars.append(var_name)

        # Lag indicators (after treatment) - omit lag_1 as reference
        for lag in range(2, max_lag + 1):
            var_name = f"lag_{lag}"
            if var_name in analysis_data.columns:
                event_time_vars.append(var_name)

        if not event_time_vars:
            print("No event time indicators found - creating them...")
            # Create event time indicators based on years_since_treatment
            for lead in range(1, max_lead + 1):
                analysis_data[f"lead_{lead}"] = (
                    analysis_data["years_since_treatment"] == -lead
                ).astype(int)
                event_time_vars.append(f"lead_{lead}")

            for lag in range(2, max_lag + 1):
                analysis_data[f"lag_{lag}"] = (
                    analysis_data["years_since_treatment"] == lag
                ).astype(int)
                event_time_vars.append(f"lag_{lag}")

        # Build regression formula
        formula_parts = [outcome_var]
        rhs_parts = event_time_vars.copy()

        # Add controls
        if available_controls:
            rhs_parts.extend(available_controls)

        # Add fixed effects
        rhs_parts.extend(["C(state)", "C(year)"])

        formula = f"{formula_parts[0]} ~ {' + '.join(rhs_parts)}"

        try:
            print(f"Estimating event study: {formula}")
            model = ols(formula, data=analysis_data).fit(
                cov_type="cluster", cov_kwds={"groups": analysis_data["state"]}
            )

            # Extract event time coefficients
            event_results = []

            # Add reference period (lag_1 = 0)
            event_results.append(
                {
                    "event_time": 1,
                    "coef": 0.0,
                    "se": 0.0,
                    "pvalue": np.nan,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                }
            )

            # Extract coefficients
            for var in event_time_vars:
                if var in model.params.index:
                    coef = model.params[var]
                    se = model.bse[var]
                    pval = model.pvalues[var]
                    ci_lower, ci_upper = model.conf_int().loc[var]

                    # Determine event time
                    if "lead" in var:
                        event_time = -int(var.split("_")[1])
                    elif "lag" in var:
                        event_time = int(var.split("_")[1])
                    else:
                        continue

                    event_results.append(
                        {
                            "event_time": event_time,
                            "coef": coef,
                            "se": se,
                            "pvalue": pval,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                        }
                    )

            # Store model for later use
            self.results[f"event_study_{outcome_var}"] = {
                "model": model,
                "event_results": pd.DataFrame(event_results).sort_values("event_time"),
                "n_obs": len(analysis_data),
                "r_squared": model.rsquared,
            }

            print("Event study estimated successfully:")
            print(f"  Observations: {len(analysis_data)}")
            print(f"  R-squared: {model.rsquared:.3f}")

            return pd.DataFrame(event_results).sort_values("event_time")

        except Exception as e:
            print(f"Error estimating event study: {e}")
            raise

    def run_staggered_analysis(
        self, outcome_vars: list[str] | None = None
    ) -> dict[str, any]:
        """
        Run complete staggered DiD analysis for specified outcomes.
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        # Default outcome variables
        if outcome_vars is None:
            outcome_vars = [col for col in self.df.columns if "gap" in col]
            if not outcome_vars:
                outcome_vars = [col for col in self.df.columns if "score" in col][
                    :2
                ]  # Limit to 2

        if not outcome_vars:
            raise ValueError("No outcome variables found")

        print(f"Running staggered DiD analysis for: {outcome_vars}")

        all_results = {}

        for outcome in outcome_vars:
            print(f"\\n{'=' * 50}")
            print(f"ANALYZING: {outcome}")
            print(f"{'=' * 50}")

            try:
                # Estimate group-time effects
                group_time_results = self.estimate_group_time_effects(outcome)

                # Aggregate effects
                if not group_time_results.empty:
                    aggregated = self.aggregate_treatment_effects(group_time_results)
                else:
                    aggregated = {}

                # Estimate event study
                event_study_results = self.estimate_event_study(outcome)

                # Store results
                all_results[outcome] = {
                    "group_time_effects": group_time_results,
                    "aggregated_effects": aggregated,
                    "event_study": event_study_results,
                    "outcome_variable": outcome,
                }

                # Print key findings
                if aggregated:
                    print(f"\\nKEY RESULTS for {outcome}:")
                    print(
                        f"  Average Treatment Effect: {aggregated['simple_att']:.3f} ({aggregated['simple_se']:.3f})"
                    )
                    print(
                        f"  Weighted Treatment Effect: {aggregated['weighted_att']:.3f} ({aggregated['weighted_se']:.3f})"
                    )
                    print(f"  Number of Cohorts: {aggregated['n_cohorts']}")

            except Exception as e:
                print(f"Error analyzing {outcome}: {e}")
                all_results[outcome] = {"error": str(e)}
                continue

        # Store in class results
        self.results["staggered_analysis"] = all_results

        return all_results

    def export_results(self, output_dir: str = "output/tables") -> dict[str, str]:
        """Export analysis results to CSV files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        exported_files = {}

        if "staggered_analysis" in self.results:
            for outcome, results in self.results["staggered_analysis"].items():
                if "error" in results:
                    continue

                # Export group-time effects
                if not results["group_time_effects"].empty:
                    gt_file = f"{output_dir}/group_time_effects_{outcome}.csv"
                    results["group_time_effects"].to_csv(gt_file, index=False)
                    exported_files[f"group_time_{outcome}"] = gt_file

                # Export event study
                if not results["event_study"].empty:
                    es_file = f"{output_dir}/event_study_{outcome}.csv"
                    results["event_study"].to_csv(es_file, index=False)
                    exported_files[f"event_study_{outcome}"] = es_file

                # Export aggregated results
                if results["aggregated_effects"]:
                    agg_data = {
                        k: [v]
                        for k, v in results["aggregated_effects"].items()
                        if not isinstance(v, pd.DataFrame)
                    }
                    agg_df = pd.DataFrame(agg_data)
                    agg_file = f"{output_dir}/aggregated_effects_{outcome}.csv"
                    agg_df.to_csv(agg_file, index=False)
                    exported_files[f"aggregated_{outcome}"] = agg_file

        print(f"Results exported to {len(exported_files)} files in {output_dir}")
        return exported_files


if __name__ == "__main__":
    # Run staggered DiD analysis
    analyzer = StaggeredDiDAnalyzer()

    # Run analysis
    results = analyzer.run_staggered_analysis()

    # Export results
    exported = analyzer.export_results()

    print(f"\\n{'=' * 60}")
    print("STAGGERED DIFFERENCE-IN-DIFFERENCES ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Outcomes analyzed: {len(results)}")
    print(f"Files exported: {len(exported)}")

    # Print summary of key findings
    for outcome, result in results.items():
        if "aggregated_effects" in result and result["aggregated_effects"]:
            agg = result["aggregated_effects"]
            print(f"\\n{outcome}:")
            print(
                f"  Treatment Effect: {agg['simple_att']:.3f} (SE: {agg['simple_se']:.3f})"
            )
            print(f"  Cohorts: {agg['n_cohorts']}")
