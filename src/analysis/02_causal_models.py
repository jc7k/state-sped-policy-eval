"""
Phase 4.2: Main Causal Analysis Module

Executes core econometric specifications for special education policy evaluation.
Implements manual difference-in-differences methods using statsmodels and linearmodels.

Models implemented:
1. Basic Two-Way Fixed Effects (TWFE)
2. Event Study Analysis
3. Manual Callaway-Sant'Anna Difference-in-Differences
4. Instrumental Variables with court orders

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)


class CausalAnalyzer:
    """
    Main causal analysis for special education policy evaluation.

    Implements multiple econometric specifications to identify
    causal effects of special education funding reforms.
    """

    def __init__(self, data_path: str = "data/final/analysis_panel.csv"):
        """
        Initialize causal analyzer.

        Args:
            data_path: Path to the analysis panel dataset
        """
        self.data_path = data_path
        self.df = None
        self.results = {}
        self.output_dir = Path("output")
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        # Create output directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load and prepare data for causal analysis."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded analysis panel: {self.df.shape}")

            # Validate required columns
            required_cols = ["state", "year", "post_treatment"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Prepare data for panel estimation
            self._prepare_panel_data()

            print(
                f"Data prepared for causal analysis. Treatment states: {self.df['post_treatment'].sum()}"
            )

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _prepare_panel_data(self) -> None:
        """Prepare data for panel estimation."""
        # Create state and year indicators for fixed effects
        self.df["state_id"] = pd.Categorical(self.df["state"]).codes
        self.df["year_id"] = pd.Categorical(self.df["year"]).codes

        # Create event time variable if treatment year available
        if "treatment_year" in self.df.columns:
            self.df["years_since_treatment"] = self.df["year"] - self.df["treatment_year"]
            self.df["years_since_treatment"] = self.df["years_since_treatment"].fillna(-999)
        else:
            # Estimate treatment year from post_treatment variable
            treatment_years = (
                self.df[self.df["post_treatment"] == 1].groupby("state")["year"].min().to_dict()
            )

            self.df["treatment_year"] = self.df["state"].map(treatment_years)
            self.df["years_since_treatment"] = np.where(
                self.df["treatment_year"].notna(), self.df["year"] - self.df["treatment_year"], -999
            )

        # Identify outcome variables
        self.outcome_vars = self._get_outcome_variables()
        print(f"Outcome variables identified: {self.outcome_vars}")

    def _get_outcome_variables(self) -> list[str]:
        """Identify main outcome variables for analysis."""
        potential_outcomes = []

        for col in self.df.columns:
            col_lower = col.lower()
            # Look for achievement gaps (primary outcome)
            if (
                "gap" in col_lower
                and any(subj in col_lower for subj in ["math", "reading"])
                or "score" in col_lower
                and "gap" not in col_lower
                or any(term in col_lower for term in ["inclusion", "placement"])
            ):
                potential_outcomes.append(col)

        # If no specific outcomes found, use numeric variables
        if not potential_outcomes:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            excluded = [
                "year",
                "state_id",
                "year_id",
                "post_treatment",
                "treatment_year",
                "years_since_treatment",
            ]
            potential_outcomes = [col for col in numeric_cols if col not in excluded][:3]

        return potential_outcomes[:5]  # Limit to 5 main outcomes

    def run_twfe_analysis(self) -> dict[str, Any]:
        """
        Run basic Two-Way Fixed Effects specification.

        Returns:
            Dictionary of TWFE results
        """
        print("Running TWFE analysis...")

        twfe_results = {}

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            try:
                # Basic TWFE specification
                formula = f"{outcome} ~ post_treatment + C(state) + C(year)"

                # First fit without clustering
                model = smf.ols(formula, data=self.df).fit()

                # Then apply cluster-robust standard errors using the numeric state_id
                try:
                    model_clustered = smf.ols(formula, data=self.df).fit(
                        cov_type="cluster", cov_kwds={"groups": self.df["state_id"]}
                    )
                    model = model_clustered
                except:
                    # Fall back to robust standard errors if clustering fails
                    model = smf.ols(formula, data=self.df).fit(cov_type="HC0")

                twfe_results[outcome] = {
                    "coefficient": model.params["post_treatment"],
                    "se": model.bse["post_treatment"],
                    "p_value": model.pvalues["post_treatment"],
                    "t_stat": model.tvalues["post_treatment"],
                    "conf_int": model.conf_int().loc["post_treatment"].tolist(),
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                    "model": model,
                }

                print(
                    f"  {outcome}: β={model.params['post_treatment']:.4f}, SE={model.bse['post_treatment']:.4f}"
                )

            except Exception as e:
                print(f"  Error with {outcome}: {e}")
                continue

        self.results["twfe"] = twfe_results
        return twfe_results

    def run_event_study(self) -> dict[str, Any]:
        """
        Run event study analysis.

        Returns:
            Dictionary of event study results
        """
        print("Running event study analysis...")

        event_results = {}

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            try:
                # Create event time indicators (-5 to +5 years)
                df_event = self.df.copy()

                # Create event time dummies with valid variable names
                for t in range(-5, 6):
                    if t != -1:  # Omit t=-1 as reference period
                        if t < 0:
                            var_name = f"event_tm{abs(t)}"  # tm for "t minus"
                        else:
                            var_name = f"event_tp{t}"  # tp for "t plus"
                        df_event[var_name] = (df_event["years_since_treatment"] == t).astype(int)

                # Build formula with valid variable names
                event_vars = []
                for t in range(-5, 6):
                    if t != -1:  # Skip reference period
                        var_name = f"event_tm{abs(t)}" if t < 0 else f"event_tp{t}"
                        if var_name in df_event.columns:
                            event_vars.append(var_name)

                if not event_vars:
                    print(f"  No event time variables for {outcome}")
                    continue

                formula = f"{outcome} ~ " + " + ".join(event_vars) + " + C(state) + C(year)"

                # Fit model with error handling for clustering
                try:
                    model = smf.ols(formula, data=df_event).fit(
                        cov_type="cluster", cov_kwds={"groups": df_event["state_id"]}
                    )
                except:
                    model = smf.ols(formula, data=df_event).fit(cov_type="HC0")

                # Extract event study coefficients
                event_coefs = {}
                event_ses = {}
                event_pvals = {}

                for t in range(-5, 6):
                    if t == -1:
                        event_coefs[t] = 0  # Reference period
                        event_ses[t] = 0
                        event_pvals[t] = np.nan
                    else:
                        var_name = f"event_tm{abs(t)}" if t < 0 else f"event_tp{t}"

                        if var_name in model.params.index:
                            event_coefs[t] = model.params[var_name]
                            event_ses[t] = model.bse[var_name]
                            event_pvals[t] = model.pvalues[var_name]
                        else:
                            event_coefs[t] = np.nan
                            event_ses[t] = np.nan
                            event_pvals[t] = np.nan

                event_results[outcome] = {
                    "coefficients": event_coefs,
                    "standard_errors": event_ses,
                    "p_values": event_pvals,
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                    "model": model,
                }

                print(f"  {outcome}: Event study completed ({len(event_vars)} periods)")

            except Exception as e:
                print(f"  Error with {outcome}: {e}")
                continue

        self.results["event_study"] = event_results
        return event_results

    def run_callaway_santanna(self) -> dict[str, Any]:
        """
        Run manual Callaway-Sant'Anna estimation.

        Since the 'did' package has dependency issues, we implement
        a simplified version manually using statsmodels.

        Returns:
            Dictionary of Callaway-Sant'Anna results
        """
        print("Running manual Callaway-Sant'Anna analysis...")

        cs_results = {}

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            try:
                # Filter to treated units only (exclude never-treated for simplicity)
                treated_df = self.df[self.df["treatment_year"].notna()].copy()

                if treated_df.empty:
                    print(f"  No treated units for {outcome}")
                    continue

                # Get unique treatment cohorts
                treatment_cohorts = sorted(treated_df["treatment_year"].unique())

                if len(treatment_cohorts) < 2:
                    print(f"  Insufficient treatment variation for {outcome}")
                    continue

                cohort_results = {}

                for cohort in treatment_cohorts:
                    # For each cohort, compare to not-yet-treated units
                    cohort_data = treated_df[
                        (treated_df["treatment_year"] == cohort)
                        | (treated_df["treatment_year"] > cohort)
                    ].copy()

                    if cohort_data.empty:
                        continue

                    # Create treatment indicator for this cohort
                    cohort_data["cohort_treated"] = (
                        (cohort_data["treatment_year"] == cohort) & (cohort_data["year"] >= cohort)
                    ).astype(int)

                    # Create state_id for clustering if not exists
                    if "state_id" not in cohort_data.columns:
                        cohort_data["state_id"] = pd.Categorical(cohort_data["state"]).codes

                    # Run TWFE for this cohort
                    formula = f"{outcome} ~ cohort_treated + C(state) + C(year)"

                    try:
                        # Try clustered standard errors first
                        try:
                            model = smf.ols(formula, data=cohort_data).fit(
                                cov_type="cluster", cov_kwds={"groups": cohort_data["state_id"]}
                            )
                        except:
                            # Fall back to robust standard errors
                            model = smf.ols(formula, data=cohort_data).fit(cov_type="HC0")

                        cohort_results[cohort] = {
                            "att": model.params["cohort_treated"],
                            "se": model.bse["cohort_treated"],
                            "p_value": model.pvalues["cohort_treated"],
                            "n_obs": int(model.nobs),
                        }

                    except Exception as e:
                        print(f"    Error with cohort {cohort}: {e}")
                        continue

                if cohort_results:
                    # Aggregate cohort-specific ATTs
                    atts = [result["att"] for result in cohort_results.values()]
                    ses = [result["se"] for result in cohort_results.values()]

                    # Simple average (could be weighted by group size)
                    overall_att = np.mean(atts)
                    overall_se = np.sqrt(np.mean(np.array(ses) ** 2))  # Simplified SE

                    # Calculate p-value using t-distribution
                    t_stat = overall_att / overall_se if overall_se > 0 else 0
                    overall_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(cohort_results) - 1))

                    cs_results[outcome] = {
                        "overall_att": overall_att,
                        "overall_se": overall_se,
                        "overall_pvalue": overall_pvalue,
                        "cohort_results": cohort_results,
                        "n_cohorts": len(cohort_results),
                    }

                    print(f"  {outcome}: ATT={overall_att:.4f}, SE={overall_se:.4f}")

            except Exception as e:
                print(f"  Error with {outcome}: {e}")
                continue

        self.results["callaway_santanna"] = cs_results
        return cs_results

    def run_instrumental_variables(self) -> dict[str, Any]:
        """
        Run instrumental variables analysis using court orders.

        Returns:
            Dictionary of IV results
        """
        print("Running instrumental variables analysis...")

        iv_results = {}

        # Check for instrumental variables
        potential_instruments = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ["court", "order", "mandate", "monitor"]):
                potential_instruments.append(col)

        if not potential_instruments:
            print("  No instrumental variables found")
            return iv_results

        instrument = potential_instruments[0]  # Use first available instrument
        print(f"  Using instrument: {instrument}")

        # Check for endogenous variable (spending)
        endogenous_vars = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ["spending", "expenditure", "per_pupil"]):
                endogenous_vars.append(col)

        if not endogenous_vars:
            print("  No endogenous spending variables found")
            return iv_results

        endogenous_var = endogenous_vars[0]  # Use first available
        print(f"  Using endogenous variable: {endogenous_var}")

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            try:
                # Prepare data for IV estimation
                iv_data = self.df[[outcome, endogenous_var, instrument, "state", "year"]].dropna()

                if iv_data.empty:
                    print(f"  No valid data for {outcome}")
                    continue

                # Use linearmodels for IV estimation instead of statsmodels
                try:
                    from linearmodels.iv import IV2SLS

                    # Add constant and fixed effects using formula approach
                    formula = (
                        f"{outcome} ~ 1 + [{endogenous_var} ~ {instrument}] + C(state) + C(year)"
                    )
                    iv_model = IV2SLS.from_formula(formula, data=iv_data).fit(cov_type="robust")

                    # Extract coefficient for endogenous variable
                    coef = iv_model.params[endogenous_var]
                    se = iv_model.std_errors[endogenous_var]
                    pval = iv_model.pvalues[endogenous_var]

                    # First stage for diagnostics
                    first_stage = smf.ols(
                        f"{endogenous_var} ~ {instrument} + C(state) + C(year)", data=iv_data
                    ).fit()
                    first_stage_f = first_stage.fvalue

                    iv_results[outcome] = {
                        "coefficient": coef,
                        "se": se,
                        "p_value": pval,
                        "first_stage_f": first_stage_f,
                        "n_obs": int(len(iv_data)),
                        "instrument": instrument,
                        "endogenous_var": endogenous_var,
                    }

                    print(f"  {outcome}: IV coef={coef:.4f}, F-stat={first_stage_f:.2f}")

                except ImportError:
                    # Fall back to basic 2SLS if linearmodels not available
                    print(f"  Warning: linearmodels not available, using basic 2SLS for {outcome}")

                    # Manual 2SLS: First stage
                    first_stage = smf.ols(
                        f"{endogenous_var} ~ {instrument} + C(state) + C(year)", data=iv_data
                    ).fit()

                    # Get predicted values
                    iv_data_pred = iv_data.copy()
                    iv_data_pred[f"{endogenous_var}_predicted"] = first_stage.fittedvalues

                    # Second stage
                    second_stage = smf.ols(
                        f"{outcome} ~ {endogenous_var}_predicted + C(state) + C(year)",
                        data=iv_data_pred,
                    ).fit(cov_type="HC0")

                    iv_results[outcome] = {
                        "coefficient": second_stage.params[f"{endogenous_var}_predicted"],
                        "se": second_stage.bse[f"{endogenous_var}_predicted"],
                        "p_value": second_stage.pvalues[f"{endogenous_var}_predicted"],
                        "first_stage_f": first_stage.fvalue,
                        "n_obs": int(len(iv_data)),
                        "instrument": instrument,
                        "endogenous_var": endogenous_var,
                    }

                    print(
                        f"  {outcome}: IV coef (manual)={second_stage.params[f'{endogenous_var}_predicted']:.4f}"
                    )

            except Exception as e:
                print(f"  Error with {outcome}: {e}")
                continue

        self.results["instrumental_variables"] = iv_results
        return iv_results

    def run_covid_analysis(self) -> dict[str, Any]:
        """
        Run COVID triple-difference analysis.

        Triple-difference specification:
        Y_st = β₁(Post-treatment) + β₂(COVID) + β₃(Post-treatment × COVID) + αₛ + γₜ + εₛₜ

        Returns:
            Dictionary of COVID analysis results
        """
        print("Running COVID triple-difference analysis...")

        covid_results = {}

        # Create COVID period indicator (2020-2022)
        if "post_covid" not in self.df.columns:
            self.df["post_covid"] = (self.df["year"] >= 2020).astype(int)

        for outcome in self.outcome_vars:
            if outcome not in self.df.columns:
                continue

            try:
                # Triple-difference specification
                formula = f"{outcome} ~ post_treatment + post_covid + post_treatment:post_covid + C(state) + C(year)"

                # Fit model with robust standard errors
                try:
                    model = smf.ols(formula, data=self.df).fit(
                        cov_type="cluster", cov_kwds={"groups": self.df["state_id"]}
                    )
                except:
                    model = smf.ols(formula, data=self.df).fit(cov_type="HC0")

                # Extract key coefficients
                covid_results[outcome] = {
                    "treatment_effect": model.params.get("post_treatment", np.nan),
                    "treatment_se": model.bse.get("post_treatment", np.nan),
                    "treatment_pval": model.pvalues.get("post_treatment", np.nan),
                    "covid_effect": model.params.get("post_covid", np.nan),
                    "covid_se": model.bse.get("post_covid", np.nan),
                    "covid_pval": model.pvalues.get("post_covid", np.nan),
                    "covid_interaction": model.params.get("post_treatment:post_covid", np.nan),
                    "covid_interaction_se": model.bse.get("post_treatment:post_covid", np.nan),
                    "covid_interaction_pval": model.pvalues.get(
                        "post_treatment:post_covid", np.nan
                    ),
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                    "model": model,
                }

                print(
                    f"  {outcome}: COVID interaction={model.params.get('post_treatment:post_covid', np.nan):.4f}"
                )

                # Save detailed results
                covid_results_df = pd.DataFrame(
                    {
                        "coefficient": [
                            covid_results[outcome]["treatment_effect"],
                            covid_results[outcome]["covid_effect"],
                            covid_results[outcome]["covid_interaction"],
                        ],
                        "std_error": [
                            covid_results[outcome]["treatment_se"],
                            covid_results[outcome]["covid_se"],
                            covid_results[outcome]["covid_interaction_se"],
                        ],
                        "p_value": [
                            covid_results[outcome]["treatment_pval"],
                            covid_results[outcome]["covid_pval"],
                            covid_results[outcome]["covid_interaction_pval"],
                        ],
                    },
                    index=["post_treatment", "post_covid", "treatment_x_covid"],
                )

                covid_results_df.to_csv(self.tables_dir / f"covid_ddd_results_{outcome}.csv")

                # Save model summary
                with open(self.tables_dir / f"covid_ddd_model_summary_{outcome}.txt", "w") as f:
                    f.write(str(model.summary()))

            except Exception as e:
                print(f"  Error with {outcome}: {e}")
                continue

        self.results["covid_analysis"] = covid_results
        return covid_results

    def create_results_table(self) -> pd.DataFrame:
        """
        Create main results table with all specifications.

        Returns:
            Results table DataFrame
        """
        print("Creating main results table...")

        if not self.results:
            print("No results available. Run analyses first.")
            return pd.DataFrame()

        # Initialize results table
        table_data = []

        for outcome in self.outcome_vars:
            row = {"outcome": outcome}

            # TWFE results
            if "twfe" in self.results and outcome in self.results["twfe"]:
                twfe = self.results["twfe"][outcome]
                row["twfe_coef"] = twfe["coefficient"]
                row["twfe_se"] = twfe["se"]
                row["twfe_pval"] = twfe["p_value"]
                row["twfe_n"] = twfe["n_obs"]

            # Event study (post-treatment average)
            if "event_study" in self.results and outcome in self.results["event_study"]:
                es = self.results["event_study"][outcome]
                # Average post-treatment effects (t=0 to t=5)
                post_coefs = [es["coefficients"].get(t, np.nan) for t in range(0, 6)]
                valid_coefs = [c for c in post_coefs if not np.isnan(c)]
                if valid_coefs:
                    row["event_avg_coef"] = np.mean(valid_coefs)
                    row["event_n"] = es["n_obs"]

            # Callaway-Sant'Anna
            if "callaway_santanna" in self.results and outcome in self.results["callaway_santanna"]:
                cs = self.results["callaway_santanna"][outcome]
                row["cs_att"] = cs["overall_att"]
                row["cs_se"] = cs["overall_se"]
                row["cs_pval"] = cs["overall_pvalue"]
                row["cs_cohorts"] = cs["n_cohorts"]

            # IV results
            if (
                "instrumental_variables" in self.results
                and outcome in self.results["instrumental_variables"]
            ):
                iv = self.results["instrumental_variables"][outcome]
                row["iv_coef"] = iv["coefficient"]
                row["iv_se"] = iv["se"]
                row["iv_pval"] = iv["p_value"]
                row["iv_fstat"] = iv["first_stage_f"]
                row["iv_n"] = iv["n_obs"]

            table_data.append(row)

        results_df = pd.DataFrame(table_data)

        # Save results
        results_df.to_csv(self.tables_dir / "table2_main_results.csv", index=False)

        # Create LaTeX table
        latex_table = self._format_results_latex(results_df)
        with open(self.tables_dir / "table2_main_results.tex", "w") as f:
            f.write(latex_table)

        print(f"Results table saved to {self.tables_dir}")
        return results_df

    def _format_results_latex(self, df: pd.DataFrame) -> str:
        """Format results table as LaTeX."""
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Main Causal Analysis Results}
\\label{tab:main_results}
\\begin{tabular}{lcccc}
\\toprule
Outcome & TWFE & Event Study & Callaway-Sant'Anna & IV \\\\
\\midrule
"""

        for _, row in df.iterrows():
            outcome_name = row["outcome"].replace("_", "\\_")

            # TWFE column
            twfe_str = "--"
            if pd.notna(row.get("twfe_coef")):
                twfe_str = f"{row['twfe_coef']:.3f}"
                if pd.notna(row.get("twfe_se")):
                    twfe_str += f" ({row['twfe_se']:.3f})"

            # Event study column
            event_str = "--"
            if pd.notna(row.get("event_avg_coef")):
                event_str = f"{row['event_avg_coef']:.3f}"

            # CS column
            cs_str = "--"
            if pd.notna(row.get("cs_att")):
                cs_str = f"{row['cs_att']:.3f}"
                if pd.notna(row.get("cs_se")):
                    cs_str += f" ({row['cs_se']:.3f})"

            # IV column
            iv_str = "--"
            if pd.notna(row.get("iv_coef")):
                iv_str = f"{row['iv_coef']:.3f}"
                if pd.notna(row.get("iv_se")):
                    iv_str += f" ({row['iv_se']:.3f})"

            latex += f"{outcome_name} & {twfe_str} & {event_str} & {cs_str} & {iv_str} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\footnotesize
\\textit{Notes:} Standard errors in parentheses. TWFE = Two-Way Fixed Effects.
Event Study shows average post-treatment effect. CS = Callaway-Sant'Anna.
IV = Instrumental Variables using court orders.
All specifications include state and year fixed effects.
\\end{table}"""

        return latex

    def create_event_study_plots(self) -> dict[str, plt.Figure]:
        """Create event study plots for each outcome."""
        if "event_study" not in self.results:
            print("No event study results to plot")
            return {}

        figures = {}

        for outcome, results in self.results["event_study"].items():
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract coefficients and confidence intervals
            periods = sorted(results["coefficients"].keys())
            coefs = [results["coefficients"][t] for t in periods]
            ses = [results["standard_errors"][t] for t in periods]

            # Remove NaN values
            valid_data = [
                (t, c, s)
                for t, c, s in zip(periods, coefs, ses, strict=False)
                if not (np.isnan(c) or np.isnan(s))
            ]

            if not valid_data:
                continue

            periods_clean, coefs_clean, ses_clean = zip(*valid_data, strict=False)

            # Calculate confidence intervals
            ci_lower = [c - 1.96 * s for c, s in zip(coefs_clean, ses_clean, strict=False)]
            ci_upper = [c + 1.96 * s for c, s in zip(coefs_clean, ses_clean, strict=False)]

            # Plot
            ax.plot(periods_clean, coefs_clean, "o-", linewidth=2, markersize=6)
            ax.fill_between(periods_clean, ci_lower, ci_upper, alpha=0.3)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
            ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.7, label="Treatment")

            ax.set_xlabel("Years Since Treatment")
            ax.set_ylabel("Treatment Effect")
            ax.set_title(f"Event Study: {outcome.replace('_', ' ').title()}")
            ax.grid(True, alpha=0.3)
            ax.legend()

            figures[outcome] = fig
            fig.savefig(
                self.figures_dir / f"event_study_{outcome}.png", dpi=300, bbox_inches="tight"
            )

        return figures

    def run_full_analysis(self) -> str:
        """
        Run complete causal analysis pipeline.

        Returns:
            Path to analysis report
        """
        print("Phase 4.2: Running Main Causal Analysis")
        print("=" * 50)

        # Load data
        if self.df is None:
            self.load_data()

        # Run all specifications
        self.run_twfe_analysis()
        self.run_event_study()
        self.run_callaway_santanna()
        self.run_instrumental_variables()
        self.run_covid_analysis()  # Add COVID analysis to pipeline

        # Create outputs
        self.create_results_table()
        self.create_event_study_plots()

        # Generate report
        report_path = self.reports_dir / "causal_analysis_report.md"

        with open(report_path, "w") as f:
            f.write(f"""# Phase 4.2: Main Causal Analysis Report

Generated by: Jeff Chen (jeffreyc1@alumni.cmu.edu)
Created in collaboration with Claude Code
Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Analysis Overview

- **Dataset**: {self.data_path}
- **Specifications**: {len(self.results)} econometric models
- **Outcomes**: {len(self.outcome_vars)} dependent variables
- **Treatment States**: {self.df["post_treatment"].sum() if self.df is not None else "N/A"}

## Model Specifications

1. **Two-Way Fixed Effects (TWFE)**: Basic DiD with state and year fixed effects
2. **Event Study**: Dynamic treatment effects from -5 to +5 years
3. **Callaway-Sant'Anna**: Manual implementation for staggered adoption
4. **Instrumental Variables**: Using court orders as instruments
5. **COVID Triple-Difference**: Policy resilience during pandemic disruption

## Key Results

{self._summarize_results()}

## Files Generated

1. **Tables**:
   - `table2_main_results.csv` - Main results in CSV format
   - `table2_main_results.tex` - LaTeX formatted table
   - `covid_ddd_results_*.csv` - COVID analysis results by outcome
   - `covid_ddd_model_summary_*.txt` - COVID model summaries

2. **Figures**:
   - Event study plots for each outcome variable

## Next Steps

Proceed to Phase 4.3: Robustness and Extensions
""")

        print("\n✅ Causal analysis completed successfully!")
        print(f"📊 Report available at: {report_path}")

        return str(report_path)

    def _summarize_results(self) -> str:
        """Summarize key findings from all specifications."""
        if not self.results:
            return "No results available."

        summary = []

        for spec, spec_results in self.results.items():
            if not spec_results:
                continue

            spec_name = spec.replace("_", " ").title()
            n_outcomes = len(spec_results)

            # Count significant results (p < 0.05)
            sig_results = 0
            for outcome_results in spec_results.values():
                if isinstance(outcome_results, dict):
                    pval_key = "p_value" if "p_value" in outcome_results else "overall_pvalue"
                    if pval_key in outcome_results and outcome_results[pval_key] < 0.05:
                        sig_results += 1

            summary.append(
                f"- **{spec_name}**: {sig_results}/{n_outcomes} outcomes significant at 5% level"
            )

        return "\n".join(summary) if summary else "No specification results to summarize."


def main():
    """Run main causal analysis pipeline."""
    try:
        analyzer = CausalAnalyzer()
        analyzer.run_full_analysis()
        return True

    except Exception as e:
        print(f"\n❌ Error in causal analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
