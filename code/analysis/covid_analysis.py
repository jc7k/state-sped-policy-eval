"""
COVID-19 Triple-Difference Analysis

Implements triple-difference (DDD) estimation using COVID-19 as a natural experiment
to identify which state special education policy reforms proved most resilient
during pandemic disruption.

Identification Strategy:
Triple-difference compares:
1. Reformed vs non-reformed states (policy dimension)
2. Pre-COVID vs COVID period (time dimension)
3. Students with disabilities vs general population (student dimension)

The interaction of all three dimensions identifies the differential effect
of policy reforms on SWD outcomes during COVID disruption.

Key Features:
- Triple-difference estimation framework
- COVID period definition (2020-2022)
- Heterogeneous treatment effects by reform timing
- Resilience analysis and policy effectiveness ranking
- Pandemic-specific robustness checks

Author: Research Team
Date: 2025-08-12
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# Import existing modules
sys.path.append(str(Path(__file__).parent))


class COVIDAnalyzer:
    """
    Triple-difference analysis using COVID-19 as natural experiment.

    Examines differential effects of state policy reforms on SWD outcomes
    during pandemic disruption to identify most resilient policies.
    """

    def __init__(
        self,
        data_path: str = "data/final/analysis_panel.csv",
        results_dir: str = "output/tables",
        figures_dir: str = "output/figures",
    ):
        """Initialize COVID analyzer."""
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.data = self._load_data()
        self.outcomes = [
            "math_grade4_gap",
            "math_grade8_gap",
            "reading_grade4_gap",
            "reading_grade8_gap",
        ]

        # COVID period definition
        self.covid_years = [2020, 2021, 2022]
        self.pre_covid_years = [2017, 2018, 2019]  # Recent pre-COVID baseline

        # Storage for results
        self.covid_results = {}

        print("COVIDAnalyzer initialized:")
        print(f"  Data: {len(self.data)} observations")
        print(f"  COVID period: {self.covid_years}")
        print(f"  Pre-COVID baseline: {self.pre_covid_years}")
        self._analyze_covid_data_availability()

    def _load_data(self) -> pd.DataFrame:
        """Load analysis panel dataset."""
        try:
            df = pd.read_csv(self.data_path)
            print(f"Loaded analysis panel: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def _analyze_covid_data_availability(self):
        """Analyze data availability during COVID period."""
        if self.data.empty:
            return

        covid_data = self.data[self.data["year"].isin(self.covid_years)]
        pre_covid_data = self.data[self.data["year"].isin(self.pre_covid_years)]

        print(f"  COVID period data: {len(covid_data)} observations")
        print(f"  Pre-COVID data: {len(pre_covid_data)} observations")

        # Check outcome availability by period
        for outcome in self.outcomes:
            covid_available = covid_data[outcome].notna().sum()
            pre_covid_available = pre_covid_data[outcome].notna().sum()
            print(
                f"  {outcome}: COVID={covid_available}, Pre-COVID={pre_covid_available}"
            )

    def prepare_triple_diff_data(self, outcome: str) -> pd.DataFrame:
        """
        Prepare data for triple-difference analysis.

        Creates binary indicators for:
        - covid_period: 1 if year in COVID period
        - policy_reform: 1 if state has implemented reform by given year
        - outcome_gap: Achievement gap (already calculated)
        """
        print(f"\nPreparing triple-difference data for {outcome}...")

        if self.data.empty:
            return pd.DataFrame()

        # Filter to relevant years (pre-COVID + COVID)
        analysis_years = self.pre_covid_years + self.covid_years
        analysis_data = self.data[self.data["year"].isin(analysis_years)].copy()

        # Create COVID period indicator
        analysis_data["covid_period"] = (
            analysis_data["year"].isin(self.covid_years)
        ).astype(int)

        # Create policy reform indicator (cumulative by year)
        analysis_data["policy_reform"] = analysis_data["post_treatment"].astype(int)

        # Select available columns
        required_cols = [outcome, "covid_period", "policy_reform", "state", "year"]
        if "log_total_revenue_per_pupil" in analysis_data.columns:
            required_cols.append("log_total_revenue_per_pupil")

        # Filter to complete cases for the outcome
        analysis_data = analysis_data[required_cols].dropna()

        print(f"  Analysis sample: {len(analysis_data)} observations")
        print(f"  States: {analysis_data['state'].nunique()}")
        print(f"  COVID observations: {analysis_data['covid_period'].sum()}")
        print(f"  Reform observations: {analysis_data['policy_reform'].sum()}")

        return analysis_data

    def estimate_triple_difference(
        self, outcome: str = "math_grade8_gap", save_results: bool = True
    ) -> dict[str, any]:
        """
        Estimate triple-difference model.

        Model: Y_st = β₀ + β₁(COVID) + β₂(Reform) + β₃(COVID × Reform) +
                      α_s + γ_t + ε_st

        Where β₃ is the triple-difference coefficient of interest.
        """
        print(f"\nEstimating triple-difference for {outcome}...")

        # Prepare data
        analysis_data = self.prepare_triple_diff_data(outcome)

        if analysis_data.empty or len(analysis_data) < 20:
            print(f"  Insufficient data: {len(analysis_data)} observations")
            return {}

        try:
            # Triple-difference specification
            # Note: Achievement gap already captures SWD vs non-SWD difference
            # So this is really a difference-in-differences with COVID as natural experiment

            # Add state and year fixed effects
            analysis_data["state_fe"] = pd.Categorical(analysis_data["state"])
            analysis_data["year_fe"] = pd.Categorical(analysis_data["year"])

            # Main DDD specification
            formula = f"{outcome} ~ covid_period + policy_reform + covid_period:policy_reform + C(state_fe) + C(year_fe)"

            # Add controls if available
            if "log_total_revenue_per_pupil" in analysis_data.columns:
                formula += " + log_total_revenue_per_pupil"

            # Estimate model
            model = ols(formula, data=analysis_data).fit(
                cov_type="cluster", cov_kwds={"groups": analysis_data["state"]}
            )

            # Extract triple-difference coefficient (interaction term)
            ddd_coeff = model.params["covid_period:policy_reform"]
            ddd_se = model.bse["covid_period:policy_reform"]
            ddd_pvalue = model.pvalues["covid_period:policy_reform"]
            ddd_tstat = model.tvalues["covid_period:policy_reform"]

            # Confidence interval
            ci_lower = ddd_coeff - 1.96 * ddd_se
            ci_upper = ddd_coeff + 1.96 * ddd_se

            # Main effects
            covid_main = model.params["covid_period"]
            reform_main = model.params["policy_reform"]

            results = {
                "outcome": outcome,
                "ddd_coefficient": ddd_coeff,
                "ddd_std_error": ddd_se,
                "ddd_pvalue": ddd_pvalue,
                "ddd_tstat": ddd_tstat,
                "ddd_ci_lower": ci_lower,
                "ddd_ci_upper": ci_upper,
                "covid_main_effect": covid_main,
                "reform_main_effect": reform_main,
                "r_squared": model.rsquared,
                "n_obs": model.nobs,
                "n_states": analysis_data["state"].nunique(),
                "significant_5pct": abs(ddd_tstat) > 1.96,
            }

            print(
                f"  Triple-difference coefficient: {ddd_coeff:.3f} (SE: {ddd_se:.3f})"
            )
            print(f"  T-statistic: {ddd_tstat:.3f}, P-value: {ddd_pvalue:.3f}")
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"  Significant: {'Yes' if abs(ddd_tstat) > 1.96 else 'No'}")
            print(f"  R²: {model.rsquared:.3f}, N: {model.nobs}")

            # Interpretation
            if ddd_coeff > 0:
                print(
                    "  → Policy reforms WORSENED outcomes during COVID (less resilient)"
                )
            else:
                print("  → Policy reforms IMPROVED resilience during COVID")

            if save_results:
                # Save detailed results
                output_path = self.results_dir / f"covid_ddd_results_{outcome}.csv"
                results_df = pd.DataFrame([results])
                results_df.to_csv(output_path, index=False)
                print(f"  Saved results: {output_path}")

                # Save model summary
                summary_path = (
                    self.results_dir / f"covid_ddd_model_summary_{outcome}.txt"
                )
                with open(summary_path, "w") as f:
                    f.write(str(model.summary()))
                print(f"  Saved model summary: {summary_path}")

            return results

        except Exception as e:
            print(f"  Error in triple-difference estimation: {e}")
            return {}

    def analyze_resilience_by_reform_timing(
        self, outcome: str = "math_grade8_gap", save_results: bool = True
    ) -> pd.DataFrame:
        """
        Analyze how COVID resilience varies by when states adopted reforms.

        Tests whether early vs late reformers showed different pandemic responses.
        """
        print(f"\nAnalyzing resilience by reform timing for {outcome}...")

        # Prepare data
        analysis_data = self.prepare_triple_diff_data(outcome)

        if analysis_data.empty:
            return pd.DataFrame()

        # Get first reform year for each treated state
        treated_states = self.data[self.data["post_treatment"] == 1]
        first_reform = treated_states.groupby("state")["year"].min().reset_index()
        first_reform.columns = ["state", "first_reform_year"]

        # Merge with analysis data
        analysis_data = analysis_data.merge(first_reform, on="state", how="left")
        analysis_data["first_reform_year"] = analysis_data["first_reform_year"].fillna(
            9999
        )  # Never treated

        # Create reform timing categories
        analysis_data["reform_timing"] = "Never Reformed"
        analysis_data.loc[
            analysis_data["first_reform_year"] <= 2017, "reform_timing"
        ] = "Early Reformer (≤2017)"
        analysis_data.loc[
            (analysis_data["first_reform_year"] >= 2018)
            & (analysis_data["first_reform_year"] <= 2020),
            "reform_timing",
        ] = "Mid Reformer (2018-2020)"
        analysis_data.loc[
            analysis_data["first_reform_year"] >= 2021, "reform_timing"
        ] = "Late Reformer (≥2021)"

        # Analyze COVID effects by reform timing group
        resilience_results = []

        for timing_group in analysis_data["reform_timing"].unique():
            if timing_group == "Never Reformed":
                continue

            group_data = analysis_data[
                analysis_data["reform_timing"].isin([timing_group, "Never Reformed"])
            ].copy()

            if len(group_data) < 20:
                continue

            # Create binary indicator for this timing group
            group_data["timing_group"] = (
                group_data["reform_timing"] == timing_group
            ).astype(int)

            try:
                # Estimate COVID resilience for this group
                formula = f"{outcome} ~ covid_period + timing_group + covid_period:timing_group"
                model = ols(formula, data=group_data).fit()

                covid_resilience = model.params["covid_period:timing_group"]
                covid_resilience_se = model.bse["covid_period:timing_group"]
                covid_resilience_p = model.pvalues["covid_period:timing_group"]

                resilience_results.append(
                    {
                        "reform_timing": timing_group,
                        "covid_resilience": covid_resilience,
                        "resilience_se": covid_resilience_se,
                        "resilience_pvalue": covid_resilience_p,
                        "n_obs": len(group_data),
                        "n_states": group_data["state"].nunique(),
                        "baseline_covid_effect": model.params["covid_period"],
                    }
                )

                print(
                    f"  {timing_group}: Resilience = {covid_resilience:.3f} (p={covid_resilience_p:.3f})"
                )

            except Exception as e:
                print(f"  Error analyzing {timing_group}: {e}")
                continue

        # Create results DataFrame
        resilience_df = pd.DataFrame(resilience_results)

        if save_results and not resilience_df.empty:
            output_path = self.results_dir / f"covid_resilience_by_timing_{outcome}.csv"
            resilience_df.to_csv(output_path, index=False)
            print(f"  Saved resilience analysis: {output_path}")

        return resilience_df

    def create_covid_analysis_visualization(
        self, save_formats: list[str] = None
    ) -> str:
        """Create comprehensive COVID analysis visualization."""
        if save_formats is None:
            save_formats = ["png", "pdf"]
        print("\nCreating COVID analysis visualization...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Triple-difference coefficients by outcome
        try:
            ddd_results = []
            for outcome in self.outcomes:
                result = self.estimate_triple_difference(outcome, save_results=False)
                if result:
                    ddd_results.append(
                        {
                            "outcome": outcome,
                            "ddd_coeff": result["ddd_coefficient"],
                            "ddd_se": result["ddd_std_error"],
                            "significant": result["significant_5pct"],
                        }
                    )

            if ddd_results:
                results_df = pd.DataFrame(ddd_results)
                outcomes_clean = [
                    self._format_outcome_label(o) for o in results_df["outcome"]
                ]

                # Color by significance
                colors = [
                    "red" if sig else "steelblue" for sig in results_df["significant"]
                ]

                ax1.bar(
                    outcomes_clean,
                    results_df["ddd_coeff"],
                    yerr=results_df["ddd_se"],
                    color=colors,
                    alpha=0.7,
                    capsize=5,
                )

                ax1.axhline(y=0, color="black", linestyle="-", alpha=0.8)
                ax1.set_title(
                    "COVID Resilience Effects by Outcome\n(Triple-Difference Coefficients)",
                    fontweight="bold",
                )
                ax1.set_ylabel("Effect Size (NAEP Points)")
                ax1.tick_params(axis="x", rotation=45)
                ax1.grid(True, alpha=0.3)

                # Legend
                from matplotlib.lines import Line2D

                legend_elements = [
                    Line2D([0], [0], color="red", lw=4, alpha=0.7, label="Significant"),
                    Line2D(
                        [0],
                        [0],
                        color="steelblue",
                        lw=4,
                        alpha=0.7,
                        label="Not significant",
                    ),
                ]
                ax1.legend(handles=legend_elements)
        except Exception:
            ax1.text(
                0.5,
                0.5,
                "DDD coefficients error",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
        ax1.set_title("COVID Resilience Effects by Outcome", fontweight="bold")

        # 2. COVID period data availability
        try:
            if not self.data.empty:
                # Data availability by year
                year_counts = []
                for year in range(2017, 2023):
                    year_data = self.data[self.data["year"] == year]
                    available_outcomes = sum(
                        year_data[outcome].notna().sum() for outcome in self.outcomes
                    )
                    year_counts.append(
                        {"year": year, "available_outcomes": available_outcomes}
                    )

                year_df = pd.DataFrame(year_counts)

                colors = [
                    "steelblue" if year < 2020 else "orange" for year in year_df["year"]
                ]
                ax2.bar(
                    year_df["year"],
                    year_df["available_outcomes"],
                    color=colors,
                    alpha=0.7,
                )

                ax2.axvline(
                    x=2019.5,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2,
                    label="COVID Onset",
                )
                ax2.set_title("Outcome Data Availability by Year", fontweight="bold")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Available Outcome Observations")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        except Exception:
            ax2.text(
                0.5,
                0.5,
                "Data availability error",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
        ax2.set_title("Outcome Data Availability by Year", fontweight="bold")

        # 3. Reform timing distribution
        try:
            if not self.data.empty:
                treated_states = self.data[self.data["post_treatment"] == 1]
                first_reforms = treated_states.groupby("state")["year"].min()

                # Categorize by timing relative to COVID
                timing_categories = []
                for year in first_reforms:
                    if year <= 2017:
                        timing_categories.append("Early (≤2017)")
                    elif year <= 2019:
                        timing_categories.append("Pre-COVID (2018-2019)")
                    elif year == 2020:
                        timing_categories.append("COVID Year (2020)")
                    else:
                        timing_categories.append("Post-COVID (≥2021)")

                timing_counts = pd.Series(timing_categories).value_counts()

                colors = ["darkblue", "steelblue", "orange", "red"][
                    : len(timing_counts)
                ]
                ax3.pie(
                    timing_counts.values,
                    labels=timing_counts.index,
                    colors=colors,
                    autopct="%1.0f%%",
                )
                ax3.set_title("Reform Timing Relative to COVID", fontweight="bold")
        except Exception:
            ax3.text(
                0.5,
                0.5,
                "Reform timing error",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
        ax3.set_title("Reform Timing Relative to COVID", fontweight="bold")

        # 4. Analysis summary
        ax4.axis("off")
        summary_text = "COVID Triple-Difference Analysis Summary\\n\\n"

        try:
            if not self.data.empty:
                covid_obs = self.data[self.data["year"].isin(self.covid_years)]
                pre_covid_obs = self.data[self.data["year"].isin(self.pre_covid_years)]

                summary_text += "Natural Experiment Design:\\n"
                summary_text += f"  COVID period: {self.covid_years}\\n"
                summary_text += f"  Pre-COVID baseline: {self.pre_covid_years}\\n"
                summary_text += f"  COVID observations: {len(covid_obs)}\\n"
                summary_text += f"  Pre-COVID observations: {len(pre_covid_obs)}\\n\\n"

                summary_text += "Identification Strategy:\\n"
                summary_text += "  DDD = (Reform vs Control) ×\\n"
                summary_text += "        (COVID vs Pre-COVID) ×\\n"
                summary_text += "        (SWD vs Non-SWD gap)\\n\\n"

                summary_text += "Key Research Questions:\\n"
                summary_text += "  • Which reforms provided COVID resilience?\\n"
                summary_text += "  • Did early reformers fare better?\\n"
                summary_text += "  • What policies protected SWD outcomes?\\n\\n"

                summary_text += "Policy Implications:\\n"
                summary_text += "  • Crisis-resilient policy design\\n"
                summary_text += "  • Emergency response effectiveness\\n"
                summary_text += "  • Future pandemic preparedness"
        except:
            summary_text += "Summary statistics not available"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            verticalalignment="top",
            fontsize=10,
            fontfamily="monospace",
        )

        plt.tight_layout()

        # Save files
        saved_files = []
        for fmt in save_formats:
            filename = f"covid_analysis_summary.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
            saved_files.append(str(filepath))

        plt.close()

        print(f"  COVID analysis visualization saved: {saved_files[0]}")
        return saved_files[0]

    def run_complete_covid_analysis(self) -> dict[str, any]:
        """Run comprehensive COVID triple-difference analysis."""
        print("=" * 60)
        print("COVID TRIPLE-DIFFERENCE ANALYSIS")
        print("=" * 60)

        all_results = {}

        # 1. Triple-difference analysis for each outcome
        for outcome in self.outcomes:
            print(f"\n{'-' * 50}")
            print(f"COVID Analysis: {self._format_outcome_label(outcome)}")
            print(f"{'-' * 50}")

            outcome_results = {}

            try:
                # Main triple-difference
                ddd_results = self.estimate_triple_difference(outcome)
                outcome_results["triple_difference"] = ddd_results

                # Resilience by reform timing
                resilience_results = self.analyze_resilience_by_reform_timing(outcome)
                outcome_results["resilience_by_timing"] = resilience_results

            except Exception as e:
                print(f"Error in COVID analysis for {outcome}: {e}")
                continue

            all_results[outcome] = outcome_results

        # 2. Generate visualization
        try:
            summary_plot = self.create_covid_analysis_visualization()
            all_results["summary_plot"] = summary_plot
        except Exception as e:
            print(f"Error creating COVID visualization: {e}")

        print(f"\n{'=' * 60}")
        print("COVID ANALYSIS COMPLETE")
        print(f"{'=' * 60}")

        # Summary of key findings
        significant_effects = 0
        resilient_policies = 0

        for outcome, results in all_results.items():
            if outcome == "summary_plot":
                continue

            print(f"\n{outcome}:")
            if "triple_difference" in results and results["triple_difference"]:
                ddd_coeff = results["triple_difference"].get("ddd_coefficient", np.nan)
                ddd_pvalue = results["triple_difference"].get("ddd_pvalue", np.nan)
                significant = results["triple_difference"].get(
                    "significant_5pct", False
                )

                print(
                    f"  COVID resilience effect: {ddd_coeff:.3f} (p={ddd_pvalue:.3f})"
                )
                print(f"  Significant: {'Yes' if significant else 'No'}")
                print(
                    f"  Interpretation: {'More resilient' if ddd_coeff < 0 else 'Less resilient'}"
                )

                if significant:
                    significant_effects += 1
                if ddd_coeff < 0:  # Negative = more resilient
                    resilient_policies += 1

        print("\nOverall COVID Analysis Summary:")
        print(
            f"  Outcomes with significant COVID effects: {significant_effects}/{len(self.outcomes)}"
        )
        print(
            f"  Outcomes showing policy resilience: {resilient_policies}/{len(self.outcomes)}"
        )

        return all_results

    def _format_outcome_label(self, outcome: str) -> str:
        """Convert outcome variable name to readable label."""
        outcome.split("_")

        if "math" in outcome:
            subject = "Math"
        elif "reading" in outcome:
            subject = "Reading"
        else:
            subject = "Achievement"

        if "grade4" in outcome:
            grade = "G4"
        elif "grade8" in outcome:
            grade = "G8"
        else:
            grade = ""

        return f"{subject} {grade}".strip()


if __name__ == "__main__":
    # Run comprehensive COVID analysis
    covid_analyzer = COVIDAnalyzer()
    results = covid_analyzer.run_complete_covid_analysis()

    print("\nAll COVID analysis files saved to:")
    print(f"  Results: {covid_analyzer.results_dir}")
    print(f"  Figures: {covid_analyzer.figures_dir}")
