"""
Simplified Instrumental Variables Analysis

Focuses on the core IV framework with robust estimation
using basic 2SLS implementation and instrument validation.

Author: Research Team
Date: 2025-08-12
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.append(str(Path(__file__).parent))


class SimpleIVAnalyzer:
    """Simplified IV analysis for policy evaluation."""

    def __init__(self, data_path: str = "data/final/analysis_panel.csv"):
        """Initialize analyzer."""
        self.data_path = Path(data_path)
        self.data = self._load_data()
        self.outcomes = [
            "math_grade4_gap",
            "math_grade8_gap",
            "reading_grade4_gap",
            "reading_grade8_gap",
        ]

        print("SimpleIVAnalyzer initialized:")
        print(f"  Data: {len(self.data)} observations")
        print(f"  Court orders: {(self.data['court_ordered'] == 1).sum()} obs")
        print(f"  Federal monitoring: {(self.data['under_monitoring'] == 1).sum()} obs")

    def _load_data(self) -> pd.DataFrame:
        """Load analysis data."""
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def manual_2sls_estimation(self, outcome: str) -> dict[str, float]:
        """Manual 2SLS estimation for transparency."""
        print(f"\nEstimating 2SLS for {outcome}...")

        # Prepare data - use complete cases only
        analysis_vars = [outcome, "post_treatment", "court_ordered", "under_monitoring"]
        if "log_total_revenue_per_pupil" in self.data.columns:
            analysis_vars.append("log_total_revenue_per_pupil")

        analysis_data = self.data[analysis_vars].dropna()

        if analysis_data.empty or len(analysis_data) < 20:
            print(f"  Insufficient data: {len(analysis_data)} observations")
            return {}

        print(f"  Sample size: {len(analysis_data)} observations")

        # Stage 1: Regress endogenous variable on instruments
        y1 = analysis_data["post_treatment"]
        X1 = analysis_data[["court_ordered", "under_monitoring"]]

        if "log_total_revenue_per_pupil" in analysis_data.columns:
            X1 = pd.concat([X1, analysis_data[["log_total_revenue_per_pupil"]]], axis=1)

        X1 = sm.add_constant(X1)

        first_stage = sm.OLS(y1, X1).fit()

        # Get predicted treatment
        treatment_predicted = first_stage.predict(X1)

        # Stage 2: Regress outcome on predicted treatment
        y2 = analysis_data[outcome]
        X2 = pd.DataFrame({"post_treatment_predicted": treatment_predicted})

        if "log_total_revenue_per_pupil" in analysis_data.columns:
            X2 = pd.concat([X2, analysis_data[["log_total_revenue_per_pupil"]]], axis=1)

        X2 = sm.add_constant(X2)

        second_stage = sm.OLS(y2, X2).fit()

        # Extract results
        iv_coeff = second_stage.params["post_treatment_predicted"]

        # First-stage F-test for instruments
        restriction_matrix = np.zeros((2, X1.shape[1]))
        restriction_matrix[0, 1] = 1  # court_ordered
        restriction_matrix[1, 2] = 1  # under_monitoring

        f_test = first_stage.f_test(restriction_matrix)
        f_stat = f_test.fvalue
        f_pvalue = f_test.pvalue

        # Calculate robust standard error (simplified)
        iv_se = second_stage.bse["post_treatment_predicted"] * np.sqrt(
            len(analysis_data) / (len(analysis_data) - X2.shape[1])
        )

        results = {
            "outcome": outcome,
            "iv_coefficient": iv_coeff,
            "iv_std_error": iv_se,
            "first_stage_f": f_stat,
            "first_stage_p": f_pvalue,
            "n_obs": len(analysis_data),
            "weak_instruments": f_stat < 10,
        }

        print(f"  IV coefficient: {iv_coeff:.3f} (SE: {iv_se:.3f})")
        print(f"  First-stage F: {f_stat:.1f} (p={f_pvalue:.3f})")
        print(f"  Weak instruments: {'Yes' if f_stat < 10 else 'No'}")

        return results

    def run_simple_iv_analysis(self) -> dict[str, dict]:
        """Run simplified IV analysis for all outcomes."""
        print("=" * 50)
        print("SIMPLIFIED IV ANALYSIS")
        print("=" * 50)

        results = {}

        for outcome in self.outcomes:
            try:
                iv_result = self.manual_2sls_estimation(outcome)
                if iv_result:
                    results[outcome] = iv_result
            except Exception as e:
                print(f"Error analyzing {outcome}: {e}")
                continue

        # Save results
        if results:
            results_list = []
            for _outcome, result in results.items():
                results_list.append(result)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv("output/tables/simple_iv_results.csv", index=False)
            print("\nSimple IV results saved to: output/tables/simple_iv_results.csv")

        print(f"\n{'=' * 50}")
        print("SIMPLE IV ANALYSIS SUMMARY")
        print(f"{'=' * 50}")

        for outcome, result in results.items():
            iv_coeff = result["iv_coefficient"]
            f_stat = result["first_stage_f"]
            weak = result["weak_instruments"]

            print(f"{outcome}: IV={iv_coeff:.3f}, F={f_stat:.1f} {'(weak)' if weak else ''}")

        return results


if __name__ == "__main__":
    analyzer = SimpleIVAnalyzer()
    results = analyzer.run_simple_iv_analysis()
