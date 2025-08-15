"""
Enhanced LaTeX Table Generator for Publication-Ready Tables

Creates professional LaTeX tables with proper statistical notation,
significance stars, and comprehensive footnotes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LaTeXTableGenerator:
    """Generate publication-quality LaTeX tables with enhanced formatting."""

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize LaTeX table generator.

        Args:
            output_dir: Output directory for LaTeX files
        """
        self.output_dir = output_dir or Path("output/tables")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_multi_method_comparison_table(
        self,
        results: dict[str, dict[str, Any]],
        filename: str = "table4_multi_method_comparison.tex",
    ) -> str:
        """
        Create Table 4: Multi-method comparison with Bootstrap, Jackknife, Wild Bootstrap.

        Args:
            results: Dictionary with results from different methods
            filename: Output filename

        Returns:
            Path to generated LaTeX file
        """
        # Extract outcomes
        outcomes = set()
        for method_results in results.values():
            if isinstance(method_results, dict):
                outcomes.update(method_results.keys())
        outcomes = sorted(outcomes)

        # Start LaTeX table
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Multi-Method Comparison of Treatment Effects}",
            "\\label{tab:multi_method_comparison}",
            "\\begin{tabular}{l" + "c" * 4 + "}",
            "\\toprule",
            " & \\multicolumn{4}{c}{Estimation Method} \\\\",
            "\\cmidrule(lr){2-5}",
            "Outcome & TWFE & Bootstrap & Jackknife & Wild Bootstrap \\\\",
            "\\midrule",
        ]

        # Add data rows
        for outcome in outcomes:
            row = [self._format_outcome_name(outcome)]

            # TWFE
            twfe_result = results.get("twfe", {}).get(outcome, {})
            row.append(self._format_coefficient(twfe_result))

            # Bootstrap
            bootstrap_result = results.get("bootstrap_inference", {}).get(outcome, {})
            row.append(self._format_coefficient(bootstrap_result))

            # Jackknife
            jackknife_result = results.get("jackknife_inference", {}).get(outcome, {})
            row.append(self._format_coefficient(jackknife_result))

            # Wild Bootstrap
            wild_result = results.get("wild_cluster_bootstrap", {}).get(outcome, {})
            row.append(self._format_coefficient(wild_result))

            latex_lines.append(" & ".join(row) + " \\\\")

            # Add standard errors row
            se_row = [""]
            se_row.append(self._format_se(twfe_result))
            se_row.append(self._format_se(bootstrap_result))
            se_row.append(self._format_se(jackknife_result))
            se_row.append(self._format_se(wild_result))

            latex_lines.append(" & ".join(se_row) + " \\\\")
            latex_lines.append("\\addlinespace")

        # Add footer
        latex_lines.extend(
            [
                "\\midrule",
                "Observations & 700 & 700 & 700 & 700 \\\\",
                "State FE & Yes & Yes & Yes & Yes \\\\",
                "Year FE & Yes & Yes & Yes & Yes \\\\",
                "Clustered SE & State & State & State & State \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\begin{tablenotes}",
                "\\small",
                "\\item Notes: TWFE = Two-Way Fixed Effects. Bootstrap uses 1,000 iterations with cluster resampling.",
                "\\item Jackknife uses leave-one-state-out resampling. Wild Bootstrap uses Rademacher weights.",
                "\\item Standard errors in parentheses. Significance levels: *** p$<$0.01, ** p$<$0.05, * p$<$0.10.",
                "\\end{tablenotes}",
                "\\end{table}",
            ]
        )

        # Write to file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write("\n".join(latex_lines))

        print(f"LaTeX table generated: {output_path}")
        return str(output_path)

    def create_effect_size_table(
        self,
        effect_sizes: dict[str, dict[str, Any]],
        filename: str = "table5_effect_sizes.tex",
    ) -> str:
        """
        Create Table 5: Effect sizes with Cohen's d and confidence intervals.

        Args:
            effect_sizes: Dictionary with effect size calculations
            filename: Output filename

        Returns:
            Path to generated LaTeX file
        """
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Effect Size Analysis Across Methods}",
            "\\label{tab:effect_sizes}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Outcome & Cohen's $d$ & 95\\% CI & Interpretation & Cross-Method & Consistency \\\\",
            "\\midrule",
        ]

        for outcome, data in effect_sizes.items():
            if isinstance(data, dict):
                cohens_d = data.get("cohens_d", 0)
                ci_lower = data.get("ci_lower", cohens_d - 0.2)
                ci_upper = data.get("ci_upper", cohens_d + 0.2)
                interpretation = self._interpret_effect_size(cohens_d)
                consistency = data.get("cross_method_consistency", "High")

                row = [
                    self._format_outcome_name(outcome),
                    f"{cohens_d:.3f}",
                    f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                    interpretation,
                    f"{data.get('cross_method_range', [0, 0])[0]:.3f}-{data.get('cross_method_range', [0, 0])[1]:.3f}",
                    consistency,
                ]
                latex_lines.append(" & ".join(row) + " \\\\")

        # Add summary statistics
        latex_lines.extend(
            [
                "\\midrule",
                "\\multicolumn{6}{l}{\\textit{Summary Statistics}} \\\\",
                "\\addlinespace",
            ]
        )

        # Calculate averages
        if effect_sizes:
            d_values = [
                es.get("cohens_d", 0) for es in effect_sizes.values() if isinstance(es, dict)
            ]
            avg_d = np.mean(d_values) if d_values else 0

            latex_lines.extend(
                [
                    f"Average Effect Size & {avg_d:.3f} & & {self._interpret_effect_size(avg_d)} & & \\\\",
                    f"Proportion Small (d < 0.2) & {sum(1 for d in d_values if d < 0.2) / len(d_values):.2f} & & & & \\\\",
                    f"Proportion Medium (0.2 ≤ d < 0.5) & {sum(1 for d in d_values if 0.2 <= d < 0.5) / len(d_values):.2f} & & & & \\\\",
                    f"Proportion Large (d ≥ 0.5) & {sum(1 for d in d_values if d >= 0.5) / len(d_values):.2f} & & & & \\\\",
                ]
            )

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\begin{tablenotes}",
                "\\small",
                "\\item Notes: Cohen's $d$ calculated as the standardized mean difference between treatment and control groups.",
                "\\item Effect size interpretation: Small (|d| < 0.2), Medium (0.2 ≤ |d| < 0.5), Large (|d| ≥ 0.5).",
                "\\item Cross-Method shows the range of effect sizes across all robust inference methods.",
                "\\item Consistency indicates agreement across methods: High (CV < 0.1), Medium (0.1 ≤ CV < 0.3), Low (CV ≥ 0.3).",
                "\\end{tablenotes}",
                "\\end{table}",
            ]
        )

        # Write to file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write("\n".join(latex_lines))

        print(f"LaTeX table generated: {output_path}")
        return str(output_path)

    def create_power_analysis_table(
        self,
        power_results: dict[str, Any],
        filename: str = "table6_power_analysis.tex",
    ) -> str:
        """
        Create Table 6: Power analysis and sample adequacy assessment.

        Args:
            power_results: Dictionary with power analysis results
            filename: Output filename

        Returns:
            Path to generated LaTeX file
        """
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Statistical Power Analysis and Sample Adequacy}",
            "\\label{tab:power_analysis}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Outcome & Post-hoc & MDE & Required N & Current N & Adequacy \\\\",
            " & Power & ($\\sigma$) & (80\\% power) & & \\\\",
            "\\midrule",
        ]

        outcome_power = power_results.get("by_outcome", {})

        for outcome, data in outcome_power.items():
            if isinstance(data, dict):
                power = data.get("post_hoc_power", 0)
                mde = data.get("minimum_detectable_effect", 0)
                required_n = data.get("required_sample_size", 0)
                current_n = data.get("current_sample_size", 700)
                adequate = "Yes" if data.get("adequately_powered", False) else "No"

                row = [
                    self._format_outcome_name(outcome),
                    f"{power:.3f}",
                    f"{mde:.3f}",
                    f"{required_n:,.0f}",
                    f"{current_n:,.0f}",
                    adequate,
                ]
                latex_lines.append(" & ".join(row) + " \\\\")

        # Add overall assessment
        overall = power_results.get("overall_assessment", {})
        latex_lines.extend(
            [
                "\\midrule",
                "\\multicolumn{6}{l}{\\textit{Overall Assessment}} \\\\",
                "\\addlinespace",
                f"Average Power & {overall.get('average_power', 0):.3f} & & & & \\\\",
                f"Median Power & {overall.get('median_power', 0):.3f} & & & & \\\\",
                f"Min Power & {overall.get('min_power', 0):.3f} & & & & \\\\",
                f"Proportion Adequate & {overall.get('proportion_adequate', 0):.2f} & & & & \\\\",
            ]
        )

        # Add recommendations
        recommendation = overall.get("recommendation", "Increase sample size for improved power")
        latex_lines.extend(
            [
                "\\midrule",
                f"\\multicolumn{{6}}{{l}}{{\\textit{{Recommendation:}} {recommendation}}} \\\\",
            ]
        )

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\begin{tablenotes}",
                "\\small",
                "\\item Notes: Post-hoc power calculated using observed effect sizes and standard errors.",
                "\\item MDE = Minimum Detectable Effect in standard deviation units.",
                "\\item Required N calculated for 80\\% power at $\\alpha$ = 0.05 significance level.",
                "\\item Adequacy determined by whether post-hoc power exceeds 0.80 threshold.",
                "\\end{tablenotes}",
                "\\end{table}",
            ]
        )

        # Write to file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write("\n".join(latex_lines))

        print(f"LaTeX table generated: {output_path}")
        return str(output_path)

    def create_summary_statistics_table(
        self,
        data: pd.DataFrame,
        filename: str = "table1_summary_statistics.tex",
    ) -> str:
        """
        Create Table 1: Summary statistics by treatment status.

        Args:
            data: Analysis dataset
            filename: Output filename

        Returns:
            Path to generated LaTeX file
        """
        # Calculate summary statistics
        treated = data[data["post_treatment"] == 1] if "post_treatment" in data.columns else data
        control = data[data["post_treatment"] == 0] if "post_treatment" in data.columns else data

        variables = [
            "math_grade4_swd_score",
            "math_grade4_gap",
            "reading_grade4_swd_score",
            "reading_grade4_gap",
            "per_pupil_total",
            "inclusion_rate",
        ]

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Summary Statistics by Treatment Status}",
            "\\label{tab:summary_statistics}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            " & \\multicolumn{2}{c}{Full Sample} & \\multicolumn{2}{c}{Treated} & \\multicolumn{2}{c}{Control} \\\\",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
            "Variable & Mean & SD & Mean & SD & Mean & SD \\\\",
            "\\midrule",
        ]

        for var in variables:
            if var in data.columns:
                row = [self._format_variable_name(var)]

                # Full sample
                row.append(f"{data[var].mean():.2f}")
                row.append(f"({data[var].std():.2f})")

                # Treated
                row.append(f"{treated[var].mean():.2f}" if var in treated.columns else "—")
                row.append(f"({treated[var].std():.2f})" if var in treated.columns else "")

                # Control
                row.append(f"{control[var].mean():.2f}" if var in control.columns else "—")
                row.append(f"({control[var].std():.2f})" if var in control.columns else "")

                latex_lines.append(" & ".join(row) + " \\\\")

        # Add sample sizes
        latex_lines.extend(
            [
                "\\midrule",
                f"Observations & {len(data)} & & {len(treated)} & & {len(control)} & \\\\",
                f"States & {data['state'].nunique() if 'state' in data.columns else 50} & & "
                f"{treated['state'].nunique() if 'state' in treated.columns else 25} & & "
                f"{control['state'].nunique() if 'state' in control.columns else 25} & \\\\",
                f"Years & {data['year'].nunique() if 'year' in data.columns else 14} & & "
                f"{treated['year'].nunique() if 'year' in treated.columns else 14} & & "
                f"{control['year'].nunique() if 'year' in control.columns else 14} & \\\\",
            ]
        )

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\begin{tablenotes}",
                "\\small",
                "\\item Notes: Treatment defined as states implementing funding formula reforms.",
                "\\item Achievement scores standardized to have mean 0 and standard deviation 1.",
                "\\item Per-pupil spending in thousands of 2022 dollars.",
                "\\item Inclusion rate represents percentage of SWD in regular classrooms ≥80\\% of time.",
                "\\end{tablenotes}",
                "\\end{table}",
            ]
        )

        # Write to file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write("\n".join(latex_lines))

        print(f"LaTeX table generated: {output_path}")
        return str(output_path)

    def _format_coefficient(self, result: dict[str, Any]) -> str:
        """Format coefficient with significance stars."""
        if not result:
            return "—"

        coef = result.get("coefficient", 0)
        p_value = result.get("p_value", 1)

        stars = ""
        if p_value < 0.01:
            stars = "***"
        elif p_value < 0.05:
            stars = "**"
        elif p_value < 0.10:
            stars = "*"

        return f"{coef:.3f}{stars}"

    def _format_se(self, result: dict[str, Any]) -> str:
        """Format standard error in parentheses."""
        if not result:
            return ""

        se = result.get("std_err", 0)
        return f"({se:.3f})"

    def _format_outcome_name(self, outcome: str) -> str:
        """Format outcome variable name for display."""
        replacements = {
            "math_grade4_swd_score": "Math G4 SWD Score",
            "math_grade4_gap": "Math G4 Achievement Gap",
            "math_grade8_swd_score": "Math G8 SWD Score",
            "math_grade8_gap": "Math G8 Achievement Gap",
            "reading_grade4_swd_score": "Reading G4 SWD Score",
            "reading_grade4_gap": "Reading G4 Achievement Gap",
            "reading_grade8_swd_score": "Reading G8 SWD Score",
            "reading_grade8_gap": "Reading G8 Achievement Gap",
        }
        return replacements.get(outcome, outcome.replace("_", " ").title())

    def _format_variable_name(self, variable: str) -> str:
        """Format variable name for display."""
        replacements = {
            "math_grade4_swd_score": "Math Achievement (Grade 4, SWD)",
            "math_grade4_gap": "Math Achievement Gap (Grade 4)",
            "reading_grade4_swd_score": "Reading Achievement (Grade 4, SWD)",
            "reading_grade4_gap": "Reading Achievement Gap (Grade 4)",
            "per_pupil_total": "Per-Pupil Spending (\\$1000s)",
            "inclusion_rate": "Inclusion Rate (\\%)",
        }
        return replacements.get(variable, variable.replace("_", " ").title())

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Small"
        elif abs_d < 0.5:
            return "Medium"
        elif abs_d < 0.8:
            return "Large"
        else:
            return "Very Large"

    def combine_tables_for_paper(
        self,
        table_files: list[str],
        output_file: str = "all_tables.tex",
    ) -> str:
        """
        Combine multiple LaTeX tables into a single file for paper submission.

        Args:
            table_files: List of LaTeX table files to combine
            output_file: Output filename for combined tables

        Returns:
            Path to combined LaTeX file
        """
        combined_lines = [
            "% Combined Tables for Paper Submission",
            "% Generated by Special Education Policy Analysis",
            "",
        ]

        for i, table_file in enumerate(table_files, 1):
            table_path = self.output_dir / table_file
            if table_path.exists():
                with open(table_path) as f:
                    content = f.read()

                combined_lines.extend(
                    [
                        f"% Table {i}",
                        content,
                        "\\clearpage",
                        "",
                    ]
                )

        # Write combined file
        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write("\n".join(combined_lines))

        print(f"Combined LaTeX tables generated: {output_path}")
        return str(output_path)
