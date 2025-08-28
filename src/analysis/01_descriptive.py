"""
Phase 4.1: Descriptive Analysis Module

Generates comprehensive descriptive statistics and trend visualizations
for the special education policy evaluation study.

This module creates:
- Table 1: Summary statistics by reform status
- Figure 1: Four-panel trend plots (achievement, gaps, funding, inclusion)
- LaTeX formatted tables for publication

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure matplotlib for publication-quality figures
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore", category=FutureWarning)


class DescriptiveAnalyzer:
    """
    Comprehensive descriptive analysis for special education policy evaluation.

    Generates summary statistics and trend visualizations comparing
    states with and without special education funding reforms.
    """

    def __init__(self, data_path: str = "data/final/analysis_panel.csv"):
        """
        Initialize descriptive analyzer.

        Args:
            data_path: Path to the analysis panel dataset
        """
        self.data_path = data_path
        self.df = None
        self.output_dir = Path("output")
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        # Create output directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load and validate analysis panel data."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded analysis panel: {self.df.shape}")

            # Validate required columns
            required_cols = ["state", "year", "post_treatment"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            print(f"Data validation passed. Years: {self.df['year'].min()}-{self.df['year'].max()}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _get_outcome_columns(self) -> dict[str, list]:
        """Identify outcome variable columns in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        outcome_mapping = {"achievement": [], "gaps": [], "inclusion": [], "finance": []}

        for col in self.df.columns:
            col_lower = col.lower()
            if "score" in col_lower and "gap" not in col_lower:
                outcome_mapping["achievement"].append(col)
            elif "gap" in col_lower:
                outcome_mapping["gaps"].append(col)
            elif any(term in col_lower for term in ["inclusion", "placement", "environment"]):
                outcome_mapping["inclusion"].append(col)
            elif any(
                term in col_lower for term in ["revenue", "expenditure", "spending", "per_pupil"]
            ):
                outcome_mapping["finance"].append(col)

        # Print what we found
        for category, cols in outcome_mapping.items():
            if cols:
                print(
                    f"{category.capitalize()} variables: {cols[:3]}{'...' if len(cols) > 3 else ''}"
                )
            else:
                print(f"{category.capitalize()} variables: None found")

        return outcome_mapping

    def create_summary_statistics(self) -> pd.DataFrame:
        """
        Create Table 1: Summary statistics by reform status.

        Returns:
            Summary statistics DataFrame
        """
        if self.df is None:
            self.load_data()

        print("Creating summary statistics table...")

        # Get outcome columns
        outcomes = self._get_outcome_columns()

        # Select key variables for summary table
        key_vars = []

        # Add representative variables from each category
        if outcomes["achievement"]:
            # Prefer math grade 8 if available, otherwise first achievement variable
            math8_vars = [
                col for col in outcomes["achievement"] if "math" in col.lower() and "8" in col
            ]
            key_vars.extend(math8_vars[:1] if math8_vars else outcomes["achievement"][:1])

        if outcomes["gaps"]:
            # Prefer math gap if available
            math_gap_vars = [col for col in outcomes["gaps"] if "math" in col.lower()]
            key_vars.extend(math_gap_vars[:1] if math_gap_vars else outcomes["gaps"][:1])

        if outcomes["inclusion"]:
            key_vars.extend(outcomes["inclusion"][:1])

        if outcomes["finance"]:
            # Prefer total per-pupil measures
            total_vars = [
                col
                for col in outcomes["finance"]
                if "total" in col.lower() and "per_pupil" in col.lower()
            ]
            key_vars.extend(total_vars[:1] if total_vars else outcomes["finance"][:1])

        # Add control variables if available
        control_vars = ["time_trend", "post_covid"]
        key_vars.extend([var for var in control_vars if var in self.df.columns])

        print(f"Summary table variables: {key_vars}")

        if not key_vars:
            # Fallback to numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            key_vars = [col for col in numeric_cols if col not in ["year", "state"]][:5]
            print(f"Using fallback variables: {key_vars}")

        # Create summary by treatment status
        summary_stats = (
            self.df.groupby("post_treatment")[key_vars].agg(["mean", "std", "count"]).round(3)
        )

        # Flatten column names
        summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]

        # Add difference between treated and control (if both groups exist)
        mean_cols = [col for col in summary_stats.columns if col.endswith("_mean")]
        if 1 in summary_stats.index and 0 in summary_stats.index:
            treated_means = summary_stats.loc[1, mean_cols]
            control_means = summary_stats.loc[0, mean_cols]

            differences = {}
            for var in key_vars:
                mean_key = f"{var}_mean"
                if mean_key in treated_means.index and mean_key in control_means.index:
                    diff = treated_means[mean_key] - control_means[mean_key]
                    differences[mean_key] = diff
                    differences[f"{var}_std"] = np.nan  # No std for differences
                    differences[f"{var}_count"] = np.nan  # No count for differences

            # Append difference row
            if differences:
                diff_series = pd.Series(differences, name="difference")
                summary_stats = pd.concat([summary_stats, diff_series.to_frame().T])

        # Save to CSV and LaTeX
        summary_stats.to_csv(self.tables_dir / "table1_summary_stats.csv")

        # Create LaTeX table
        # Escape underscores in column headers for LaTeX only (keep CSV headers unchanged)
        latex_table = self._format_latex_table(
            summary_stats.rename(columns=lambda c: str(c).replace("_", "\\_")),
            title="Summary Statistics by Reform Status",
            label="tab:summary_stats",
        )

        with open(self.tables_dir / "table1_summary_stats.tex", "w") as f:
            f.write(latex_table)

        print(f"Summary statistics saved to {self.tables_dir}")
        return summary_stats

    def _format_latex_table(self, df: pd.DataFrame, title: str, label: str) -> str:
        """Format DataFrame as LaTeX table."""
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{title}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{"c" * len(df.columns)}}}
\\toprule
"""
        # Add column headers
        latex += "Variable & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"

        # Add rows
        for idx, row in df.iterrows():
            row_name = "Treated" if idx == 1 else "Control" if idx == 0 else "Difference"
            latex += (
                f"{row_name} & "
                + " & ".join([f"{val:.3f}" if pd.notna(val) else "--" for val in row])
                + " \\\\\n"
            )

        latex += """\\bottomrule
\\end{tabular}
\\footnotesize
\\textit{Notes:} Sample includes all state-year observations 2009-2023. 
Treated = 1 for states that implemented special education funding reforms.
Standard deviations in parentheses.
\\end{table}"""

        return latex

    def create_trend_figures(self) -> dict[str, plt.Figure]:
        """
        Create Figure 1: Four-panel trend plots.

        Returns:
            Dictionary of matplotlib figures
        """
        if self.df is None:
            self.load_data()

        print("Creating trend figures...")

        outcomes = self._get_outcome_columns()

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Trends by Reform Status", fontsize=16, y=0.98)

        # Panel A: Achievement trends
        if outcomes["achievement"]:
            achievement_var = outcomes["achievement"][0]  # Use first available
            self._plot_trends(axes[0, 0], achievement_var, "Achievement Scores", "Panel A")
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "No achievement data",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_title("Panel A: Achievement (No Data)")

        # Panel B: Gap trends
        if outcomes["gaps"]:
            gap_var = outcomes["gaps"][0]  # Use first available
            self._plot_trends(axes[0, 1], gap_var, "Achievement Gap (points)", "Panel B")
        else:
            axes[0, 1].text(
                0.5, 0.5, "No gap data", ha="center", va="center", transform=axes[0, 1].transAxes
            )
            axes[0, 1].set_title("Panel B: Achievement Gap (No Data)")

        # Panel C: Funding trends
        if outcomes["finance"]:
            finance_var = outcomes["finance"][0]  # Use first available
            self._plot_trends(axes[1, 0], finance_var, "Per-Pupil Funding ($1000s)", "Panel C")
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No finance data",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Panel C: Funding (No Data)")

        # Panel D: Inclusion trends
        if outcomes["inclusion"]:
            inclusion_var = outcomes["inclusion"][0]  # Use first available
            self._plot_trends(axes[1, 1], inclusion_var, "Inclusion Rate (%)", "Panel D")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No inclusion data",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Panel D: Inclusion (No Data)")

        plt.tight_layout()

        # Save figure
        fig.savefig(self.figures_dir / "figure1_trends.png", dpi=300, bbox_inches="tight")
        fig.savefig(self.figures_dir / "figure1_trends.pdf", bbox_inches="tight")

        print(f"Trend figures saved to {self.figures_dir}")

        return {"trends": fig}

    def _plot_trends(self, ax: plt.Axes, variable: str, ylabel: str, title: str) -> None:
        """Plot trends for a specific variable."""
        if variable not in self.df.columns:
            ax.text(
                0.5,
                0.5,
                f"Variable {variable}\nnot found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{title}: {variable} (Missing)")
            return

        # Calculate yearly means by treatment status
        yearly_trends = (
            self.df.groupby(["year", "post_treatment"])[variable].mean().unstack(fill_value=np.nan)
        )

        # Plot lines
        if 0 in yearly_trends.columns:
            ax.plot(
                yearly_trends.index,
                yearly_trends[0],
                "o-",
                label="Control States",
                linewidth=2,
                markersize=4,
            )
        if 1 in yearly_trends.columns:
            ax.plot(
                yearly_trends.index,
                yearly_trends[1],
                "s-",
                label="Treated States",
                linewidth=2,
                markersize=4,
            )

        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {variable.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add vertical line at typical reform period (2015)
        if 2015 in yearly_trends.index:
            ax.axvline(x=2015, color="red", linestyle="--", alpha=0.5, label="Reform Period")

    def generate_descriptive_report(self) -> str:
        """
        Generate comprehensive descriptive analysis report.

        Returns:
            Path to generated report
        """
        print("Generating comprehensive descriptive analysis...")

        # Load data if not already loaded
        if self.df is None:
            self.load_data()

        # Create summary statistics
        summary_stats = self.create_summary_statistics()

        # Create trend figures
        self.create_trend_figures()

        # Generate markdown report
        report_path = self.reports_dir / "descriptive_analysis_report.md"

        with open(report_path, "w") as f:
            f.write(f"""# Phase 4.1: Descriptive Analysis Report

Generated by: Jeff Chen (jeffreyc1@alumni.cmu.edu)
Created in collaboration with Claude Code
Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Data Overview

- **Dataset**: {self.data_path}
- **Observations**: {len(self.df):,}
- **States**: {self.df["state"].nunique()}
- **Years**: {self.df["year"].min()}-{self.df["year"].max()}
- **Treated States**: {len(self.df[self.df["post_treatment"] == 1]["state"].unique())}
- **Control States**: {len(self.df[self.df["post_treatment"] == 0]["state"].unique())}

## Key Findings

### Summary Statistics
{self._format_summary_findings(summary_stats)}

### Trend Analysis
- See Figure 1 (figure1_trends.png) for visual trends
- Trends show patterns by reform status across key outcomes

## Files Generated

1. **Tables**:
   - `table1_summary_stats.csv` - Summary statistics in CSV format
   - `table1_summary_stats.tex` - LaTeX formatted table

2. **Figures**:
   - `figure1_trends.png` - Four-panel trend plots (high resolution)
   - `figure1_trends.pdf` - PDF version for publication

## Next Steps

Proceed to Phase 4.2: Main Causal Analysis
""")

        print(f"Descriptive analysis complete! Report saved to {report_path}")
        return str(report_path)

    def _format_summary_findings(self, summary_stats: pd.DataFrame) -> str:
        """Format key findings from summary statistics."""
        if summary_stats.empty:
            return "No summary statistics available."

        findings = []

        # Look for difference row
        if "difference" in summary_stats.index:
            diff_row = summary_stats.loc["difference"]
            mean_cols = [col for col in diff_row.index if col.endswith("_mean")]

            for col in mean_cols[:3]:  # Show first 3 variables
                var_name = col.replace("_mean", "").replace("_", " ").title()
                diff_val = diff_row[col]
                if pd.notna(diff_val):
                    direction = "higher" if diff_val > 0 else "lower"
                    findings.append(
                        f"- **{var_name}**: Treated states are {abs(diff_val):.3f} units {direction} on average"
                    )

        return (
            "\n".join(findings)
            if findings
            else "No significant differences found in available variables."
        )


def main():
    """Run descriptive analysis pipeline."""
    print("Phase 4.1: Running Descriptive Analysis")
    print("=" * 50)

    try:
        analyzer = DescriptiveAnalyzer()
        report_path = analyzer.generate_descriptive_report()
        print("\n✅ Descriptive analysis completed successfully!")
        print(f"📊 Report available at: {report_path}")

        return True

    except Exception as e:
        print(f"\n❌ Error in descriptive analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
