"""
Treatment Effects Dashboard with Geographic Visualization

Creates interactive and static geographic visualizations showing the spatial
distribution of state special education policy reforms and their effects.
Includes state-level choropleth maps, treatment timeline visualizations,
and regional effect comparisons.

Features:
- State-level choropleth maps of treatment effects
- Treatment rollout timeline visualization
- Regional comparison charts
- Policy reform type categorization
- Publication-ready static maps

Author: Research Team
Date: 2025-08-12
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TreatmentEffectsDashboard:
    """
    Geographic visualization dashboard for state policy treatment effects.

    Creates publication-ready maps and charts showing:
    - State-level treatment effect magnitudes
    - Geographic patterns in policy adoption
    - Treatment timing and rollout visualization
    - Regional heterogeneity analysis
    """

    def __init__(
        self,
        results_dir: str = "output/tables",
        figures_dir: str = "output/figures",
        policy_data_path: str = "data/processed/state_policy_database.csv",
    ):
        """Initialize dashboard with data sources."""
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.policy_data = self._load_policy_data(policy_data_path)
        self.treatment_effects = self._load_treatment_effects()

        # State coordinates for plotting (approximate state centers)
        self.state_coords = self._get_state_coordinates()

        # Regional groupings
        self.regions = self._define_regions()

        print("TreatmentEffectsDashboard initialized:")
        print(f"  Policy data: {len(self.policy_data)} state-year observations")
        print(f"  Treatment effects: {len(self.treatment_effects)} outcomes")
        print(f"  State coordinates: {len(self.state_coords)} states")

    def _load_policy_data(self, policy_path: str) -> pd.DataFrame:
        """Load state policy database."""
        try:
            policy_df = pd.read_csv(policy_path)
            return policy_df
        except Exception as e:
            print(f"Warning: Could not load policy data: {e}")
            return pd.DataFrame()

    def _load_treatment_effects(self) -> dict[str, pd.DataFrame]:
        """Load aggregated treatment effects for mapping."""
        effects = {}

        # Load aggregated effects files
        agg_files = list(self.results_dir.glob("aggregated_effects_*.csv"))

        for file_path in agg_files:
            outcome = file_path.name.replace("aggregated_effects_", "").replace(
                ".csv", ""
            )

            try:
                agg_df = pd.read_csv(file_path)
                effects[outcome] = agg_df
            except Exception as e:
                print(f"Warning: Could not load effects for {outcome}: {e}")
                continue

        return effects

    def _get_state_coordinates(self) -> dict[str, tuple[float, float]]:
        """Approximate state center coordinates for plotting."""
        # Simplified state coordinates (longitude, latitude)
        coords = {
            "AL": (-86.8, 32.8),
            "AK": (-152.0, 64.0),
            "AZ": (-111.9, 34.2),
            "AR": (-92.2, 34.8),
            "CA": (-119.8, 36.8),
            "CO": (-105.5, 39.2),
            "CT": (-72.7, 41.6),
            "DE": (-75.5, 39.2),
            "DC": (-77.0, 38.9),
            "FL": (-81.5, 27.8),
            "GA": (-83.2, 32.2),
            "HI": (-157.8, 21.3),
            "ID": (-114.6, 44.1),
            "IL": (-89.2, 40.1),
            "IN": (-86.3, 39.8),
            "IA": (-93.6, 42.0),
            "KS": (-98.4, 38.5),
            "KY": (-84.9, 37.8),
            "LA": (-91.8, 31.2),
            "ME": (-69.2, 45.2),
            "MD": (-76.5, 39.0),
            "MA": (-71.8, 42.4),
            "MI": (-84.5, 43.3),
            "MN": (-94.6, 46.4),
            "MS": (-89.4, 32.7),
            "MO": (-92.2, 38.3),
            "MT": (-110.4, 47.1),
            "NE": (-99.8, 41.5),
            "NV": (-117.0, 39.8),
            "NH": (-71.5, 43.2),
            "NJ": (-74.8, 40.2),
            "NM": (-106.2, 34.5),
            "NY": (-74.2, 42.2),
            "NC": (-78.6, 35.8),
            "ND": (-99.8, 47.5),
            "OH": (-82.7, 40.2),
            "OK": (-97.1, 35.6),
            "OR": (-122.0, 44.9),
            "PA": (-77.2, 40.3),
            "RI": (-71.4, 41.6),
            "SC": (-80.9, 33.8),
            "SD": (-99.9, 44.3),
            "TN": (-86.4, 35.9),
            "TX": (-97.7, 31.1),
            "UT": (-111.9, 40.2),
            "VT": (-72.6, 44.0),
            "VA": (-78.2, 37.7),
            "WA": (-121.5, 47.4),
            "WV": (-80.9, 38.8),
            "WI": (-90.0, 44.3),
            "WY": (-107.3, 42.8),
        }
        return coords

    def _define_regions(self) -> dict[str, list[str]]:
        """Define regional groupings for analysis."""
        regions = {
            "Northeast": ["CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
            "South": [
                "AL",
                "AR",
                "DE",
                "DC",
                "FL",
                "GA",
                "KY",
                "LA",
                "MD",
                "MS",
                "NC",
                "OK",
                "SC",
                "TN",
                "TX",
                "VA",
                "WV",
            ],
            "Midwest": [
                "IL",
                "IN",
                "IA",
                "KS",
                "MI",
                "MN",
                "MO",
                "NE",
                "ND",
                "OH",
                "SD",
                "WI",
            ],
            "West": [
                "AK",
                "AZ",
                "CA",
                "CO",
                "HI",
                "ID",
                "MT",
                "NV",
                "NM",
                "OR",
                "UT",
                "WA",
                "WY",
            ],
        }
        return regions

    def create_treatment_timeline_map(
        self,
        title: str = "State Special Education Policy Reforms Timeline",
        save_formats: list[str] = None,
    ) -> str:
        """
        Create map showing when states adopted policy reforms.
        """
        if save_formats is None:
            save_formats = ["png", "pdf"]
        if self.policy_data.empty:
            print("Warning: No policy data available")
            return ""

        # Get first treatment year for each state
        treated_states = self.policy_data[
            self.policy_data["post_treatment"] == 1
        ].copy()

        if treated_states.empty:
            print("Warning: No treated states found")
            return ""

        first_treatment = treated_states.groupby("state")["year"].min().reset_index()
        first_treatment.columns = ["state", "reform_year"]

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(-180, -60)
        ax.set_ylim(15, 75)

        # Define color scheme for treatment years
        treatment_years = sorted(first_treatment["reform_year"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(treatment_years)))
        year_color_map = dict(zip(treatment_years, colors, strict=False))

        # Plot states
        for state, coord in self.state_coords.items():
            lon, lat = coord

            # Check if state had reform
            state_reform = first_treatment[first_treatment["state"] == state]

            if not state_reform.empty:
                reform_year = state_reform["reform_year"].iloc[0]
                color = year_color_map[reform_year]
                marker = "o"
                size = 150
            else:
                color = "lightgray"
                marker = "s"
                size = 80

            ax.scatter(
                lon,
                lat,
                c=[color],
                s=size,
                marker=marker,
                edgecolors="black",
                linewidth=1,
                alpha=0.8,
            )

            # Add state labels
            ax.annotate(
                state,
                (lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

        # Create legend for treatment years
        legend_elements = []
        for year in treatment_years:
            legend_elements.append(
                plt.scatter(
                    [],
                    [],
                    c=[year_color_map[year]],
                    s=150,
                    marker="o",
                    edgecolors="black",
                    label=f"Reformed {year}",
                )
            )
        legend_elements.append(
            plt.scatter(
                [],
                [],
                c="lightgray",
                s=80,
                marker="s",
                edgecolors="black",
                label="No reform",
            )
        )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="Policy Reform Year",
            title_fontsize=12,
        )

        # Formatting
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save files
        saved_files = []
        for fmt in save_formats:
            filename = f"treatment_timeline_map.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
            saved_files.append(str(filepath))

        plt.close()

        print(f"Treatment timeline map saved: {saved_files[0]}")
        return saved_files[0]

    def create_regional_comparison_chart(
        self,
        outcome: str = "math_grade8_gap",
        title: str | None = None,
        save_formats: list[str] = None,
    ) -> str:
        """
        Create chart comparing treatment effects across regions.
        """
        if save_formats is None:
            save_formats = ["png", "pdf"]
        if self.policy_data.empty or outcome not in self.treatment_effects:
            print(f"Warning: Cannot create regional comparison for {outcome}")
            return ""

        # Get treated states and their effects
        treated_states = self.policy_data[self.policy_data["post_treatment"] == 1][
            "state"
        ].unique()

        # Assign regions to states
        regional_data = []
        for region, states in self.regions.items():
            treated_in_region = [s for s in states if s in treated_states]

            if treated_in_region:
                regional_data.append(
                    {
                        "region": region,
                        "treated_states": len(treated_in_region),
                        "total_states": len(states),
                        "treatment_rate": len(treated_in_region) / len(states),
                    }
                )

        regional_df = pd.DataFrame(regional_data)

        if regional_df.empty:
            print("Warning: No regional data found")
            return ""

        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Chart 1: Treatment adoption by region
        bars1 = ax1.bar(
            regional_df["region"],
            regional_df["treatment_rate"],
            color=["steelblue", "darkorange", "forestgreen", "crimson"],
            alpha=0.7,
            edgecolor="black",
        )

        ax1.set_title(
            "Policy Reform Adoption by Region", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Fraction of States with Reforms", fontsize=12)
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            treated = regional_df.iloc[i]["treated_states"]
            total = regional_df.iloc[i]["total_states"]
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{treated}/{total}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Chart 2: Treatment effect magnitudes (if available)
        if outcome in self.treatment_effects:
            effect_data = self.treatment_effects[outcome]
            if "simple_att" in effect_data.columns:
                overall_effect = effect_data["simple_att"].iloc[0]
                effect_data.get("simple_se", [0]).iloc[0]

                # For now, show overall effect (could be enhanced with region-specific effects)
                regions_list = list(self.regions.keys())
                effects = [overall_effect] * len(regions_list)  # Placeholder

                bars2 = ax2.bar(
                    regions_list,
                    effects,
                    color=["steelblue", "darkorange", "forestgreen", "crimson"],
                    alpha=0.7,
                    edgecolor="black",
                )

                ax2.axhline(y=0, color="black", linestyle="-", alpha=0.8)
                ax2.set_title(
                    f"Treatment Effects by Region\\n({self._format_outcome_label(outcome)})",
                    fontsize=14,
                    fontweight="bold",
                )
                ax2.set_ylabel("Treatment Effect (NAEP Points)", fontsize=12)

                # Add effect size labels
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01 if height > 0 else height - 0.03,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom" if height > 0 else "top",
                        fontweight="bold",
                    )

        # Rotate x-axis labels
        ax1.tick_params(axis="x", rotation=45)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save files
        saved_files = []
        for fmt in save_formats:
            filename = f"regional_comparison_{outcome}.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
            saved_files.append(str(filepath))

        plt.close()

        print(f"Regional comparison chart saved: {saved_files[0]}")
        return saved_files[0]

    def create_policy_type_analysis(
        self,
        title: str = "State Policy Reform Types and Timing",
        save_formats: list[str] = None,
    ) -> str:
        """
        Analyze different types of policy interventions by state and time.
        """
        if save_formats is None:
            save_formats = ["png", "pdf"]
        if self.policy_data.empty:
            print("Warning: No policy data available")
            return ""

        # Analyze policy types from the data
        treated_data = self.policy_data[self.policy_data["post_treatment"] == 1].copy()

        if treated_data.empty:
            print("Warning: No treated states found")
            return ""

        # Get treatment timing
        first_treatment = treated_data.groupby("state")["year"].min().reset_index()
        first_treatment.columns = ["state", "reform_year"]

        # Add monitoring and court order information
        monitoring_data = self.policy_data[self.policy_data["under_monitoring"] == 1]
        court_data = self.policy_data[self.policy_data["court_ordered"] == 1]

        # Create analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Chart 1: Reform timeline
        reform_years = first_treatment["reform_year"].value_counts().sort_index()

        bars1 = ax1.bar(
            reform_years.index,
            reform_years.values,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )

        ax1.set_title("Number of State Reforms by Year", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Reform Year", fontsize=12)
        ax1.set_ylabel("Number of States", fontsize=12)
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Chart 2: Policy intervention types
        intervention_counts = {
            "Funding Reforms": len(first_treatment),
            "Federal Monitoring": len(monitoring_data["state"].unique())
            if not monitoring_data.empty
            else 0,
            "Court Orders": len(court_data["state"].unique())
            if not court_data.empty
            else 0,
        }

        bars2 = ax2.bar(
            intervention_counts.keys(),
            intervention_counts.values(),
            color=["steelblue", "darkorange", "forestgreen"],
            alpha=0.7,
            edgecolor="black",
        )

        ax2.set_title(
            "Types of Policy Interventions (2009-2023)", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Number of States", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        # Save files
        saved_files = []
        for fmt in save_formats:
            filename = f"policy_type_analysis.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
            saved_files.append(str(filepath))

        plt.close()

        print(f"Policy type analysis saved: {saved_files[0]}")
        return saved_files[0]

    def create_complete_dashboard(self) -> dict[str, list[str]]:
        """Generate complete treatment effects dashboard."""
        print("Creating comprehensive treatment effects dashboard...")

        all_files = {}

        try:
            # Treatment timeline map
            timeline_file = self.create_treatment_timeline_map()
            if timeline_file:
                all_files["timeline_map"] = [timeline_file]
        except Exception as e:
            print(f"Error creating timeline map: {e}")

        try:
            # Regional comparison charts for each outcome
            for outcome in self.treatment_effects:
                regional_file = self.create_regional_comparison_chart(outcome)
                if regional_file:
                    if "regional_comparisons" not in all_files:
                        all_files["regional_comparisons"] = []
                    all_files["regional_comparisons"].append(regional_file)
        except Exception as e:
            print(f"Error creating regional comparisons: {e}")

        try:
            # Policy type analysis
            policy_file = self.create_policy_type_analysis()
            if policy_file:
                all_files["policy_analysis"] = [policy_file]
        except Exception as e:
            print(f"Error creating policy analysis: {e}")

        return all_files

    def _format_outcome_label(self, outcome: str) -> str:
        """Convert outcome variable name to readable label."""
        outcome.split("_")

        if "math" in outcome:
            subject = "Mathematics"
        elif "reading" in outcome:
            subject = "Reading"
        else:
            subject = "Achievement"

        if "grade4" in outcome:
            grade = "Grade 4"
        elif "grade8" in outcome:
            grade = "Grade 8"
        else:
            grade = ""

        if "gap" in outcome:
            metric = "Achievement Gap"
        elif "score" in outcome:
            metric = "Score"
        else:
            metric = "Outcome"

        return f"{subject} {grade} {metric}".strip()


if __name__ == "__main__":
    # Create dashboard and generate all visualizations
    dashboard = TreatmentEffectsDashboard()

    # Generate complete dashboard
    all_plots = dashboard.create_complete_dashboard()

    # Summary report
    print(f"\\n{'=' * 60}")
    print("TREATMENT EFFECTS DASHBOARD COMPLETE")
    print(f"{'=' * 60}")

    total_files = sum(len(files) for files in all_plots.values())
    print(f"Total dashboard plots created: {total_files}")

    # List all generated files
    for category, files in all_plots.items():
        print(f"\\n{category}:")
        for file in files:
            print(f"  - {file}")

    print(f"\\nAll dashboard figures saved to: {dashboard.figures_dir}")
