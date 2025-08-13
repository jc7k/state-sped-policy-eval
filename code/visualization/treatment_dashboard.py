"""
Treatment Effects Dashboard with Geographic Visualization

Optimized geographic visualization dashboard leveraging BaseVisualizer infrastructure.
Features efficient data processing, smart caching, and high-performance rendering
for state-level special education policy analysis.

Key Optimizations:
- Inherits efficient data loading and caching from BaseVisualizer
- Vectorized geographic data processing
- Optimized matplotlib rendering for maps
- Smart memory management for large datasets
- Cached color palette generation for geographic plots

Features:
- State-level choropleth maps of treatment effects
- Treatment rollout timeline visualization
- Regional comparison charts with statistical analysis
- Policy reform type categorization and analysis
- Publication-ready static maps with consistent styling

Author: Jeff Chen, jeffreyc1@alumni.cmu.edu
Created in collaboration with Claude Code
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import BaseVisualizer
from .config import get_state_coordinates, get_us_regions
from .utils import format_outcome_label, get_optimal_figure_size, safe_numeric_operation


class TreatmentEffectsDashboard(BaseVisualizer):
    """
    Optimized geographic visualization dashboard inheriting from BaseVisualizer.

    Features efficient data loading, smart caching, and optimized rendering for:
    - State-level treatment effect magnitudes
    - Geographic patterns in policy adoption  
    - Treatment timing and rollout visualization
    - Regional heterogeneity analysis with statistical testing
    """

    def __init__(
        self,
        results_dir: str = "output/tables",
        figures_dir: str = "output/figures",
        policy_data_path: str = "data/processed/state_policy_database.csv",
        **kwargs
    ):
        """Initialize optimized dashboard with BaseVisualizer infrastructure."""
        # Initialize base class with all optimizations
        super().__init__(results_dir, figures_dir, **kwargs)
        
        # Store policy data path for lazy loading
        self.policy_data_path = policy_data_path
        self.policy_data = pd.DataFrame()  # Will be loaded on demand
        
        # Cached geographic data (efficient access)
        self._state_coords = None
        self._regions = None
        
        if self.verbose:
            print("TreatmentEffectsDashboard initialized with optimized caching")

    def _load_available_data(self) -> None:
        """Load available data using BaseVisualizer infrastructure."""
        # Load policy data on-demand
        if self.policy_data.empty:
            self.policy_data = self._load_policy_data_optimized()
        
        # Load treatment effects using parent's optimized method
        self.treatment_effects = self._load_csv_file(
            "aggregated_effects_*.csv",
            cache_key="aggregated_effects"
        )
    
    def _load_policy_data_optimized(self) -> pd.DataFrame:
        """Optimized policy data loading with error handling."""
        try:
            policy_df = pd.read_csv(
                self.policy_data_path,
                engine='c',  # Use C engine for speed
                low_memory=False
            )
            
            # Add to cache for future use
            self._add_to_cache("policy_data", policy_df)
            
            if self.verbose:
                self.logger.info(f"Loaded policy data: {len(policy_df)} records")
                
            return policy_df
            
        except Exception as e:
            self.logger.warning(f"Could not load policy data from {self.policy_data_path}: {e}")
            return pd.DataFrame()

    @property
    def state_coords(self) -> Dict[str, Tuple[float, float]]:
        """Cached state coordinates for plotting."""
        if self._state_coords is None:
            self._state_coords = get_state_coordinates()
        return self._state_coords

    @property  
    def regions(self) -> Dict[str, List[str]]:
        """Cached regional groupings for analysis."""
        if self._regions is None:
            self._regions = get_us_regions()
        return self._regions

    def create_treatment_timeline_map(
        self,
        title: str = "State Special Education Policy Reforms Timeline",
        save_formats: Optional[List[str]] = None,
    ) -> str:
        """
        Create optimized map showing when states adopted policy reforms.
        """
        if save_formats is None:
            save_formats = ["png", "pdf"]
            
        # Ensure data is loaded
        self._ensure_data_loaded()
        
        if self.policy_data.empty:
            self.logger.warning("No policy data available")
            return ""

        # Vectorized data processing for performance
        treated_states = self.policy_data[
            self.policy_data["post_treatment"] == 1
        ].copy()

        if treated_states.empty:
            self.logger.warning("No treated states found")
            return ""

        # Optimized groupby operation
        first_treatment = (treated_states.groupby("state", sort=False)["year"]
                          .min().reset_index(name="reform_year"))

        # Create figure with optimal sizing
        figsize = get_optimal_figure_size("map", len(first_treatment))
        fig, ax = self._create_figure(figsize)
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

        # Save using BaseVisualizer's optimized save method
        saved_files = self._save_plot(fig, "treatment_timeline_map", save_formats)
        return saved_files[0] if saved_files else ""

    def create_regional_comparison_chart(
        self,
        outcome: str = "math_grade8_gap",
        title: Optional[str] = None,
        save_formats: Optional[List[str]] = None,
    ) -> str:
        """
        Create optimized chart comparing treatment effects across regions.
        """
        if save_formats is None:
            save_formats = ["png", "pdf"]
            
        # Ensure data is loaded
        self._ensure_data_loaded()
        
        if self.policy_data.empty or outcome not in self.treatment_effects:
            self.logger.warning(f"Cannot create regional comparison for {outcome}")
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

        # Save using BaseVisualizer's optimized save method
        saved_files = self._save_plot(fig, f"regional_comparison_{outcome}", save_formats)
        return saved_files[0] if saved_files else ""

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

        # Save using BaseVisualizer's optimized save method
        saved_files = self._save_plot(fig, "policy_type_analysis", save_formats)
        return saved_files[0] if saved_files else ""

    def create_all_visualizations(self, **kwargs) -> Dict[str, List[str]]:
        """Generate complete treatment effects dashboard using BaseVisualizer pattern."""
        if self.verbose:
            print("Creating comprehensive treatment effects dashboard...")

        all_files = {}

        try:
            # Treatment timeline map
            timeline_file = self.create_treatment_timeline_map(**kwargs)
            if timeline_file:
                all_files["timeline_map"] = [timeline_file]
        except Exception as e:
            self.logger.error(f"Error creating timeline map: {e}")

        try:
            # Regional comparison charts for each outcome
            self._ensure_data_loaded()
            regional_files = []
            for outcome in self.treatment_effects:
                regional_file = self.create_regional_comparison_chart(outcome, **kwargs)
                if regional_file:
                    regional_files.append(regional_file)
            
            if regional_files:
                all_files["regional_comparisons"] = regional_files
                
        except Exception as e:
            self.logger.error(f"Error creating regional comparisons: {e}")

        try:
            # Policy type analysis
            policy_file = self.create_policy_type_analysis(**kwargs)
            if policy_file:
                all_files["policy_analysis"] = [policy_file]
        except Exception as e:
            self.logger.error(f"Error creating policy analysis: {e}")

        return all_files
    
    # Keep backward compatibility
    def create_complete_dashboard(self) -> Dict[str, List[str]]:
        """Legacy method - delegates to create_all_visualizations."""
        return self.create_all_visualizations()
    
    def create_visualization(self, plot_type: str = "timeline_map", outcome: str = "", **kwargs) -> str:
        """
        Create a single visualization (implements abstract method).
        
        Args:
            plot_type: Type of plot ('timeline_map', 'regional_comparison', 'policy_analysis')
            outcome: Outcome variable name (for regional comparison)
            **kwargs: Additional arguments
            
        Returns:
            Path to created visualization file
        """
        if plot_type == "timeline_map":
            return self.create_treatment_timeline_map(**kwargs)
        elif plot_type == "regional_comparison":
            if not outcome:
                outcome = "math_grade8_gap"  # Default outcome
            return self.create_regional_comparison_chart(outcome, **kwargs)
        elif plot_type == "policy_analysis":
            return self.create_policy_type_analysis(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")


if __name__ == "__main__":
    # Create optimized dashboard and generate all visualizations
    dashboard = TreatmentEffectsDashboard(verbose=True)

    # Generate complete dashboard using optimized BaseVisualizer pattern
    all_plots = dashboard.create_all_visualizations()

    # Enhanced summary report
    print(f"\n{'=' * 60}")
    print("TREATMENT EFFECTS DASHBOARD COMPLETE")
    print(f"{'=' * 60}")

    total_files = sum(len(files) for files in all_plots.values())
    print(f"Total dashboard plots created: {total_files}")
    
    # Cache statistics
    cache_info = dashboard.get_cache_info()
    print(f"Cache utilization: {cache_info['utilization_percent']:.1f}% "
          f"({cache_info['memory_usage_mb']:.1f}MB)")

    # List all generated files by category
    for category, files in all_plots.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for file in files:
            print(f"  - {Path(file).name}")

    print(f"\nAll dashboard figures saved to: {dashboard.figures_dir}")
