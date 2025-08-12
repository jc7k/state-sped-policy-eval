"""
Event Study Visualization Engine

Creates publication-ready event study plots from staggered difference-in-differences
analysis results. Generates coefficient plots with confidence intervals, parallel
trends validation, and treatment effect visualizations.

Features:
- Professional matplotlib styling for academic publications
- Automatic confidence interval plotting
- Parallel trends testing visualization
- Multiple output formats (PDF, PNG, EPS)
- Customizable styling and annotations

Author: Research Team
Date: 2025-08-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
from scipy import stats


class EventStudyVisualizer:
    """
    Professional event study visualization engine for economic research.
    
    Creates publication-ready plots from econometric analysis results including:
    - Event study coefficient plots with confidence intervals
    - Parallel trends validation visualizations  
    - Treatment effect forest plots
    - Robustness comparison charts
    """
    
    def __init__(self, 
                 results_dir: str = "output/tables",
                 figures_dir: str = "output/figures"):
        """Initialize visualizer with input and output directories."""
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-ready style
        self._setup_plotting_style()
        
        # Load existing results
        self.results = self._load_analysis_results()
        
        print(f"EventStudyVisualizer initialized:")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Figures directory: {self.figures_dir}")
        print(f"  Available outcomes: {list(self.results.keys())}")
    
    def _setup_plotting_style(self):
        """Configure matplotlib for publication-ready figures."""
        plt.style.use('default')  # Start with clean slate
        
        # Set publication parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Computer Modern Roman'],
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def _load_analysis_results(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load existing staggered DiD analysis results."""
        results = {}
        
        # Find all event study result files
        event_study_files = list(self.results_dir.glob("event_study_*.csv"))
        
        for file_path in event_study_files:
            # Extract outcome name from filename
            outcome = file_path.name.replace("event_study_", "").replace(".csv", "")
            
            try:
                event_df = pd.read_csv(file_path)
                
                # Also load corresponding aggregated effects if available
                agg_file = self.results_dir / f"aggregated_effects_{outcome}.csv"
                group_file = self.results_dir / f"group_time_effects_{outcome}.csv"
                
                outcome_results = {'event_study': event_df}
                
                if agg_file.exists():
                    outcome_results['aggregated'] = pd.read_csv(agg_file)
                
                if group_file.exists():
                    outcome_results['group_time'] = pd.read_csv(group_file)
                
                results[outcome] = outcome_results
                
            except Exception as e:
                print(f"Warning: Could not load results for {outcome}: {e}")
                continue
        
        return results
    
    def plot_event_study(self, 
                        outcome: str,
                        title: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        save_formats: List[str] = ['png', 'pdf']) -> str:
        """
        Create professional event study plot with confidence intervals.
        
        Args:
            outcome: Outcome variable name (e.g., 'math_grade4_gap')
            title: Custom plot title (auto-generated if None)
            ylabel: Custom y-axis label (auto-generated if None)
            save_formats: List of formats to save ('png', 'pdf', 'eps')
            
        Returns:
            Path to main saved figure
        """
        if outcome not in self.results:
            raise ValueError(f"Outcome {outcome} not found. Available: {list(self.results.keys())}")
        
        event_data = self.results[outcome]['event_study'].copy()
        
        if event_data.empty:
            print(f"Warning: No event study data for {outcome}")
            return ""
        
        # Sort by event time
        event_data = event_data.sort_values('event_time')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        event_times = event_data['event_time']
        coefficients = event_data['coef']
        
        # Handle confidence intervals
        if 'ci_lower' in event_data.columns and 'ci_upper' in event_data.columns:
            ci_lower = event_data['ci_lower']
            ci_upper = event_data['ci_upper']
            has_ci = True
        elif 'se' in event_data.columns:
            # Calculate 95% CIs from standard errors
            ci_lower = coefficients - 1.96 * event_data['se']
            ci_upper = coefficients + 1.96 * event_data['se']
            has_ci = True
        else:
            has_ci = False
        
        # Plot reference line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        
        # Plot treatment period divider
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='Treatment Begins')
        
        # Plot coefficients
        ax.plot(event_times, coefficients, 'o-', color='steelblue', linewidth=2.5, 
               markersize=8, markerfacecolor='steelblue', markeredgecolor='white', 
               markeredgewidth=1.5, label='Treatment Effects')
        
        # Plot confidence intervals
        if has_ci:
            ax.fill_between(event_times, ci_lower, ci_upper, alpha=0.2, 
                           color='steelblue', label='95% Confidence Interval')
        
        # Formatting
        if title is None:
            title = self._generate_title(outcome, "Event Study")
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if ylabel is None:
            ylabel = self._generate_ylabel(outcome)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Years Relative to Policy Reform', fontsize=14, fontweight='bold')
        
        # Customize x-axis
        ax.set_xticks(event_times)
        ax.set_xticklabels([f'{int(t)}' if t != 1 else 'Reform' for t in event_times])
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        
        # Tight layout
        plt.tight_layout()
        
        # Save in multiple formats
        saved_files = []
        for fmt in save_formats:
            filename = f"event_study_{outcome}.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
        
        plt.close()
        
        print(f"Event study plot saved: {saved_files[0]}")
        return saved_files[0]
    
    def plot_treatment_effects_summary(self,
                                     outcomes: Optional[List[str]] = None,
                                     title: str = "Treatment Effects Summary",
                                     save_formats: List[str] = ['png', 'pdf']) -> str:
        """
        Create forest plot showing treatment effects across outcomes.
        """
        if outcomes is None:
            outcomes = list(self.results.keys())
        
        # Extract aggregated effects
        effects_data = []
        for outcome in outcomes:
            if outcome not in self.results or 'aggregated' not in self.results[outcome]:
                continue
                
            agg_data = self.results[outcome]['aggregated']
            if not agg_data.empty and 'simple_att' in agg_data.columns:
                effect = agg_data['simple_att'].iloc[0]
                se = agg_data.get('simple_se', [0]).iloc[0]
                
                effects_data.append({
                    'outcome': self._format_outcome_label(outcome),
                    'effect': effect,
                    'se': se,
                    'ci_lower': effect - 1.96 * se if se > 0 else effect - 0.5,
                    'ci_upper': effect + 1.96 * se if se > 0 else effect + 0.5
                })
        
        if not effects_data:
            print("Warning: No aggregated effects data found")
            return ""
        
        effects_df = pd.DataFrame(effects_data)
        
        # Create forest plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_positions = range(len(effects_df))
        
        # Plot confidence intervals
        for i, row in effects_df.iterrows():
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 'k-', linewidth=2)
            ax.plot([row['ci_lower'], row['ci_lower']], [i-0.1, i+0.1], 'k-', linewidth=2)
            ax.plot([row['ci_upper'], row['ci_upper']], [i-0.1, i+0.1], 'k-', linewidth=2)
        
        # Plot point estimates
        colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson'][:len(effects_df)]
        ax.scatter(effects_df['effect'], y_positions, s=100, c=colors, 
                  edgecolors='white', linewidths=1.5, zorder=5)
        
        # Reference line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(effects_df['outcome'])
        ax.set_xlabel('Treatment Effect (NAEP Points)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save files
        saved_files = []
        for fmt in save_formats:
            filename = f"treatment_effects_summary.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
        
        plt.close()
        
        print(f"Treatment effects summary saved: {saved_files[0]}")
        return saved_files[0]
    
    def plot_parallel_trends_test(self,
                                outcome: str,
                                pre_periods: int = 3,
                                title: Optional[str] = None,
                                save_formats: List[str] = ['png', 'pdf']) -> str:
        """
        Create parallel trends validation plot focusing on pre-treatment periods.
        """
        if outcome not in self.results:
            raise ValueError(f"Outcome {outcome} not found")
        
        event_data = self.results[outcome]['event_study'].copy()
        
        if event_data.empty:
            print(f"Warning: No event study data for {outcome}")
            return ""
        
        # Filter to pre-treatment periods
        pre_treatment = event_data[event_data['event_time'] < 1].copy()
        pre_treatment = pre_treatment.sort_values('event_time')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot coefficients
        event_times = pre_treatment['event_time']
        coefficients = pre_treatment['coef']
        
        # Confidence intervals
        if 'ci_lower' in pre_treatment.columns:
            ci_lower = pre_treatment['ci_lower']
            ci_upper = pre_treatment['ci_upper']
        elif 'se' in pre_treatment.columns:
            ci_lower = coefficients - 1.96 * pre_treatment['se']
            ci_upper = coefficients + 1.96 * pre_treatment['se']
        else:
            ci_lower = ci_upper = None
        
        # Plot reference line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        
        # Plot coefficients and CIs
        ax.plot(event_times, coefficients, 'o-', color='steelblue', linewidth=2.5,
               markersize=10, markerfacecolor='steelblue', markeredgecolor='white',
               markeredgewidth=2, label='Pre-treatment Effects')
        
        if ci_lower is not None and ci_upper is not None:
            ax.fill_between(event_times, ci_lower, ci_upper, alpha=0.2,
                           color='steelblue', label='95% Confidence Interval')
        
        # Test statistical significance of pre-trends
        if len(coefficients) > 1 and 'pvalue' in pre_treatment.columns:
            p_values = pre_treatment['pvalue'].dropna()
            significant = (p_values < 0.05).sum()
            
            # Add text box with test results
            textstr = f'Pre-treatment periods: {len(coefficients)}\n'
            textstr += f'Significant effects (p<0.05): {significant}\n'
            if significant == 0:
                textstr += 'Parallel trends supported ✓'
            else:
                textstr += 'Parallel trends violated ✗'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        # Formatting
        if title is None:
            title = f"Parallel Trends Test: {self._format_outcome_label(outcome)}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Years Before Reform', fontsize=14, fontweight='bold')
        ax.set_ylabel(self._generate_ylabel(outcome), fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save files
        saved_files = []
        for fmt in save_formats:
            filename = f"parallel_trends_{outcome}.{fmt}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
        
        plt.close()
        
        print(f"Parallel trends plot saved: {saved_files[0]}")
        return saved_files[0]
    
    def create_all_visualizations(self) -> Dict[str, List[str]]:
        """Generate complete set of visualizations for all outcomes."""
        print("Creating comprehensive visualization suite...")
        
        all_files = {}
        
        # Event study plots for each outcome
        for outcome in self.results.keys():
            print(f"\\nProcessing {outcome}...")
            
            outcome_files = []
            
            try:
                # Event study plot
                event_file = self.plot_event_study(outcome)
                if event_file:
                    outcome_files.append(event_file)
                
                # Parallel trends test
                trends_file = self.plot_parallel_trends_test(outcome)
                if trends_file:
                    outcome_files.append(trends_file)
                
                all_files[outcome] = outcome_files
                
            except Exception as e:
                print(f"Error creating plots for {outcome}: {e}")
                continue
        
        # Summary plots
        try:
            summary_file = self.plot_treatment_effects_summary()
            if summary_file:
                all_files['summary'] = [summary_file]
        except Exception as e:
            print(f"Error creating summary plot: {e}")
        
        return all_files
    
    def _generate_title(self, outcome: str, plot_type: str) -> str:
        """Generate descriptive title for plot."""
        outcome_label = self._format_outcome_label(outcome)
        return f"{plot_type}: {outcome_label}"
    
    def _format_outcome_label(self, outcome: str) -> str:
        """Convert outcome variable name to readable label."""
        # Parse outcome name
        parts = outcome.split('_')
        
        if 'math' in outcome:
            subject = 'Mathematics'
        elif 'reading' in outcome:
            subject = 'Reading'
        else:
            subject = 'Achievement'
        
        if 'grade4' in outcome:
            grade = 'Grade 4'
        elif 'grade8' in outcome:
            grade = 'Grade 8'
        else:
            grade = ''
        
        if 'gap' in outcome:
            metric = 'Achievement Gap'
        elif 'score' in outcome:
            metric = 'Score'
        else:
            metric = 'Outcome'
        
        return f"{subject} {grade} {metric}".strip()
    
    def _generate_ylabel(self, outcome: str) -> str:
        """Generate y-axis label."""
        if 'gap' in outcome:
            return "Achievement Gap (NAEP Points)"
        elif 'score' in outcome:
            return "Achievement Score (NAEP Points)" 
        else:
            return "Treatment Effect"


if __name__ == "__main__":
    # Create visualizer and generate all plots
    visualizer = EventStudyVisualizer()
    
    # Generate complete visualization suite
    all_plots = visualizer.create_all_visualizations()
    
    # Summary report
    print(f"\\n{'='*60}")
    print("EVENT STUDY VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    
    total_files = sum(len(files) for files in all_plots.values())
    print(f"Total plots created: {total_files}")
    print(f"Outcomes processed: {len([k for k in all_plots.keys() if k != 'summary'])}")
    
    # List all generated files
    for outcome, files in all_plots.items():
        print(f"\\n{outcome}:")
        for file in files:
            print(f"  - {file}")
    
    print(f"\\nAll figures saved to: {visualizer.figures_dir}")