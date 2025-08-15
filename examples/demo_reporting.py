#!/usr/bin/env python
"""
Demo script for Phase 4 reporting capabilities.

This script demonstrates the new reporting modules with sample data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reporting import (
    LaTeXTableGenerator,
    VisualizationSuite,
    PolicyBriefGenerator,
    TechnicalAppendixGenerator
)


def create_sample_results():
    """Create sample analysis results for demonstration."""
    return {
        "twfe": {
            "math_grade4_gap": {"coefficient": 0.15, "std_err": 0.08, "p_value": 0.06},
            "math_grade8_gap": {"coefficient": 0.12, "std_err": 0.09, "p_value": 0.18},
            "reading_grade4_gap": {"coefficient": 0.08, "std_err": 0.07, "p_value": 0.25}
        },
        "bootstrap_inference": {
            "math_grade4_gap": {"coefficient": 0.14, "std_err": 0.09, "p_value": 0.12},
            "math_grade8_gap": {"coefficient": 0.11, "std_err": 0.10, "p_value": 0.27},
            "reading_grade4_gap": {"coefficient": 0.07, "std_err": 0.08, "p_value": 0.38}
        },
        "jackknife_inference": {
            "math_grade4_gap": {"coefficient": 0.13, "std_err": 0.10, "p_value": 0.19},
            "math_grade8_gap": {"coefficient": 0.10, "std_err": 0.11, "p_value": 0.35},
            "reading_grade4_gap": {"coefficient": 0.06, "std_err": 0.09, "p_value": 0.48}
        },
        "wild_cluster_bootstrap": {
            "math_grade4_gap": {"coefficient": 0.12, "std_err": 0.11, "p_value": 0.28},
            "math_grade8_gap": {"coefficient": 0.09, "std_err": 0.12, "p_value": 0.45},
            "reading_grade4_gap": {"coefficient": 0.05, "std_err": 0.10, "p_value": 0.62}
        }
    }


def create_sample_effect_sizes():
    """Create sample effect size data."""
    return {
        "math_grade4_gap": {
            "cohens_d": 0.12,
            "ci_lower": 0.02,
            "ci_upper": 0.22,
            "cross_method_range": [0.10, 0.16],
            "cross_method_consistency": "High"
        },
        "math_grade8_gap": {
            "cohens_d": 0.08,
            "ci_lower": -0.02,
            "ci_upper": 0.18,
            "cross_method_range": [0.06, 0.12],
            "cross_method_consistency": "High"
        },
        "reading_grade4_gap": {
            "cohens_d": 0.05,
            "ci_lower": -0.05,
            "ci_upper": 0.15,
            "cross_method_range": [0.03, 0.08],
            "cross_method_consistency": "Medium"
        }
    }


def create_sample_power_results():
    """Create sample power analysis results."""
    return {
        "by_outcome": {
            "math_grade4_gap": {
                "post_hoc_power": 0.65,
                "minimum_detectable_effect": 0.25,
                "required_sample_size": 1200,
                "current_sample_size": 700,
                "adequately_powered": False
            },
            "math_grade8_gap": {
                "post_hoc_power": 0.48,
                "minimum_detectable_effect": 0.28,
                "required_sample_size": 1500,
                "current_sample_size": 700,
                "adequately_powered": False
            },
            "reading_grade4_gap": {
                "post_hoc_power": 0.35,
                "minimum_detectable_effect": 0.32,
                "required_sample_size": 1800,
                "current_sample_size": 700,
                "adequately_powered": False
            }
        },
        "overall_assessment": {
            "average_power": 0.49,
            "median_power": 0.48,
            "min_power": 0.35,
            "proportion_adequate": 0.00,
            "recommendation": "Increase sample size significantly for improved power"
        }
    }


def create_sample_data():
    """Create sample dataset."""
    return pd.DataFrame({
        "state": ["CA", "TX", "NY", "FL", "PA"] * 20,
        "year": [2020] * 100,
        "math_grade4_swd_score": [250, 260, 240, 245, 255] * 20,
        "math_grade4_gap": [25, 30, 20, 22, 28] * 20,
        "reading_grade4_swd_score": [245, 255, 235, 240, 250] * 20,
        "reading_grade4_gap": [20, 25, 15, 18, 23] * 20,
        "per_pupil_total": [12000, 11000, 13000, 11500, 12500] * 20,
        "inclusion_rate": [75, 80, 70, 72, 78] * 20,
        "post_treatment": [1, 0, 1, 0, 1] * 20
    })


def demo_latex_tables():
    """Demonstrate LaTeX table generation."""
    print("📊 Generating LaTeX Tables...")
    
    generator = LaTeXTableGenerator()
    results = create_sample_results()
    effect_sizes = create_sample_effect_sizes()
    power_results = create_sample_power_results()
    sample_data = create_sample_data()
    
    # Generate tables
    table4_path = generator.create_multi_method_comparison_table(results)
    table5_path = generator.create_effect_size_table(effect_sizes)
    table6_path = generator.create_power_analysis_table(power_results)
    table1_path = generator.create_summary_statistics_table(sample_data)
    
    print(f"✅ Table 4 (Multi-method comparison): {table4_path}")
    print(f"✅ Table 5 (Effect sizes): {table5_path}")
    print(f"✅ Table 6 (Power analysis): {table6_path}")
    print(f"✅ Table 1 (Summary statistics): {table1_path}")
    
    # Combine for paper
    combined_path = generator.combine_tables_for_paper([
        "table4_multi_method_comparison.tex",
        "table5_effect_sizes.tex",
        "table6_power_analysis.tex",
        "table1_summary_statistics.tex"
    ])
    print(f"✅ Combined tables: {combined_path}")


def demo_visualizations():
    """Demonstrate advanced visualization creation."""
    print("\n📈 Generating Advanced Visualizations...")
    
    suite = VisualizationSuite()
    results = create_sample_results()
    
    # Create specification curve data
    spec_results = {
        "math_grade4_gap": [
            {"coefficient": 0.15, "std_err": 0.08},
            {"coefficient": 0.12, "std_err": 0.09},
            {"coefficient": 0.18, "std_err": 0.07},
            {"coefficient": 0.11, "std_err": 0.10}
        ]
    }
    
    # Generate visualizations
    forest_path = suite.create_forest_plot_grid(results)
    funnel_path = suite.create_robustness_funnel_plot(spec_results)
    heatmap_path = suite.create_method_reliability_heatmap(results)
    spec_curve_path = suite.create_enhanced_specification_curve(spec_results)
    
    print(f"✅ Forest plot grid: {forest_path}")
    print(f"✅ Funnel plot: {funnel_path}")
    print(f"✅ Reliability heatmap: {heatmap_path}")
    print(f"✅ Enhanced specification curve: {spec_curve_path}")


def demo_policy_brief():
    """Demonstrate policy brief generation."""
    print("\n📝 Generating Policy Brief...")
    
    generator = PolicyBriefGenerator()
    results = create_sample_results()
    
    brief_path = generator.create_policy_brief(results)
    infographic_data = generator.create_infographic_data(results)
    
    print(f"✅ Policy brief: {brief_path}")
    print(f"✅ Infographic data: {infographic_data}")


def demo_technical_appendix():
    """Demonstrate technical appendix generation."""
    print("\n📚 Generating Technical Appendix...")
    
    generator = TechnicalAppendixGenerator()
    results = {
        "robustness": create_sample_results(),
        "power": create_sample_power_results(),
        "effect_sizes": create_sample_effect_sizes()
    }
    
    appendix_path = generator.create_technical_appendix(results)
    metadata_path = generator.export_metadata(results)
    
    print(f"✅ Technical appendix: {appendix_path}")
    print(f"✅ Analysis metadata: {metadata_path}")


def main():
    """Run the reporting demonstration."""
    print("🚀 Phase 4: Report Generation Demonstration")
    print("=" * 50)
    
    try:
        demo_latex_tables()
        demo_visualizations()
        demo_policy_brief()
        demo_technical_appendix()
        
        print("\n🎉 All reporting modules demonstrated successfully!")
        print("\nGenerated files:")
        print("📁 output/tables/     - LaTeX publication tables")
        print("📁 output/figures/    - Publication-quality visualizations") 
        print("📁 output/briefs/     - Policy briefs for stakeholders")
        print("📁 output/reports/    - Technical documentation")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())