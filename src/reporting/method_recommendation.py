#!/usr/bin/env python
"""
Phase 5.2: Method Comparison and Recommendation Engine
Automated selection and ranking of robust analysis methods based on data characteristics.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class MethodPerformanceMetric:
    """Performance metrics for a robust analysis method."""
    method_name: str
    reliability_score: float  # 0-100
    computational_cost: str  # "low", "medium", "high"
    sample_size_requirement: int
    cluster_requirement: int
    strengths: List[str]
    weaknesses: List[str]
    recommended_when: List[str]
    not_recommended_when: List[str]


class MethodRecommendationEngine:
    """
    Intelligent method selection and comparison framework.
    
    Provides:
    - Data-driven method recommendation
    - Performance comparison across methods
    - Robustness ranking and scoring
    - Situational guidance for method selection
    """
    
    def __init__(self, data: pd.DataFrame, output_dir: Path = None):
        """
        Initialize method recommendation engine.
        
        Args:
            data: Analysis panel dataset
            output_dir: Directory for recommendation outputs
        """
        self.data = data.copy()
        self.output_dir = Path(output_dir) if output_dir else Path("output/method_recommendations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.data_characteristics = {}
        self.method_scores = {}
        
        # Initialize method database
        self.method_database = self._initialize_method_database()
        
        # Configure logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup dedicated logging for method recommendations."""
        log_file = self.output_dir / "method_recommendation.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def _initialize_method_database(self) -> Dict[str, MethodPerformanceMetric]:
        """Initialize database of robust analysis methods."""
        methods = {
            "cluster_robust_se": MethodPerformanceMetric(
                method_name="Cluster Robust Standard Errors",
                reliability_score=85.0,
                computational_cost="low",
                sample_size_requirement=200,
                cluster_requirement=30,
                strengths=[
                    "Fast computation",
                    "Standard in econometrics",
                    "Good asymptotic properties"
                ],
                weaknesses=[
                    "Poor performance with few clusters",
                    "Assumes large cluster asymptotics"
                ],
                recommended_when=[
                    "Large number of clusters (>30)",
                    "Balanced cluster sizes",
                    "Standard DiD applications"
                ],
                not_recommended_when=[
                    "Few clusters (<20)",
                    "Very unbalanced clusters",
                    "Small sample sizes"
                ]
            ),
            
            "wild_cluster_bootstrap": MethodPerformanceMetric(
                method_name="Wild Cluster Bootstrap",
                reliability_score=92.0,
                computational_cost="medium",
                sample_size_requirement=100,
                cluster_requirement=10,
                strengths=[
                    "Robust to few clusters",
                    "Good finite sample properties",
                    "Handles unbalanced clusters well"
                ],
                weaknesses=[
                    "Computationally intensive",
                    "Implementation complexity"
                ],
                recommended_when=[
                    "Few clusters (10-30)",
                    "Unbalanced cluster sizes",
                    "Small to medium samples"
                ],
                not_recommended_when=[
                    "Very few clusters (<5)",
                    "Extremely large datasets",
                    "Time-sensitive analyses"
                ]
            ),
            
            "cluster_bootstrap": MethodPerformanceMetric(
                method_name="Cluster Bootstrap",
                reliability_score=88.0,
                computational_cost="medium",
                sample_size_requirement=150,
                cluster_requirement=15,
                strengths=[
                    "Good cluster resampling",
                    "Preserves dependence structure",
                    "Flexible implementation"
                ],
                weaknesses=[
                    "Moderate computational cost",
                    "Bootstrap may not converge"
                ],
                recommended_when=[
                    "Moderate number of clusters (15-50)",
                    "Complex dependence structure",
                    "Non-normal error distributions"
                ],
                not_recommended_when=[
                    "Very few clusters",
                    "Extremely large datasets",
                    "Simple linear models"
                ]
            ),
            
            "jackknife_inference": MethodPerformanceMetric(
                method_name="Jackknife Inference",
                reliability_score=83.0,
                computational_cost="low",
                sample_size_requirement=100,
                cluster_requirement=10,
                strengths=[
                    "Simple implementation",
                    "Good bias correction",
                    "Robust to outliers"
                ],
                weaknesses=[
                    "Conservative inference",
                    "Limited theoretical justification for clustering"
                ],
                recommended_when=[
                    "Exploratory analysis",
                    "Suspected outliers",
                    "Simple robustness check"
                ],
                not_recommended_when=[
                    "Primary inference method",
                    "Complex models",
                    "Publication-quality analysis"
                ]
            ),
            
            "permutation_test": MethodPerformanceMetric(
                method_name="Permutation Test",
                reliability_score=95.0,
                computational_cost="high",
                sample_size_requirement=50,
                cluster_requirement=5,
                strengths=[
                    "Exact finite sample inference",
                    "No distributional assumptions",
                    "Robust to all violations"
                ],
                weaknesses=[
                    "Very computationally intensive",
                    "Limited to simple hypotheses",
                    "May be conservative"
                ],
                recommended_when=[
                    "Very few clusters (<10)",
                    "Non-standard distributions",
                    "Exact inference required"
                ],
                not_recommended_when=[
                    "Large datasets",
                    "Complex models",
                    "Multiple hypotheses"
                ]
            ),
            
            "randomization_inference": MethodPerformanceMetric(
                method_name="Randomization Inference",
                reliability_score=93.0,
                computational_cost="high",
                sample_size_requirement=30,
                cluster_requirement=5,
                strengths=[
                    "Model-free inference",
                    "Exact p-values",
                    "Robust to heterogeneity"
                ],
                weaknesses=[
                    "Computationally demanding",
                    "Limited to randomized designs",
                    "Complex implementation"
                ],
                recommended_when=[
                    "Randomized or quasi-randomized design",
                    "Very few clusters",
                    "Skeptical audience"
                ],
                not_recommended_when=[
                    "Observational studies",
                    "Large datasets",
                    "Standard applications"
                ]
            )
        }
        
        return methods
        
    def analyze_data_characteristics(self) -> Dict[str, Any]:
        """
        Analyze dataset characteristics relevant for method selection.
        
        Returns:
            Dict containing data characteristics
        """
        self.logger.info("Analyzing data characteristics for method recommendation...")
        
        characteristics = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": {
                "total_observations": len(self.data),
                "states": self.data['state'].nunique(),
                "years": self.data['year'].nunique(),
                "avg_obs_per_state": len(self.data) / self.data['state'].nunique()
            },
            "treatment_structure": {
                "treated_states": self.data[self.data['post_treatment'] == 1]['state'].nunique(),
                "control_states": self.data[self.data['post_treatment'] == 0]['state'].nunique(),
                "treatment_variation": self.data['post_treatment'].var(),
                "staggered_adoption": self._check_staggered_adoption()
            },
            "clustering_characteristics": {
                "n_clusters": self.data['state'].nunique(),
                "cluster_sizes": self.data['state'].value_counts().to_dict(),
                "min_cluster_size": self.data['state'].value_counts().min(),
                "max_cluster_size": self.data['state'].value_counts().max(),
                "cluster_balance": self._assess_cluster_balance()
            },
            "outcome_characteristics": {
                "outcome_variables": self._identify_outcome_variables(),
                "missing_patterns": self._analyze_missing_patterns(),
                "outlier_presence": self._detect_outliers()
            },
            "computational_constraints": {
                "large_dataset": len(self.data) > 10000,
                "many_clusters": self.data['state'].nunique() > 50,
                "complex_structure": self._assess_complexity()
            }
        }
        
        self.data_characteristics = characteristics
        self.logger.info("Data characteristics analysis complete")
        
        return characteristics
        
    def _check_staggered_adoption(self) -> bool:
        """Check if treatment adoption is staggered over time."""
        if 'treatment_year' not in self.data.columns:
            return False
            
        treatment_years = self.data[self.data['post_treatment'] == 1]['treatment_year'].dropna()
        return treatment_years.nunique() > 1
        
    def _assess_cluster_balance(self) -> Dict[str, Any]:
        """Assess balance of cluster sizes."""
        cluster_sizes = self.data['state'].value_counts()
        
        return {
            "coefficient_of_variation": cluster_sizes.std() / cluster_sizes.mean(),
            "size_range_ratio": cluster_sizes.max() / cluster_sizes.min(),
            "balanced": (cluster_sizes.std() / cluster_sizes.mean()) < 0.2
        }
        
    def _identify_outcome_variables(self) -> List[str]:
        """Identify outcome variables in the dataset."""
        outcome_vars = []
        
        for col in self.data.columns:
            if any(x in col.lower() for x in ['score', 'gap', 'achievement', 'rate']):
                if col not in ['time_trend']:  # Exclude non-outcome variables
                    outcome_vars.append(col)
                    
        return outcome_vars
        
    def _analyze_missing_patterns(self) -> Dict[str, float]:
        """Analyze missing data patterns for outcome variables."""
        outcome_vars = self._identify_outcome_variables()
        missing_patterns = {}
        
        for var in outcome_vars:
            if var in self.data.columns:
                missing_patterns[var] = self.data[var].isna().mean()
                
        return missing_patterns
        
    def _detect_outliers(self) -> Dict[str, bool]:
        """Detect presence of outliers in outcome variables."""
        outcome_vars = self._identify_outcome_variables()
        outlier_presence = {}
        
        for var in outcome_vars:
            if var in self.data.columns and self.data[var].dtype in ['float64', 'int64']:
                # Use IQR method for outlier detection
                Q1 = self.data[var].quantile(0.25)
                Q3 = self.data[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.data[var] < lower_bound) | 
                           (self.data[var] > upper_bound)).sum()
                outlier_presence[var] = outliers > 0
                
        return outlier_presence
        
    def _assess_complexity(self) -> bool:
        """Assess whether the data structure is complex."""
        complexity_factors = [
            len(self.data) > 5000,  # Large dataset
            self.data['state'].nunique() > 40,  # Many clusters
            len(self._identify_outcome_variables()) > 5,  # Many outcomes
            self._check_staggered_adoption()  # Staggered treatment
        ]
        
        return sum(complexity_factors) >= 2
        
    def score_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Score each method based on data characteristics.
        
        Returns:
            Dict containing method scores and recommendations
        """
        self.logger.info("Scoring methods based on data characteristics...")
        
        if not self.data_characteristics:
            self.analyze_data_characteristics()
            
        method_scores = {}
        
        for method_name, method_info in self.method_database.items():
            score = self._calculate_method_score(method_info)
            
            method_scores[method_name] = {
                "base_reliability": method_info.reliability_score,
                "data_suitability_score": score["suitability"],
                "overall_score": score["overall"],
                "recommendation_strength": score["strength"],
                "specific_concerns": score["concerns"],
                "computational_feasibility": score["computational"],
                "method_info": method_info
            }
            
        # Rank methods by overall score
        ranked_methods = sorted(
            method_scores.items(), 
            key=lambda x: x[1]["overall_score"], 
            reverse=True
        )
        
        self.method_scores = dict(ranked_methods)
        self.logger.info("Method scoring complete")
        
        return self.method_scores
        
    def _calculate_method_score(self, method: MethodPerformanceMetric) -> Dict[str, Any]:
        """Calculate overall score for a method given data characteristics."""
        chars = self.data_characteristics
        
        # Suitability scoring based on requirements
        suitability_score = 100.0
        concerns = []
        
        # Sample size check
        if chars["sample_size"]["total_observations"] < method.sample_size_requirement:
            penalty = 30.0
            suitability_score -= penalty
            concerns.append(f"Sample size ({chars['sample_size']['total_observations']}) below requirement ({method.sample_size_requirement})")
            
        # Cluster requirement check
        if chars["clustering_characteristics"]["n_clusters"] < method.cluster_requirement:
            penalty = 40.0
            suitability_score -= penalty
            concerns.append(f"Too few clusters ({chars['clustering_characteristics']['n_clusters']}) for method requirement ({method.cluster_requirement})")
            
        # Cluster balance assessment
        if not chars["clustering_characteristics"]["cluster_balance"]["balanced"]:
            if method.method_name == "Cluster Robust Standard Errors":
                penalty = 25.0
                suitability_score -= penalty
                concerns.append("Unbalanced clusters may affect cluster robust SE performance")
            elif method.method_name == "Wild Cluster Bootstrap":
                # Wild bootstrap handles unbalanced clusters well
                suitability_score += 5.0
                
        # Few clusters penalty/bonus (refined for econometric practice)
        n_clusters = chars["clustering_characteristics"]["n_clusters"]
        if n_clusters < 15:
            if method.method_name in ["Wild Cluster Bootstrap", "Permutation Test", "Randomization Inference"]:
                suitability_score += 10.0  # Bonus for few-cluster methods
            else:
                penalty = 20.0
                suitability_score -= penalty
                concerns.append(f"Method not recommended for few clusters ({n_clusters})")
        elif n_clusters >= 30:
            # For sufficient clusters, prioritize practical methods
            if method.method_name == "Cluster Robust Standard Errors":
                suitability_score += 10.0  # Bonus for standard method with sufficient clusters
            elif method.method_name == "Wild Cluster Bootstrap":
                suitability_score += 8.0   # Still good but not as critical
            elif method.method_name in ["Permutation Test", "Randomization Inference"]:
                # Reduce preference for computationally intensive methods when simpler ones suffice
                suitability_score -= 5.0
                
        # Large dataset considerations (enhanced)
        if chars["computational_constraints"]["large_dataset"]:
            if method.computational_cost == "high":
                penalty = 20.0  # Increased penalty for computational intensity
                suitability_score -= penalty
                concerns.append("Computationally intensive for large dataset")
            elif method.computational_cost == "low":
                suitability_score += 10.0  # Increased bonus for efficiency
        
        # Standard econometric practice preference
        if n_clusters >= 30 and chars["sample_size"]["total_observations"] >= 300:
            if method.method_name == "Cluster Robust Standard Errors":
                suitability_score += 5.0  # Additional bonus for standard practice
            elif method.method_name == "Wild Cluster Bootstrap":
                suitability_score += 3.0  # Good alternative
                
        # Computational feasibility
        computational_score = 100.0
        if method.computational_cost == "high" and chars["computational_constraints"]["large_dataset"]:
            computational_score = 60.0
        elif method.computational_cost == "medium":
            computational_score = 80.0
            
        # Overall score (weighted combination)
        overall_score = (
            0.4 * method.reliability_score +
            0.4 * max(0, suitability_score) +
            0.2 * computational_score
        )
        
        # Recommendation strength
        if overall_score >= 85:
            strength = "Strongly Recommended"
        elif overall_score >= 70:
            strength = "Recommended"
        elif overall_score >= 55:
            strength = "Conditionally Recommended"
        else:
            strength = "Not Recommended"
            
        return {
            "suitability": max(0, suitability_score),
            "overall": overall_score,
            "strength": strength,
            "concerns": concerns,
            "computational": computational_score
        }
        
    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate comprehensive method recommendations.
        
        Returns:
            Dict containing recommendations and rationale
        """
        self.logger.info("Generating method recommendations...")
        
        if not self.method_scores:
            self.score_methods()
            
        # Get top recommendations
        top_methods = list(self.method_scores.keys())[:3]
        
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "sample_size": self.data_characteristics["sample_size"]["total_observations"],
                "n_clusters": self.data_characteristics["clustering_characteristics"]["n_clusters"],
                "treatment_variation": self.data_characteristics["treatment_structure"]["treatment_variation"]
            },
            "primary_recommendation": {
                "method": top_methods[0],
                "score": self.method_scores[top_methods[0]]["overall_score"],
                "rationale": self._generate_rationale(top_methods[0])
            },
            "alternative_methods": [
                {
                    "method": method,
                    "score": self.method_scores[method]["overall_score"],
                    "rationale": self._generate_rationale(method)
                }
                for method in top_methods[1:3]
            ],
            "method_comparison": self._create_method_comparison(),
            "specific_guidance": self._generate_specific_guidance(),
            "implementation_notes": self._generate_implementation_notes()
        }
        
        self.logger.info("Method recommendations generated successfully")
        
        return recommendations
        
    def _generate_rationale(self, method_name: str) -> str:
        """Generate rationale for method recommendation."""
        method_score = self.method_scores[method_name]
        method_info = method_score["method_info"]
        
        rationale_parts = [
            f"Overall score: {method_score['overall_score']:.1f}/100",
            f"Reliability: {method_info.reliability_score}/100"
        ]
        
        # Add specific strengths
        relevant_strengths = []
        chars = self.data_characteristics
        
        if chars["clustering_characteristics"]["n_clusters"] < 20 and "few clusters" in str(method_info.strengths).lower():
            relevant_strengths.append("Handles few clusters well")
            
        if not chars["clustering_characteristics"]["cluster_balance"]["balanced"] and "unbalanced" in str(method_info.strengths).lower():
            relevant_strengths.append("Robust to unbalanced clusters")
            
        if relevant_strengths:
            rationale_parts.append(f"Key advantages: {', '.join(relevant_strengths)}")
            
        # Add concerns
        if method_score["specific_concerns"]:
            rationale_parts.append(f"Concerns: {'; '.join(method_score['specific_concerns'])}")
            
        return ". ".join(rationale_parts)
        
    def _create_method_comparison(self) -> Dict[str, Any]:
        """Create detailed method comparison table."""
        comparison = {
            "methods": [],
            "criteria": [
                "Overall Score",
                "Reliability",
                "Data Suitability", 
                "Computational Cost",
                "Recommendation"
            ]
        }
        
        for method_name, scores in self.method_scores.items():
            method_data = {
                "method_name": method_name,
                "overall_score": scores["overall_score"],
                "reliability": scores["base_reliability"],
                "suitability": scores["data_suitability_score"],
                "computational_cost": scores["method_info"].computational_cost,
                "recommendation": scores["recommendation_strength"],
                "concerns": scores["specific_concerns"]
            }
            comparison["methods"].append(method_data)
            
        return comparison
        
    def _generate_specific_guidance(self) -> Dict[str, str]:
        """Generate specific guidance based on data characteristics."""
        chars = self.data_characteristics
        guidance = {}
        
        # Cluster-specific guidance
        n_clusters = chars["clustering_characteristics"]["n_clusters"]
        if n_clusters < 10:
            guidance["clustering"] = ("Very few clusters detected. Consider wild cluster bootstrap, "
                                    "permutation tests, or randomization inference for robust results.")
        elif n_clusters < 20:
            guidance["clustering"] = ("Moderate number of clusters. Wild cluster bootstrap recommended "
                                    "over standard cluster robust standard errors.")
        else:
            guidance["clustering"] = ("Sufficient clusters for standard methods. Cluster robust "
                                    "standard errors should perform well.")
            
        # Sample size guidance
        n_obs = chars["sample_size"]["total_observations"]
        if n_obs < 200:
            guidance["sample_size"] = ("Small sample size. Consider exact or bootstrap methods "
                                     "rather than asymptotic approaches.")
        elif n_obs > 5000:
            guidance["sample_size"] = ("Large sample size. Computational efficiency becomes important. "
                                     "Consider avoiding permutation-based methods.")
            
        # Balance guidance
        if not chars["clustering_characteristics"]["cluster_balance"]["balanced"]:
            guidance["balance"] = ("Unbalanced cluster sizes detected. Wild cluster bootstrap "
                                 "recommended over standard cluster robust standard errors.")
            
        return guidance
        
    def _generate_implementation_notes(self) -> Dict[str, List[str]]:
        """Generate implementation notes for recommended methods."""
        top_methods = list(self.method_scores.keys())[:3]
        
        implementation_notes = {}
        
        for method in top_methods:
            method_info = self.method_scores[method]["method_info"]
            
            notes = [
                f"Computational cost: {method_info.computational_cost}",
                f"Minimum sample size: {method_info.sample_size_requirement}",
                f"Minimum clusters: {method_info.cluster_requirement}"
            ]
            
            # Add method-specific implementation details
            if method == "wild_cluster_bootstrap":
                notes.extend([
                    "Use Rademacher weights for best performance",
                    "Bootstrap iterations: 999 or 1999 (odd numbers)",
                    "Cluster at state level for policy analysis"
                ])
            elif method == "cluster_robust_se":
                notes.extend([
                    "Ensure degrees of freedom correction",
                    "Use HC1 or HC2 finite sample corrections",
                    "Check for sufficient within-cluster variation"
                ])
            elif method == "permutation_test":
                notes.extend([
                    "Requires careful null hypothesis specification",
                    "Use at least 999 permutations",
                    "Consider computational time for large datasets"
                ])
                
            implementation_notes[method] = notes
            
        return implementation_notes
        
    def export_recommendations(self, format: str = "both") -> List[str]:
        """
        Export method recommendations to files.
        
        Args:
            format: "json", "markdown", or "both"
            
        Returns:
            List of generated file paths
        """
        recommendations = self.generate_recommendations()
        output_files = []
        
        if format in ["json", "both"]:
            json_path = self.output_dir / "method_recommendations.json"
            with open(json_path, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            output_files.append(str(json_path))
            
        if format in ["markdown", "both"]:
            md_path = self.output_dir / "method_recommendations.md"
            markdown_content = self._create_markdown_report(recommendations)
            with open(md_path, 'w') as f:
                f.write(markdown_content)
            output_files.append(str(md_path))
            
        return output_files
        
    def _create_markdown_report(self, recommendations: Dict) -> str:
        """Create markdown report for method recommendations."""
        report = f"""# Method Recommendation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Executive Summary

**Primary Recommendation**: {recommendations['primary_recommendation']['method']}  
**Score**: {recommendations['primary_recommendation']['score']:.1f}/100  

### Data Characteristics
- **Sample Size**: {recommendations['data_summary']['sample_size']:,} observations
- **Clusters**: {recommendations['data_summary']['n_clusters']} states
- **Treatment Variation**: {recommendations['data_summary']['treatment_variation']:.3f}

## Detailed Recommendations

### 🥇 Primary Method: {recommendations['primary_recommendation']['method']}

**Score**: {recommendations['primary_recommendation']['score']:.1f}/100

**Rationale**: {recommendations['primary_recommendation']['rationale']}

### 🥈 Alternative Methods
"""

        for i, alt in enumerate(recommendations['alternative_methods'], 2):
            report += f"""
#### Alternative {i-1}: {alt['method']}

**Score**: {alt['score']:.1f}/100

**Rationale**: {alt['rationale']}
"""

        # Method comparison table
        report += """
## Method Comparison

| Method | Overall Score | Reliability | Suitability | Cost | Recommendation |
|--------|---------------|-------------|-------------|------|----------------|
"""
        
        for method in recommendations['method_comparison']['methods']:
            report += f"| {method['method_name']} | {method['overall_score']:.1f} | {method['reliability']:.1f} | {method['suitability']:.1f} | {method['computational_cost']} | {method['recommendation']} |\n"
            
        # Specific guidance
        report += "\n## Specific Guidance\n\n"
        for category, guidance in recommendations['specific_guidance'].items():
            report += f"**{category.title()}**: {guidance}\n\n"
            
        # Implementation notes
        report += "## Implementation Notes\n\n"
        for method, notes in recommendations['implementation_notes'].items():
            report += f"### {method}\n"
            for note in notes:
                report += f"- {note}\n"
            report += "\n"
            
        report += "\n---\n*This recommendation report provides data-driven method selection for robust policy evaluation.*"
        
        return report


def run_method_recommendation(data_path: str = "data/final/analysis_panel.csv") -> List[str]:
    """
    Run comprehensive method recommendation analysis.
    
    Args:
        data_path: Path to analysis panel dataset
        
    Returns:
        List of generated file paths
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize recommendation engine
    engine = MethodRecommendationEngine(data)
    
    # Generate recommendations
    output_files = engine.export_recommendations()
    
    return output_files


if __name__ == "__main__":
    # Run method recommendation if called directly
    output_files = run_method_recommendation()
    print(f"✅ Method recommendation analysis complete!")
    for file_path in output_files:
        print(f"📋 Report available at: {file_path}")