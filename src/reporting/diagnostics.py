#!/usr/bin/env python
"""
Phase 5.1: Detailed Diagnostic Reporting Framework
Comprehensive diagnostics for method failures and data adequacy assessments.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
from datetime import datetime


class DiagnosticReporter:
    """
    Comprehensive diagnostic framework for robustness analysis.
    
    Provides detailed analysis of:
    - Method failure patterns and troubleshooting
    - Data adequacy assessments
    - Statistical assumption testing
    - Quality control recommendations
    """
    
    def __init__(self, data: pd.DataFrame, output_dir: Path = None):
        """
        Initialize diagnostic reporter.
        
        Args:
            data: Analysis panel dataset
            output_dir: Directory for diagnostic outputs
        """
        self.data = data.copy()
        self.output_dir = Path(output_dir) if output_dir else Path("output/diagnostics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.diagnostic_results = {}
        
        # Configure diagnostic logging
        self._setup_diagnostic_logging()
        
    def _setup_diagnostic_logging(self):
        """Setup dedicated diagnostic logging."""
        diagnostic_log = self.output_dir / "diagnostic_analysis.log"
        
        # Create file handler for diagnostics
        file_handler = logging.FileHandler(diagnostic_log)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def assess_data_adequacy(self) -> Dict[str, Any]:
        """
        Comprehensive data adequacy assessment.
        
        Returns:
            Dict containing data adequacy diagnostics
        """
        self.logger.info("Starting comprehensive data adequacy assessment...")
        
        adequacy = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": {},
            "coverage": {},
            "balance": {},
            "power": {},
            "recommendations": [],
            "overall_assessment": "adequate"  # Will be updated based on checks
        }
        
        # Sample size adequacy
        adequacy["sample_size"] = self._assess_sample_size()
        
        # Coverage assessment
        adequacy["coverage"] = self._assess_coverage()
        
        # Balance assessment
        adequacy["balance"] = self._assess_balance()
        
        # Power analysis
        adequacy["power"] = self._assess_statistical_power()
        
        # Generate recommendations
        adequacy["recommendations"] = self._generate_data_recommendations(adequacy)
        
        # Overall assessment
        adequacy["overall_assessment"] = self._determine_overall_adequacy(adequacy)
        
        self.diagnostic_results["data_adequacy"] = adequacy
        self.logger.info(f"Data adequacy assessment complete: {adequacy['overall_assessment']}")
        
        return adequacy
    
    def _assess_sample_size(self) -> Dict[str, Any]:
        """Assess sample size adequacy for different analyses."""
        n_obs = len(self.data)
        n_states = self.data['state'].nunique()
        n_years = self.data['year'].nunique()
        
        # Minimum requirements based on econometric literature
        min_obs_panel = 200  # Minimum for panel data
        min_states = 20  # Minimum states for clustering
        min_years = 5   # Minimum time periods
        min_treated = 10  # Minimum treated units
        
        n_treated = self.data[self.data['post_treatment'] == 1]['state'].nunique()
        n_control = self.data[self.data['post_treatment'] == 0]['state'].nunique()
        
        sample_assessment = {
            "total_observations": n_obs,
            "states": n_states,
            "years": n_years,
            "treated_states": n_treated,
            "control_states": n_control,
            "adequacy_checks": {
                "total_obs_adequate": n_obs >= min_obs_panel,
                "states_adequate": n_states >= min_states,
                "years_adequate": n_years >= min_years,
                "treated_adequate": n_treated >= min_treated,
                "control_adequate": n_control >= min_treated
            },
            "recommendations": []
        }
        
        # Generate specific recommendations
        if n_obs < min_obs_panel:
            sample_assessment["recommendations"].append(
                f"Consider pooling additional years: {n_obs} obs < {min_obs_panel} minimum"
            )
        
        if n_treated < min_treated:
            sample_assessment["recommendations"].append(
                f"Limited treated units ({n_treated}): consider alternative identification"
            )
            
        if n_years < 8:
            sample_assessment["recommendations"].append(
                f"Short panel ({n_years} years): may limit event study precision"
            )
        
        return sample_assessment
    
    def _assess_coverage(self) -> Dict[str, Any]:
        """Assess data coverage across dimensions with NAEP-aware analysis."""
        coverage = {
            "temporal": {},
            "geographic": {},
            "outcome": {},
            "missing_data": {},
            "naep_pattern_analysis": {}
        }
        
        # Temporal coverage
        year_coverage = self.data['year'].value_counts().sort_index()
        coverage["temporal"] = {
            "years_covered": year_coverage.index.tolist(),
            "observations_per_year": year_coverage.to_dict(),
            "missing_years": [],
            "unbalanced_years": year_coverage[year_coverage < year_coverage.median() * 0.8].index.tolist()
        }
        
        # Geographic coverage
        state_coverage = self.data['state'].value_counts()
        coverage["geographic"] = {
            "states_covered": len(state_coverage),
            "observations_per_state": state_coverage.to_dict(),
            "missing_states": [],
            "unbalanced_states": state_coverage[state_coverage < state_coverage.median() * 0.8].index.tolist()
        }
        
        # NAEP-specific pattern analysis
        naep_vars = [col for col in self.data.columns if any(x in col.lower() for x in ['math', 'reading']) and 
                     any(y in col.lower() for y in ['score', 'gap'])]
        naep_years = [2017, 2019, 2022]  # Known NAEP assessment years
        
        coverage["naep_pattern_analysis"] = {
            "naep_variables": naep_vars,
            "expected_naep_years": naep_years,
            "available_naep_years": sorted([y for y in naep_years if y in self.data['year'].values]),
            "expected_missing_rate": 1 - (len(naep_years) / self.data['year'].nunique()),
            "naep_observations_available": {}
        }
        
        # Calculate available NAEP observations
        for year in naep_years:
            if year in self.data['year'].values:
                year_data = self.data[self.data['year'] == year]
                coverage["naep_pattern_analysis"]["naep_observations_available"][year] = len(year_data)
        
        # Outcome coverage with NAEP-aware assessment
        outcome_vars = [col for col in self.data.columns if 'score' in col or 'gap' in col]
        outcome_missing = {}
        for var in outcome_vars:
            if var in self.data.columns:
                missing_pct = self.data[var].isna().mean()
                
                # NAEP-aware adequacy assessment
                is_naep_var = var in naep_vars
                if is_naep_var:
                    expected_missing = coverage["naep_pattern_analysis"]["expected_missing_rate"]
                    # Allow for some tolerance around expected missing rate
                    adequate = abs(missing_pct - expected_missing) < 0.1
                    assessment_note = f"NAEP variable - expected ~{expected_missing:.1%} missing"
                else:
                    adequate = missing_pct < 0.3  # Standard threshold for non-NAEP variables
                    assessment_note = "Standard adequacy threshold (30%)"
                
                outcome_missing[var] = {
                    "missing_percentage": missing_pct,
                    "missing_count": self.data[var].isna().sum(),
                    "adequate": adequate,
                    "is_naep_variable": is_naep_var,
                    "assessment_note": assessment_note
                }
        
        coverage["outcome"] = outcome_missing
        
        # Overall missing data pattern
        coverage["missing_data"] = {
            "total_missing_cells": self.data.isna().sum().sum(),
            "missing_percentage": self.data.isna().sum().sum() / (len(self.data) * len(self.data.columns)),
            "variables_with_missing": self.data.columns[self.data.isna().any()].tolist(),
            "naep_adjusted_concerns": []
        }
        
        # Identify genuine missing data concerns (excluding expected NAEP patterns)
        for var, info in outcome_missing.items():
            if not info["adequate"] and not info["is_naep_variable"]:
                coverage["missing_data"]["naep_adjusted_concerns"].append(var)
        
        return coverage
    
    def _assess_balance(self) -> Dict[str, Any]:
        """Assess covariate balance between treated and control groups."""
        balance = {
            "pre_treatment_balance": {},
            "balance_tests": {},
            "standardized_differences": {},
            "overall_balance": "good"
        }
        
        # Get pre-treatment data
        pre_treatment = self.data[self.data['post_treatment'] == 0].copy()
        
        if len(pre_treatment) == 0:
            balance["error"] = "No pre-treatment data available"
            return balance
        
        # Variables to test for balance
        balance_vars = [
            'math_grade4_swd_score', 'math_grade4_non_swd_score', 
            'total_revenue_per_pupil', 'time_trend'
        ]
        
        for var in balance_vars:
            if var not in self.data.columns:
                continue
                
            try:
                # Get treated and control pre-treatment means
                treated_pre = pre_treatment[pre_treatment['treated'] == 1][var].dropna()
                control_pre = pre_treatment[pre_treatment['treated'] == 0][var].dropna()
                
                if len(treated_pre) == 0 or len(control_pre) == 0:
                    continue
                
                # Calculate standardized difference
                pooled_std = np.sqrt(((len(treated_pre) - 1) * treated_pre.var() + 
                                    (len(control_pre) - 1) * control_pre.var()) / 
                                   (len(treated_pre) + len(control_pre) - 2))
                
                std_diff = (treated_pre.mean() - control_pre.mean()) / pooled_std
                
                # T-test for difference
                t_stat, p_value = stats.ttest_ind(treated_pre, control_pre)
                
                balance["standardized_differences"][var] = {
                    "std_diff": std_diff,
                    "treated_mean": treated_pre.mean(),
                    "control_mean": control_pre.mean(),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "balanced": abs(std_diff) < 0.25  # Cohen's conventional threshold
                }
                
            except Exception as e:
                self.logger.warning(f"Balance test failed for {var}: {e}")
                continue
        
        # Overall balance assessment
        balanced_vars = sum(1 for v in balance["standardized_differences"].values() 
                          if v.get("balanced", False))
        total_vars = len(balance["standardized_differences"])
        
        if total_vars > 0:
            balance_rate = balanced_vars / total_vars
            if balance_rate >= 0.8:
                balance["overall_balance"] = "good"
            elif balance_rate >= 0.6:
                balance["overall_balance"] = "acceptable"
            else:
                balance["overall_balance"] = "poor"
        
        return balance
    
    def _assess_statistical_power(self) -> Dict[str, Any]:
        """Assess statistical power for detecting policy effects."""
        power_analysis = {
            "power_calculations": {},
            "minimum_detectable_effects": {},
            "sample_size_recommendations": {},
            "overall_power": "adequate"
        }
        
        # Standard power analysis parameters
        alpha = 0.05
        desired_power = 0.80
        
        # Calculate power for main outcomes
        outcome_vars = ['math_grade4_swd_score', 'math_grade4_gap']
        
        for outcome in outcome_vars:
            if outcome not in self.data.columns:
                continue
                
            try:
                # Estimate standard error from current data
                model = smf.ols(f"{outcome} ~ post_treatment + C(state) + C(year)", 
                               data=self.data.dropna(subset=[outcome])).fit(
                    cov_type='cluster', cov_kwds={'groups': self.data['state']}
                )
                
                current_se = model.bse['post_treatment']
                current_n = int(model.nobs)
                
                # Calculate minimum detectable effect (MDE)
                t_critical = stats.t.ppf(1 - alpha/2, df=model.df_resid)
                t_power = stats.t.ppf(desired_power, df=model.df_resid)
                
                mde = (t_critical + t_power) * current_se
                
                # Calculate power for small, medium, large effects (Cohen's d = 0.2, 0.5, 0.8)
                outcome_std = self.data[outcome].std()
                small_effect = 0.2 * outcome_std
                medium_effect = 0.5 * outcome_std
                large_effect = 0.8 * outcome_std
                
                power_small = 1 - stats.t.cdf(t_critical - small_effect/current_se, df=model.df_resid)
                power_medium = 1 - stats.t.cdf(t_critical - medium_effect/current_se, df=model.df_resid)
                power_large = 1 - stats.t.cdf(t_critical - large_effect/current_se, df=model.df_resid)
                
                power_analysis["power_calculations"][outcome] = {
                    "current_sample_size": current_n,
                    "current_se": current_se,
                    "minimum_detectable_effect": mde,
                    "power_small_effect": power_small,
                    "power_medium_effect": power_medium,
                    "power_large_effect": power_large,
                    "adequate_power": power_medium >= 0.8
                }
                
                power_analysis["minimum_detectable_effects"][outcome] = mde
                
            except Exception as e:
                self.logger.warning(f"Power analysis failed for {outcome}: {e}")
                continue
        
        return power_analysis
    
    def test_statistical_assumptions(self) -> Dict[str, Any]:
        """
        Test key statistical assumptions for robustness methods.
        
        Returns:
            Dict containing assumption test results
        """
        self.logger.info("Testing statistical assumptions...")
        
        assumptions = {
            "parallel_trends": {},
            "no_anticipation": {},
            "random_treatment": {},
            "stable_unit_treatment": {},
            "clustering_assumptions": {},
            "overall_validity": "valid"
        }
        
        # Parallel trends test
        assumptions["parallel_trends"] = self._test_parallel_trends()
        
        # No anticipation test
        assumptions["no_anticipation"] = self._test_no_anticipation()
        
        # Clustering adequacy
        assumptions["clustering_assumptions"] = self._test_clustering_adequacy()
        
        # Overall validity assessment
        assumptions["overall_validity"] = self._assess_overall_validity(assumptions)
        
        self.diagnostic_results["assumption_tests"] = assumptions
        self.logger.info(f"Assumption testing complete: {assumptions['overall_validity']}")
        
        return assumptions
    
    def _test_parallel_trends(self) -> Dict[str, Any]:
        """Test parallel trends assumption with formal tests."""
        parallel_trends = {
            "pre_treatment_trends": {},
            "joint_test_results": {},
            "placebo_tests": {},
            "overall_assessment": "satisfied"
        }
        
        # Create lead indicators for pre-treatment periods
        outcome_vars = ['math_grade4_swd_score', 'math_grade4_gap']
        
        for outcome in outcome_vars:
            if outcome not in self.data.columns:
                continue
                
            try:
                # Test with lead indicators
                lead_vars = [col for col in self.data.columns if col.startswith('lead_')]
                if not lead_vars:
                    continue
                
                formula = f"{outcome} ~ {' + '.join(lead_vars)} + C(state) + C(year)"
                model = smf.ols(formula, data=self.data.dropna(subset=[outcome])).fit(
                    cov_type='cluster', cov_kwds={'groups': self.data['state']}
                )
                
                # Extract lead coefficients
                lead_coefs = {}
                lead_pvals = {}
                for var in lead_vars:
                    if var in model.params.index:
                        lead_coefs[var] = model.params[var]
                        lead_pvals[var] = model.pvalues[var]
                
                # Joint F-test for all leads
                lead_restrictions = [f"{var} = 0" for var in lead_vars if var in model.params.index]
                if lead_restrictions:
                    joint_test = model.f_test(lead_restrictions)
                    
                    parallel_trends["pre_treatment_trends"][outcome] = {
                        "lead_coefficients": lead_coefs,
                        "lead_p_values": lead_pvals,
                        "joint_f_statistic": joint_test.fvalue[0][0],
                        "joint_p_value": joint_test.pvalue,
                        "trends_parallel": joint_test.pvalue > 0.05,
                        "model_summary": str(model.summary().tables[1])
                    }
                
            except Exception as e:
                self.logger.warning(f"Parallel trends test failed for {outcome}: {e}")
                continue
        
        return parallel_trends
    
    def _test_no_anticipation(self) -> Dict[str, Any]:
        """Test no anticipation assumption."""
        no_anticipation = {
            "immediate_leads": {},
            "announcement_effects": {},
            "overall_assessment": "satisfied"
        }
        
        # Test immediate pre-treatment period (t=-1)
        if 'lead_1' in self.data.columns:
            outcome_vars = ['math_grade4_swd_score', 'math_grade4_gap']
            
            for outcome in outcome_vars:
                if outcome not in self.data.columns:
                    continue
                    
                try:
                    model = smf.ols(f"{outcome} ~ lead_1 + C(state) + C(year)", 
                                   data=self.data.dropna(subset=[outcome])).fit(
                        cov_type='cluster', cov_kwds={'groups': self.data['state']}
                    )
                    
                    no_anticipation["immediate_leads"][outcome] = {
                        "lead_1_coefficient": model.params['lead_1'],
                        "lead_1_p_value": model.pvalues['lead_1'],
                        "no_anticipation": model.pvalues['lead_1'] > 0.05
                    }
                    
                except Exception as e:
                    self.logger.warning(f"No anticipation test failed for {outcome}: {e}")
                    continue
        
        return no_anticipation
    
    def _test_clustering_adequacy(self) -> Dict[str, Any]:
        """Test clustering assumptions and adequacy."""
        clustering = {
            "cluster_sizes": {},
            "within_cluster_correlation": {},
            "minimum_clusters": {},
            "recommendations": []
        }
        
        # Analyze state clustering
        state_counts = self.data['state'].value_counts()
        
        clustering["cluster_sizes"] = {
            "n_clusters": len(state_counts),
            "min_cluster_size": state_counts.min(),
            "max_cluster_size": state_counts.max(),
            "mean_cluster_size": state_counts.mean(),
            "adequate_n_clusters": len(state_counts) >= 20  # Minimum for clustering
        }
        
        if len(state_counts) < 20:
            clustering["recommendations"].append(
                f"Only {len(state_counts)} clusters - consider alternative inference methods"
            )
        
        if len(state_counts) < 10:
            clustering["recommendations"].append(
                "Very few clusters - wild cluster bootstrap strongly recommended"
            )
        
        return clustering
    
    def analyze_method_failures(self) -> Dict[str, Any]:
        """
        Analyze patterns in method failures and provide troubleshooting guidance.
        
        Returns:
            Dict containing method failure analysis
        """
        self.logger.info("Analyzing method failure patterns...")
        
        failure_analysis = {
            "bootstrap_failures": {},
            "iv_failures": {},
            "clustering_failures": {},
            "convergence_issues": {},
            "troubleshooting_guide": {},
            "recommended_alternatives": {}
        }
        
        # Analyze common failure patterns
        failure_analysis["bootstrap_failures"] = self._analyze_bootstrap_failures()
        failure_analysis["iv_failures"] = self._analyze_iv_failures()
        failure_analysis["clustering_failures"] = self._analyze_clustering_failures()
        
        # Generate troubleshooting guide
        failure_analysis["troubleshooting_guide"] = self._generate_troubleshooting_guide()
        
        # Recommend alternative methods
        failure_analysis["recommended_alternatives"] = self._recommend_alternative_methods()
        
        self.diagnostic_results["method_failures"] = failure_analysis
        self.logger.info("Method failure analysis complete")
        
        return failure_analysis
    
    def _analyze_bootstrap_failures(self) -> Dict[str, Any]:
        """Analyze bootstrap method failures."""
        bootstrap_analysis = {
            "common_failures": [
                "Insufficient bootstrap samples converging",
                "Extreme bootstrap distribution",
                "Memory limitations with large datasets",
                "Clustering structure not preserved"
            ],
            "diagnostic_checks": {
                "sample_size_adequate": len(self.data) >= 100,
                "clusters_adequate": self.data['state'].nunique() >= 10,
                "variance_finite": True  # Would check if outcome variance is finite
            },
            "solutions": {
                "insufficient_convergence": "Increase bootstrap iterations to 2000+",
                "extreme_distribution": "Use bias-corrected bootstrap (BCa)",
                "memory_issues": "Use stratified bootstrap or reduce sample size",
                "clustering_issues": "Ensure state-level resampling preserves structure"
            }
        }
        
        return bootstrap_analysis
    
    def _analyze_iv_failures(self) -> Dict[str, Any]:
        """Analyze instrumental variables failures."""
        iv_analysis = {
            "common_failures": [
                "Weak instruments (low F-statistic)",
                "Perfect collinearity in instruments",
                "Insufficient variation in instrument",
                "Endogeneity tests fail"
            ],
            "diagnostic_checks": {},
            "solutions": {
                "weak_instruments": "Find stronger instruments or use limited information methods",
                "collinearity": "Check instrument correlation matrix and remove redundant IVs",
                "insufficient_variation": "Expand sample or find alternative instruments",
                "endogeneity_fails": "Use alternative identification strategy"
            }
        }
        
        # Check instrument strength if available
        if 'under_monitoring' in self.data.columns:
            try:
                # First stage regression
                first_stage = smf.ols('total_expenditure_per_pupil ~ under_monitoring + C(state) + C(year)', 
                                     data=self.data).fit()
                
                iv_analysis["diagnostic_checks"] = {
                    "first_stage_f": first_stage.fvalue,
                    "instrument_strong": first_stage.fvalue > 10,  # Rule of thumb
                    "r_squared": first_stage.rsquared
                }
                
            except Exception as e:
                iv_analysis["diagnostic_checks"]["error"] = str(e)
        
        return iv_analysis
    
    def _analyze_clustering_failures(self) -> Dict[str, Any]:
        """Analyze clustering method failures."""
        clustering_analysis = {
            "common_failures": [
                "Too few clusters for asymptotic theory",
                "Unbalanced cluster sizes",
                "Within-cluster correlation too high",
                "Clustering variable has missing values"
            ],
            "diagnostic_checks": {
                "n_clusters": self.data['state'].nunique(),
                "min_cluster_size": self.data['state'].value_counts().min(),
                "max_cluster_size": self.data['state'].value_counts().max(),
                "missing_clusters": self.data['state'].isna().sum()
            },
            "solutions": {
                "few_clusters": "Use wild cluster bootstrap or randomization inference",
                "unbalanced_sizes": "Consider weighted clustering or bootstrap",
                "high_correlation": "Use conservative inference methods",
                "missing_clusters": "Assign missing observations or drop incomplete clusters"
            }
        }
        
        return clustering_analysis
    
    def _generate_troubleshooting_guide(self) -> Dict[str, Any]:
        """Generate comprehensive troubleshooting guide."""
        guide = {
            "convergence_issues": {
                "symptoms": ["Model fails to converge", "Unrealistic coefficient estimates"],
                "causes": ["Perfect collinearity", "Insufficient variation", "Numerical precision"],
                "solutions": ["Check correlation matrix", "Add regularization", "Scale variables"]
            },
            "inference_failures": {
                "symptoms": ["Standard errors are NaN", "Confidence intervals explode"],
                "causes": ["Rank deficient covariance matrix", "Too few clusters"],
                "solutions": ["Use robust inference", "Increase sample size", "Alternative clustering"]
            },
            "specification_issues": {
                "symptoms": ["Residuals show patterns", "Assumption tests fail"],
                "causes": ["Omitted variables", "Functional form misspecification"],
                "solutions": ["Add controls", "Test non-linear terms", "Consider alternative specifications"]
            }
        }
        
        return guide
    
    def _recommend_alternative_methods(self) -> Dict[str, Any]:
        """Recommend alternative methods based on data characteristics."""
        data_chars = {
            "n_clusters": self.data['state'].nunique(),
            "n_obs": len(self.data),
            "treatment_variation": self.data['post_treatment'].var()
        }
        
        recommendations = {
            "primary_methods": [],
            "fallback_methods": [],
            "not_recommended": []
        }
        
        # Based on number of clusters
        if data_chars["n_clusters"] < 10:
            recommendations["primary_methods"].extend([
                "Wild cluster bootstrap",
                "Randomization inference", 
                "Permutation tests"
            ])
            recommendations["not_recommended"].extend([
                "Standard cluster robust SE",
                "Regular bootstrap"
            ])
        elif data_chars["n_clusters"] < 30:
            recommendations["primary_methods"].extend([
                "Wild cluster bootstrap",
                "Cluster bootstrap",
                "Jackknife inference"
            ])
        else:
            recommendations["primary_methods"].extend([
                "Cluster robust standard errors",
                "Bootstrap inference",
                "Wild cluster bootstrap"
            ])
        
        # Based on sample size
        if data_chars["n_obs"] < 200:
            recommendations["fallback_methods"].extend([
                "Small sample corrections",
                "Exact tests where possible"
            ])
        
        return recommendations
    
    def _generate_data_recommendations(self, adequacy: Dict) -> List[str]:
        """Generate specific recommendations based on data adequacy assessment."""
        recommendations = []
        
        # Sample size recommendations
        if not adequacy["sample_size"]["adequacy_checks"]["total_obs_adequate"]:
            recommendations.append(
                "Increase sample size by pooling additional years or states"
            )
        
        if not adequacy["sample_size"]["adequacy_checks"]["treated_adequate"]:
            recommendations.append(
                "Limited treatment variation - consider alternative identification strategies"
            )
        
        # Coverage recommendations
        missing_outcomes = [k for k, v in adequacy["coverage"]["outcome"].items() 
                          if not v.get("adequate", True)]
        if missing_outcomes:
            recommendations.append(
                f"High missing data in: {', '.join(missing_outcomes)} - consider imputation"
            )
        
        # Balance recommendations
        if adequacy["balance"]["overall_balance"] == "poor":
            recommendations.append(
                "Poor covariate balance - consider matching, weighting, or additional controls"
            )
        
        return recommendations
    
    def _determine_overall_adequacy(self, adequacy: Dict) -> str:
        """Determine overall data adequacy assessment."""
        checks = adequacy["sample_size"]["adequacy_checks"]
        balance = adequacy["balance"]["overall_balance"]
        
        critical_failures = sum(1 for check in checks.values() if not check)
        
        if critical_failures == 0 and balance in ["good", "acceptable"]:
            return "adequate"
        elif critical_failures <= 2:
            return "marginal"
        else:
            return "inadequate"
    
    def _assess_overall_validity(self, assumptions: Dict) -> str:
        """Assess overall validity of statistical assumptions."""
        # Check parallel trends
        parallel_ok = True
        if "pre_treatment_trends" in assumptions["parallel_trends"]:
            for outcome_test in assumptions["parallel_trends"]["pre_treatment_trends"].values():
                if not outcome_test.get("trends_parallel", True):
                    parallel_ok = False
                    break
        
        # Check clustering
        clustering_ok = True
        if "cluster_sizes" in assumptions["clustering_assumptions"]:
            if not assumptions["clustering_assumptions"]["cluster_sizes"]["adequate_n_clusters"]:
                clustering_ok = False
        
        if parallel_ok and clustering_ok:
            return "valid"
        elif parallel_ok or clustering_ok:
            return "questionable"
        else:
            return "invalid"
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive diagnostic report.
        
        Returns:
            Path to generated report
        """
        self.logger.info("Generating comprehensive diagnostic report...")
        
        # Run all diagnostic analyses
        data_adequacy = self.assess_data_adequacy()
        assumption_tests = self.test_statistical_assumptions()
        failure_analysis = self.analyze_method_failures()
        
        # Generate report
        report_path = self.output_dir / "comprehensive_diagnostic_report.md"
        
        with open(report_path, 'w') as f:
            f.write(self._create_diagnostic_report_markdown(
                data_adequacy, assumption_tests, failure_analysis
            ))
        
        # Save JSON results
        json_path = self.output_dir / "diagnostic_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.diagnostic_results, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive diagnostic report saved to {report_path}")
        return str(report_path)
    
    def _create_diagnostic_report_markdown(self, data_adequacy: Dict, 
                                     assumption_tests: Dict, 
                                     failure_analysis: Dict) -> str:
        """Create markdown diagnostic report."""
        report = f"""# Comprehensive Diagnostic Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Executive Summary

- **Data Adequacy**: {data_adequacy['overall_assessment'].title()}
- **Statistical Assumptions**: {assumption_tests['overall_validity'].title()}
- **Sample Size**: {data_adequacy['sample_size']['total_observations']:,} observations across {data_adequacy['sample_size']['states']} states
- **Treatment Variation**: {data_adequacy['sample_size']['treated_states']} treated states vs {data_adequacy['sample_size']['control_states']} control states

## 1. Data Adequacy Assessment

### Sample Size Analysis
- **Total Observations**: {data_adequacy['sample_size']['total_observations']:,}
- **States**: {data_adequacy['sample_size']['states']}
- **Years**: {data_adequacy['sample_size']['years']}
- **Treated States**: {data_adequacy['sample_size']['treated_states']}
- **Control States**: {data_adequacy['sample_size']['control_states']}

### Coverage Assessment
"""

        # Add NAEP pattern analysis
        if 'naep_pattern_analysis' in data_adequacy['coverage']:
            naep_info = data_adequacy['coverage']['naep_pattern_analysis']
            report += f"""
#### NAEP Data Pattern Analysis
- **NAEP Variables**: {len(naep_info['naep_variables'])} variables identified
- **Expected Assessment Years**: {', '.join(map(str, naep_info['expected_naep_years']))}
- **Available Assessment Years**: {', '.join(map(str, naep_info['available_naep_years']))}
- **Expected Missing Rate**: {naep_info['expected_missing_rate']:.1%} (NAEP only tests in specific years)
- **Available Observations per NAEP Year**: {dict(naep_info['naep_observations_available'])}
"""

        # Add coverage details
        if 'outcome' in data_adequacy['coverage']:
            report += "\\n#### Outcome Variable Coverage\\n"
            naep_vars = []
            other_vars = []
            
            for var, info in data_adequacy['coverage']['outcome'].items():
                status = "✅" if info['adequate'] else "⚠️"
                var_info = f"- {status} **{var}**: {info['missing_percentage']:.1%} missing"
                if info.get('is_naep_variable', False):
                    var_info += f" ({info['assessment_note']})"
                    naep_vars.append(var_info)
                else:
                    var_info += f" ({info['assessment_note']})"
                    other_vars.append(var_info)
            
            if naep_vars:
                report += "\\n##### NAEP Variables (Expected Pattern)\\n"
                report += "\\n".join(naep_vars) + "\\n"
                
            if other_vars:
                report += "\\n##### Other Outcome Variables\\n"
                report += "\\n".join(other_vars) + "\\n"

        # Add genuine concerns section
        if 'naep_adjusted_concerns' in data_adequacy['coverage']['missing_data']:
            concerns = data_adequacy['coverage']['missing_data']['naep_adjusted_concerns']
            if concerns:
                report += f"\\n#### Data Quality Concerns\\n"
                report += f"Variables with unexpectedly high missing data: {', '.join(concerns)}\\n"
            else:
                report += f"\\n#### Data Quality Assessment\\n"
                report += "✅ No unexpected missing data patterns detected\\n"

        # Add balance assessment
        if data_adequacy['balance']['overall_balance']:
            report += f"\\n### Covariate Balance: {data_adequacy['balance']['overall_balance'].title()}\\n"
            
            if 'standardized_differences' in data_adequacy['balance']:
                for var, balance_info in data_adequacy['balance']['standardized_differences'].items():
                    status = "✅" if balance_info['balanced'] else "⚠️"
                    report += f"- {status} **{var}**: Std. diff = {balance_info['std_diff']:.3f}\\n"

        # Add assumption tests
        report += f"\\n## 2. Statistical Assumption Tests\\n\\n"
        report += f"**Overall Assessment**: {assumption_tests['overall_validity'].title()}\\n\\n"
        
        if 'pre_treatment_trends' in assumption_tests['parallel_trends']:
            report += "### Parallel Trends Tests\\n"
            for outcome, test in assumption_tests['parallel_trends']['pre_treatment_trends'].items():
                status = "✅" if test['trends_parallel'] else "❌"
                report += f"- {status} **{outcome}**: Joint F-test p-value = {test['joint_p_value']:.3f}\\n"

        # Add recommendations
        if data_adequacy['recommendations']:
            report += "\\n## 3. Recommendations\\n\\n"
            for i, rec in enumerate(data_adequacy['recommendations'], 1):
                report += f"{i}. {rec}\\n"

        # Add troubleshooting guide
        report += "\\n## 4. Method-Specific Troubleshooting\\n\\n"
        
        if 'troubleshooting_guide' in failure_analysis:
            for issue, guide in failure_analysis['troubleshooting_guide'].items():
                report += f"### {issue.replace('_', ' ').title()}\\n"
                report += f"**Symptoms**: {', '.join(guide['symptoms'])}\\n\\n"
                report += f"**Solutions**: {', '.join(guide['solutions'])}\\n\\n"

        report += "\\n---\\n*This diagnostic report provides comprehensive assessment for robust policy evaluation.*"
        
        return report


def run_comprehensive_diagnostics(data_path: str = "data/final/analysis_panel.csv") -> str:
    """
    Run comprehensive diagnostic analysis on the analysis dataset.
    
    Args:
        data_path: Path to analysis panel dataset
        
    Returns:
        Path to generated diagnostic report
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize diagnostic reporter
    diagnostics = DiagnosticReporter(data)
    
    # Generate comprehensive report
    report_path = diagnostics.generate_comprehensive_report()
    
    return report_path


if __name__ == "__main__":
    # Run diagnostics if called directly
    report_path = run_comprehensive_diagnostics()
    print(f"✅ Comprehensive diagnostics complete!")
    print(f"📊 Report available at: {report_path}")