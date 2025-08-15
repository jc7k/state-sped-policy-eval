#!/usr/bin/env python
"""
Phase 5.3: Comprehensive Methodology Documentation
Technical documentation generator for robust analysis methods and best practices.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class MethodDocumentation:
    """Complete documentation for a robust analysis method."""
    name: str
    category: str
    description: str
    theoretical_foundation: str
    implementation_details: str
    assumptions: List[str]
    advantages: List[str]
    disadvantages: List[str]
    when_to_use: List[str]
    when_not_to_use: List[str]
    code_example: str
    references: List[str]
    monte_carlo_evidence: Optional[str] = None
    practical_considerations: Optional[List[str]] = None


class MethodologyDocumentationGenerator:
    """
    Comprehensive methodology documentation system.
    
    Generates:
    - Technical method descriptions
    - Implementation guides
    - Best practices documentation
    - Code examples and tutorials
    - Troubleshooting guides
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize methodology documentation generator.
        
        Args:
            output_dir: Directory for documentation outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output/methodology_docs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize method documentation database
        self.method_docs = self._initialize_method_documentation()
        
        # Configure logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup dedicated logging for methodology documentation."""
        log_file = self.output_dir / "methodology_documentation.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def _initialize_method_documentation(self) -> Dict[str, MethodDocumentation]:
        """Initialize comprehensive method documentation database."""
        
        methods = {
            "cluster_robust_se": MethodDocumentation(
                name="Cluster Robust Standard Errors",
                category="Asymptotic Methods",
                description="""
Cluster robust standard errors adjust for within-cluster correlation in panel data.
They provide consistent variance estimation when observations are correlated within
clusters (e.g., states, schools, firms) but independent across clusters.
""",
                theoretical_foundation="""
Based on the work of Arellano (1987) and extended by Cameron, Gelbach & Miller (2011).
The method relaxes the independence assumption and allows for arbitrary correlation
within clusters while maintaining independence across clusters.

The cluster robust variance estimator is:
V_cluster = (X'X)^(-1) * Σ_g (X_g' ε_g ε_g' X_g) * (X'X)^(-1)

where g indexes clusters, X_g are regressors for cluster g, and ε_g are residuals.
""",
                implementation_details="""
1. Estimate model using OLS
2. Calculate cluster-specific score vectors
3. Compute sandwich variance estimator
4. Apply finite sample corrections (HC1, HC2, HC3)
5. Use t-distribution with G-1 degrees of freedom
""",
                assumptions=[
                    "Independence across clusters",
                    "Sufficient number of clusters (typically 30+)",
                    "Cluster sizes can be unequal",
                    "Model is correctly specified"
                ],
                advantages=[
                    "Computationally efficient",
                    "Standard in econometric practice",
                    "Good asymptotic properties",
                    "Handles unbalanced clusters",
                    "Easy to implement"
                ],
                disadvantages=[
                    "Poor performance with few clusters (<20)",
                    "Over-rejection in finite samples",
                    "Sensitive to cluster definition",
                    "Assumes large-cluster asymptotics"
                ],
                when_to_use=[
                    "Panel data with natural clustering",
                    "Large number of clusters (>30)",
                    "Standard policy evaluation",
                    "When computational efficiency matters"
                ],
                when_not_to_use=[
                    "Few clusters (<15)",
                    "Suspected model misspecification",
                    "Non-standard distributions",
                    "Very unbalanced cluster sizes"
                ],
                code_example="""
import statsmodels.formula.api as smf

# Estimate model with cluster robust standard errors
model = smf.ols('outcome ~ treatment + C(state) + C(year)', data=df)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['state']})

# Display results
print(results.summary())
""",
                references=[
                    "Arellano, M. (1987). Computing robust standard errors for within-groups estimators.",
                    "Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.",
                    "Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data."
                ],
                practical_considerations=[
                    "Use HC1 finite sample correction for better performance",
                    "Consider clustering at multiple levels if relevant",
                    "Check sensitivity to cluster definition",
                    "Verify sufficient within-cluster variation"
                ]
            ),
            
            "wild_cluster_bootstrap": MethodDocumentation(
                name="Wild Cluster Bootstrap",
                category="Bootstrap Methods",
                description="""
Wild cluster bootstrap is a resampling method specifically designed for inference
with clustered data, particularly when the number of clusters is small. It provides
robust inference by resampling entire clusters with random weights.
""",
                theoretical_foundation="""
Developed by Cameron, Gelbach & Miller (2008) based on Wu's (1986) wild bootstrap.
The method assigns random weights to cluster-level residuals while preserving
the original regressors, allowing for robust inference under clustering.

The wild bootstrap implements:
y*_g = X_g β̂ + ω_g ε̂_g

where ω_g are iid random weights (typically Rademacher: ±1 with equal probability)
and ε̂_g are cluster-level residuals.
""",
                implementation_details="""
1. Estimate restricted model under null hypothesis
2. Calculate cluster-level residuals
3. For each bootstrap iteration:
   a. Draw random weights ω_g for each cluster
   b. Construct bootstrap sample: y*_g = X_g β̂_0 + ω_g ε̂_g
   c. Re-estimate model and save test statistic
4. Compare original statistic to bootstrap distribution
5. Calculate bootstrap p-value
""",
                assumptions=[
                    "Model is linear in parameters",
                    "Clusters are independent",
                    "Residuals can be heteroskedastic within clusters",
                    "No assumption on cluster sizes"
                ],
                advantages=[
                    "Robust to few clusters (as few as 5-10)",
                    "Handles unbalanced cluster sizes well",
                    "Good finite sample properties",
                    "Robust to heteroskedasticity",
                    "Provides exact finite sample inference"
                ],
                disadvantages=[
                    "Computationally intensive",
                    "More complex to implement",
                    "Requires careful null hypothesis specification",
                    "May be conservative in some settings"
                ],
                when_to_use=[
                    "Few clusters (5-30)",
                    "Unbalanced cluster sizes",
                    "Suspected heteroskedasticity",
                    "Small sample inference needed",
                    "Standard errors seem unreliable"
                ],
                when_not_to_use=[
                    "Very large datasets (computational burden)",
                    "Many clusters (>50) with balanced sizes",
                    "Non-linear models",
                    "Time-sensitive analysis"
                ],
                code_example="""
import numpy as np
from scipy import stats

def wild_cluster_bootstrap(X, y, cluster_id, n_boot=999):
    '''
    Wild cluster bootstrap implementation
    '''
    # Estimate restricted model
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    
    # Get cluster-specific residuals
    clusters = np.unique(cluster_id)
    n_clusters = len(clusters)
    
    bootstrap_stats = []
    
    for b in range(n_boot):
        # Draw Rademacher weights
        weights = np.random.choice([-1, 1], size=n_clusters)
        
        # Create bootstrap sample
        y_boot = X @ model.params.copy()
        for i, cluster in enumerate(clusters):
            cluster_mask = cluster_id == cluster
            y_boot[cluster_mask] += weights[i] * residuals[cluster_mask]
        
        # Re-estimate and save test statistic
        boot_model = sm.OLS(y_boot, X).fit()
        bootstrap_stats.append(boot_model.tvalues[1])  # Treatment coefficient
    
    return np.array(bootstrap_stats)
""",
                references=[
                    "Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors.",
                    "Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling methods in regression analysis.",
                    "MacKinnon, J. G., & Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes."
                ],
                monte_carlo_evidence="""
Extensive Monte Carlo studies show:
- Good size control even with 5-10 clusters
- Superior to cluster robust SE with few clusters
- Robust to unbalanced cluster sizes
- Slight loss of power compared to infeasible methods
""",
                practical_considerations=[
                    "Use odd number of bootstrap replications (999, 1999)",
                    "Rademacher weights generally perform best",
                    "Consider computational time for large datasets",
                    "Verify bootstrap distribution is well-behaved"
                ]
            ),
            
            "cluster_bootstrap": MethodDocumentation(
                name="Cluster Bootstrap",
                category="Bootstrap Methods", 
                description="""
Cluster bootstrap resamples entire clusters with replacement to preserve
within-cluster dependence structure. It's particularly useful when the
number of clusters is moderate and cluster structure is important.
""",
                theoretical_foundation="""
Based on standard bootstrap theory (Efron, 1979) adapted for clustered data.
The method resamples clusters rather than individual observations to preserve
the correlation structure within clusters.

Bootstrap samples θ* from:
F*_n = (1/G) Σ_{g∈S*} δ_{cluster_g}

where S* is a bootstrap sample of clusters drawn with replacement.
""",
                implementation_details="""
1. Identify all unique clusters
2. For each bootstrap iteration:
   a. Sample G clusters with replacement from original clusters
   b. Form bootstrap dataset by stacking selected clusters
   c. Estimate model on bootstrap sample
   d. Save parameter estimates or test statistics
3. Construct bootstrap distribution
4. Calculate confidence intervals or p-values
""",
                assumptions=[
                    "Clusters are independent and identically distributed",
                    "Within-cluster correlation can be arbitrary",
                    "Moderate number of clusters (15+)",
                    "Cluster sizes can vary"
                ],
                advantages=[
                    "Preserves exact cluster dependence structure",
                    "Flexible for various estimators",
                    "Intuitive resampling scheme",
                    "Can handle complex within-cluster patterns"
                ],
                disadvantages=[
                    "Requires moderate number of clusters",
                    "Computationally intensive",
                    "Bootstrap samples may be unbalanced",
                    "Convergence not always guaranteed"
                ],
                when_to_use=[
                    "Moderate number of clusters (15-50)",
                    "Complex within-cluster dependence",
                    "Non-standard estimators",
                    "Robustness check for other methods"
                ],
                when_not_to_use=[
                    "Very few clusters (<10)",
                    "Very large datasets",
                    "Simple linear models with many clusters",
                    "When wild bootstrap is available"
                ],
                code_example="""
def cluster_bootstrap(data, cluster_col, n_boot=999):
    '''
    Cluster bootstrap implementation
    '''
    clusters = data[cluster_col].unique()
    n_clusters = len(clusters)
    
    bootstrap_results = []
    
    for b in range(n_boot):
        # Sample clusters with replacement
        boot_clusters = np.random.choice(clusters, size=n_clusters, replace=True)
        
        # Create bootstrap sample
        boot_data = []
        for cluster in boot_clusters:
            cluster_data = data[data[cluster_col] == cluster].copy()
            boot_data.append(cluster_data)
        
        boot_sample = pd.concat(boot_data, ignore_index=True)
        
        # Estimate model
        model = smf.ols('outcome ~ treatment + controls', data=boot_sample)
        result = model.fit()
        bootstrap_results.append(result.params['treatment'])
    
    return np.array(bootstrap_results)
""",
                references=[
                    "Efron, B. (1979). Bootstrap methods: another look at the jackknife.",
                    "Davison, A. C., & Hinkley, D. V. (1997). Bootstrap methods and their applications.",
                    "Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data."
                ]
            ),
            
            "permutation_test": MethodDocumentation(
                name="Permutation Test",
                category="Exact Methods",
                description="""
Permutation tests provide exact finite sample inference by comparing the
observed test statistic to its distribution under random reassignment of
treatment status. They require minimal assumptions and provide exact p-values.
""",
                theoretical_foundation="""
Based on Fisher's exact test principle (Fisher, 1935). Under the sharp null
hypothesis of no treatment effect for any unit, all possible treatment
assignments are equally likely, providing an exact reference distribution.

The test compares observed statistic T_obs to the permutation distribution:
T_perm = {T(π): π ∈ Π}

where Π is the set of all possible treatment permutations.
""",
                implementation_details="""
1. Specify sharp null hypothesis (typically zero effect)
2. Calculate observed test statistic
3. Generate all possible (or random sample of) treatment permutations
4. For each permutation:
   a. Reassign treatment status
   b. Calculate test statistic under permutation
5. Compare observed statistic to permutation distribution
6. Calculate exact p-value as proportion of permuted statistics ≥ observed
""",
                assumptions=[
                    "Sharp null hypothesis (no effect for any unit)",
                    "Treatment assignment mechanism known",
                    "Observations exchangeable under null",
                    "Minimal distributional assumptions"
                ],
                advantages=[
                    "Exact finite sample inference",
                    "No distributional assumptions",
                    "Robust to all model violations",
                    "Valid with any sample size",
                    "Intuitive interpretation"
                ],
                disadvantages=[
                    "Computationally very intensive",
                    "Limited to simple null hypotheses",
                    "May be conservative",
                    "Requires careful treatment assignment modeling"
                ],
                when_to_use=[
                    "Very few clusters (<10)",
                    "Non-standard distributions",
                    "Exact inference required",
                    "Skeptical audience",
                    "Small samples"
                ],
                when_not_to_use=[
                    "Large datasets (computational burden)",
                    "Complex models with many parameters",
                    "Continuous treatment variables",
                    "Standard applications with sufficient clusters"
                ],
                code_example="""
def permutation_test(data, treatment_col, outcome_col, n_perm=999):
    '''
    Permutation test implementation
    '''
    # Calculate observed test statistic
    treated = data[data[treatment_col] == 1][outcome_col]
    control = data[data[treatment_col] == 0][outcome_col]
    obs_stat = treated.mean() - control.mean()
    
    # Generate permutation distribution
    perm_stats = []
    outcomes = data[outcome_col].values
    n_treated = (data[treatment_col] == 1).sum()
    
    for p in range(n_perm):
        # Randomly permute treatment assignment
        perm_indices = np.random.permutation(len(outcomes))
        perm_treated = outcomes[perm_indices[:n_treated]]
        perm_control = outcomes[perm_indices[n_treated:]]
        
        perm_stat = perm_treated.mean() - perm_control.mean()
        perm_stats.append(perm_stat)
    
    # Calculate p-value
    p_value = (np.sum(np.abs(perm_stats) >= np.abs(obs_stat)) + 1) / (n_perm + 1)
    
    return obs_stat, p_value, perm_stats
""",
                references=[
                    "Fisher, R. A. (1935). The design of experiments.",
                    "Rosenbaum, P. R. (2002). Observational studies.",
                    "Ernst, M. D. (2004). Permutation methods: a basis for exact inference."
                ],
                monte_carlo_evidence="""
Simulation studies demonstrate:
- Exact size control under null hypothesis
- Good power properties for large effects
- Robust to all distributional violations
- Conservative for small effects in some settings
""",
                practical_considerations=[
                    "Ensure sufficient permutations (≥999)",
                    "Consider clustering in permutation scheme",
                    "Verify computational feasibility",
                    "Document exact null hypothesis tested"
                ]
            )
        }
        
        return methods
        
    def generate_method_guides(self) -> Dict[str, str]:
        """
        Generate detailed method guides for all documented methods.
        
        Returns:
            Dict mapping method names to guide file paths
        """
        self.logger.info("Generating detailed method guides...")
        
        method_guides = {}
        
        for method_name, method_doc in self.method_docs.items():
            guide_content = self._create_method_guide(method_doc)
            guide_path = self.output_dir / f"{method_name}_guide.md"
            
            with open(guide_path, 'w') as f:
                f.write(guide_content)
                
            method_guides[method_name] = str(guide_path)
            
        self.logger.info(f"Generated {len(method_guides)} method guides")
        return method_guides
        
    def _create_method_guide(self, method_doc: MethodDocumentation) -> str:
        """Create detailed markdown guide for a specific method."""
        
        guide = f"""# {method_doc.name} - Complete Guide

**Category**: {method_doc.category}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Overview

{method_doc.description.strip()}

## Theoretical Foundation

{method_doc.theoretical_foundation.strip()}

## Implementation Details

{method_doc.implementation_details.strip()}

## Assumptions

"""
        
        for assumption in method_doc.assumptions:
            guide += f"- {assumption}\n"
            
        guide += f"""
## Advantages

"""
        for advantage in method_doc.advantages:
            guide += f"- {advantage}\n"
            
        guide += f"""
## Disadvantages

"""
        for disadvantage in method_doc.disadvantages:
            guide += f"- {disadvantage}\n"
            
        guide += f"""
## When to Use

"""
        for use_case in method_doc.when_to_use:
            guide += f"- {use_case}\n"
            
        guide += f"""
## When NOT to Use

"""
        for avoid_case in method_doc.when_not_to_use:
            guide += f"- {avoid_case}\n"
            
        guide += f"""
## Code Example

```python
{method_doc.code_example.strip()}
```

"""

        if method_doc.monte_carlo_evidence:
            guide += f"""
## Monte Carlo Evidence

{method_doc.monte_carlo_evidence.strip()}

"""

        if method_doc.practical_considerations:
            guide += f"""
## Practical Considerations

"""
            for consideration in method_doc.practical_considerations:
                guide += f"- {consideration}\n"
                
        guide += f"""
## References

"""
        for reference in method_doc.references:
            guide += f"- {reference}\n"
            
        guide += f"""

---
*This guide is part of the comprehensive robustness analysis methodology documentation.*
"""
        
        return guide
        
    def generate_best_practices_guide(self) -> str:
        """
        Generate comprehensive best practices guide.
        
        Returns:
            Path to best practices guide
        """
        self.logger.info("Generating best practices guide...")
        
        best_practices = f"""# Robust Analysis Best Practices Guide

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Executive Summary

This guide provides comprehensive best practices for conducting robust policy evaluation
with clustered data, drawing from econometric theory and extensive Monte Carlo evidence.

## Method Selection Framework

### Step 1: Assess Data Characteristics

1. **Count clusters**: How many independent units (states, schools, firms)?
2. **Examine cluster balance**: Are cluster sizes roughly equal?
3. **Check sample size**: How many total observations?
4. **Evaluate treatment variation**: Sufficient variation in treatment status?

### Step 2: Apply Selection Rules

#### With 30+ clusters:
- **Primary choice**: Cluster robust standard errors
- **Alternative**: Wild cluster bootstrap (robustness check)
- **Avoid**: Permutation tests (computationally inefficient)

#### With 15-30 clusters:
- **Primary choice**: Wild cluster bootstrap
- **Alternative**: Cluster bootstrap
- **Caution**: Standard cluster robust SE may over-reject

#### With 5-15 clusters:
- **Primary choice**: Wild cluster bootstrap
- **Alternative**: Permutation test
- **Avoid**: Standard cluster robust SE

#### With <5 clusters:
- **Primary choice**: Permutation test or randomization inference
- **Consider**: Aggregating to higher level
- **Avoid**: All asymptotic methods

### Step 3: Implement Multiple Methods

Always implement at least two methods for robustness:
1. Primary method based on data characteristics
2. Alternative method as sensitivity check
3. Document any differences in conclusions

## Implementation Best Practices

### Cluster Robust Standard Errors

```python
# Use appropriate finite sample corrections
results = model.fit(cov_type='cluster', 
                   cov_kwds={{'groups': data['state'],
                            'use_correction': True}})

# Check degrees of freedom: should be n_clusters - 1
print(f"Degrees of freedom: {{results.df_resid}}")
```

**Key considerations**:
- Use HC1 finite sample correction
- Verify sufficient within-cluster variation
- Check for balanced clusters
- Consider two-way clustering if relevant

### Wild Cluster Bootstrap

```python
# Use odd number of iterations
n_boot = 999  # or 1999 for publication

# Implement Rademacher weights
weights = np.random.choice([-1, 1], size=n_clusters)

# Verify bootstrap distribution
plt.hist(bootstrap_stats)
plt.axvline(observed_stat, color='red', label='Observed')
plt.legend()
```

**Key considerations**:
- Always use odd number of bootstrap replications
- Rademacher weights typically perform best
- Check bootstrap distribution for anomalies
- Document computational time

### Permutation Tests

```python
# Ensure sufficient permutations
n_perm = max(999, int(10**6 / n_possible_permutations))

# Account for clustering in permutation scheme
# Permute at cluster level, not individual level
```

**Key considerations**:
- Specify exact null hypothesis clearly
- Use cluster-level permutation for clustered data
- Document permutation scheme
- Consider computational feasibility

## Common Pitfalls and Solutions

### Pitfall 1: Wrong Clustering Level
**Problem**: Clustering at individual level when policy varies at state level
**Solution**: Always cluster at the level of treatment assignment

### Pitfall 2: Too Few Clusters
**Problem**: Using cluster robust SE with <20 clusters
**Solution**: Switch to wild cluster bootstrap or permutation tests

### Pitfall 3: Ignoring Unbalanced Clusters
**Problem**: Very different cluster sizes affecting inference
**Solution**: Use wild cluster bootstrap which handles imbalance well

### Pitfall 4: Computational Shortcuts
**Problem**: Using too few bootstrap replications to save time
**Solution**: Use minimum 999 replications, more for final results

### Pitfall 5: Single Method Reporting
**Problem**: Reporting only one inference method
**Solution**: Always report multiple methods for robustness

## Reporting Standards

### Minimum Requirements
1. Document data characteristics (n, clusters, balance)
2. Justify method selection based on characteristics
3. Report results from at least two methods
4. Discuss any differences between methods
5. Include computational details (bootstrap replications, etc.)

### Publication Quality
1. Include method comparison table
2. Provide detailed implementation notes
3. Report diagnostic tests and assumption checks
4. Include robustness section with multiple methods
5. Make replication code available

## Troubleshooting Guide

### Problem: Results differ across methods
**Diagnosis**: Check for few clusters, imbalanced data, or model misspecification
**Solution**: Investigate data quality, consider alternative specifications

### Problem: Bootstrap doesn't converge
**Diagnosis**: Check for perfect collinearity, insufficient variation
**Solution**: Verify model specification, check cluster sizes

### Problem: Computational time too long
**Diagnosis**: Large dataset with computationally intensive method
**Solution**: Use stratified sampling, parallel processing, or switch methods

### Problem: Unrealistic standard errors
**Diagnosis**: Model misspecification, wrong clustering level
**Solution**: Check model specification, verify clustering assumption

## Advanced Topics

### Two-Way Clustering
When treatment varies at multiple levels (e.g., state-year policies):
```python
# Implement two-way clustering
results = model.fit(cov_type='cluster',
                   cov_kwds={{'groups': [data['state'], data['year']]}})
```

### Temporal Correlation
For panel data with time trends:
- Include time fixed effects
- Consider cluster-robust with time clustering
- Use wild bootstrap with block structure

### Non-Linear Models
For logit, probit, and other non-linear models:
- Bootstrap methods often more reliable
- Score bootstrap for maximum likelihood
- Parametric bootstrap for complex models

## Conclusion

Robust inference with clustered data requires careful attention to data characteristics
and appropriate method selection. When in doubt, implement multiple methods and
report all results transparently.

---
*This best practices guide synthesizes current econometric research and practical experience.*
"""
        
        guide_path = self.output_dir / "best_practices_guide.md"
        with open(guide_path, 'w') as f:
            f.write(best_practices)
            
        self.logger.info(f"Best practices guide saved to {guide_path}")
        return str(guide_path)
        
    def generate_troubleshooting_guide(self) -> str:
        """
        Generate comprehensive troubleshooting guide.
        
        Returns:
            Path to troubleshooting guide
        """
        self.logger.info("Generating troubleshooting guide...")
        
        troubleshooting = f"""# Robustness Analysis Troubleshooting Guide

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Quick Diagnostic Checklist

Before diving into specific issues, run through this checklist:

- [ ] Data has been properly cleaned and validated
- [ ] Clustering variable is correctly specified
- [ ] Treatment assignment level matches clustering level
- [ ] Sufficient observations and clusters for chosen method
- [ ] Model specification has been validated
- [ ] Software implementation is correct

## Common Issues and Solutions

### Issue 1: "Standard errors are NaN"

**Symptoms**:
- Standard errors show as NaN or missing
- Confidence intervals are infinite
- T-statistics are undefined

**Likely Causes**:
- Perfect multicollinearity in regressors
- Insufficient variation in treatment within clusters
- Rank-deficient design matrix

**Diagnostic Steps**:
```python
# Check for multicollinearity
import pandas as pd
corr_matrix = X.corr()
print("High correlations (>0.95):")
print(corr_matrix[corr_matrix > 0.95].stack())

# Check treatment variation
treatment_by_cluster = data.groupby('cluster')['treatment'].agg(['mean', 'std'])
print("Clusters with no treatment variation:")
print(treatment_by_cluster[treatment_by_cluster['std'] == 0])
```

**Solutions**:
1. Remove perfectly correlated variables
2. Check for constant regressors within clusters
3. Increase sample size or reduce model complexity
4. Use regularization techniques

### Issue 2: "Results differ dramatically across methods"

**Symptoms**:
- Cluster robust SE gives significant results, bootstrap doesn't
- P-values differ by orders of magnitude
- Confidence intervals don't overlap

**Likely Causes**:
- Too few clusters for asymptotic methods
- Severe cluster imbalance
- Model misspecification
- Outliers affecting specific methods

**Diagnostic Steps**:
```python
# Check cluster characteristics
cluster_sizes = data.groupby('cluster').size()
print(f"Number of clusters: {{len(cluster_sizes)}}")
print(f"Cluster size range: {{cluster_sizes.min()}} - {{cluster_sizes.max()}}")
print(f"Coefficient of variation: {{cluster_sizes.std() / cluster_sizes.mean()}}")

# Check for outliers
from scipy import stats
outliers = stats.zscore(data['outcome']) > 3
print(f"Potential outliers: {{outliers.sum()}} observations")
```

**Solutions**:
1. Use methods appropriate for number of clusters
2. Investigate and handle outliers
3. Check model specification
4. Consider alternative clustering schemes

### Issue 3: "Bootstrap doesn't converge"

**Symptoms**:
- Bootstrap distribution is degenerate
- Extreme values in bootstrap samples
- Error messages during bootstrap

**Likely Causes**:
- Insufficient bootstrap replications
- Perfect collinearity in bootstrap samples
- Very small cluster sizes
- Numerical precision issues

**Solutions**:
```python
# Increase bootstrap replications
n_boot = 1999  # Instead of 999

# Check for numerical issues
print(f"Condition number: {{np.linalg.cond(X.T @ X)}}")

# Use more stable computation
# Implement bias-corrected bootstrap
```

### Issue 4: "Computational time is excessive"

**Symptoms**:
- Methods take hours to complete
- Memory usage is very high
- System becomes unresponsive

**Solutions**:
1. **Reduce bootstrap replications for testing**:
```python
# Use fewer replications during development
n_boot_test = 99   # For testing
n_boot_final = 999 # For final results
```

2. **Implement parallel processing**:
```python
from multiprocessing import Pool

def bootstrap_parallel(args):
    # Bootstrap function here
    pass

with Pool() as pool:
    results = pool.map(bootstrap_parallel, range(n_boot))
```

3. **Use stratified sampling for large datasets**
4. **Switch to computationally efficient methods**

### Issue 5: "Wild bootstrap gives conservative results"

**Symptoms**:
- P-values consistently higher than other methods
- Confidence intervals are wider
- Low power to detect known effects

**Likely Causes**:
- Wrong weight distribution choice
- Insufficient bootstrap replications
- Method inherently conservative for this setting

**Solutions**:
```python
# Try different weight distributions
# Rademacher weights (standard)
weights_rademacher = np.random.choice([-1, 1], size=n_clusters)

# Mammen weights (alternative)
weights_mammen = np.random.choice([-(np.sqrt(5)-1)/2, (np.sqrt(5)+1)/2], 
                                 size=n_clusters,
                                 p=[(np.sqrt(5)+1)/(2*np.sqrt(5)), 
                                    (np.sqrt(5)-1)/(2*np.sqrt(5))])
```

### Issue 6: "Permutation test is too slow"

**Symptoms**:
- Test takes extremely long to complete
- Combinatorial explosion with sample size

**Solutions**:
1. **Use random permutations instead of complete enumeration**:
```python
# Instead of all possible permutations
n_perm = min(999, math.comb(n, n_treated))
```

2. **Implement early stopping**:
```python
# Stop early if result is clear
if p_value_estimate > 0.1 and n_completed > 100:
    break
```

3. **Consider approximate tests**

### Issue 7: "Results not reproducible"

**Symptoms**:
- Different results each time code is run
- Collaborators get different answers
- Random seed doesn't seem to work

**Solutions**:
```python
# Set comprehensive random seed
import numpy as np
import random

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # For sklearn, tensorflow, etc.
    # sklearn.utils.check_random_state(seed)

set_seeds(42)
```

## Method-Specific Troubleshooting

### Cluster Robust Standard Errors

**Common Issues**:
- Over-rejection with few clusters
- Sensitivity to cluster definition

**Fixes**:
- Use finite sample corrections
- Try two-way clustering
- Switch to bootstrap methods

### Wild Cluster Bootstrap

**Common Issues**:
- Fails with very few clusters (<5)
- Sensitive to weight choice

**Fixes**:
- Increase number of clusters by aggregation
- Try different weight distributions
- Use permutation test instead

### Permutation Test

**Common Issues**:
- Computationally infeasible
- Results depend on permutation scheme

**Fixes**:
- Use random subset of permutations
- Carefully specify permutation scheme
- Document exact procedure

## When to Seek Help

Contact a statistician or econometrician if:
- Multiple methods give contradictory results
- Data characteristics are unusual
- Implementing novel identification strategy
- Results seem too good/bad to be true
- Computational methods consistently fail

## Additional Resources

- Cameron, Gelbach & Miller (2011) for clustering guidance
- MacKinnon & Webb (2017) for wild bootstrap details
- Angrist & Pischke (2009) for general econometric practice
- Efron & Tibshirani (1993) for bootstrap theory

---
*This troubleshooting guide addresses the most common issues in robust inference.*
"""
        
        guide_path = self.output_dir / "troubleshooting_guide.md"
        with open(guide_path, 'w') as f:
            f.write(troubleshooting)
            
        self.logger.info(f"Troubleshooting guide saved to {guide_path}")
        return str(guide_path)
        
    def generate_complete_documentation(self) -> Dict[str, Any]:
        """
        Generate complete methodology documentation suite.
        
        Returns:
            Dict containing all generated documentation paths
        """
        self.logger.info("Generating complete methodology documentation...")
        
        documentation = {
            "timestamp": datetime.now().isoformat(),
            "method_guides": self.generate_method_guides(),
            "best_practices": self.generate_best_practices_guide(),
            "troubleshooting": self.generate_troubleshooting_guide(),
            "summary": self._generate_documentation_summary()
        }
        
        # Save complete documentation index
        index_path = self.output_dir / "documentation_index.json"
        with open(index_path, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
            
        self.logger.info("Complete methodology documentation generated successfully")
        return documentation
        
    def _generate_documentation_summary(self) -> str:
        """Generate summary of all documentation."""
        summary = f"""# Methodology Documentation Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Available Documentation

### Method-Specific Guides
{len(self.method_docs)} detailed method guides available:

"""
        
        for method_name, method_doc in self.method_docs.items():
            summary += f"- **{method_doc.name}** ({method_doc.category}): `{method_name}_guide.md`\n"
            
        summary += f"""

### General Guides
- **Best Practices Guide**: `best_practices_guide.md`
- **Troubleshooting Guide**: `troubleshooting_guide.md`

### Documentation Index
- **Complete Index**: `documentation_index.json`

## Quick Navigation

For practitioners new to robust methods:
1. Start with `best_practices_guide.md`
2. Read method-specific guides for your use case
3. Consult `troubleshooting_guide.md` for issues

For researchers implementing new methods:
1. Review theoretical foundations in method guides
2. Study implementation details and code examples
3. Check Monte Carlo evidence where available

For policy evaluators:
1. Use best practices for method selection
2. Implement multiple methods for robustness
3. Follow reporting standards

---
*This documentation provides comprehensive guidance for robust policy evaluation.*
"""
        
        summary_path = self.output_dir / "documentation_summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
            
        return str(summary_path)


def run_methodology_documentation() -> Dict[str, Any]:
    """
    Run complete methodology documentation generation.
    
    Returns:
        Dict containing all generated documentation
    """
    # Initialize documentation generator
    doc_generator = MethodologyDocumentationGenerator()
    
    # Generate complete documentation
    documentation = doc_generator.generate_complete_documentation()
    
    return documentation


if __name__ == "__main__":
    # Run methodology documentation if called directly
    documentation = run_methodology_documentation()
    print(f"✅ Methodology documentation complete!")
    print(f"📚 Documentation index: {documentation['summary']}")
    print(f"📋 Method guides: {len(documentation['method_guides'])} files")
    print(f"📖 Best practices: {documentation['best_practices']}")
    print(f"🔧 Troubleshooting: {documentation['troubleshooting']}")