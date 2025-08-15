# Robustness Analysis Troubleshooting Guide

**Generated**: 2025-08-15 14:09:53  
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
print(f"Number of clusters: {len(cluster_sizes)}")
print(f"Cluster size range: {cluster_sizes.min()} - {cluster_sizes.max()}")
print(f"Coefficient of variation: {cluster_sizes.std() / cluster_sizes.mean()}")

# Check for outliers
from scipy import stats
outliers = stats.zscore(data['outcome']) > 3
print(f"Potential outliers: {outliers.sum()} observations")
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
print(f"Condition number: {np.linalg.cond(X.T @ X)}")

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
