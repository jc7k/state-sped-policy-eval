# Wild Cluster Bootstrap - Complete Guide

**Category**: Bootstrap Methods  
**Generated**: 2025-08-15 14:09:53  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Overview

Wild cluster bootstrap is a resampling method specifically designed for inference
with clustered data, particularly when the number of clusters is small. It provides
robust inference by resampling entire clusters with random weights.

## Theoretical Foundation

Developed by Cameron, Gelbach & Miller (2008) based on Wu's (1986) wild bootstrap.
The method assigns random weights to cluster-level residuals while preserving
the original regressors, allowing for robust inference under clustering.

The wild bootstrap implements:
y*_g = X_g β̂ + ω_g ε̂_g

where ω_g are iid random weights (typically Rademacher: ±1 with equal probability)
and ε̂_g are cluster-level residuals.

## Implementation Details

1. Estimate restricted model under null hypothesis
2. Calculate cluster-level residuals
3. For each bootstrap iteration:
   a. Draw random weights ω_g for each cluster
   b. Construct bootstrap sample: y*_g = X_g β̂_0 + ω_g ε̂_g
   c. Re-estimate model and save test statistic
4. Compare original statistic to bootstrap distribution
5. Calculate bootstrap p-value

## Assumptions

- Model is linear in parameters
- Clusters are independent
- Residuals can be heteroskedastic within clusters
- No assumption on cluster sizes

## Advantages

- Robust to few clusters (as few as 5-10)
- Handles unbalanced cluster sizes well
- Good finite sample properties
- Robust to heteroskedasticity
- Provides exact finite sample inference

## Disadvantages

- Computationally intensive
- More complex to implement
- Requires careful null hypothesis specification
- May be conservative in some settings

## When to Use

- Few clusters (5-30)
- Unbalanced cluster sizes
- Suspected heteroskedasticity
- Small sample inference needed
- Standard errors seem unreliable

## When NOT to Use

- Very large datasets (computational burden)
- Many clusters (>50) with balanced sizes
- Non-linear models
- Time-sensitive analysis

## Code Example

```python
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
```


## Monte Carlo Evidence

Extensive Monte Carlo studies show:
- Good size control even with 5-10 clusters
- Superior to cluster robust SE with few clusters
- Robust to unbalanced cluster sizes
- Slight loss of power compared to infeasible methods


## Practical Considerations

- Use odd number of bootstrap replications (999, 1999)
- Rademacher weights generally perform best
- Consider computational time for large datasets
- Verify bootstrap distribution is well-behaved

## References

- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors.
- Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling methods in regression analysis.
- MacKinnon, J. G., & Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes.


---
*This guide is part of the comprehensive robustness analysis methodology documentation.*
