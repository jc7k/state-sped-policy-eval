# Cluster Robust Standard Errors - Complete Guide

**Category**: Asymptotic Methods  
**Generated**: 2025-08-15 14:09:53  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Overview

Cluster robust standard errors adjust for within-cluster correlation in panel data.
They provide consistent variance estimation when observations are correlated within
clusters (e.g., states, schools, firms) but independent across clusters.

## Theoretical Foundation

Based on the work of Arellano (1987) and extended by Cameron, Gelbach & Miller (2011).
The method relaxes the independence assumption and allows for arbitrary correlation
within clusters while maintaining independence across clusters.

The cluster robust variance estimator is:
V_cluster = (X'X)^(-1) * Σ_g (X_g' ε_g ε_g' X_g) * (X'X)^(-1)

where g indexes clusters, X_g are regressors for cluster g, and ε_g are residuals.

## Implementation Details

1. Estimate model using OLS
2. Calculate cluster-specific score vectors
3. Compute sandwich variance estimator
4. Apply finite sample corrections (HC1, HC2, HC3)
5. Use t-distribution with G-1 degrees of freedom

## Assumptions

- Independence across clusters
- Sufficient number of clusters (typically 30+)
- Cluster sizes can be unequal
- Model is correctly specified

## Advantages

- Computationally efficient
- Standard in econometric practice
- Good asymptotic properties
- Handles unbalanced clusters
- Easy to implement

## Disadvantages

- Poor performance with few clusters (<20)
- Over-rejection in finite samples
- Sensitive to cluster definition
- Assumes large-cluster asymptotics

## When to Use

- Panel data with natural clustering
- Large number of clusters (>30)
- Standard policy evaluation
- When computational efficiency matters

## When NOT to Use

- Few clusters (<15)
- Suspected model misspecification
- Non-standard distributions
- Very unbalanced cluster sizes

## Code Example

```python
import statsmodels.formula.api as smf

# Estimate model with cluster robust standard errors
model = smf.ols('outcome ~ treatment + C(state) + C(year)', data=df)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['state']})

# Display results
print(results.summary())
```


## Practical Considerations

- Use HC1 finite sample correction for better performance
- Consider clustering at multiple levels if relevant
- Check sensitivity to cluster definition
- Verify sufficient within-cluster variation

## References

- Arellano, M. (1987). Computing robust standard errors for within-groups estimators.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
- Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data.


---
*This guide is part of the comprehensive robustness analysis methodology documentation.*
