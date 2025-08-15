# Cluster Bootstrap - Complete Guide

**Category**: Bootstrap Methods  
**Generated**: 2025-08-15 14:09:53  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Overview

Cluster bootstrap resamples entire clusters with replacement to preserve
within-cluster dependence structure. It's particularly useful when the
number of clusters is moderate and cluster structure is important.

## Theoretical Foundation

Based on standard bootstrap theory (Efron, 1979) adapted for clustered data.
The method resamples clusters rather than individual observations to preserve
the correlation structure within clusters.

Bootstrap samples θ* from:
F*_n = (1/G) Σ_{g∈S*} δ_{cluster_g}

where S* is a bootstrap sample of clusters drawn with replacement.

## Implementation Details

1. Identify all unique clusters
2. For each bootstrap iteration:
   a. Sample G clusters with replacement from original clusters
   b. Form bootstrap dataset by stacking selected clusters
   c. Estimate model on bootstrap sample
   d. Save parameter estimates or test statistics
3. Construct bootstrap distribution
4. Calculate confidence intervals or p-values

## Assumptions

- Clusters are independent and identically distributed
- Within-cluster correlation can be arbitrary
- Moderate number of clusters (15+)
- Cluster sizes can vary

## Advantages

- Preserves exact cluster dependence structure
- Flexible for various estimators
- Intuitive resampling scheme
- Can handle complex within-cluster patterns

## Disadvantages

- Requires moderate number of clusters
- Computationally intensive
- Bootstrap samples may be unbalanced
- Convergence not always guaranteed

## When to Use

- Moderate number of clusters (15-50)
- Complex within-cluster dependence
- Non-standard estimators
- Robustness check for other methods

## When NOT to Use

- Very few clusters (<10)
- Very large datasets
- Simple linear models with many clusters
- When wild bootstrap is available

## Code Example

```python
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
```


## References

- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
- Davison, A. C., & Hinkley, D. V. (1997). Bootstrap methods and their applications.
- Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data.


---
*This guide is part of the comprehensive robustness analysis methodology documentation.*
