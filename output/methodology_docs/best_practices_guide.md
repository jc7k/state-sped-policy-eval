# Robust Analysis Best Practices Guide

**Generated**: 2025-08-15 14:09:53  
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
                   cov_kwds={'groups': data['state'],
                            'use_correction': True})

# Check degrees of freedom: should be n_clusters - 1
print(f"Degrees of freedom: {results.df_resid}")
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
                   cov_kwds={'groups': [data['state'], data['year']]})
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
