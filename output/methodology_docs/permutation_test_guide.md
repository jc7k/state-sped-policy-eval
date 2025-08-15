# Permutation Test - Complete Guide

**Category**: Exact Methods  
**Generated**: 2025-08-15 14:09:53  
**Author**: Jeff Chen (jeffreyc1@alumni.cmu.edu)  
**Created with**: Claude Code  

## Overview

Permutation tests provide exact finite sample inference by comparing the
observed test statistic to its distribution under random reassignment of
treatment status. They require minimal assumptions and provide exact p-values.

## Theoretical Foundation

Based on Fisher's exact test principle (Fisher, 1935). Under the sharp null
hypothesis of no treatment effect for any unit, all possible treatment
assignments are equally likely, providing an exact reference distribution.

The test compares observed statistic T_obs to the permutation distribution:
T_perm = {T(π): π ∈ Π}

where Π is the set of all possible treatment permutations.

## Implementation Details

1. Specify sharp null hypothesis (typically zero effect)
2. Calculate observed test statistic
3. Generate all possible (or random sample of) treatment permutations
4. For each permutation:
   a. Reassign treatment status
   b. Calculate test statistic under permutation
5. Compare observed statistic to permutation distribution
6. Calculate exact p-value as proportion of permuted statistics ≥ observed

## Assumptions

- Sharp null hypothesis (no effect for any unit)
- Treatment assignment mechanism known
- Observations exchangeable under null
- Minimal distributional assumptions

## Advantages

- Exact finite sample inference
- No distributional assumptions
- Robust to all model violations
- Valid with any sample size
- Intuitive interpretation

## Disadvantages

- Computationally very intensive
- Limited to simple null hypotheses
- May be conservative
- Requires careful treatment assignment modeling

## When to Use

- Very few clusters (<10)
- Non-standard distributions
- Exact inference required
- Skeptical audience
- Small samples

## When NOT to Use

- Large datasets (computational burden)
- Complex models with many parameters
- Continuous treatment variables
- Standard applications with sufficient clusters

## Code Example

```python
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
```


## Monte Carlo Evidence

Simulation studies demonstrate:
- Exact size control under null hypothesis
- Good power properties for large effects
- Robust to all distributional violations
- Conservative for small effects in some settings


## Practical Considerations

- Ensure sufficient permutations (≥999)
- Consider clustering in permutation scheme
- Verify computational feasibility
- Document exact null hypothesis tested

## References

- Fisher, R. A. (1935). The design of experiments.
- Rosenbaum, P. R. (2002). Observational studies.
- Ernst, M. D. (2004). Permutation methods: a basis for exact inference.


---
*This guide is part of the comprehensive robustness analysis methodology documentation.*
