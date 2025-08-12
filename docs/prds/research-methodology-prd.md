# Research Methodology PRD: Special Education State Policy Analysis

## Document Purpose
**Audience**: Academic researchers, peer reviewers, methodologists  
**Scope**: Core research design and econometric methodology  
**Status**: ✅ COMPLETED - Full methodology implemented with detailed report  
**Related Documents**: [Data Collection PRD](data-collection-prd.md), [COVID Analysis PRD](covid-analysis-prd.md)

---

## 1. Executive Summary

This project examines how state-level special education policies affect outcomes for students with disabilities (SWD) using quasi-experimental methods. By exploiting variation in state funding reforms, federal monitoring status, and COVID-19 responses, we identify causal effects of policy choices on achievement, inclusion, and equity. Using current data through 2023, this research provides timely evidence for the ongoing $190B IDEA reauthorization debate.

**Key Innovation:** First study to combine post-COVID special education outcomes with state policy variation, enabling identification of which state approaches proved most resilient and effective.

---

## 2. Research Questions

### Primary (Causal):
1. **What is the causal effect of state funding formula reforms on special education achievement?**
   - Exploit staggered timing of reforms across 15+ states (2009-2023)
   - Estimate elasticity of achievement with respect to per-pupil funding

2. **How does federal IDEA monitoring pressure affect state performance?**
   - Use monitoring status changes as instrument for state effort
   - Identify spillover effects on general education students

3. **Which state policies enhanced resilience during COVID-19?**
   - Triple-difference design comparing pre/post COVID by reform status
   - Identify protective factors against learning loss

### Secondary (Mechanisms):
- Do funding increases work through inclusion, staffing, or services?
- How do political economy factors affect policy adoption and implementation?
- What are optimal funding weights for different disability categories?

---

## 3. Identification Strategy

### 3.1 Primary Approach: Staggered Difference-in-Differences
```
Y_st = Σ_τ β_τ · 1[t - T*_s = τ] + X_st + α_s + δ_t + ε_st

Where:
- T*_s = year state s reformed funding formula
- τ ∈ [-5, +5] event window
- Callaway-Sant'Anna (2021) estimator for staggered adoption
```

**Requirements**:
- Parallel trends assumption validation
- Robust standard errors clustered at state level
- Wild cluster bootstrap for small N inference

### 3.2 Instrumental Variables Strategy
```
First Stage: Funding_st = γ·Z_st + X_st + α_s + δ_t + ν_st
Second Stage: Y_st = β·Funding_st_hat + X_st + α_s + δ_t + ε_st

Instruments (Z_st):
- Court-ordered funding increases
- Federal monitoring status changes  
- Legislative turnover in education committees
- Neighboring state reforms (spatial IV)
```

**Validity Requirements**:
- First stage F-statistic > 10
- Exclusion restriction tests
- Overidentification tests when applicable

### 3.3 COVID Natural Experiment
```
Y_st = β₁·Reform_s + β₂·Post_COVID_t + β₃·Reform×Post_COVID_st + X_st + α_s + δ_t + ε_st

Tests whether pre-COVID reforms provided resilience
```

**Implementation**:
- See [COVID Analysis PRD](covid-analysis-prd.md) for detailed specifications

---

## 4. Data Requirements

### 4.1 Outcome Data
- **NAEP State Assessments** (2009-2022): Achievement by disability status, only nationally representative state-level SWD outcomes
- **EdFacts/IDEA Reports** (2009-2023): Graduation, inclusion rates, discipline
- **State Longitudinal Data Systems**: Post-school employment/college (select states)

### 4.2 Policy Variables (Hand-Collected)
- **Funding Formulas**: Type (census/weighted/cost), weights, changes
- **Court Orders**: Special education adequacy rulings
- **Federal Status**: IDEA monitoring/enforcement levels
- **Inclusion Policies**: LRE targets, mainstreaming requirements

### 4.3 Controls
- **Demographics**: Census/ACS state characteristics
- **Economic**: State GDP, unemployment, fiscal capacity
- **Political**: Governor/legislature party, union strength
- **Education**: Overall per-pupil spending, teacher wages

**Data Collection**: See [Data Collection PRD](data-collection-prd.md) for technical specifications

---

## 5. Methods

### 5.1 Main Specifications

**Event Study:**
```stata
reghdfe achievement i.years_to_treatment##i.state [aw=enrollment], ///
    absorb(state year) cluster(state)
```

**Continuous Treatment:**
```stata
ivreghdfe achievement (log_sped_funding = court_order neighbor_reform) ///
    controls, absorb(state year) cluster(state)
```

**Heterogeneity Analysis:**
- By disability type (learning disabilities vs. autism vs. intellectual)
- By initial achievement levels (floor/ceiling effects)
- By state capacity (fiscal health, urbanicity)

### 5.2 Robustness Requirements

**Mandatory Robustness Checks**:
- Synthetic control for each treated state
- Permutation inference for small N
- Leave-one-state-out validation
- Alternative clustering (region, political affiliation)

**Specification Curve Analysis**:
- Vary control sets
- Alternative outcome measures
- Different sample restrictions
- Multiple inference corrections

### 5.3 Statistical Implementation

**Software Requirements**:
- Primary: Python with statsmodels, linearmodels
- Secondary: R with did package for Callaway-Sant'Anna
- Validation: Stata for robustness checks

**Inference**:
- Cluster-robust standard errors at state level
- Wild cluster bootstrap for small cluster robust inference
- Multiple hypothesis testing corrections (Bonferroni, FDR)

---

## 6. Timeline and Deliverables

| Phase | Duration | Activities | Key Outputs |
|-------|----------|------------|-------------|
| 1 | Month 1 | Data collection via APIs | Clean NAEP, EdFacts, F-33 datasets |
| 2 | Month 2 | Policy database construction | Coded reform dates, court cases |
| 3 | Month 3 | Descriptive analysis & validation | Summary statistics, event study plots |
| 4 | Month 4 | Main causal estimates | DiD, IV, and COVID interaction results |
| 5 | Month 5 | Robustness & mechanisms | Synthetic controls, heterogeneity |
| 6 | Month 6 | Writing & dissemination | Full paper, policy brief, presentations |

---

## 7. Expected Contributions

### 7.1 Academic Contributions
- **Novel Identification**: First causal estimates using post-COVID special education data
- **Methodological Innovation**: COVID as exogenous shock for policy evaluation
- **Empirical Evidence**: Quantified effects of state funding formula reforms
- **Mechanism Analysis**: Decomposition of funding effects through channels

### 7.2 Policy Relevance
- **IDEA Reauthorization**: Direct input for $190B federal program
- **State Guidance**: Evidence-based funding formula recommendations
- **Minimum Thresholds**: Data-driven per-pupil spending targets
- **ROI Analysis**: Cost-effectiveness of special education investments

### 7.3 Methodological Advances
- **Staggered DiD**: Application to education policy with small N
- **COVID Identification**: Framework for pandemic impact studies
- **State-Level Focus**: Demonstration of feasible policy evaluation approach

---

## 8. Quality Assurance

### 8.1 Validation Requirements
- **Pre-trends Testing**: Parallel trends in 3+ pre-treatment periods
- **Placebo Tests**: Randomized treatment assignment
- **Sensitivity Analysis**: Vary treatment timing definitions
- **External Validity**: Compare with district-level studies where available

### 8.2 Reproducibility Standards
- **Code Documentation**: All analysis scripts with clear comments
- **Version Control**: Git repository with tagged releases
- **Data Provenance**: Complete documentation of data sources and transformations
- **Computational Environment**: Requirements file and environment specifications

### 8.3 Peer Review Preparation
- **Methods Transparency**: Detailed technical appendix
- **Results Robustness**: Comprehensive sensitivity analysis
- **Alternative Explanations**: Address potential confounds
- **Policy Implications**: Clear causal interpretation

---

## 9. Success Metrics

### 9.1 Academic Success
- **Statistical Power**: 80% power to detect 0.1σ effect sizes
- **Robustness**: Results stable across 90% of specifications
- **Publication Target**: Top field journal (Journal of Public Economics, AEJ: Economic Policy)

### 9.2 Policy Impact
- **State Adoption**: 5+ states request technical assistance
- **Federal Reference**: Cited in IDEA reauthorization hearings
- **Media Coverage**: Coverage in education policy publications

### 9.3 Technical Success
- **Automation**: Full pipeline executable with single command
- **Reproduction**: Independent replication by research assistant
- **Extension**: Framework adaptable to other policy domains

---

## 10. Risk Mitigation

### 10.1 Data Risks
- **API Changes**: Multiple data sources for each outcome
- **Missing Data**: Interpolation and imputation strategies
- **Quality Issues**: Comprehensive validation protocols

### 10.2 Methodological Risks
- **Weak Instruments**: Multiple IV specifications and testing
- **Small N**: Bootstrap and permutation inference
- **Heterogeneous Effects**: Explicit modeling of treatment heterogeneity

### 10.3 Timeline Risks
- **Technical Delays**: Modular implementation allows parallel development
- **Scope Creep**: Clear prioritization of core vs. extension analyses
- **Resource Constraints**: Solo-researcher feasible design

---

## Appendices

### A. Related Literature
Key papers establishing methodological foundations and policy context.

### B. Technical Specifications
Detailed econometric specifications and assumption requirements.

### C. Sample Size Calculations
Power analysis for main specifications.

---

**Document Control**  
- Version: 1.0  
- Last Updated: 2025-08-11  
- Next Review: Monthly during active development  
- Contact: Project lead researcher