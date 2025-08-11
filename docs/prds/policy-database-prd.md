# Policy Database PRD: Special Education State Policy Coding System

## Document Purpose
**Audience**: Policy researchers, research assistants, legal analysts  
**Scope**: Requirements for systematic state policy coding and database construction  
**Status**: Active development  
**Related Documents**: [Research Methodology PRD](research-methodology-prd.md), [Data Collection PRD](data-collection-prd.md)

---

## 1. Overview

This PRD defines the requirements for creating a comprehensive database of state-level special education policy changes from 2009-2023. The system must systematically code funding formula reforms, court orders, federal monitoring changes, and political variables to enable causal analysis of policy effects.

### 1.1 Core Objectives

- **Systematic Coding**: Standardized approach to categorizing state policy changes
- **Temporal Precision**: Exact timing of policy implementations
- **Treatment Variation**: Clear identification of treated vs. control states
- **Validation**: Quality assurance and inter-coder reliability
- **Automation**: Semi-automated search and coding where possible

---

## 2. Policy Coding Framework

### 2.1 Base Panel Structure

**Requirements**:
- State-year panel: 50 states × 15 years = 750 observations
- Consistent state identifiers (two-letter codes)
- Complete temporal coverage (2009-2023)
- Balanced panel structure

```python
class PolicyDatabaseBuilder:
    """
    Systematic approach to coding state special education policies
    """
    
    def create_base_panel(self) -> pd.DataFrame:
        """
        Create state-year panel structure with consistent identifiers
        
        Output Schema:
        - state: str (two-letter code)
        - year: int (2009-2023)
        - state_year: str (unique identifier)
        - fips_code: str (federal state code)
        - region: str (census region)
        """
```

### 2.2 Data Structure Requirements

**Primary Keys**: `state`, `year`  
**Foreign Keys**: Links to outcome datasets via state-year  
**Temporal Consistency**: No gaps in time series by state

---

## 3. Funding Formula Coding

### 3.1 Reform Classification System

**Primary Categories**:
1. **Census-based**: Per-pupil allocation regardless of actual costs
2. **Weighted formula**: Disability-specific multipliers
3. **Cost reimbursement**: Percentage of actual expenditures
4. **Circuit breaker**: State aid kicks in above threshold
5. **Evidence-based**: Funding tied to research-based practices

### 3.2 Known Major Reforms (2009-2023)

| State | Year | Reform Type | Description | Implementation Status |
|-------|------|-------------|-------------|----------------------|
| CA | 2013 | Census-plus | Local Control Funding Formula | ✓ Implemented |
| MA | 2019 | Circuit breaker enhanced | Increased reimbursement rate | ✓ Implemented |
| PA | 2014 | Court-ordered study | Special education funding commission | ✓ Implemented |
| TX | 2019 | Weight increase | HB3 increased special ed weights | ✓ Implemented |
| IL | 2017 | Evidence-based | Complete funding formula overhaul | ✓ Implemented |
| TN | 2016 | Weighted student | Tennessee Education Finance Act | ✓ Implemented |
| WA | 2018 | Court-mandated | McCleary decision implementation | ✓ Implemented |
| KS | 2017 | Court-mandated | Gannon v. Kansas settlement | ✓ Implemented |
| CT | 2018 | Excess cost reform | Special education grant changes | ✓ Implemented |
| NJ | 2018 | Census-plus | S2 school funding reform | ✓ Implemented |
| VT | 2019 | Census to weighted | Act 173 special education reform | ✓ Implemented |
| NV | 2019 | Weighted funding | SB543 pupil-centered funding | ✓ Implemented |
| MD | 2020 | Blueprint implementation | Kirwan Commission recommendations | ✓ Implemented |
| NH | 2020 | Court settlement | ConVal adequate education decision | ✓ Implemented |
| MI | 2019 | Weighted formula | Foundation formula enhancement | ✓ Implemented |

### 3.3 Coding Requirements

**For Each Reform**:
- `reform_year`: Exact year of implementation
- `reform_type`: Category from classification system
- `reform_status`: Binary indicator (0/1)
- `pre_post_indicator`: Separate variables for analysis periods
- `reform_description`: Detailed text description
- `funding_mechanism_change`: Specific changes to funding calculation

**Validation Requirements**:
- Cross-reference with 3+ independent sources
- Verify implementation vs. passage dates
- Document phase-in periods for gradual reforms

---

## 4. Court Order Database

### 4.1 Legal Case Tracking

**Scope**: Special education adequacy litigation affecting state funding

**Required Information**:
- Case name and citation
- Filing date and decision dates
- Court level (state supreme, federal district, etc.)
- Remedy ordered (funding increase, formula change, etc.)
- Compliance timeline
- Current status (active, settled, appealed)

### 4.2 Known Cases (2009-2023)

```python
court_orders = {
    'PA': {
        'case': 'Gaskin v. Pennsylvania',
        'start_year': 2014,
        'end_year': 2023,
        'type': 'adequacy',
        'remedy': 'funding formula study and reform',
        'status': 'active'
    },
    'WA': {
        'case': 'McCleary v. State of Washington', 
        'start_year': 2012,
        'end_year': 2018,
        'type': 'adequacy',
        'remedy': 'increased K-12 funding including special ed',
        'status': 'resolved'
    },
    'KS': {
        'case': 'Gannon v. Kansas',
        'start_year': 2014, 
        'end_year': 2019,
        'type': 'adequacy',
        'remedy': 'increased base funding and weights',
        'status': 'resolved'
    }
    # ... additional cases
}
```

### 4.3 Coding Variables

- `court_order_active`: Binary indicator for active litigation
- `court_case_name`: Full case citation
- `remedy_type`: Category of court-ordered remedy
- `compliance_pressure`: Intensity measure (1-5 scale)
- `federal_vs_state`: Court jurisdiction

---

## 5. Federal Monitoring Database

### 5.1 IDEA Monitoring Status

**Data Source**: OSEP State Determination Letters (annual)  
**Categories**:
1. **Meets Requirements**: No significant compliance issues
2. **Needs Assistance**: Some compliance problems identified  
3. **Needs Intervention**: Serious compliance issues requiring federal action
4. **Needs Substantial Intervention**: Most serious category

### 5.2 Implementation Requirements

```python
def code_federal_monitoring(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Code IDEA federal monitoring and enforcement status
    
    Data Source: OSEP State Determination Letters 2009-2023
    
    Variables Created:
    - federal_status: Categorical status
    - needs_assistance: Binary indicator
    - intervention_level: Ordinal scale (0-3)
    - status_change: Year-over-year status change
    """
```

### 5.3 Historical Data (2020-2023)

**2023 Status**:
- Needs Assistance: CA, DC, NY, TX
- Needs Intervention: PR
- Meets Requirements: Remaining 46 states

**2022 Status**:  
- Needs Assistance: CA, DC, HI, NY, TX, WV
- Needs Intervention: PR

**Coding Challenge**: Historical data requires manual collection from archived OSEP reports

---

## 6. Political Control Variables

### 6.1 Required Variables

**Electoral Control**:
- Governor party affiliation
- State legislature control (unified/divided)
- Education committee leadership changes
- Superintendent selection method (elected/appointed)

**Interest Group Influence**:
- Teacher union strength (membership rates)
- Parent advocacy organization presence
- Business coalition involvement
- Disability rights organization activity

### 6.2 Data Sources

**Primary Sources**:
- National Conference of State Legislatures (NCSL)
- Ballotpedia election results
- Council of Chief State School Officers (CCSSO)
- Education Commission of the States (ECS)

**Implementation**:
```python
def code_political_variables(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add political control variables from external sources
    
    Sources:
    - NCSL partisan control database
    - Ballotpedia governor elections
    - State education agency websites
    
    Variables:
    - governor_party: R/D/I
    - legislature_control: unified_R/unified_D/divided
    - unified_government: Binary
    - education_governance: Elected/appointed superintendent
    """
```

---

## 7. Semi-Automated Policy Search

### 7.1 Legislative Database Integration

**LegiScan API Integration**:
- Search state legislation by keywords
- Filter for education and disability-related bills
- Track bill progress and passage
- Identify policy changes requiring manual review

```python
def search_state_legislation(state: str, keywords: List[str]) -> pd.DataFrame:
    """
    Search state legislation databases for policy changes
    
    API: LegiScan (requires subscription)
    Alternative: State legislature websites with web scraping
    
    Keywords:
    - "special education"
    - "disability funding"  
    - "IDEA"
    - "weighted funding"
    - "inclusive education"
    
    Output: Candidate bills for manual policy coding review
    """
```

### 7.2 Quality Control Process

**Automated Pre-filtering**:
1. Keyword matching in bill titles and summaries
2. Education committee assignment
3. Bill status (passed vs. failed)
4. Effective date within study period

**Manual Review Requirements**:
- Research assistant reviews all flagged bills
- Determines substantive policy changes
- Codes implementation timing
- Cross-references with news coverage

---

## 8. Validation and Quality Assurance

### 8.1 Inter-coder Reliability

**Process**:
- 20% random sample coded by two independent researchers
- Kappa statistics calculated for categorical variables
- Discrepancies resolved through discussion
- Target: κ > 0.80 for major variables

### 8.2 External Validation

**Cross-reference Sources**:
- Education Week policy tracking
- State education department websites
- Academic literature on state reforms
- Interest group reports and newsletters

### 8.3 Temporal Consistency Checks

**Automated Checks**:
- No backward time travel (reform_year decreases)
- Reform status changes are monotonic
- Court order dates align with legal records
- Federal monitoring changes match OSEP letters

```python
def validate_coding(self, df: pd.DataFrame) -> Dict:
    """
    Comprehensive validation of policy database
    
    Checks:
    - Temporal consistency
    - Cross-variable logical relationships  
    - Coverage completeness
    - Missing data patterns
    - Outlier identification
    
    Returns validation report with pass/fail status
    """
```

---

## 9. Database Schema and Output

### 9.1 Core Variables

| Variable Name | Type | Description | Source |
|--------------|------|-------------|---------|
| `state` | str | Two-letter state code | Standard |
| `year` | int | Calendar year | Standard |
| `reform_status` | binary | Any major reform active | Derived |
| `reform_year` | int | Year of reform implementation | Manual coding |
| `reform_type` | categorical | Type of funding reform | Manual coding |
| `court_order_active` | binary | Active litigation | Legal databases |
| `federal_monitoring` | categorical | OSEP determination status | OSEP reports |
| `governor_party` | categorical | Governor party affiliation | NCSL/Ballotpedia |
| `legislature_control` | categorical | Legislative control | NCSL |
| `funding_weight` | float | Special ed weight in formula | Statute review |

### 9.2 Quality Indicators

**Completeness Metrics**:
- Percentage of state-years with complete data
- Missing data patterns by variable and year
- Coverage by region and state characteristics

**Reliability Metrics**:
- Inter-coder agreement statistics
- Source triangulation success rates
- Temporal consistency measures

---

## 10. Implementation Timeline

### 10.1 Phase 1: Foundation (Month 1)
- Create base panel structure
- Collect and review major reform documentation
- Set up validation framework
- Begin known case coding

### 10.2 Phase 2: Systematic Coding (Month 2)  
- Manual coding of all identified reforms
- Court order database construction
- Federal monitoring data entry
- Political variable integration

### 10.3 Phase 3: Validation (Month 2-3)
- Inter-coder reliability testing
- External source cross-referencing  
- Automated consistency checking
- Quality report generation

### 10.4 Phase 4: Finalization (Month 3)
- Final database assembly
- Documentation completion
- Delivery to analysis team
- Archive and version control

---

## 11. Resource Requirements

### 11.1 Personnel
- **Lead Policy Researcher**: 0.5 FTE for 3 months
- **Research Assistant**: 0.25 FTE for 2 months  
- **Legal Research Support**: 20 hours consultation

### 11.2 Data Access
- LegiScan API subscription ($500)
- State legal database access
- News archive subscriptions (LexisNexis)
- Academic database access

### 11.3 Technology
- Reference management software (Zotero)
- Collaborative coding platform
- Version control system (Git)
- Database hosting and backup

---

## 12. Deliverables

### 12.1 Primary Database
- **Format**: CSV, Stata, and JSON formats
- **Documentation**: Complete data dictionary
- **Validation**: Quality assurance report
- **Version Control**: Tagged releases with change logs

### 12.2 Supporting Documentation
- **Codebook**: Variable definitions and coding rules
- **Source Bibliography**: Complete reference list
- **Methodology Report**: Coding procedures and validation
- **Known Issues**: Limitations and caveats

### 12.3 Replication Materials
- **Coding Sheets**: Templates for future updates
- **Search Protocols**: Systematic search procedures
- **Validation Scripts**: Automated checking code
- **Update Procedures**: Process for extending database

---

**Document Control**  
- Version: 1.0  
- Last Updated: 2025-08-11  
- Next Review: Monthly during active coding  
- Quality Assurance: Independent validation required