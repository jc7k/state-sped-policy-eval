# Data Collection PRD: Special Education State Policy Analysis

## Document Purpose
**Audience**: Data engineers, research assistants, developers  
**Scope**: Technical specifications for automated data acquisition  
**Status**: Active development  
**Related Documents**: [Research Methodology PRD](research-methodology-prd.md), [Policy Database PRD](policy-database-prd.md)

---

## 1. Overview

This PRD defines the technical requirements for collecting, processing, and validating all datasets needed for the state-level special education policy analysis. The system must automatically gather data from multiple federal APIs and web sources, handle rate limiting, validate data quality, and produce analysis-ready datasets.

### 1.1 Data Sources Summary

| Source | Coverage | Update Frequency | API Available | Key Variables |
|--------|----------|------------------|---------------|---------------|
| NAEP | 2009-2022 (biennial) | Every 2 years | Yes | SWD achievement scores |
| EdFacts | 2009-2023 | Annual | Yes | Inclusion, graduation, discipline |
| Census F-33 | 2009-2022 | Annual | Yes | Per-pupil spending, revenues |
| OCR CRDC | 2009-2020 (biennial) | Every 2 years | CSV downloads | Discipline, access measures |

---

## 2. NAEP Data Collection

### 2.1 API Specifications

**Base URL**: `https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx`  
**Documentation**: https://www.nationsreportcard.gov/api_documentation.aspx  
**Rate Limit**: 10 requests/minute  

### 2.2 Implementation Requirements

```python
class NAEPDataCollector:
    """
    Automated NAEP data collection for state-level special education analysis
    """
    
    def __init__(self):
        self.base_url = "https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx"
        self.results = []
        self.rate_limit_delay = 6  # seconds
        
    def fetch_state_swd_data(self, years: List[int], grades: List[int] = [4, 8],
                             subjects: List[str] = ['mathematics', 'reading']) -> pd.DataFrame:
        """
        Fetch NAEP data by state for students with disabilities
        
        Required parameters:
        - type: 'data'
        - subject: 'mathematics' | 'reading'
        - grade: 4 | 8
        - year: valid assessment year
        - jurisdiction: 'states'
        - variable: 'SDRACEM' (Students with disabilities)
        - stattype: 'MN:MN,RP:RP' (Mean scores and percentiles)
        """
```

### 2.3 Data Validation Requirements

**Mandatory Checks**:
- All 50 states + DC represented for available years
- No duplicate state-year-subject-grade combinations
- Score values within reasonable ranges (0-500 scale)
- Standard errors present and non-zero
- Missing data patterns documented

**Output Schema**:
```python
{
    'state': str,           # Two-letter state code
    'year': int,            # Assessment year
    'grade': int,           # 4 or 8
    'subject': str,         # 'mathematics' or 'reading'
    'swd_mean': float,      # Mean scale score for SWD
    'swd_se': float,        # Standard error
    'non_swd_mean': float,  # Mean scale score for non-SWD
    'non_swd_se': float,    # Standard error
    'gap': float,           # Achievement gap (non-SWD - SWD)
    'gap_se': float         # Gap standard error
}
```

---

## 3. EdFacts Data Collection

### 3.1 API Specifications

**Base URL**: `https://www2.ed.gov/data/api/edfacts/v1/`  
**Authentication**: None required  
**Rate Limit**: Not specified (use 1 request/second)

### 3.2 Required Datasets

#### 3.2.1 IDEA Child Count
**Endpoint**: `/idea/childcount/{year}`  
**Purpose**: Students served by disability category  
**Years**: 2009-2023

#### 3.2.2 Educational Environments
**Endpoint**: `/idea/environments/{year}`  
**Purpose**: Inclusion rates (LRE data)  
**Years**: 2009-2023

#### 3.2.3 Exiting Data
**Endpoint**: `/idea/exiting/{year}`  
**Purpose**: Graduation and dropout rates  
**Years**: 2009-2023

### 3.3 Implementation Requirements

```python
class EdFactsCollector:
    """
    EdFacts state-level data collection
    New API endpoint as of 2023
    """
    
    def __init__(self):
        self.base_url = "https://www2.ed.gov/data/api/edfacts/v1/"
        self.rate_limit_delay = 1.0
        
    def fetch_idea_data(self, years: List[int]) -> pd.DataFrame:
        """
        Comprehensive IDEA data collection
        
        Returns merged dataset with:
        - Child count by disability category
        - Educational environment percentages
        - Exit outcome percentages
        """
```

### 3.4 Derived Variables

**Required Calculations**:
- `inclusion_rate`: % in regular class 80%+ of day
- `restrictive_rate`: % in separate schools/facilities
- `graduation_rate`: % graduating with regular diploma
- `dropout_rate`: % dropping out

---

## 4. Census Education Finance Data

### 4.1 API Specifications

**Base URL**: `https://api.census.gov/data/`  
**Authentication**: API key required from census.gov  
**Rate Limit**: 500 requests/day without key, unlimited with key

### 4.2 Implementation Requirements

```python
class CensusEducationFinance:
    """
    Census Bureau education finance data
    Includes special education expenditures starting 2015
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.census.gov/data/"
        
    def fetch_state_finance(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch F-33 survey data with special education breakouts
        
        Key variables:
        - TOTALEXP: Total expenditures
        - TCURINST: Current instruction spending
        - TCURSSVC: Student support services
        - TCUROTH: Other current expenditures
        - ENROLL: Student enrollment
        """
```

### 4.3 Special Education Spending

**Challenge**: Special education spending not separately reported in all years  
**Solution**: Use proxy measures and state supplemental data where available

**Proxy Variables**:
- Student support services expenditures
- Special programs expenditures
- Federal IDEA funding received

---

## 5. OCR Civil Rights Data

### 5.1 Data Access Method

**Method**: Direct CSV download (no API available)  
**URL Pattern**: `https://ocrdata.ed.gov/assets/downloads/{year}/CRDC-{year}-State-Discipline.csv`  
**Years**: 2009, 2011, 2013, 2015, 2017, 2020

### 5.2 Implementation Requirements

```python
class OCRDataCollector:
    """
    Office for Civil Rights state-aggregated data
    Focus on discipline and access measures
    """
    
    def fetch_discipline_data(self, year: int) -> pd.DataFrame:
        """
        Fetch discipline data by disability status
        
        Calculate disproportionality metrics:
        - Suspension risk ratio (SWD vs. non-SWD)
        - Expulsion risk ratio
        - Referral to law enforcement ratio
        """
```

### 5.3 Key Variables

- Out-of-school suspensions by disability status
- Expulsions by disability status  
- Referrals to law enforcement
- School-related arrests
- Access to advanced coursework

---

## 6. Master Data Pipeline

### 6.1 Pipeline Architecture

```python
def run_full_data_collection():
    """
    Master function to collect all datasets with error handling and logging
    """
    
    # Configuration
    config = {
        'census_api_key': os.environ.get('CENSUS_API_KEY'),
        'output_dir': 'data/raw/',
        'log_file': 'data_collection.log',
        'years': list(range(2009, 2024))
    }
    
    # Initialize collectors
    collectors = {
        'naep': NAEPDataCollector(),
        'edfacts': EdFactsCollector(),
        'census': CensusEducationFinance(config['census_api_key']),
        'ocr': OCRDataCollector()
    }
    
    # Execute with progress tracking and error handling
    results = {}
    for name, collector in collectors.items():
        try:
            logger.info(f"Starting {name} collection...")
            df = collector.collect_all(config['years'])
            
            # Validate
            validation = validate_dataset(df, name)
            if not validation['passed']:
                logger.error(f"{name} validation failed: {validation['errors']}")
                
            # Save
            output_path = f"{config['output_dir']}/{name}_raw.csv"
            df.to_csv(output_path, index=False)
            results[name] = {'records': len(df), 'path': output_path}
            
            logger.info(f"✓ Completed {name}: {len(df)} records")
            
        except Exception as e:
            logger.error(f"✗ Failed {name}: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results
```

### 6.2 Error Handling Requirements

**Network Errors**:
- Retry failed requests up to 3 times
- Exponential backoff for rate limiting
- Graceful degradation for partial failures

**Data Quality Issues**:
- Log all validation failures
- Continue processing with warnings
- Generate quality report for manual review

**Rate Limiting**:
- Respect documented limits
- Implement conservative delays
- Monitor for 429 status codes

---

## 7. Data Validation Framework

### 7.1 Validation Requirements

```python
def validate_dataset(df: pd.DataFrame, source: str) -> Dict:
    """
    Comprehensive data validation for each source
    
    Returns:
    - passed: bool
    - errors: List[str]
    - warnings: List[str]
    - summary: Dict[str, any]
    """
    
    validation = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Universal checks
    validation = check_required_columns(df, source, validation)
    validation = check_data_types(df, source, validation)
    validation = check_year_coverage(df, source, validation)
    validation = check_state_coverage(df, source, validation)
    validation = check_duplicates(df, source, validation)
    
    # Source-specific checks
    if source == 'naep':
        validation = validate_naep_specific(df, validation)
    elif source == 'edfacts':
        validation = validate_edfacts_specific(df, validation)
    # ... etc
    
    return validation
```

### 7.2 Quality Metrics

**Coverage Metrics**:
- Percentage of expected state-year combinations
- Missing data patterns by variable
- Temporal consistency checks

**Value Range Checks**:
- Achievement scores within valid ranges
- Rates and percentages between 0-100
- Spending amounts positive and reasonable

**Consistency Checks**:
- Cross-variable logical relationships
- Time series plausibility
- State ranking stability

---

## 8. Output Specifications

### 8.1 File Structure

```
data/
├── raw/
│   ├── naep_raw.csv
│   ├── edfacts_raw.csv
│   ├── census_raw.csv
│   ├── ocr_raw.csv
│   └── collection_log.txt
├── processed/
│   └── [cleaned individual files]
└── final/
    └── [merged analysis dataset]
```

### 8.2 Documentation Requirements

**Data Dictionary**: Complete variable definitions and sources  
**Collection Log**: Timestamps, success rates, error descriptions  
**Quality Report**: Validation results and recommended actions  
**Lineage**: Complete data provenance from source to final dataset

---

## 9. Performance Requirements

### 9.1 Execution Time
- **Target**: Complete collection in < 2 hours
- **Acceptable**: < 4 hours with full validation
- **Critical**: Must complete within 8 hours

### 9.2 Reliability
- **Success Rate**: 95% of expected records collected
- **Error Recovery**: Automatic retry for temporary failures
- **Monitoring**: Real-time progress reporting

### 9.3 Scalability
- **Years**: Easy to add new years of data
- **Sources**: Modular design for additional data sources
- **States**: Support for territories if needed

---

## 10. Security and Compliance

### 10.1 Data Privacy
- **Public Data**: All sources contain only aggregate, public data
- **No PII**: No individual student information collected
- **FERPA Compliance**: Not applicable (aggregate data only)

### 10.2 API Security
- **Keys**: Store API keys in environment variables
- **HTTPS**: All connections encrypted
- **Authentication**: Follow each provider's requirements

---

## 11. Testing Requirements

### 11.1 Testing Framework and Standards

**Framework**: pytest with coverage reporting  
**Minimum Coverage**: 80% overall line coverage  
**Critical Module Coverage**: 90%+ for data collectors and validation functions  
**Complex Logic Coverage**: 95%+ for calculations and API parsing logic

### 11.2 Unit Tests

**Coverage Target**: 90%+ for individual modules

**Required Test Categories**:
- Individual collector classes (NAEPDataCollector, EdFactsCollector, etc.)
- Validation functions with comprehensive edge case testing
- Data transformation utilities and calculation accuracy
- Error handling and recovery mechanisms
- State name/code conversions and mappings
- API parameter validation and formatting

**Testing Patterns**:
- Mock all external API calls using `pytest-httpx`
- Property-based testing with `hypothesis` for data validation
- Parameterized tests for multiple scenarios
- Fixture-based test data management

### 11.3 Integration Tests

**Coverage Target**: 85%+ for multi-component interactions

**Required Test Scenarios**:
- Full pipeline execution with mocked external dependencies
- Error propagation and recovery across modules  
- Data quality validation end-to-end
- Cross-module data consistency verification
- Master pipeline orchestration testing

### 11.4 Performance Tests

**Requirements**:
- Large dataset handling (10,000+ records simulation)
- Memory usage monitoring and leak detection
- Network timeout handling and retry logic
- API rate limiting compliance verification
- Execution time benchmarks (<60 seconds for test suite)

### 11.5 Test Coverage Reporting

**Tools Required**:
- `pytest-cov>=4.0.0` for coverage measurement
- Coverage reports in HTML and XML formats
- Integration with CI/CD for coverage trending
- Coverage badges for documentation

**Quality Gates**:
- All tests must pass before code integration
- Coverage cannot decrease without explicit justification
- Critical paths (data collection, validation) require 90%+ coverage
- New features must include comprehensive test coverage

### 11.6 Mock Strategy and Test Data

**API Mocking**:
- Use `pytest-httpx` for HTTP request/response mocking
- Realistic mock responses matching actual API schemas
- Error scenario simulation (timeouts, 404s, malformed data)
- Rate limiting behavior simulation

**Test Fixtures**:
- Sample API responses for each data source
- Edge case data (missing values, special characters, boundary conditions)  
- State mapping test data (all 50 states + DC)
- Policy reform test scenarios with known dates

**Test Data Management**:
- Fixtures stored in `tests/fixtures/` directory
- JSON format for API responses
- CSV format for tabular test data
- Synthetic data generation for large-scale testing

### 11.7 Continuous Integration Requirements

**CI/CD Pipeline** (GitHub Actions):
- Run tests on Python 3.12 and 3.13
- Execute full test suite on push and pull request
- Generate and upload coverage reports
- Fail builds on coverage decrease or test failures
- Performance regression detection

**Pre-commit Hooks**:
- Run fast unit tests before commits
- Code formatting validation (black, isort)
- Type checking with mypy
- Linting with flake8

### 11.8 Test Organization Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── unit/
│   ├── collection/
│   │   ├── test_naep_collector.py          # 95%+ coverage target
│   │   ├── test_edfacts_collector.py       # 90%+ coverage target  
│   │   ├── test_census_collector.py        # 85%+ coverage target
│   │   ├── test_ocr_collector.py           # 80%+ coverage target
│   │   ├── test_policy_builder.py          # 90%+ coverage target
│   │   └── test_validation.py              # 95%+ coverage target
│   ├── cleaning/
│   └── analysis/
├── integration/
│   ├── test_full_pipeline.py               # End-to-end pipeline
│   └── test_data_quality.py                # Cross-module validation
├── performance/
│   └── test_performance.py                 # Memory, speed benchmarks
└── fixtures/
    ├── naep_sample_response.json           # Realistic API responses
    ├── edfacts_sample_data.json
    ├── census_sample_response.json
    └── policy_test_data.csv
```

### 11.9 Testing Configuration Files

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=code
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-report=term-missing
    --cov-fail-under=80
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
```

**Test Dependencies** (added to pyproject.toml dev section):
- `pytest>=7.0.0` (testing framework)
- `pytest-cov>=4.0.0` (coverage reporting)
- `pytest-mock>=3.10.0` (mocking utilities)
- `pytest-httpx>=0.21.0` (HTTP request mocking)
- `factory-boy>=3.2.0` (test data generation)
- `hypothesis>=6.0.0` (property-based testing)
- `pytest-xdist>=3.0.0` (parallel test execution)

---

## 12. Deployment

### 12.1 Environment Setup
```bash
# Required Python packages
pip install pandas numpy requests beautifulsoup4 
pip install pytest logging pathlib

# Environment variables
export CENSUS_API_KEY="your_census_key"
export DATA_OUTPUT_DIR="data/"
```

### 12.2 Execution
```bash
# Full pipeline
python -m code.collection.run_full_collection

# Individual sources
python -m code.collection.naep_collector
python -m code.collection.edfacts_collector
```

---

**Document Control**  
- Version: 1.0  
- Last Updated: 2025-08-11  
- Next Review: Weekly during development  
- Dependencies: Census API key, stable internet connection