# Technology Stack

## Core Requirements
- **Python**: >=3.12 (downgraded from 3.13 for better package compatibility)
- **Package Manager**: UV (preferred over pip for faster dependency management)

## Core Data Science Libraries
- **pandas**: >=2.0.0 - Data manipulation and analysis
- **numpy**: >=1.24.0 - Numerical computing
- **scipy**: >=1.10.0 - Scientific computing

## Econometric Analysis
- **statsmodels**: >=0.14.0 - Statistical modeling
- **linearmodels**: >=5.0.0 - Panel data and econometric models
- **econtools**: >=0.3.0 - Additional econometric utilities
- **Manual DiD Implementation**: Using statsmodels + linearmodels instead of 'did' package (due to dependency conflicts)

## Data Collection & Processing
- **requests**: >=2.28.0 - HTTP requests for API data collection
- **beautifulsoup4**: >=4.11.0 - Web scraping
- **openpyxl**: >=3.1.0 - Excel file processing
- **xlrd**: >=2.0.2 - Excel file reading

## Visualization
- **matplotlib**: >=3.6.0 - Plotting and visualization
- **seaborn**: >=0.12.0 - Statistical data visualization

## Development Tools
- **ruff**: >=0.12.8 - Fast Python linter and formatter (replaces black/isort/flake8)
- **mypy**: >=1.0.0 - Static type checking
- **pytest**: >=7.0.0 - Testing framework
- **pytest-cov**: >=4.0.0 - Coverage reporting

## Optional Extensions
- **scikit-learn**: >=1.3.0 - Machine learning
- **jupyter**: >=1.0.0 - Notebook support
- **python-dotenv**: >=1.0.0 - Environment configuration

## Dependency Management Notes
- **Removed 'did' package**: Requires system Kerberos libraries causing build failures
- **Removed 'synthdid' package**: Has build issues with missing files
- **Solution**: Manual implementation of Callaway-Sant'Anna and synthetic control methods