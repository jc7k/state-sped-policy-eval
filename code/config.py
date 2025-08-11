#!/usr/bin/env python
"""
Configuration Management for Special Education Policy Analysis
Loads environment variables and provides configuration validation
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration management for data collection pipeline
    """
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults"""
        
        # API Keys
        self.census_api_key = os.getenv('CENSUS_API_KEY')
        
        # Data directories
        self.data_output_dir = Path(os.getenv('DATA_OUTPUT_DIR', 'data/'))
        self.raw_data_dir = Path(os.getenv('RAW_DATA_DIR', 'data/raw/'))
        self.processed_data_dir = Path(os.getenv('PROCESSED_DATA_DIR', 'data/processed/'))
        self.final_data_dir = Path(os.getenv('FINAL_DATA_DIR', 'data/final/'))
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'data_collection.log')
        
        # Rate limiting (seconds between requests)
        self.naep_rate_limit_delay = float(os.getenv('NAEP_RATE_LIMIT_DELAY', '6.0'))
        self.edfacts_rate_limit_delay = float(os.getenv('EDFACTS_RATE_LIMIT_DELAY', '1.0'))
        self.ocr_rate_limit_delay = float(os.getenv('OCR_RATE_LIMIT_DELAY', '1.0'))
        
        # Data collection settings
        self.collection_years = self._parse_years(os.getenv('COLLECTION_YEARS', '2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023'))
        self.naep_years = self._parse_years(os.getenv('NAEP_YEARS', '2009,2011,2013,2015,2017,2019,2022'))
        self.ocr_years = self._parse_years(os.getenv('OCR_YEARS', '2009,2011,2013,2015,2017,2020'))
        self.naep_subjects = self._parse_list(os.getenv('NAEP_SUBJECTS', 'mathematics,reading'))
        self.naep_grades = self._parse_int_list(os.getenv('NAEP_GRADES', '4,8'))
        
        # Validation settings
        self.min_state_coverage = int(os.getenv('MIN_STATE_COVERAGE', '50'))
        self.max_missing_data_rate = float(os.getenv('MAX_MISSING_DATA_RATE', '0.30'))
        self.enable_validation = os.getenv('ENABLE_VALIDATION', 'true').lower() == 'true'
        
        # Development and testing
        self.development_mode = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'
        self.test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
        self.temp_dir = Path(os.getenv('TEMP_DIR', 'tmp/'))
        
        # Backup settings
        self.enable_backup = os.getenv('ENABLE_BACKUP', 'true').lower() == 'true'
        self.backup_dir = Path(os.getenv('BACKUP_DIR', 'backups/'))
        self.backup_retention = int(os.getenv('BACKUP_RETENTION', '5'))
        
        # Performance settings
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '3'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_base_delay = int(os.getenv('RETRY_BASE_DELAY', '2'))
        
        # Setup logging
        self._setup_logging()
        
        # Create directories
        self._create_directories()
        
    def _parse_years(self, years_str: str) -> List[int]:
        """Parse comma-separated years string into list of integers"""
        try:
            return [int(year.strip()) for year in years_str.split(',')]
        except (ValueError, AttributeError):
            return []
            
    def _parse_list(self, list_str: str) -> List[str]:
        """Parse comma-separated string into list of strings"""
        try:
            return [item.strip() for item in list_str.split(',')]
        except AttributeError:
            return []
            
    def _parse_int_list(self, list_str: str) -> List[int]:
        """Parse comma-separated string into list of integers"""
        try:
            return [int(item.strip()) for item in list_str.split(',')]
        except (ValueError, AttributeError):
            return []
            
    def _setup_logging(self):
        """Configure logging based on environment settings"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_output_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.final_data_dir,
            self.temp_dir,
            self.backup_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration and return validation results
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'required_keys': [],
            'optional_keys': []
        }
        
        # Check required API keys
        if not self.census_api_key:
            validation['errors'].append(
                "CENSUS_API_KEY is required. Get one at: https://api.census.gov/data/key_signup.html"
            )
            validation['valid'] = False
            validation['required_keys'].append('CENSUS_API_KEY')
        else:
            validation['required_keys'].append('CENSUS_API_KEY ✓')
            
        # Check data years
        if not self.collection_years:
            validation['warnings'].append("No collection years specified, using defaults")
            
        if not self.naep_years:
            validation['warnings'].append("No NAEP years specified, using defaults")
            
        # Check directories are writable
        try:
            test_file = self.raw_data_dir / '.test_write'
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            validation['errors'].append(f"Cannot write to data directory: {e}")
            validation['valid'] = False
            
        # Check reasonable rate limits
        if self.naep_rate_limit_delay < 1.0:
            validation['warnings'].append("NAEP rate limit very low, may cause API blocking")
            
        # Check performance settings
        if self.max_concurrent_requests > 10:
            validation['warnings'].append("High concurrent requests may overwhelm APIs")
            
        # Optional configuration status
        validation['optional_keys'] = [
            f"LOG_LEVEL: {self.log_level}",
            f"DEVELOPMENT_MODE: {self.development_mode}",
            f"ENABLE_VALIDATION: {self.enable_validation}",
            f"ENABLE_BACKUP: {self.enable_backup}",
        ]
        
        return validation
        
    def get_collector_config(self, collector_name: str) -> Dict[str, Any]:
        """
        Get configuration specific to a data collector
        
        Args:
            collector_name: Name of collector ('naep', 'census', 'edfacts', 'ocr')
            
        Returns:
            Configuration dictionary for the collector
        """
        
        base_config = {
            'raw_data_dir': self.raw_data_dir,
            'enable_validation': self.enable_validation,
            'development_mode': self.development_mode,
            'test_mode': self.test_mode,
            'request_timeout': self.request_timeout,
            'max_retries': self.max_retries
        }
        
        if collector_name.lower() == 'naep':
            return {
                **base_config,
                'rate_limit_delay': self.naep_rate_limit_delay,
                'years': self.naep_years,
                'subjects': self.naep_subjects,
                'grades': self.naep_grades
            }
            
        elif collector_name.lower() == 'census':
            return {
                **base_config,
                'api_key': self.census_api_key,
                'rate_limit_delay': 1.0,  # Conservative for Census API
                'years': self.collection_years
            }
            
        elif collector_name.lower() == 'edfacts':
            return {
                **base_config,
                'rate_limit_delay': self.edfacts_rate_limit_delay,
                'years': self.collection_years
            }
            
        elif collector_name.lower() == 'ocr':
            return {
                **base_config,
                'rate_limit_delay': self.ocr_rate_limit_delay,
                'years': self.ocr_years
            }
            
        else:
            return base_config
            
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
Configuration Summary:
- Census API Key: {'✓ Set' if self.census_api_key else '✗ Missing'}
- Data Directory: {self.data_output_dir}
- Collection Years: {len(self.collection_years)} years
- NAEP Years: {len(self.naep_years)} years  
- Log Level: {self.log_level}
- Development Mode: {self.development_mode}
- Validation Enabled: {self.enable_validation}
        """.strip()


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def validate_environment() -> bool:
    """
    Validate environment configuration and print results
    
    Returns:
        True if configuration is valid, False otherwise
    """
    validation = config.validate_configuration()
    
    logger = logging.getLogger(__name__)
    
    # Print validation results
    print("\n" + "="*60)
    print("ENVIRONMENT CONFIGURATION VALIDATION")
    print("="*60)
    
    if validation['required_keys']:
        print("\nRequired Keys:")
        for key in validation['required_keys']:
            print(f"  {key}")
            
    if validation['optional_keys']:
        print(f"\nOptional Configuration:")
        for key in validation['optional_keys']:
            print(f"  {key}")
            
    if validation['warnings']:
        print(f"\nWarnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
            logger.warning(warning)
            
    if validation['errors']:
        print(f"\nErrors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
            logger.error(error)
    else:
        print(f"\n✅ Configuration validation passed!")
        
    print("="*60 + "\n")
    
    return validation['valid']


if __name__ == "__main__":
    # Validate configuration when run directly
    print(config)
    is_valid = validate_environment()
    exit(0 if is_valid else 1)