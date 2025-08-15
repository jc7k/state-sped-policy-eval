"""
Data Collection Module
Automated collection from NAEP, EdFacts, Census, and OCR sources
"""

# Import common utilities
# Import base classes
from .base_collector import APIBasedCollector, BaseDataCollector, FileBasedCollector

# Import specific collectors
from .census_collector import CensusEducationFinance
from .census_data_parser import CensusDataParser
from .census_file_downloader import CensusFileDownloader
from .common import APIClient, DataValidator, FileUtils, SafeTypeConverter, StateUtils
from .edfacts_collector import EdFactsCollector
from .naep_collector import NAEPDataCollector
from .ocr_collector import OCRCollector

__all__ = [
    # Common utilities
    "APIClient",
    "DataValidator",
    "FileUtils",
    "SafeTypeConverter",
    "StateUtils",
    # Base classes
    "APIBasedCollector",
    "BaseDataCollector",
    "FileBasedCollector",
    # Specific collectors
    "CensusEducationFinance",
    "CensusDataParser",
    "CensusFileDownloader",
    "EdFactsCollector",
    "NAEPDataCollector",
    "OCRCollector",
]
