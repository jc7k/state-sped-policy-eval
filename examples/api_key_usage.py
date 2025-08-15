#!/usr/bin/env python
"""
Example: Using API Keys with Environment Variables
Demonstrates proper API key configuration for data collectors
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.collection.naep_collector import NAEPDataCollector  
from src.collection.census_collector import CensusEducationFinance
from src.config import get_config, validate_environment

def main():
    """
    Example of using API keys from environment variables
    """
    
    print("="*60)
    print("API KEY USAGE EXAMPLE")
    print("="*60)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Validate environment configuration
    print("\n1. Validating Environment Configuration...")
    is_valid = validate_environment()
    
    if not is_valid:
        print("\n❌ Environment validation failed!")
        print("Please check your .env file and fix the errors above.")
        return
        
    # Get configuration
    config = get_config()
    print(f"\n2. Configuration loaded successfully:")
    print(config)
    
    # Example 1: NAEP Collector (no API key required)
    print("\n3. NAEP Data Collector (Public API)...")
    try:
        naep_collector = NAEPDataCollector()
        print("✅ NAEP collector initialized successfully")
        print(f"   Rate limit: {naep_collector.rate_limit_delay} seconds")
        print(f"   Base URL: {naep_collector.base_url}")
    except Exception as e:
        print(f"❌ NAEP collector failed: {e}")
        
    # Example 2: Census Collector (API key required)
    print("\n4. Census Finance Collector (API Key Required)...")
    try:
        census_collector = CensusEducationFinance()
        print("✅ Census collector initialized successfully")
        print(f"   API Key: {'*' * 20}...{census_collector.api_key[-4:] if census_collector.api_key else 'None'}")
        print(f"   Base URL: {census_collector.base_url}")
    except ValueError as e:
        print(f"❌ Census collector failed: {e}")
        print("\nTo fix this:")
        print("1. Get a free API key from: https://api.census.gov/data/key_signup.html")
        print("2. Add it to your .env file: CENSUS_API_KEY=your_key_here")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return
        
    # Example 3: Configuration-based collector setup
    print("\n5. Using Configuration for Collectors...")
    
    naep_config = config.get_collector_config('naep')
    census_config = config.get_collector_config('census')
    
    print(f"NAEP Config:")
    print(f"  Years: {naep_config['years']}")
    print(f"  Subjects: {naep_config['subjects']}")
    print(f"  Grades: {naep_config['grades']}")
    
    print(f"Census Config:")
    print(f"  Years: {census_config['years']}")
    print(f"  API Key Set: {'✓' if census_config.get('api_key') else '✗'}")
    
    # Example 4: Directory structure
    print(f"\n6. Data Directories:")
    print(f"  Raw Data: {config.raw_data_dir}")
    print(f"  Processed: {config.processed_data_dir}")
    print(f"  Final: {config.final_data_dir}")
    
    # Check if directories exist and are writable
    for name, directory in [
        ("Raw", config.raw_data_dir),
        ("Processed", config.processed_data_dir), 
        ("Final", config.final_data_dir)
    ]:
        if directory.exists():
            try:
                test_file = directory / '.test_write'
                test_file.touch()
                test_file.unlink()
                status = "✓ Writable"
            except:
                status = "✗ Not writable"
        else:
            status = "✗ Does not exist"
            
        print(f"    {name}: {status}")
        
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    # Summary of what's working
    collectors_working = []
    collectors_failed = []
    
    try:
        NAEPDataCollector()
        collectors_working.append("✅ NAEP (public API)")
    except:
        collectors_failed.append("❌ NAEP")
        
    try:
        CensusEducationFinance()
        collectors_working.append("✅ Census (API key required)")
    except:
        collectors_failed.append("❌ Census (needs API key)")
        
    if collectors_working:
        print(f"\nWorking Collectors:")
        for collector in collectors_working:
            print(f"  {collector}")
            
    if collectors_failed:
        print(f"\nFailed Collectors:")
        for collector in collectors_failed:
            print(f"  {collector}")
            
    print(f"\nNext Steps:")
    if not collectors_failed:
        print("🎉 All collectors ready! You can now run data collection.")
        print("\nTry running:")
        print("  python -m src.collection.naep_collector")
        print("  python -m src.collection.census_collector")
    else:
        print("📋 Fix the failed collectors above, then you're ready to go!")
        
    print("\n" + "="*60)


if __name__ == "__main__":
    main()