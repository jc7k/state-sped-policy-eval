#!/usr/bin/env python
"""
Data Integration Module - Refactored

Orchestrates the complete data integration pipeline using modular components.
Combines NAEP, Census, EdFacts, and OCR data into master analysis dataset.

Refactored for improved maintainability:
- Modular data loaders for each source
- Centralized configuration management
- Consistent error handling patterns
- Separation of concerns
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import IntegrationConfig, DEFAULT_CONFIG
from .data_loaders import DataLoaderFactory
from .panel_creator import PanelDataCreator
from .utils import ErrorHandlingMixin

warnings.filterwarnings("ignore")


class DataIntegrator(ErrorHandlingMixin):
    """
    Orchestrates the complete data integration pipeline.
    
    Refactored to use modular components for improved maintainability:
    - Uses specialized data loaders for each data source
    - Leverages centralized configuration management
    - Delegates panel creation to specialized class
    - Provides consistent error handling and logging
    """

    def __init__(
        self, 
        raw_data_dir: Optional[Path] = None, 
        output_dir: Optional[Path] = None,
        config: Optional[IntegrationConfig] = None
    ):
        """
        Initialize data integrator with modular components.

        Args:
            raw_data_dir: Directory containing raw data files (optional)
            output_dir: Directory to save processed datasets (optional)
            config: Integration configuration (uses default if not provided)
        """
        super().__init__()
        
        # Use provided config or default
        if config is None:
            self.config = IntegrationConfig(raw_data_dir, output_dir)
        else:
            self.config = config
        
        # Ensure directories exist
        self.config.ensure_directories()
        
        # Initialize modular components
        self.data_loaders = DataLoaderFactory.create_all_loaders(self.config.raw_data_dir)
        self.panel_creator = PanelDataCreator(self.config.processing_params.DEFAULT_YEAR_RANGE)
        
        self.log_info(f"Initialized DataIntegrator with:")
        self.log_info(f"  Raw data dir: {self.config.raw_data_dir}")
        self.log_info(f"  Output dir: {self.config.output_dir}")
        self.log_info(f"  Available loaders: {list(self.data_loaders.keys())}")

    def load_datasets(self, sources: Optional[list] = None) -> Dict[str, pd.DataFrame]:
        """
        Load and clean data from all or specified sources.
        
        Args:
            sources: List of data sources to load (default: all available)
            
        Returns:
            Dictionary mapping source names to cleaned DataFrames
        """
        if sources is None:
            sources = list(self.data_loaders.keys())
        
        datasets = {}
        
        for source in sources:
            if source not in self.data_loaders:
                self.log_warning(f"Unknown data source: {source}")
                continue
            
            self.log_info(f"Loading {source} data...")
            
            try:
                loader = self.data_loaders[source]
                df = loader.load_and_clean()
                
                # Check for loader errors
                if loader.has_errors():
                    error_summary = loader.get_error_summary()
                    for error in error_summary["errors"]:
                        self.log_error(f"Loader error for {source}: {error}")
                    for warning in error_summary["warnings"]:
                        self.log_warning(f"Loader warning for {source}: {warning}")
                
                datasets[source] = df
                
                if not df.empty:
                    summary = loader.get_data_summary(df)
                    self.log_info(
                        f"Successfully loaded {source}: {summary['records']} records, "
                        f"{summary['states']} states, years {summary['years']}"
                    )
                else:
                    self.log_warning(f"No data loaded for {source}")
                    
            except Exception as e:
                self.log_error(f"Failed to load {source} data", e)
                datasets[source] = pd.DataFrame()
        
        return datasets

    def save_individual_datasets(
        self, 
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Path]:
        """
        Save individual cleaned datasets to files.
        
        Args:
            datasets: Dictionary of cleaned datasets
            
        Returns:
            Dictionary mapping dataset names to output file paths
        """
        output_files = {}
        
        for source, df in datasets.items():
            if df.empty:
                self.log_warning(f"Skipping empty dataset: {source}")
                continue
            
            filename = self.config.output_settings.OUTPUT_FILES.get(
                f"{source}_cleaned", f"{source}_cleaned.csv"
            )
            output_path = self.config.get_output_path(filename)
            
            try:
                df.to_csv(output_path, **self.config.output_settings.CSV_SETTINGS)
                output_files[f"{source}_cleaned"] = output_path
                self.log_info(f"Saved {source} cleaned data: {output_path}")
            except Exception as e:
                self.log_error(f"Failed to save {source} data to {output_path}", e)
        
        return output_files

    def create_master_dataset(
        self, 
        datasets: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create master state-year panel dataset from individual sources.
        
        Args:
            datasets: Dictionary of cleaned datasets
            
        Returns:
            Master panel dataset with derived variables
        """
        self.log_info("Creating master state-year panel dataset...")
        
        try:
            master_df = self.panel_creator.create_panel_dataset(datasets)
            
            # Check for panel creator errors/warnings
            if self.panel_creator.has_errors():
                error_summary = self.panel_creator.get_error_summary()
                for error in error_summary["errors"]:
                    self.log_error(f"Panel creation error: {error}")
                for warning in error_summary["warnings"]:
                    self.log_warning(f"Panel creation warning: {warning}")
            
            # Get panel summary
            panel_summary = self.panel_creator.get_panel_summary(master_df)
            self.log_info(f"Master dataset summary: {panel_summary}")
            
            return master_df
            
        except Exception as e:
            self.log_error("Failed to create master dataset", e)
            return pd.DataFrame()

    def save_master_dataset(
        self, 
        master_df: pd.DataFrame
    ) -> Path:
        """
        Save master dataset to file.
        
        Args:
            master_df: Master panel dataset
            
        Returns:
            Path to saved file
        """
        filename = self.config.output_settings.OUTPUT_FILES["master_dataset"]
        output_path = self.config.get_output_path(filename)
        
        try:
            master_df.to_csv(output_path, **self.config.output_settings.CSV_SETTINGS)
            self.log_info(f"Saved master dataset: {output_path}")
            return output_path
        except Exception as e:
            self.log_error(f"Failed to save master dataset to {output_path}", e)
            raise

    def get_integration_summary(
        self, 
        datasets: Dict[str, pd.DataFrame],
        master_df: pd.DataFrame
    ) -> Dict:
        """
        Get comprehensive summary of integration results.
        
        Args:
            datasets: Individual cleaned datasets
            master_df: Master panel dataset
            
        Returns:
            Integration summary dictionary
        """
        summary = {
            "individual_datasets": {},
            "master_dataset": {},
            "errors_warnings": self.get_error_summary()
        }
        
        # Summarize individual datasets
        for source, df in datasets.items():
            if source in self.data_loaders:
                summary["individual_datasets"][source] = self.data_loaders[source].get_data_summary(df)
        
        # Summarize master dataset
        summary["master_dataset"] = self.panel_creator.get_panel_summary(master_df)
        
        return summary


    def run_integration(
        self, 
        sources: Optional[list] = None,
        save_individual: bool = True
    ) -> pd.DataFrame:
        """
        Run complete data integration pipeline.
        
        Args:
            sources: List of data sources to include (default: all)
            save_individual: Whether to save individual cleaned datasets
            
        Returns:
            Master panel dataset
        """
        self.log_info("Starting refactored data integration pipeline...")
        
        try:
            # Clear any previous errors/warnings
            self.clear_logs()
            
            # Load all datasets using modular loaders
            datasets = self.load_datasets(sources)
            
            # Create master panel using specialized panel creator
            master_df = self.create_master_dataset(datasets)
            
            # Save individual datasets if requested
            output_files = {}
            if save_individual:
                output_files = self.save_individual_datasets(datasets)
            
            # Save master dataset
            if not master_df.empty:
                master_path = self.save_master_dataset(master_df)
                output_files["master"] = master_path
            
            # Get and log integration summary
            summary = self.get_integration_summary(datasets, master_df)
            self.log_info(f"Integration summary: {summary}")
            
            self.log_info(
                f"Integration complete. Output files: {list(output_files.keys())}"
            )
            
            return master_df
            
        except Exception as e:
            self.log_error("Integration pipeline failed", e)
            raise


def main():
    """Main function for running refactored data integration pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate multiple data sources using refactored modular pipeline"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default="data/raw",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/processed",
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["naep", "census", "edfacts", "ocr"],
        help="Specific data sources to load (default: all)",
    )
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("MIN_YEAR", "MAX_YEAR"),
        help="Year range for panel (e.g., --year-range 2015 2022)",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip saving individual cleaned datasets",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create configuration
    year_range = tuple(args.year_range) if args.year_range else None
    config = IntegrationConfig(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        year_range=year_range
    )

    # Run integration with refactored pipeline
    integrator = DataIntegrator(config=config)
    master_df = integrator.run_integration(
        sources=args.sources,
        save_individual=not args.skip_individual
    )

    # Display summary statistics
    print("\n=== Refactored Integration Summary ===")
    if master_df.empty:
        print("ERROR: No data integrated successfully")
        return
    
    print(f"Master dataset shape: {master_df.shape}")
    print(f"States: {master_df['state'].nunique()}")
    print(f"Years: {master_df['year'].min()}-{master_df['year'].max()}")
    
    # Check data coverage for key variables
    print("\nData coverage by key variables:")
    key_vars = [
        "naep_avg_scale_score",
        "total_expenditure", 
        "edfacts_child_count",
        "ocr_total_enrollment",
    ]
    
    for var in key_vars:
        if var in master_df.columns:
            non_missing = master_df[var].notna().sum()
            coverage_pct = (non_missing / len(master_df)) * 100
            print(f"  {var}: {non_missing} ({coverage_pct:.1f}%)")
        else:
            print(f"  {var}: NOT FOUND")
    
    # Show any errors or warnings
    error_summary = integrator.get_error_summary()
    if error_summary["errors"]:
        print(f"\nERRORS ({len(error_summary['errors'])}):")
        for error in error_summary["errors"]:
            print(f"  - {error}")
    
    if error_summary["warnings"]:
        print(f"\nWARNINGS ({len(error_summary['warnings'])}):")
        for warning in error_summary["warnings"]:
            print(f"  - {warning}")
    
    print("\nIntegration completed successfully!")


if __name__ == "__main__":
    main()
