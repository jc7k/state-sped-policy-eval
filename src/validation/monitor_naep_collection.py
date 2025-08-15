#!/usr/bin/env python
"""
NAEP Collection Monitoring Script
Monitors the background NAEP collection and triggers validation when complete
"""

import logging
import subprocess
import time

from src.config import get_config
from src.validation.naep_data_validator import NAEPDataValidator


def check_collection_status(bash_id: str = "bash_2") -> tuple[bool, str]:
    """
    Check if NAEP collection is still running

    Returns:
        Tuple of (is_running, status_info)
    """
    try:
        subprocess.run(
            [
                "python",
                "-c",
                '''
import subprocess
result = subprocess.run(["python", "-c", """
import sys
import json
import requests

# This would be the actual BashOutput tool call logic
# For now, just check if the process exists
try:
    import psutil
    found_process = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and 'naep_collector' in ' '.join(proc.info['cmdline']):
                found_process = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print(json.dumps({"running": found_process, "info": "Process check complete"}))
except ImportError:
    # Fallback: assume still running if we can't check
    print(json.dumps({"running": True, "info": "Cannot check process status - assuming running"}))
"""], capture_output=True, text=True)
print(result.stdout)
        ''',
            ],
            capture_output=True,
            text=True,
        )

        # For now, let's check if the expected output file exists and has expected size
        config = get_config()
        output_file = config.raw_data_dir / "naep_state_swd_data.csv"

        if output_file.exists():
            # Check file size - should be around 1,200 records
            import pandas as pd

            try:
                df = pd.read_csv(output_file)
                record_count = len(df)

                # Expected: 1,200 records (600 requests × 2 records each)
                if record_count >= 1200:
                    return False, f"Collection complete - {record_count} records found"
                else:
                    return (
                        True,
                        f"Collection in progress - {record_count} records so far",
                    )
            except Exception as e:
                return (
                    True,
                    f"Collection in progress - file exists but still being written: {e}",
                )
        else:
            return True, "Collection in progress - output file not yet created"

    except Exception as e:
        return True, f"Cannot determine status: {e}"


def wait_for_collection_completion(check_interval: int = 30, max_wait_hours: int = 2):
    """
    Wait for NAEP collection to complete, then run validation

    Args:
        check_interval: Seconds between status checks
        max_wait_hours: Maximum hours to wait before timing out
    """
    logger = logging.getLogger(__name__)
    config = get_config()

    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600

    logger.info("Starting NAEP collection monitoring...")

    while True:
        # Check if we've exceeded max wait time
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            logger.error(f"Timeout: Collection did not complete within {max_wait_hours} hours")
            return False

        # Check collection status
        is_running, status_info = check_collection_status()

        logger.info(f"Status: {status_info}")

        if not is_running:
            logger.info("🎉 NAEP collection completed!")

            # Run validation
            output_file = config.raw_data_dir / "naep_state_swd_data.csv"
            if output_file.exists():
                logger.info("Starting data validation...")

                validator = NAEPDataValidator()
                validation_results = validator.validate_dataset(output_file)

                # Generate and save report
                report_file = config.raw_data_dir.parent / "reports" / "naep_validation_report.txt"
                report_file.parent.mkdir(parents=True, exist_ok=True)

                report_text = validator.generate_report(validation_results, report_file)

                logger.info(f"✅ Validation complete - report saved to {report_file}")
                print("\n" + report_text)

                return validation_results["summary"]["valid"]
            else:
                logger.error("Collection marked as complete but output file not found")
                return False

        # Wait before next check
        logger.info(f"Waiting {check_interval} seconds before next check...")
        time.sleep(check_interval)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Monitor NAEP data collection")
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Seconds between status checks (default: 30)",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=int,
        default=2,
        help="Maximum hours to wait (default: 2)",
    )

    args = parser.parse_args()

    success = wait_for_collection_completion(
        check_interval=args.check_interval, max_wait_hours=args.max_wait_hours
    )

    exit(0 if success else 1)
