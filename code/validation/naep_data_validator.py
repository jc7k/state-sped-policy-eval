#!/usr/bin/env python
"""
NAEP Data Validation Module
Validates collected NAEP data for completeness, quality, and consistency
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd


class NAEPDataValidator:
    """
    Comprehensive validation for NAEP achievement data
    Checks completeness, quality, and consistency of collected data
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Expected data structure parameters
        self.expected_states = [
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
        ]
        self.expected_years = [2017, 2019, 2022]
        self.expected_grades = [4, 8]
        self.expected_subjects = ["mathematics", "reading"]
        self.expected_disability_statuses = ["SWD", "non-SWD"]

        # Expected total records: 50 states × 3 years × 2 grades × 2 subjects × 2 statuses = 1,200
        self.expected_total_records = (
            len(self.expected_states)
            * len(self.expected_years)
            * len(self.expected_grades)
            * len(self.expected_subjects)
            * len(self.expected_disability_statuses)
        )

    def validate_dataset(self, data_file: Path) -> dict[str, Any]:
        """
        Comprehensive validation of NAEP dataset

        Args:
            data_file: Path to NAEP CSV data file

        Returns:
            Dictionary with validation results and metrics
        """
        validation_results = {
            "file_info": {},
            "completeness": {},
            "quality": {},
            "consistency": {},
            "summary": {"valid": True, "issues": [], "warnings": []},
        }

        try:
            # Load and basic file info
            self.logger.info(f"Loading NAEP data from {data_file}")
            df = pd.read_csv(data_file)

            validation_results["file_info"] = {
                "file_path": str(data_file),
                "file_size_mb": data_file.stat().st_size / (1024 * 1024),
                "total_records": len(df),
                "columns": list(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            }

            # Completeness validation
            validation_results["completeness"] = self._validate_completeness(df)

            # Quality validation
            validation_results["quality"] = self._validate_quality(df)

            # Consistency validation
            validation_results["consistency"] = self._validate_consistency(df)

            # Generate summary
            validation_results["summary"] = self._generate_summary(validation_results)

            self.logger.info(f"Validation completed for {len(df)} records")

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results["summary"]["valid"] = False
            validation_results["summary"]["issues"].append(
                f"Validation error: {str(e)}"
            )

        return validation_results

    def _validate_completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data completeness and coverage"""
        completeness = {
            "total_records": len(df),
            "expected_records": self.expected_total_records,
            "coverage_percentage": (len(df) / self.expected_total_records) * 100,
            "missing_combinations": [],
            "coverage_by_dimension": {},
        }

        # Check coverage by each dimension
        completeness["coverage_by_dimension"] = {
            "states": {
                "found": df["state"].nunique(),
                "expected": len(self.expected_states),
                "missing": list(set(self.expected_states) - set(df["state"].unique())),
                "unexpected": list(
                    set(df["state"].unique()) - set(self.expected_states)
                ),
            },
            "years": {
                "found": df["year"].nunique(),
                "expected": len(self.expected_years),
                "missing": list(set(self.expected_years) - set(df["year"].unique())),
                "unexpected": list(set(df["year"].unique()) - set(self.expected_years)),
            },
            "grades": {
                "found": df["grade"].nunique(),
                "expected": len(self.expected_grades),
                "missing": list(set(self.expected_grades) - set(df["grade"].unique())),
                "unexpected": list(
                    set(df["grade"].unique()) - set(self.expected_grades)
                ),
            },
            "subjects": {
                "found": df["subject"].nunique(),
                "expected": len(self.expected_subjects),
                "missing": list(
                    set(self.expected_subjects) - set(df["subject"].unique())
                ),
                "unexpected": list(
                    set(df["subject"].unique()) - set(self.expected_subjects)
                ),
            },
            "disability_statuses": {
                "found": df["disability_status"].nunique(),
                "expected": len(self.expected_disability_statuses),
                "missing": list(
                    set(self.expected_disability_statuses)
                    - set(df["disability_status"].unique())
                ),
                "unexpected": list(
                    set(df["disability_status"].unique())
                    - set(self.expected_disability_statuses)
                ),
            },
        }

        # Find missing combinations
        expected_combinations = []
        for state in self.expected_states:
            for year in self.expected_years:
                for grade in self.expected_grades:
                    for subject in self.expected_subjects:
                        for status in self.expected_disability_statuses:
                            expected_combinations.append(
                                (state, year, grade, subject, status)
                            )

        existing_combinations = set(
            zip(
                df["state"],
                df["year"],
                df["grade"],
                df["subject"],
                df["disability_status"],
                strict=False,
            )
        )

        missing_combinations = []
        for combo in expected_combinations:
            if combo not in existing_combinations:
                missing_combinations.append(
                    {
                        "state": combo[0],
                        "year": combo[1],
                        "grade": combo[2],
                        "subject": combo[3],
                        "disability_status": combo[4],
                    }
                )

        completeness["missing_combinations"] = missing_combinations
        completeness["missing_combinations_count"] = len(missing_combinations)

        return completeness

    def _validate_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data quality and reasonableness"""
        quality = {
            "missing_values": {},
            "score_statistics": {},
            "achievement_gaps": {},
            "outliers": {},
            "data_types": {},
        }

        # Missing values check
        quality["missing_values"] = {
            col: {
                "count": df[col].isnull().sum(),
                "percentage": (df[col].isnull().sum() / len(df)) * 100,
            }
            for col in df.columns
        }

        # Score statistics (should be reasonable NAEP scale scores)
        if "mean_score" in df.columns:
            scores = df["mean_score"].dropna()
            quality["score_statistics"] = {
                "count": len(scores),
                "mean": float(scores.mean()),
                "median": float(scores.median()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "range": float(scores.max() - scores.min()),
            }

            # Check for unreasonable scores (NAEP scores typically 0-500 range)
            unreasonable_scores = scores[(scores < 0) | (scores > 500)]
            quality["outliers"]["unreasonable_scores"] = {
                "count": len(unreasonable_scores),
                "values": unreasonable_scores.tolist(),
            }

        # Achievement gap analysis
        if "mean_score" in df.columns and "disability_status" in df.columns:
            quality["achievement_gaps"] = self._analyze_achievement_gaps(df)

        # Data type validation
        quality["data_types"] = {col: str(df[col].dtype) for col in df.columns}

        return quality

    def _validate_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data consistency and logical relationships"""
        consistency = {
            "duplicate_records": {},
            "logical_consistency": {},
            "value_consistency": {},
        }

        # Duplicate records check
        duplicate_cols = ["state", "year", "grade", "subject", "disability_status"]
        if all(col in df.columns for col in duplicate_cols):
            duplicates = df.duplicated(subset=duplicate_cols, keep=False)
            consistency["duplicate_records"] = {
                "count": duplicates.sum(),
                "percentage": (duplicates.sum() / len(df)) * 100,
            }

            if duplicates.sum() > 0:
                consistency["duplicate_records"]["examples"] = (
                    df[duplicates][duplicate_cols].head().to_dict("records")
                )

        # Value consistency checks
        consistency["value_consistency"] = {
            "var_value_mapping": self._check_var_value_consistency(df),
            "state_name_consistency": self._check_state_name_consistency(df),
        }

        return consistency

    def _analyze_achievement_gaps(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze achievement gaps between SWD and non-SWD students"""
        gaps = {
            "overall": {},
            "by_year": {},
            "by_grade": {},
            "by_subject": {},
            "by_state": {},
        }

        # Calculate gaps by various dimensions
        for group_by in [None, "year", "grade", "subject", "state"]:
            if group_by is None:
                # Overall gap
                swd_scores = df[df["disability_status"] == "SWD"]["mean_score"]
                non_swd_scores = df[df["disability_status"] == "non-SWD"]["mean_score"]

                if len(swd_scores) > 0 and len(non_swd_scores) > 0:
                    gap = non_swd_scores.mean() - swd_scores.mean()
                    gaps["overall"] = {
                        "mean_gap": float(gap),
                        "swd_mean": float(swd_scores.mean()),
                        "non_swd_mean": float(non_swd_scores.mean()),
                        "swd_count": len(swd_scores),
                        "non_swd_count": len(non_swd_scores),
                    }
            else:
                # Gaps by dimension
                group_gaps = {}
                for group_value in df[group_by].unique():
                    group_data = df[df[group_by] == group_value]
                    swd_scores = group_data[group_data["disability_status"] == "SWD"][
                        "mean_score"
                    ]
                    non_swd_scores = group_data[
                        group_data["disability_status"] == "non-SWD"
                    ]["mean_score"]

                    if len(swd_scores) > 0 and len(non_swd_scores) > 0:
                        gap = non_swd_scores.mean() - swd_scores.mean()
                        group_gaps[group_value] = {
                            "mean_gap": float(gap),
                            "swd_mean": float(swd_scores.mean()),
                            "non_swd_mean": float(non_swd_scores.mean()),
                        }

                gaps[f"by_{group_by}"] = group_gaps

        return gaps

    def _check_var_value_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check consistency of var_value mapping to disability_status"""
        if "var_value" not in df.columns or "disability_status" not in df.columns:
            return {"error": "Required columns not found"}

        # Expected mapping: var_value "1" = SWD, var_value "2" = non-SWD
        mapping_check = (
            df.groupby(["var_value", "disability_status"])
            .size()
            .reset_index(name="count")
        )

        expected_mappings = {"1": "SWD", "2": "non-SWD"}

        consistency_issues = []
        for _, row in mapping_check.iterrows():
            var_val = str(row["var_value"])
            status = row["disability_status"]

            if var_val in expected_mappings and expected_mappings[var_val] != status:
                consistency_issues.append(
                    {
                        "var_value": var_val,
                        "found_status": status,
                        "expected_status": expected_mappings[var_val],
                        "count": row["count"],
                    }
                )

        return {
            "mapping_table": mapping_check.to_dict("records"),
            "consistency_issues": consistency_issues,
            "issues_count": len(consistency_issues),
        }

    def _check_state_name_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check consistency of state codes to state names"""
        if "state" not in df.columns or "state_name" not in df.columns:
            return {"error": "Required columns not found"}

        # Check for one-to-one mapping between state codes and names
        state_mapping = df.groupby("state")["state_name"].unique()
        inconsistent_states = {}

        for state_code, names in state_mapping.items():
            if len(names) > 1:
                inconsistent_states[state_code] = names.tolist()

        return {
            "total_states": len(state_mapping),
            "inconsistent_mappings": inconsistent_states,
            "inconsistent_count": len(inconsistent_states),
        }

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall validation summary"""
        summary = {"valid": True, "issues": [], "warnings": []}

        # Check completeness issues
        completeness = results.get("completeness", {})
        if completeness.get("missing_combinations_count", 0) > 0:
            summary["issues"].append(
                f"Missing {completeness['missing_combinations_count']} expected data combinations"
            )
            summary["valid"] = False

        if completeness.get("coverage_percentage", 0) < 95:
            summary["warnings"].append(
                f"Data coverage is only {completeness.get('coverage_percentage', 0):.1f}%"
            )

        # Check quality issues
        quality = results.get("quality", {})
        if (
            quality.get("outliers", {}).get("unreasonable_scores", {}).get("count", 0)
            > 0
        ):
            summary["issues"].append(
                f"Found {quality['outliers']['unreasonable_scores']['count']} unreasonable score values"
            )

        # Check consistency issues
        consistency = results.get("consistency", {})
        if consistency.get("duplicate_records", {}).get("count", 0) > 0:
            summary["issues"].append(
                f"Found {consistency['duplicate_records']['count']} duplicate records"
            )
            summary["valid"] = False

        # Check achievement gaps are reasonable (should be 15-40 points typically)
        overall_gap = (
            quality.get("achievement_gaps", {}).get("overall", {}).get("mean_gap")
        )
        if overall_gap is not None and (overall_gap < 10 or overall_gap > 50):
            summary["warnings"].append(
                f"Overall achievement gap of {overall_gap:.1f} points seems unusual (expected 15-40)"
            )

        return summary

    def generate_report(
        self, validation_results: dict[str, Any], output_file: Path = None
    ) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("=" * 80)
        report.append("NAEP DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # File info
        file_info = validation_results.get("file_info", {})
        report.append("📁 FILE INFORMATION")
        report.append(f"File: {file_info.get('file_path', 'N/A')}")
        report.append(f"Size: {file_info.get('file_size_mb', 0):.2f} MB")
        report.append(f"Records: {file_info.get('total_records', 0):,}")
        report.append(f"Columns: {len(file_info.get('columns', []))}")
        report.append("")

        # Summary
        summary = validation_results.get("summary", {})
        status = "✅ VALID" if summary.get("valid", False) else "❌ INVALID"
        report.append(f"📊 VALIDATION STATUS: {status}")
        report.append("")

        if summary.get("issues"):
            report.append("🚨 ISSUES FOUND:")
            for issue in summary["issues"]:
                report.append(f"  • {issue}")
            report.append("")

        if summary.get("warnings"):
            report.append("⚠️  WARNINGS:")
            for warning in summary["warnings"]:
                report.append(f"  • {warning}")
            report.append("")

        # Completeness
        completeness = validation_results.get("completeness", {})
        report.append("📈 DATA COMPLETENESS")
        report.append(
            f"Coverage: {completeness.get('coverage_percentage', 0):.1f}% ({completeness.get('total_records', 0):,}/{completeness.get('expected_records', 0):,} records)"
        )

        coverage_by_dim = completeness.get("coverage_by_dimension", {})
        for dim, info in coverage_by_dim.items():
            found = info.get("found", 0)
            expected = info.get("expected", 0)
            report.append(f"  {dim.title()}: {found}/{expected}")
            if info.get("missing"):
                report.append(f"    Missing: {info['missing']}")
        report.append("")

        # Quality
        quality = validation_results.get("quality", {})
        if "achievement_gaps" in quality and "overall" in quality["achievement_gaps"]:
            gap_info = quality["achievement_gaps"]["overall"]
            report.append("🎯 ACHIEVEMENT GAP ANALYSIS")
            report.append(f"Overall Gap: {gap_info.get('mean_gap', 0):.1f} points")
            report.append(f"SWD Mean Score: {gap_info.get('swd_mean', 0):.1f}")
            report.append(f"Non-SWD Mean Score: {gap_info.get('non_swd_mean', 0):.1f}")
            report.append("")

        # Score statistics
        if "score_statistics" in quality:
            stats = quality["score_statistics"]
            report.append("📊 SCORE STATISTICS")
            report.append(f"Mean: {stats.get('mean', 0):.1f}")
            report.append(
                f"Range: {stats.get('min', 0):.1f} - {stats.get('max', 0):.1f}"
            )
            report.append(f"Standard Deviation: {stats.get('std', 0):.1f}")
            report.append("")

        report_text = "\n".join(report)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report_text)

        return report_text


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate NAEP dataset")
    parser.add_argument("data_file", help="Path to NAEP CSV data file")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Validate data
    validator = NAEPDataValidator()
    results = validator.validate_dataset(Path(args.data_file))

    # Generate report
    report_file = Path(args.output) if args.output else None
    report = validator.generate_report(results, report_file)

    print(report)

    # Exit with error code if validation failed
    exit(0 if results["summary"]["valid"] else 1)


if __name__ == "__main__":
    main()
