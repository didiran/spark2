"""
Standalone data validation engine for the ML Training Pipeline.

Implements schema enforcement, null checks, outlier detection,
value-range constraints, and composite data quality scoring
using pandas DataFrames for local development and testing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RuleSeverity(Enum):
    """Severity levels for validation rules."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationRule:
    """A single data validation rule definition."""
    name: str
    description: str
    check_fn: Callable[[pd.DataFrame], Tuple[bool, Dict[str, Any]]]
    severity: RuleSeverity = RuleSeverity.WARNING
    tags: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a single validation rule execution."""
    rule_name: str
    passed: bool
    severity: str
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ValidationReport:
    """Aggregated report for a full validation run."""
    total_rules: int
    passed_rules: int
    failed_rules: int
    critical_failures: int
    warning_failures: int
    quality_score: float
    results: List[ValidationResult]
    is_valid: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a serializable dictionary."""
        return {
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "critical_failures": self.critical_failures,
            "warning_failures": self.warning_failures,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "results": [
                {
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class StandaloneDataValidator:
    """
    Pandas-based data quality validation engine.

    Applies configurable validation rules to pandas DataFrames,
    computes data quality scores, and produces structured
    validation reports.

    Attributes:
        rules: List of registered validation rules.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []
        logger.info("StandaloneDataValidator initialized")

    def add_rule(self, rule: ValidationRule) -> "StandaloneDataValidator":
        """Register a validation rule (fluent interface)."""
        self.rules.append(rule)
        logger.info(f"Rule added | name={rule.name} | severity={rule.severity.value}")
        return self

    def expect_column_not_null(
        self,
        column: str,
        max_null_fraction: float = 0.0,
        severity: RuleSeverity = RuleSeverity.CRITICAL,
    ) -> "StandaloneDataValidator":
        """
        Add rule: column null fraction must not exceed threshold.

        Args:
            column: Column name to check.
            max_null_fraction: Maximum allowed fraction of nulls.
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            total = len(df)
            if total == 0:
                return True, {"total": 0, "null_count": 0, "null_fraction": 0.0}
            null_count = int(df[column].isna().sum())
            null_fraction = null_count / total
            passed = null_fraction <= max_null_fraction
            return passed, {
                "column": column,
                "total": total,
                "null_count": null_count,
                "null_fraction": round(null_fraction, 6),
                "threshold": max_null_fraction,
            }

        rule = ValidationRule(
            name=f"not_null_{column}",
            description=f"Column '{column}' null fraction <= {max_null_fraction}",
            check_fn=_check,
            severity=severity,
            tags=["null_check", column],
        )
        return self.add_rule(rule)

    def expect_column_values_in_range(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_out_of_range_fraction: float = 0.0,
        severity: RuleSeverity = RuleSeverity.WARNING,
    ) -> "StandaloneDataValidator":
        """
        Add rule: column values must fall within a numeric range.

        Args:
            column: Numeric column to check.
            min_value: Minimum allowed value (inclusive).
            max_value: Maximum allowed value (inclusive).
            max_out_of_range_fraction: Maximum fraction outside range.
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            total = len(df)
            if total == 0:
                return True, {"total": 0, "out_of_range": 0}

            valid = df[column].dropna()
            condition = pd.Series(False, index=valid.index)
            if min_value is not None:
                condition = condition | (valid < min_value)
            if max_value is not None:
                condition = condition | (valid > max_value)

            oor_count = int(condition.sum())
            oor_fraction = oor_count / total
            passed = oor_fraction <= max_out_of_range_fraction

            return passed, {
                "column": column,
                "min_value": min_value,
                "max_value": max_value,
                "total": total,
                "out_of_range_count": oor_count,
                "out_of_range_fraction": round(oor_fraction, 6),
                "threshold": max_out_of_range_fraction,
            }

        rule = ValidationRule(
            name=f"range_{column}",
            description=(
                f"Column '{column}' values in [{min_value}, {max_value}] "
                f"(tolerance={max_out_of_range_fraction})"
            ),
            check_fn=_check,
            severity=severity,
            tags=["range_check", column],
        )
        return self.add_rule(rule)

    def expect_column_values_in_set(
        self,
        column: str,
        allowed_values: List[Any],
        severity: RuleSeverity = RuleSeverity.WARNING,
    ) -> "StandaloneDataValidator":
        """
        Add rule: column values must belong to a predefined set.

        Args:
            column: Column to check.
            allowed_values: List of valid values.
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            total = len(df)
            if total == 0:
                return True, {"total": 0}

            invalid_count = int((~df[column].isin(allowed_values)).sum())
            passed = invalid_count == 0

            return passed, {
                "column": column,
                "allowed_values": allowed_values,
                "total": total,
                "invalid_count": invalid_count,
            }

        rule = ValidationRule(
            name=f"in_set_{column}",
            description=f"Column '{column}' values in {allowed_values}",
            check_fn=_check,
            severity=severity,
            tags=["set_check", column],
        )
        return self.add_rule(rule)

    def expect_column_unique(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.WARNING,
    ) -> "StandaloneDataValidator":
        """
        Add rule: column values must be unique.

        Args:
            column: Column to check for uniqueness.
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            total = len(df)
            distinct_count = df[column].nunique()
            duplicate_count = total - distinct_count
            passed = duplicate_count == 0

            return passed, {
                "column": column,
                "total": total,
                "distinct_count": distinct_count,
                "duplicate_count": duplicate_count,
            }

        rule = ValidationRule(
            name=f"unique_{column}",
            description=f"Column '{column}' values are unique",
            check_fn=_check,
            severity=severity,
            tags=["uniqueness_check", column],
        )
        return self.add_rule(rule)

    def expect_schema_match(
        self,
        expected_columns: Dict[str, str],
        allow_extra_columns: bool = True,
        severity: RuleSeverity = RuleSeverity.CRITICAL,
    ) -> "StandaloneDataValidator":
        """
        Add rule: DataFrame schema must match expected columns.

        Args:
            expected_columns: Dict of column_name -> expected_dtype.
            allow_extra_columns: Whether extra columns are acceptable.
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            actual_cols = set(df.columns)
            expected_cols = set(expected_columns.keys())

            missing_columns = expected_cols - actual_cols
            extra_columns = actual_cols - expected_cols

            type_mismatches = {}
            for col, expected_type in expected_columns.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type and actual_type not in expected_type:
                        type_mismatches[col] = {
                            "expected": expected_type,
                            "actual": actual_type,
                        }

            passed = (
                len(missing_columns) == 0
                and len(type_mismatches) == 0
                and (allow_extra_columns or len(extra_columns) == 0)
            )

            return passed, {
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns),
                "type_mismatches": type_mismatches,
                "allow_extra_columns": allow_extra_columns,
            }

        rule = ValidationRule(
            name="schema_match",
            description="DataFrame schema matches expected columns",
            check_fn=_check,
            severity=severity,
            tags=["schema_check"],
        )
        return self.add_rule(rule)

    def detect_outliers_iqr(
        self,
        column: str,
        iqr_multiplier: float = 1.5,
        max_outlier_fraction: float = 0.05,
        severity: RuleSeverity = RuleSeverity.WARNING,
    ) -> "StandaloneDataValidator":
        """
        Add rule: detect outliers using the IQR method.

        Args:
            column: Numeric column for outlier detection.
            iqr_multiplier: IQR multiplier for bounds.
            max_outlier_fraction: Maximum tolerated outlier fraction.
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            total = len(df)
            if total == 0:
                return True, {"total": 0}

            values = df[column].dropna()
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
            outlier_fraction = outlier_count / total
            passed = outlier_fraction <= max_outlier_fraction

            return passed, {
                "column": column,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
                "outlier_fraction": round(outlier_fraction, 6),
                "threshold": max_outlier_fraction,
            }

        rule = ValidationRule(
            name=f"outlier_iqr_{column}",
            description=(
                f"Column '{column}' outlier fraction <= {max_outlier_fraction} "
                f"(IQR x{iqr_multiplier})"
            ),
            check_fn=_check,
            severity=severity,
            tags=["outlier_check", column],
        )
        return self.add_rule(rule)

    def expect_row_count(
        self,
        min_count: int = 1,
        max_count: Optional[int] = None,
        severity: RuleSeverity = RuleSeverity.CRITICAL,
    ) -> "StandaloneDataValidator":
        """
        Add rule: DataFrame row count must be within bounds.

        Args:
            min_count: Minimum expected row count.
            max_count: Maximum expected row count (optional).
            severity: Rule severity level.

        Returns:
            Self for method chaining.
        """
        def _check(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
            count = len(df)
            passed = count >= min_count
            if max_count is not None:
                passed = passed and count <= max_count

            return passed, {
                "row_count": count,
                "min_count": min_count,
                "max_count": max_count,
            }

        rule = ValidationRule(
            name="row_count",
            description=f"Row count in [{min_count}, {max_count or 'inf'}]",
            check_fn=_check,
            severity=severity,
            tags=["completeness_check"],
        )
        return self.add_rule(rule)

    def validate(
        self,
        df: pd.DataFrame,
        fail_on_critical: bool = True,
    ) -> ValidationReport:
        """
        Run all registered validation rules against the DataFrame.

        Args:
            df: pandas DataFrame to validate.
            fail_on_critical: If True, raises on any critical failure.

        Returns:
            ValidationReport with per-rule results and quality score.

        Raises:
            ValueError: If fail_on_critical is True and a critical rule fails.
        """
        results: List[ValidationResult] = []
        passed_count = 0
        critical_failures = 0
        warning_failures = 0

        for rule in self.rules:
            try:
                passed, details = rule.check_fn(df)
                result = ValidationResult(
                    rule_name=rule.name,
                    passed=passed,
                    severity=rule.severity.value,
                    details=details,
                )
                results.append(result)

                if passed:
                    passed_count += 1
                    logger.info(f"PASS | {rule.name}")
                else:
                    if rule.severity == RuleSeverity.CRITICAL:
                        critical_failures += 1
                        logger.error(f"FAIL [CRITICAL] | {rule.name} | {details}")
                    elif rule.severity == RuleSeverity.WARNING:
                        warning_failures += 1
                        logger.warning(f"FAIL [WARNING] | {rule.name} | {details}")
                    else:
                        logger.info(f"FAIL [INFO] | {rule.name} | {details}")

            except Exception as exc:
                logger.error(f"ERROR | {rule.name} | {exc}")
                results.append(ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity=rule.severity.value,
                    details={"error": str(exc)},
                ))
                if rule.severity == RuleSeverity.CRITICAL:
                    critical_failures += 1

        total_rules = len(self.rules)
        quality_score = (passed_count / total_rules * 100.0) if total_rules > 0 else 0.0
        is_valid = critical_failures == 0

        report = ValidationReport(
            total_rules=total_rules,
            passed_rules=passed_count,
            failed_rules=total_rules - passed_count,
            critical_failures=critical_failures,
            warning_failures=warning_failures,
            quality_score=round(quality_score, 2),
            results=results,
            is_valid=is_valid,
        )

        logger.info(
            f"Validation complete | score={quality_score:.1f}% "
            f"| passed={passed_count}/{total_rules} | critical_fails={critical_failures}"
        )

        if fail_on_critical and not is_valid:
            failed_rules = [
                r.rule_name for r in results
                if not r.passed and r.severity == "critical"
            ]
            raise ValueError(
                f"Critical validation failures: {failed_rules}. "
                f"Quality score: {quality_score:.1f}%"
            )

        return report

    def clear_rules(self) -> None:
        """Remove all registered validation rules."""
        self.rules.clear()
        logger.info("All validation rules cleared")
