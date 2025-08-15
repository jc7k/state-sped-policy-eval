#!/usr/bin/env python
"""
Common Utilities for Data Collection Module

Shared functionality to eliminate code duplication across data collectors.
Includes state mappings, API client patterns, validation logic, and file utilities.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class StateUtils:
    """
    Centralized state mapping utilities - SINGLE SOURCE OF TRUTH.
    
    Eliminates code duplication across collectors and provides comprehensive
    state validation, conversion, and metadata management.
    """

    # Comprehensive state mapping: full name -> abbreviation
    STATE_NAME_TO_ABBREV = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ", 
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
    }

    # FIPS codes for Census API (includes only states with IDEA data)
    FIPS_TO_ABBREV = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
    }

    # Reverse mappings (computed once for efficiency)
    ABBREV_TO_NAME = {v: k for k, v in STATE_NAME_TO_ABBREV.items()}
    ABBREV_TO_FIPS = {v: k for k, v in FIPS_TO_ABBREV.items()}
    
    # All state abbreviations (50 states + DC = 51 total)
    ALL_STATE_CODES = list(STATE_NAME_TO_ABBREV.values())
    
    # Special education policy research states (for filtering if needed)
    MAJOR_POLICY_STATES = [
        "CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
    ]

    @classmethod
    def name_to_abbrev(cls, state_name: str) -> str | None:
        """
        Convert state name to abbreviation with robust error handling.
        
        Args:
            state_name: Full state name (case-insensitive)
            
        Returns:
            Two-letter state code or None if invalid
        """
        if not state_name or not isinstance(state_name, str):
            return None
            
        # Normalize whitespace and try exact match first
        normalized = state_name.strip()
        if normalized in cls.STATE_NAME_TO_ABBREV:
            return cls.STATE_NAME_TO_ABBREV[normalized]
            
        # Try case-insensitive match
        normalized_lower = normalized.lower()
        for name, abbrev in cls.STATE_NAME_TO_ABBREV.items():
            if name.lower() == normalized_lower:
                return abbrev
                
        # Try partial matches for common variations
        partial_matches = {
            "washington dc": "DC",
            "d.c.": "DC", 
            "dc": "DC",
            "rhode island": "RI",
            "r.i.": "RI",
        }
        
        if normalized_lower in partial_matches:
            return partial_matches[normalized_lower]
            
        return None

    @classmethod
    def fips_to_abbrev(cls, fips_code: str) -> str | None:
        """
        Convert FIPS code to state abbreviation.
        
        Args:
            fips_code: 2-digit FIPS code (string or int)
            
        Returns:
            Two-letter state code or None if invalid
        """
        if not fips_code:
            return None
            
        # Handle both string and integer inputs
        fips_str = str(fips_code).strip().zfill(2)  # Pad with leading zero
        return cls.FIPS_TO_ABBREV.get(fips_str)

    @classmethod
    def abbrev_to_name(cls, state_abbrev: str) -> str | None:
        """Convert state abbreviation to full name."""
        if not state_abbrev:
            return None
        return cls.ABBREV_TO_NAME.get(state_abbrev.upper().strip())

    @classmethod
    def abbrev_to_fips(cls, state_abbrev: str) -> str | None:
        """Convert state abbreviation to FIPS code."""
        if not state_abbrev:
            return None
        return cls.ABBREV_TO_FIPS.get(state_abbrev.upper().strip())

    @classmethod
    def is_valid_state(cls, state_code: str) -> bool:
        """
        Check if state code is valid.
        
        Args:
            state_code: Two-letter state abbreviation
            
        Returns:
            True if valid state code
        """
        if not state_code or not isinstance(state_code, str):
            return False
        return state_code.upper().strip() in cls.ALL_STATE_CODES

    @classmethod
    def get_all_states(cls) -> list[str]:
        """Get list of all valid state codes (50 states + DC)."""
        return cls.ALL_STATE_CODES.copy()

    @classmethod
    def get_policy_states(cls) -> list[str]:
        """Get list of major policy research states."""
        return cls.MAJOR_POLICY_STATES.copy()

    @classmethod
    def validate_state_coverage(cls, state_list: list[str]) -> dict[str, Any]:
        """
        Validate state coverage for research purposes.
        
        Args:
            state_list: List of state codes to validate
            
        Returns:
            Validation report with coverage statistics
        """
        valid_states = [s for s in state_list if cls.is_valid_state(s)]
        invalid_states = [s for s in state_list if not cls.is_valid_state(s)]
        
        coverage_pct = (len(valid_states) / 51) * 100 if valid_states else 0
        
        return {
            "total_provided": len(state_list),
            "valid_states": len(valid_states),
            "invalid_states": invalid_states,
            "coverage_percentage": coverage_pct,
            "missing_states": [s for s in cls.ALL_STATE_CODES if s not in valid_states],
            "has_full_coverage": len(valid_states) == 51,
            "has_minimum_coverage": len(valid_states) >= 45,  # 90% threshold
            "policy_state_coverage": len([s for s in valid_states if s in cls.MAJOR_POLICY_STATES])
        }

    @classmethod
    def normalize_state_identifier(cls, identifier: str) -> str | None:
        """
        Normalize any state identifier (name, abbreviation, FIPS) to standard abbreviation.
        
        Args:
            identifier: State name, abbreviation, or FIPS code
            
        Returns:
            Standardized two-letter state code or None
        """
        if not identifier:
            return None
            
        identifier = str(identifier).strip()
        
        # Try as abbreviation first (most common)
        if len(identifier) == 2 and cls.is_valid_state(identifier):
            return identifier.upper()
            
        # Try as FIPS code
        if identifier.isdigit() and len(identifier) <= 2:
            return cls.fips_to_abbrev(identifier)
            
        # Try as state name
        return cls.name_to_abbrev(identifier)


class RateLimiter:
    """
    Advanced rate limiting with exponential backoff and API-specific configurations.
    
    Eliminates hardcoded rate limits and provides centralized, configurable
    rate limiting strategy across all data collectors.
    """
    
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst of requests allowed
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.burst_size = burst_size
        self.request_times: list[float] = []
        self.consecutive_errors = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old request times outside the window
        cutoff_time = now - 1.0  # 1 second window
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check if we need to wait
        if len(self.request_times) >= self.burst_size:
            # Calculate wait time based on oldest request in burst
            wait_time = self.request_times[0] + 1.0 - now
            if wait_time > 0:
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                
        # Apply minimum interval between requests
        if self.request_times and self.min_interval > 0:
            time_since_last = now - self.request_times[-1]
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                self.logger.debug(f"Minimum interval: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                
        # Record this request time
        self.request_times.append(time.time())
        
    def on_request_success(self) -> None:
        """Called when a request succeeds."""
        self.consecutive_errors = 0
        
    def on_request_error(self) -> None:
        """Called when a request fails - implements exponential backoff."""
        self.consecutive_errors += 1
        
        if self.consecutive_errors > 1:
            # Exponential backoff: 2^(errors-1) seconds, max 60 seconds
            backoff_time = min(2 ** (self.consecutive_errors - 1), 60)
            self.logger.warning(f"Request failed, backing off for {backoff_time}s")
            time.sleep(backoff_time)


class APIRateLimitConfig:
    """Configuration for API-specific rate limits."""
    
    # Default rate limits for known APIs (requests per second)
    DEFAULT_LIMITS = {
        'naep': 0.17,  # ~6 second delay (conservative for education data)
        'census': 1.0,  # 1 request per second
        'edfacts': 0.5,  # 2 second delay
        'ocr': 0.5,     # 2 second delay
        'default': 1.0,  # Default 1 request per second
    }
    
    # Burst sizes for different APIs
    BURST_SIZES = {
        'naep': 1,      # No bursts for NAEP (most restrictive)
        'census': 3,    # Small bursts allowed
        'edfacts': 2,   # Small bursts
        'ocr': 2,       # Small bursts
        'default': 3,   # Default burst size
    }
    
    @classmethod
    def get_rate_limit(cls, api_name: str) -> float:
        """Get rate limit for API."""
        return cls.DEFAULT_LIMITS.get(api_name.lower(), cls.DEFAULT_LIMITS['default'])
        
    @classmethod
    def get_burst_size(cls, api_name: str) -> int:
        """Get burst size for API."""
        return cls.BURST_SIZES.get(api_name.lower(), cls.BURST_SIZES['default'])
        
    @classmethod
    def create_rate_limiter(cls, api_name: str) -> RateLimiter:
        """Create appropriately configured rate limiter for API."""
        rate_limit = cls.get_rate_limit(api_name)
        burst_size = cls.get_burst_size(api_name)
        return RateLimiter(rate_limit, burst_size)


class APIClient:
    """Enhanced HTTP client with unified rate limiting and error handling."""

    def __init__(self, api_name: str = 'default', rate_limiter: RateLimiter | None = None, timeout: int = 30):
        """
        Initialize API client with unified rate limiting and error reporting.

        Args:
            api_name: Name of API for appropriate rate limiting
            rate_limiter: Custom rate limiter (creates default if None)
            timeout: Request timeout in seconds
        """
        self.api_name = api_name
        self.timeout = timeout
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{api_name}")
        self._request_count = 0
        
        # Initialize structured error reporting
        self.error_reporter = ErrorReporter(self.logger)
        
        # Use provided rate limiter or create appropriate one
        if rate_limiter is None:
            self.rate_limiter = APIRateLimitConfig.create_rate_limiter(api_name)
        else:
            self.rate_limiter = rate_limiter

    def get(self, url: str, params: dict | None = None, **kwargs) -> requests.Response:
        """
        Make GET request with unified rate limiting and error handling.

        Args:
            url: Request URL
            params: Query parameters
            **kwargs: Additional requests parameters

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        self._request_count += 1
        
        # Apply unified rate limiting
        self.rate_limiter.wait_if_needed()

        kwargs.setdefault("timeout", self.timeout)
        
        # Add user agent for better API compliance
        headers = kwargs.get('headers', {})
        headers['User-Agent'] = f'State-SPED-Policy-Research/1.0 ({self.api_name})'
        kwargs['headers'] = headers

        try:
            self.logger.debug(f"Making request {self._request_count} to {url}")
            response = requests.get(url, params=params, **kwargs)
            response.raise_for_status()
            
            # Track successful requests
            self.rate_limiter.on_request_success()
            
            # Log rate limiting headers if present
            self._log_rate_limit_headers(response)
            
            return response
            
        except requests.exceptions.RequestException as e:
            # Track failed requests for backoff
            self.rate_limiter.on_request_error()
            
            # Report structured error with context
            self.error_reporter.report_api_error(
                operation='http_get',
                component=self.api_name,
                exception=e,
                attempt=self.rate_limiter.consecutive_errors,
                url=url
            )
            
            raise
            
    def _log_rate_limit_headers(self, response: requests.Response) -> None:
        """Log rate limiting information from response headers."""
        rate_headers = {
            'X-RateLimit-Limit': 'limit',
            'X-RateLimit-Remaining': 'remaining', 
            'X-RateLimit-Reset': 'reset',
            'Retry-After': 'retry_after'
        }
        
        rate_info = {}
        for header, key in rate_headers.items():
            if header in response.headers:
                rate_info[key] = response.headers[header]
                
        if rate_info:
            self.logger.debug(f"API rate limit info: {rate_info}")
            
            # Warn if approaching rate limit
            if 'remaining' in rate_info and 'limit' in rate_info:
                try:
                    remaining = int(rate_info['remaining'])
                    limit = int(rate_info['limit'])
                    if remaining < limit * 0.1:  # Less than 10% remaining
                        self.logger.warning(f"API rate limit nearly exhausted: {remaining}/{limit}")
                except ValueError:
                    pass


@dataclass
class ErrorContext:
    """Context information for error reporting."""
    operation: str
    component: str
    state: Optional[str] = None
    year: Optional[int] = None
    attempt: int = 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ErrorReport:
    """Structured error report with context and severity."""
    timestamp: str
    error_type: str
    severity: str  # 'critical', 'error', 'warning', 'info'
    message: str
    context: ErrorContext
    exception: Optional[str] = None
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'severity': self.severity,
            'message': self.message,
            'context': {
                'operation': self.context.operation,
                'component': self.context.component,
                'state': self.context.state,
                'year': self.context.year,
                'attempt': self.context.attempt,
                'metadata': self.context.metadata
            },
            'exception': self.exception,
            'suggested_action': self.suggested_action
        }


class ErrorReporter:
    """Centralized structured error reporting and handling."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error reporter with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        self.errors: List[ErrorReport] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    def report_error(
        self,
        error_type: str,
        severity: str,
        message: str,
        context: ErrorContext,
        exception: Optional[Exception] = None,
        suggested_action: Optional[str] = None
    ) -> ErrorReport:
        """Report a structured error with context."""
        from datetime import datetime
        
        error_report = ErrorReport(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            severity=severity,
            message=message,
            context=context,
            exception=str(exception) if exception else None,
            suggested_action=suggested_action
        )
        
        self.errors.append(error_report)
        self.error_counts[f"{severity}_{error_type}"] += 1
        
        # Log based on severity
        log_method = {
            'critical': self.logger.critical,
            'error': self.logger.error,
            'warning': self.logger.warning,
            'info': self.logger.info
        }.get(severity, self.logger.error)
        
        log_message = f"[{error_type}] {message}"
        if context.state:
            log_message += f" (State: {context.state})"
        if context.year:
            log_message += f" (Year: {context.year})"
        if context.attempt > 1:
            log_message += f" (Attempt: {context.attempt})"
            
        log_method(log_message)
        
        return error_report
    
    def report_api_error(
        self,
        operation: str,
        component: str,
        exception: Exception,
        state: Optional[str] = None,
        year: Optional[int] = None,
        attempt: int = 1,
        url: Optional[str] = None
    ) -> ErrorReport:
        """Report API-specific errors with standardized context."""
        context = ErrorContext(
            operation=operation,
            component=component,
            state=state,
            year=year,
            attempt=attempt,
            metadata={'url': url} if url else {}
        )
        
        # Determine severity and suggested action based on exception type
        if isinstance(exception, requests.exceptions.Timeout):
            severity = 'warning'
            suggested_action = 'Retry with longer timeout or during off-peak hours'
        elif isinstance(exception, requests.exceptions.ConnectionError):
            severity = 'error'
            suggested_action = 'Check network connectivity and API endpoint availability'
        elif isinstance(exception, requests.exceptions.HTTPError):
            if hasattr(exception, 'response') and exception.response:
                status_code = exception.response.status_code
                if status_code == 429:
                    severity = 'warning'
                    suggested_action = 'Increase rate limiting delays'
                elif status_code >= 500:
                    severity = 'error'
                    suggested_action = 'Server error - retry later'
                else:
                    severity = 'error'
                    suggested_action = f'Client error (HTTP {status_code}) - check request parameters'
            else:
                severity = 'error'
                suggested_action = 'Check HTTP request format and parameters'
        else:
            severity = 'error'
            suggested_action = 'Review error details and adjust request parameters'
            
        return self.report_error(
            error_type='api_request_failed',
            severity=severity,
            message=f"API request failed: {str(exception)}",
            context=context,
            exception=exception,
            suggested_action=suggested_action
        )
    
    def report_data_validation_error(
        self,
        validation_type: str,
        message: str,
        state: Optional[str] = None,
        year: Optional[int] = None,
        severity: str = 'error',
        data_context: Optional[Dict[str, Any]] = None
    ) -> ErrorReport:
        """Report data validation errors with context."""
        context = ErrorContext(
            operation='data_validation',
            component='validator',
            state=state,
            year=year,
            metadata=data_context or {}
        )
        
        suggested_actions = {
            'missing_required_column': 'Verify data source schema and collection parameters',
            'invalid_data_type': 'Check data parsing logic and type conversion methods',
            'out_of_range_values': 'Review data quality and apply appropriate filters',
            'duplicate_records': 'Implement deduplication logic or verify data source integrity',
            'missing_data_threshold': 'Investigate data source completeness or adjust collection parameters'
        }
        
        return self.report_error(
            error_type=f'validation_{validation_type}',
            severity=severity,
            message=message,
            context=context,
            suggested_action=suggested_actions.get(validation_type, 'Review validation logic and data source')
        )
    
    def report_parsing_error(
        self,
        parser_type: str,
        message: str,
        raw_data: Optional[Any] = None,
        state: Optional[str] = None,
        year: Optional[int] = None
    ) -> ErrorReport:
        """Report data parsing errors with context."""
        context = ErrorContext(
            operation='data_parsing',
            component=f'{parser_type}_parser',
            state=state,
            year=year,
            metadata={
                'raw_data_type': type(raw_data).__name__ if raw_data else None,
                'raw_data_preview': str(raw_data)[:200] if raw_data else None
            }
        )
        
        return self.report_error(
            error_type='parsing_failed',
            severity='error',
            message=message,
            context=context,
            suggested_action='Review API response format and update parsing logic'
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all reported errors."""
        total_errors = len(self.errors)
        
        severity_counts = defaultdict(int)
        error_type_counts = defaultdict(int)
        state_error_counts = defaultdict(int)
        component_error_counts = defaultdict(int)
        
        for error in self.errors:
            severity_counts[error.severity] += 1
            error_type_counts[error.error_type] += 1
            if error.context.state:
                state_error_counts[error.context.state] += 1
            component_error_counts[error.context.component] += 1
        
        return {
            'total_errors': total_errors,
            'severity_breakdown': dict(severity_counts),
            'error_type_breakdown': dict(error_type_counts),
            'errors_by_state': dict(state_error_counts),
            'errors_by_component': dict(component_error_counts),
            'most_recent_errors': [error.to_dict() for error in self.errors[-5:]],
            'critical_errors': [
                error.to_dict() for error in self.errors 
                if error.severity == 'critical'
            ]
        }
    
    def export_error_report(self, file_path: str) -> None:
        """Export detailed error report to JSON file."""
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_errors': len(self.errors),
                'report_version': '1.0'
            },
            'summary': self.get_error_summary(),
            'detailed_errors': [error.to_dict() for error in self.errors]
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"Error report exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export error report: {str(e)}")
    
    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()
        self.error_counts.clear()
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors have been reported."""
        return any(error.severity == 'critical' for error in self.errors)
    
    def get_retry_recommendations(self) -> List[str]:
        """Get recommendations for retry strategies based on error patterns."""
        recommendations = []
        
        api_errors = [e for e in self.errors if e.error_type == 'api_request_failed']
        if api_errors:
            timeout_errors = [e for e in api_errors if 'timeout' in e.message.lower()]
            if len(timeout_errors) > 3:
                recommendations.append("Consider increasing timeout values for API requests")
            
            rate_limit_errors = [e for e in api_errors if 'rate limit' in e.message.lower() or '429' in e.message]
            if rate_limit_errors:
                recommendations.append("Implement more aggressive rate limiting to avoid API quotas")
        
        parsing_errors = [e for e in self.errors if e.error_type == 'parsing_failed']
        if len(parsing_errors) > 5:
            recommendations.append("Review API response format - may have changed")
        
        validation_errors = [e for e in self.errors if 'validation_' in e.error_type]
        if len(validation_errors) > 10:
            recommendations.append("Data quality issues detected - consider data source validation")
        
        return recommendations


class DataValidator:
    """
    Comprehensive data validation framework for research data quality.
    
    Provides standardized validation across all data sources with domain-specific
    business rules, statistical validation, and automated data quality scoring.
    """

    def __init__(self, state_utils: StateUtils):
        self.state_utils = state_utils
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_reporter = ErrorReporter(self.logger)

    def validate_dataset(self, df: pd.DataFrame, data_source: str, **validation_config) -> dict[str, Any]:
        """
        Comprehensive dataset validation with automated scoring.
        
        Args:
            df: DataFrame to validate
            data_source: Type of data source ('naep', 'census', etc.)
            **validation_config: Source-specific validation parameters
            
        Returns:
            Comprehensive validation report with quality score
        """
        validation_report = {
            'data_source': data_source,
            'total_records': len(df),
            'validation_timestamp': pd.Timestamp.now(),
            'passed': True,
            'quality_score': 0.0,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'recommendations': []
        }
        
        if df.empty:
            validation_report.update({
                'passed': False,
                'quality_score': 0.0,
                'errors': ['Dataset is empty']
            })
            return validation_report
            
        # Run all validation checks
        checks = [
            self._validate_state_coverage,
            self._validate_temporal_consistency,
            self._validate_data_types,
            self._validate_business_rules,
            self._validate_statistical_patterns,
            self._check_duplicates,
            self._check_missing_data_patterns
        ]
        
        total_points = 0
        earned_points = 0
        
        for check in checks:
            try:
                check_result = check(df, data_source, validation_config)
                total_points += check_result.get('max_points', 10)
                earned_points += check_result.get('earned_points', 0)
                
                validation_report['errors'].extend(check_result.get('errors', []))
                validation_report['warnings'].extend(check_result.get('warnings', []))
                validation_report['metrics'].update(check_result.get('metrics', {}))
                validation_report['recommendations'].extend(check_result.get('recommendations', []))
                
                if check_result.get('critical_failure', False):
                    validation_report['passed'] = False
                    
            except Exception as e:
                self.logger.error(f"Validation check failed: {e}")
                validation_report['errors'].append(f"Validation check error: {str(e)}")
                
        # Calculate overall quality score
        validation_report['quality_score'] = (earned_points / total_points) * 100 if total_points > 0 else 0
        
        # Add overall assessment
        validation_report['assessment'] = self._get_quality_assessment(validation_report['quality_score'])
        
        return validation_report

    def _validate_state_coverage(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Validate state coverage with research-specific requirements."""
        state_column = config.get('state_column', 'state')
        
        if state_column not in df.columns:
            # Report structured error for missing state column
            self.error_reporter.report_data_validation_error(
                validation_type='missing_required_column',
                message=f"Required state column '{state_column}' not found in {data_source} data",
                severity='critical',
                data_context={'available_columns': df.columns.tolist(), 'data_source': data_source}
            )
            
            return {
                'max_points': 15,
                'earned_points': 0,
                'critical_failure': True,
                'errors': [f"Required state column '{state_column}' not found"],
                'metrics': {'states_covered': 0}
            }
            
        # Validate state codes using centralized StateUtils
        state_validation = self.state_utils.validate_state_coverage(df[state_column].unique().tolist())
        
        result = {
            'max_points': 15,
            'earned_points': 0,
            'errors': [],
            'warnings': [],
            'metrics': {
                'states_covered': state_validation['valid_states'],
                'coverage_percentage': state_validation['coverage_percentage'],
                'invalid_states': state_validation['invalid_states'],
                'missing_states': state_validation['missing_states']
            }
        }
        
        # Scoring based on coverage
        if state_validation['has_full_coverage']:
            result['earned_points'] = 15
        elif state_validation['has_minimum_coverage']:
            result['earned_points'] = 12
            result['warnings'].append(
                f"Incomplete state coverage: {state_validation['valid_states']}/51 states"
            )
        elif state_validation['valid_states'] >= 20:
            result['earned_points'] = 8
            result['warnings'].append(
                f"Limited state coverage: {state_validation['valid_states']}/51 states"
            )
        else:
            # Report critical coverage issue
            self.error_reporter.report_data_validation_error(
                validation_type='insufficient_coverage',
                message=f"Insufficient state coverage in {data_source}: only {state_validation['valid_states']} valid states",
                severity='critical',
                data_context={
                    'valid_states': state_validation['valid_states'],
                    'coverage_percentage': state_validation['coverage_percentage'],
                    'data_source': data_source
                }
            )
            
            result['earned_points'] = 0
            result['critical_failure'] = True
            result['errors'].append(
                f"Insufficient state coverage: only {state_validation['valid_states']} states"
            )
            
        # Check for invalid state codes
        if state_validation['invalid_states']:
            # Report invalid state codes
            self.error_reporter.report_data_validation_error(
                validation_type='invalid_state_codes',
                message=f"Invalid state codes found in {data_source} data: {state_validation['invalid_states']}",
                severity='error',
                data_context={
                    'invalid_states': state_validation['invalid_states'],
                    'data_source': data_source
                }
            )
            
            result['errors'].append(f"Invalid state codes found: {state_validation['invalid_states']}")
            result['earned_points'] = max(0, result['earned_points'] - 3)
            
        return result

    def _validate_temporal_consistency(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Validate temporal consistency and logical date progressions."""
        year_column = config.get('year_column', 'year')
        
        result = {
            'max_points': 10,
            'earned_points': 10,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        if year_column not in df.columns:
            result.update({
                'earned_points': 0,
                'errors': [f"Required year column '{year_column}' not found"]
            })
            return result
            
        years = df[year_column].dropna().unique()
        
        if len(years) == 0:
            result.update({
                'earned_points': 0,
                'errors': ['No valid years found in data']
            })
            return result
            
        result['metrics'] = {
            'year_range': f"{years.min()}-{years.max()}",
            'years_covered': len(years),
            'year_list': sorted(years.tolist())
        }
        
        # Check for reasonable year range
        current_year = pd.Timestamp.now().year
        if years.min() < 2009:
            result['warnings'].append(f"Data includes years before 2009 (IDEA reauth): {years.min()}")
            result['earned_points'] -= 2
            
        if years.max() > current_year:
            result['errors'].append(f"Data includes future years: {years.max()}")
            result['earned_points'] -= 3
            
        # Check for data gaps in time series
        if len(years) > 1:
            year_gaps = []
            sorted_years = sorted(years)
            for i in range(1, len(sorted_years)):
                gap = sorted_years[i] - sorted_years[i-1]
                if gap > 3:  # Gap larger than 3 years
                    year_gaps.append(f"{sorted_years[i-1]}-{sorted_years[i]}")
                    
            if year_gaps:
                result['warnings'].append(f"Large temporal gaps detected: {year_gaps}")
                result['recommendations'].append("Consider imputation or note data limitations")
                
        return result

    def _validate_data_types(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Validate data types and numeric ranges."""
        result = {
            'max_points': 10,
            'earned_points': 10,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Data source specific type validation
        type_rules = self._get_type_rules(data_source)
        
        for column, expected_type in type_rules.items():
            if column not in df.columns:
                continue
                
            actual_type = df[column].dtype
            result['metrics'][f'{column}_type'] = str(actual_type)
            
            # Check numeric columns for proper types
            if expected_type in ['int', 'float'] and not pd.api.types.is_numeric_dtype(actual_type):
                result['warnings'].append(f"Column '{column}' should be {expected_type}, found {actual_type}")
                result['earned_points'] -= 1
                
        return result

    def _validate_business_rules(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Validate domain-specific business rules."""
        result = {
            'max_points': 15,
            'earned_points': 15,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        if data_source.lower() == 'naep':
            return self._validate_naep_business_rules(df, result)
        elif data_source.lower() == 'census':
            return self._validate_census_business_rules(df, result)
        else:
            # Generic business rules for other sources
            return result

    def _validate_naep_business_rules(self, df: pd.DataFrame, result: dict) -> dict:
        """NAEP-specific business rule validation."""
        
        # Check NAEP score ranges (0-500 scale)
        score_columns = [col for col in df.columns if 'score' in col.lower() or 'mean' in col.lower()]
        
        for col in score_columns:
            if col in df.columns:
                valid_scores = df[col].dropna()
                if len(valid_scores) > 0:
                    out_of_range = ((valid_scores < 0) | (valid_scores > 500)).sum()
                    result['metrics'][f'{col}_out_of_range'] = out_of_range
                    
                    if out_of_range > 0:
                        pct_invalid = (out_of_range / len(valid_scores)) * 100
                        if pct_invalid > 5:  # More than 5% invalid
                            result['errors'].append(
                                f"Column '{col}': {out_of_range} scores outside valid NAEP range (0-500)"
                            )
                            result['earned_points'] -= 3
                        else:
                            result['warnings'].append(
                                f"Column '{col}': {out_of_range} scores outside typical range"
                            )
                            result['earned_points'] -= 1
                            
        # Check achievement gaps are reasonable
        if 'gap' in df.columns:
            gaps = df['gap'].dropna()
            if len(gaps) > 0:
                unreasonable_gaps = ((gaps < 0) | (gaps > 100)).sum()
                if unreasonable_gaps > 0:
                    result['warnings'].append(f"{unreasonable_gaps} unreasonable achievement gaps found")
                    
        return result

    def _validate_census_business_rules(self, df: pd.DataFrame, result: dict) -> dict:
        """Census-specific business rule validation."""
        
        # Check expenditure reasonableness
        expenditure_cols = [col for col in df.columns if 'expenditure' in col.lower() or 'spending' in col.lower()]
        
        for col in expenditure_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    negative_values = (values < 0).sum()
                    extremely_high = (values > 100000).sum()  # >$100k per pupil
                    
                    if negative_values > 0:
                        result['errors'].append(f"Column '{col}': {negative_values} negative expenditures")
                        result['earned_points'] -= 3
                        
                    if extremely_high > 0:
                        result['warnings'].append(f"Column '{col}': {extremely_high} extremely high values")
                        
        return result

    def _validate_statistical_patterns(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Validate statistical patterns and detect outliers."""
        result = {
            'max_points': 10,
            'earned_points': 10,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        numeric_columns = df.select_dtypes(include=[int, float]).columns
        
        for col in numeric_columns:
            if df[col].notna().sum() < 10:  # Skip columns with too few values
                continue
                
            col_data = df[col].dropna()
            
            # Calculate basic statistics
            q1, median, q3 = col_data.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            
            # Detect outliers using IQR method
            outlier_threshold = 3 * iqr
            outliers = ((col_data < (q1 - outlier_threshold)) | 
                       (col_data > (q3 + outlier_threshold))).sum()
            
            outlier_pct = (outliers / len(col_data)) * 100
            
            result['metrics'][f'{col}_outliers'] = {
                'count': outliers,
                'percentage': outlier_pct,
                'median': median,
                'iqr': iqr
            }
            
            if outlier_pct > 10:  # More than 10% outliers
                result['warnings'].append(f"Column '{col}': High outlier rate ({outlier_pct:.1f}%)")
                result['recommendations'].append(f"Review '{col}' for data quality issues")
                result['earned_points'] -= 1
                
        return result

    def _check_duplicates(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Check for duplicate records using appropriate keys."""
        key_columns = config.get('unique_keys', ['state', 'year'])
        
        result = {
            'max_points': 10,
            'earned_points': 10,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check if key columns exist
        missing_keys = [col for col in key_columns if col not in df.columns]
        if missing_keys:
            result.update({
                'earned_points': 0,
                'errors': [f"Missing key columns for duplicate check: {missing_keys}"]
            })
            return result
            
        # Check for duplicates
        duplicates = df.duplicated(subset=key_columns).sum()
        total_records = len(df)
        duplicate_pct = (duplicates / total_records) * 100 if total_records > 0 else 0
        
        result['metrics']['duplicates'] = {
            'count': duplicates,
            'percentage': duplicate_pct,
            'key_columns': key_columns
        }
        
        if duplicates > 0:
            if duplicate_pct > 1:  # More than 1% duplicates
                result['errors'].append(f"{duplicates} duplicate records found ({duplicate_pct:.1f}%)")
                result['earned_points'] = 0
            else:
                result['warnings'].append(f"{duplicates} duplicate records found")
                result['earned_points'] -= 3
                
        return result

    def _check_missing_data_patterns(self, df: pd.DataFrame, data_source: str, config: dict) -> dict[str, Any]:
        """Analyze missing data patterns and rates."""
        result = {
            'max_points': 10,
            'earned_points': 10,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        missing_threshold = config.get('missing_threshold', 0.3)
        critical_columns = config.get('critical_columns', [])
        
        missing_analysis = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_rate = missing_count / len(df) if len(df) > 0 else 0
            
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_rate': missing_rate
            }
            
            # Check critical columns
            if col in critical_columns and missing_rate > 0.1:  # 10% threshold for critical
                result['errors'].append(f"Critical column '{col}' has {missing_rate:.1%} missing data")
                result['earned_points'] -= 3
                
            # Check general missing data rates
            elif missing_rate > missing_threshold:
                result['warnings'].append(f"Column '{col}' has high missing rate: {missing_rate:.1%}")
                result['earned_points'] -= 1
                
        result['metrics']['missing_data_analysis'] = missing_analysis
        
        # Overall missing data assessment
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()
        overall_missing_rate = total_missing / total_cells if total_cells > 0 else 0
        
        result['metrics']['overall_missing_rate'] = overall_missing_rate
        
        if overall_missing_rate > 0.5:  # More than 50% missing overall
            result['errors'].append(f"Extremely high overall missing rate: {overall_missing_rate:.1%}")
            result['earned_points'] = max(0, result['earned_points'] - 5)
            
        return result

    def _get_type_rules(self, data_source: str) -> dict[str, str]:
        """Get data type validation rules for each data source."""
        rules = {
            'naep': {
                'state': 'str',
                'year': 'int',
                'grade': 'int',
                'mean_score': 'float',
                'gap': 'float'
            },
            'census': {
                'state': 'str',
                'year': 'int',
                'total_expenditures': 'int',
                'enrollment': 'int',
                'per_pupil_spending': 'float'
            }
        }
        return rules.get(data_source.lower(), {})

    def _get_quality_assessment(self, score: float) -> str:
        """Get qualitative assessment of data quality score."""
        if score >= 90:
            return "Excellent - Ready for production analysis"
        elif score >= 80:
            return "Good - Minor issues to address"
        elif score >= 70:
            return "Fair - Several quality concerns"
        elif score >= 60:
            return "Poor - Significant data quality issues"
        else:
            return "Critical - Data not suitable for analysis"


class FileUtils:
    """File handling utilities for data collectors."""

    @staticmethod
    def ensure_directory(file_path: str | Path) -> Path:
        """
        Ensure directory exists for the given file path.

        Args:
            file_path: Path to file

        Returns:
            Path object for the directory
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent

    @staticmethod
    def save_dataframe(
        df: pd.DataFrame, file_path: str | Path, logger: logging.Logger | None = None
    ) -> bool:
        """
        Save DataFrame to CSV with directory creation.

        Args:
            df: DataFrame to save
            file_path: Output file path
            logger: Optional logger instance

        Returns:
            True if successful, False otherwise
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            FileUtils.ensure_directory(file_path)
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}: {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
            return False

    @staticmethod
    def get_default_output_path(
        filename: str, env_var: str = "RAW_DATA_DIR", default_dir: str = "data/raw"
    ) -> str:
        """
        Get default output path from environment or use default.

        Args:
            filename: Name of output file
            env_var: Environment variable for directory
            default_dir: Default directory if env var not set

        Returns:
            Full file path
        """
        data_dir = os.getenv(env_var, default_dir)
        return os.path.join(data_dir, filename)


class SafeTypeConverter:
    """Safe type conversion utilities for API data."""

    @staticmethod
    def safe_float(value: Any, invalid_values: list | None = None) -> float | None:
        """
        Safely convert value to float, handling API special codes.

        Args:
            value: Value to convert
            invalid_values: List of values to treat as None

        Returns:
            Float value or None if invalid/missing
        """
        if invalid_values is None:
            invalid_values = [None, "", "null", "‡", "*", "N/A", "#", "N", "X", "S", "D"]

        if value in invalid_values:
            return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_int(value: Any, invalid_values: list | None = None) -> int | None:
        """
        Safely convert value to int, handling API special codes.

        Args:
            value: Value to convert
            invalid_values: List of values to treat as None

        Returns:
            Integer value or None if invalid/missing
        """
        if invalid_values is None:
            invalid_values = [None, "", "null", "N", "X", "S", "D"]

        if value in invalid_values:
            return None

        try:
            return int(value)
        except (ValueError, TypeError):
            return None


class BaseAPIResponseParser(ABC):
    """
    Abstract base class for standardized API response parsing.
    
    Eliminates inconsistent parsing logic across different data collectors
    and provides a unified interface for handling various API response formats.
    """
    
    def __init__(self, state_utils: StateUtils):
        self.state_utils = state_utils
        self.converter = SafeTypeConverter()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse_response(self, raw_response: dict[str, Any], **context) -> list[dict[str, Any]]:
        """
        Parse raw API response into standardized records.
        
        Args:
            raw_response: Raw API response dictionary
            **context: Additional context (year, grade, subject, etc.)
            
        Returns:
            List of standardized record dictionaries
        """
        pass
    
    @abstractmethod
    def validate_response_structure(self, raw_response: dict[str, Any]) -> bool:
        """
        Validate that API response has expected structure.
        
        Args:
            raw_response: Raw API response
            
        Returns:
            True if structure is valid
        """
        pass
    
    def extract_state_identifier(self, record: dict[str, Any]) -> str | None:
        """
        Extract and normalize state identifier from API record.
        
        Args:
            record: Individual record from API response
            
        Returns:
            Standardized state code or None if not found
        """
        # Try common field names for state identification
        state_fields = [
            'jurisdiction', 'state', 'state_code', 'state_abbrev',
            'jurisLabel', 'state_name', 'NAME', 'name'
        ]
        
        for field in state_fields:
            if field in record and record[field]:
                state_id = self.state_utils.normalize_state_identifier(str(record[field]))
                if state_id:
                    return state_id
                    
        return None
    
    def create_base_record(self, **kwargs) -> dict[str, Any]:
        """
        Create base record with standard fields.
        
        Args:
            **kwargs: Field values
            
        Returns:
            Base record dictionary with standard structure
        """
        return {
            'state': kwargs.get('state'),
            'year': kwargs.get('year'),
            'data_source': kwargs.get('data_source', self.__class__.__name__),
            'collection_timestamp': pd.Timestamp.now(),
            'raw_record_id': kwargs.get('raw_record_id'),
        }


class NAEPResponseParser(BaseAPIResponseParser):
    """Parser for NAEP API responses with IEP (disability status) breakdowns."""
    
    def validate_response_structure(self, raw_response: dict[str, Any]) -> bool:
        """Validate NAEP API response structure."""
        if not isinstance(raw_response, dict):
            return False
            
        if 'result' not in raw_response:
            return False
            
        if not isinstance(raw_response['result'], list):
            return False
            
        return True
    
    def parse_response(self, raw_response: dict[str, Any], **context) -> list[dict[str, Any]]:
        """
        Parse NAEP API response into standardized records.
        
        Expected context: year, grade, subject
        """
        if not self.validate_response_structure(raw_response):
            self.logger.warning("Invalid NAEP response structure")
            return []
            
        records = []
        year = context.get('year')
        grade = context.get('grade') 
        subject = context.get('subject')
        
        for state_data in raw_response.get('result', []):
            try:
                # Extract state information
                state_code = self.extract_state_identifier(state_data)
                if not state_code:
                    continue
                
                # Handle new API format (varValue/varValueLabel)
                if 'varValue' in state_data:
                    record = self._parse_new_format(state_data, state_code, year, grade, subject)
                    if record:
                        records.append(record)
                        
                # Handle legacy format (datavalue array)
                elif 'datavalue' in state_data:
                    parsed_records = self._parse_legacy_format(state_data, state_code, year, grade, subject)
                    records.extend(parsed_records)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse NAEP record: {e}")
                continue
                
        return records
    
    def _parse_new_format(self, state_data: dict, state_code: str, year: int, grade: int, subject: str) -> dict | None:
        """Parse new NAEP API format with varValue/varValueLabel."""
        var_value = state_data.get('varValue', '')
        score = self.converter.safe_float(state_data.get('value'))
        
        if score is None:
            return None
            
        # Determine disability status from varValue
        is_swd = var_value == "1"  # "1" = Students with disabilities
        
        record = self.create_base_record(
            state=state_code,
            year=year,
            data_source='NAEP'
        )
        
        record.update({
            'grade': grade,
            'subject': subject,
            'disability_status': 'SWD' if is_swd else 'non-SWD',
            'disability_label': state_data.get('varValueLabel', ''),
            'mean_score': score,
            'error_flag': state_data.get('errorFlag'),
            'is_displayable': state_data.get('isStatDisplayable', 0) == 1,
            'var_value': var_value,
        })
        
        return record
    
    def _parse_legacy_format(self, state_data: dict, state_code: str, year: int, grade: int, subject: str) -> list[dict]:
        """Parse legacy NAEP API format with datavalue array."""
        records = []
        
        for data_item in state_data.get('datavalue', []):
            category = data_item.get('categoryname', '')
            score = self.converter.safe_float(data_item.get('value'))
            
            if score is None:
                continue
                
            # Determine disability status from category name
            is_swd = 'IEP - Yes' in category or 'with IEP' in category
            
            record = self.create_base_record(
                state=state_code,
                year=year,
                data_source='NAEP'
            )
            
            record.update({
                'grade': grade,
                'subject': subject,
                'disability_status': 'SWD' if is_swd else 'non-SWD',
                'disability_label': category,
                'mean_score': score,
                'error_flag': data_item.get('errorFlag'),
                'category_name': category,
            })
            
            records.append(record)
            
        return records


class CensusResponseParser(BaseAPIResponseParser):
    """Parser for Census API responses with education finance data."""
    
    def validate_response_structure(self, raw_response: dict[str, Any]) -> bool:
        """Validate Census API response structure."""
        if not isinstance(raw_response, list):
            return False
            
        if len(raw_response) < 2:  # Need headers + at least one data row
            return False
            
        return True
    
    def parse_response(self, raw_response: list[list[str]], **context) -> list[dict[str, Any]]:
        """
        Parse Census API response into standardized records.
        
        Expected context: year
        """
        if not self.validate_response_structure(raw_response):
            self.logger.warning("Invalid Census response structure")
            return []
            
        records = []
        year = context.get('year')
        
        # First row contains headers
        headers = raw_response[0]
        data_rows = raw_response[1:]
        
        for row in data_rows:
            try:
                row_dict = dict(zip(headers, row, strict=False))
                record = self._parse_finance_record(row_dict, year)
                if record:
                    records.append(record)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse Census record: {e}")
                continue
                
        return records
    
    def _parse_finance_record(self, row_data: dict[str, str], year: int) -> dict | None:
        """Parse individual Census finance record."""
        # Extract state identifier
        state_code = self.extract_state_identifier(row_data)
        if not state_code:
            return None
            
        record = self.create_base_record(
            state=state_code,
            year=year,
            data_source='Census'
        )
        
        # Parse financial fields
        record.update({
            'total_expenditures': self.converter.safe_int(row_data.get('TOTALEXP')),
            'current_instruction': self.converter.safe_int(row_data.get('TCURINST')),
            'student_support_services': self.converter.safe_int(row_data.get('TCURSSVC')),
            'other_current_expenditures': self.converter.safe_int(row_data.get('TCUROTH')),
            'enrollment': self.converter.safe_int(row_data.get('ENROLL')),
        })
        
        # Calculate derived fields
        if record['total_expenditures'] and record['enrollment'] and record['enrollment'] > 0:
            record['per_pupil_spending'] = record['total_expenditures'] / record['enrollment']
        else:
            record['per_pupil_spending'] = None
            
        if record['student_support_services'] and record['enrollment'] and record['enrollment'] > 0:
            record['support_services_per_pupil'] = record['student_support_services'] / record['enrollment']
        else:
            record['support_services_per_pupil'] = None
            
        return record


class ResponseParserFactory:
    """Factory for creating appropriate response parsers based on data source."""
    
    @staticmethod
    def create_parser(data_source: str, state_utils: StateUtils) -> BaseAPIResponseParser:
        """
        Create appropriate response parser for data source.
        
        Args:
            data_source: Name of data source ('naep', 'census', etc.)
            state_utils: StateUtils instance
            
        Returns:
            Appropriate parser instance
            
        Raises:
            ValueError: If data source is not supported
        """
        parsers = {
            'naep': NAEPResponseParser,
            'census': CensusResponseParser,
        }
        
        if data_source.lower() not in parsers:
            raise ValueError(f"Unsupported data source: {data_source}")
            
        return parsers[data_source.lower()](state_utils)
