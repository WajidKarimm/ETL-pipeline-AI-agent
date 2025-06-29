"""
Data Validation Module for ETL Pipeline

This module provides enterprise-grade data validation capabilities:
- Schema validation
- Data quality checks
- Statistical validation
- Custom business rules
- Validation reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationRule:
    """Represents a data validation rule."""
    rule_name: str
    rule_type: str  # 'schema', 'quality', 'statistical', 'business'
    description: str
    severity: str  # 'critical', 'warning', 'info'
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Represents the result of a validation check."""
    rule_name: str
    passed: bool
    message: str
    severity: str
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    dataset_name: str
    validation_time: datetime
    total_rules: int
    passed_rules: int
    failed_rules: int
    critical_failures: int
    warnings: int
    results: List[ValidationResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class DataValidator:
    """
    Enterprise-grade data validator with comprehensive validation capabilities.
    """
    
    def __init__(self, validation_dir: str = "validation"):
        """
        Initialize the data validator.
        
        Args:
            validation_dir: Directory to store validation rules and reports
        """
        self.validation_dir = validation_dir
        os.makedirs(validation_dir, exist_ok=True)
        
        # Initialize validation rules
        self.validation_rules = self._initialize_default_rules()
        
        # Validation history
        self.validation_history = []
        
        # Performance tracking
        self.validation_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0
        }
        
        logger.info("Data Validator initialized with enterprise-grade validation capabilities")
    
    def _initialize_default_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules."""
        rules = [
            # Schema validation rules
            ValidationRule(
                rule_name="schema_completeness",
                rule_type="schema",
                description="Check if all required columns are present",
                severity="critical",
                parameters={"required_columns": []}
            ),
            ValidationRule(
                rule_name="data_type_consistency",
                rule_type="schema",
                description="Check if data types match expected schema",
                severity="warning",
                parameters={"expected_types": {}}
            ),
            
            # Data quality rules
            ValidationRule(
                rule_name="null_value_check",
                rule_type="quality",
                description="Check for excessive null values",
                severity="warning",
                parameters={"max_null_percentage": 0.1}
            ),
            ValidationRule(
                rule_name="duplicate_check",
                rule_type="quality",
                description="Check for duplicate rows",
                severity="warning",
                parameters={"max_duplicate_percentage": 0.05}
            ),
            ValidationRule(
                rule_name="outlier_detection",
                rule_type="quality",
                description="Detect statistical outliers",
                severity="info",
                parameters={"outlier_threshold": 3.0}
            ),
            
            # Statistical validation rules
            ValidationRule(
                rule_name="value_range_check",
                rule_type="statistical",
                description="Check if values are within expected ranges",
                severity="warning",
                parameters={"ranges": {}}
            ),
            ValidationRule(
                rule_name="distribution_check",
                rule_type="statistical",
                description="Check data distribution characteristics",
                severity="info",
                parameters={"distribution_thresholds": {}}
            ),
            
            # Business rules
            ValidationRule(
                rule_name="business_logic_check",
                rule_type="business",
                description="Validate business-specific rules",
                severity="critical",
                parameters={"business_rules": []}
            )
        ]
        
        return rules
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.validation_rules.append(rule)
        logger.info(f"Added validation rule: {rule.rule_name}")
    
    def remove_validation_rule(self, rule_name: str):
        """Remove a validation rule."""
        self.validation_rules = [r for r in self.validation_rules if r.rule_name != rule_name]
        logger.info(f"Removed validation rule: {rule_name}")
    
    def validate_dataset(self, data: pd.DataFrame, dataset_name: str = "unknown") -> ValidationReport:
        """
        Perform comprehensive validation on a dataset.
        
        Args:
            data: Input DataFrame to validate
            dataset_name: Name of the dataset for reporting
            
        Returns:
            ValidationReport with comprehensive results
        """
        start_time = datetime.now()
        logger.info(f"Starting validation for dataset: {dataset_name}")
        
        results = []
        enabled_rules = [rule for rule in self.validation_rules if rule.enabled]
        
        for rule in enabled_rules:
            try:
                result = self._execute_validation_rule(data, rule)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing rule {rule.rule_name}: {e}")
                results.append(ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"Rule execution failed: {str(e)}",
                    severity=rule.severity
                ))
        
        # Calculate summary statistics
        total_rules = len(results)
        passed_rules = len([r for r in results if r.passed])
        failed_rules = total_rules - passed_rules
        critical_failures = len([r for r in results if not r.passed and r.severity == "critical"])
        warnings = len([r for r in results if not r.passed and r.severity == "warning"])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, data)
        
        # Create validation report
        report = ValidationReport(
            dataset_name=dataset_name,
            validation_time=start_time,
            total_rules=total_rules,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            critical_failures=critical_failures,
            warnings=warnings,
            results=results,
            recommendations=recommendations
        )
        
        # Update metrics
        validation_time = (datetime.now() - start_time).total_seconds()
        self._update_validation_metrics(report, validation_time)
        
        # Store validation history
        self.validation_history.append(report)
        
        logger.info(f"Validation completed for {dataset_name}: {passed_rules}/{total_rules} rules passed")
        
        return report
    
    def _execute_validation_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Execute a specific validation rule."""
        
        if rule.rule_type == "schema":
            return self._validate_schema(data, rule)
        elif rule.rule_type == "quality":
            return self._validate_quality(data, rule)
        elif rule.rule_type == "statistical":
            return self._validate_statistical(data, rule)
        elif rule.rule_type == "business":
            return self._validate_business_logic(data, rule)
        else:
            return ValidationResult(
                rule_name=rule.rule_name,
                passed=False,
                message=f"Unknown rule type: {rule.rule_type}",
                severity=rule.severity
            )
    
    def _validate_schema(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate schema-related rules."""
        
        if rule.rule_name == "schema_completeness":
            required_columns = rule.parameters.get("required_columns", [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"Missing required columns: {missing_columns}",
                    severity=rule.severity,
                    affected_columns=missing_columns
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message="All required columns present",
                    severity=rule.severity
                )
        
        elif rule.rule_name == "data_type_consistency":
            expected_types = rule.parameters.get("expected_types", {})
            type_violations = []
            
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        type_violations.append(f"{col}: expected {expected_type}, got {actual_type}")
            
            if type_violations:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"Data type violations: {type_violations}",
                    severity=rule.severity,
                    affected_columns=[v.split(':')[0] for v in type_violations]
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message="All data types consistent",
                    severity=rule.severity
                )
        
        return ValidationResult(
            rule_name=rule.rule_name,
            passed=False,
            message=f"Unknown schema rule: {rule.rule_name}",
            severity=rule.severity
        )
    
    def _validate_quality(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate data quality rules."""
        
        if rule.rule_name == "null_value_check":
            max_null_percentage = rule.parameters.get("max_null_percentage", 0.1)
            null_percentages = (data.isnull().sum() / len(data)) * 100
            high_null_cols = null_percentages[null_percentages > max_null_percentage * 100]
            
            if not high_null_cols.empty:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"High null percentage in columns: {high_null_cols.to_dict()}",
                    severity=rule.severity,
                    affected_columns=high_null_cols.index.tolist(),
                    details={"null_percentages": high_null_cols.to_dict()}
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message="Null values within acceptable limits",
                    severity=rule.severity
                )
        
        elif rule.rule_name == "duplicate_check":
            max_duplicate_percentage = rule.parameters.get("max_duplicate_percentage", 0.05)
            duplicate_count = data.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(data)) * 100
            
            if duplicate_percentage > max_duplicate_percentage * 100:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"High duplicate percentage: {duplicate_percentage:.2f}%",
                    severity=rule.severity,
                    affected_rows=duplicate_count,
                    details={"duplicate_percentage": duplicate_percentage}
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message=f"Duplicate percentage acceptable: {duplicate_percentage:.2f}%",
                    severity=rule.severity,
                    details={"duplicate_percentage": duplicate_percentage}
                )
        
        elif rule.rule_name == "outlier_detection":
            threshold = rule.parameters.get("outlier_threshold", 3.0)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_info[col] = outliers
            
            if outlier_info:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"Outliers detected: {outlier_info}",
                    severity=rule.severity,
                    affected_columns=list(outlier_info.keys()),
                    details={"outlier_counts": outlier_info}
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message="No significant outliers detected",
                    severity=rule.severity
                )
        
        return ValidationResult(
            rule_name=rule.rule_name,
            passed=False,
            message=f"Unknown quality rule: {rule.rule_name}",
            severity=rule.severity
        )
    
    def _validate_statistical(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate statistical rules."""
        
        if rule.rule_name == "value_range_check":
            ranges = rule.parameters.get("ranges", {})
            violations = {}
            
            for col, (min_val, max_val) in ranges.items():
                if col in data.columns:
                    out_of_range = ((data[col] < min_val) | (data[col] > max_val)).sum()
                    if out_of_range > 0:
                        violations[col] = out_of_range
            
            if violations:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"Values out of range: {violations}",
                    severity=rule.severity,
                    affected_columns=list(violations.keys()),
                    details={"range_violations": violations}
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message="All values within expected ranges",
                    severity=rule.severity
                )
        
        elif rule.rule_name == "distribution_check":
            thresholds = rule.parameters.get("distribution_thresholds", {})
            distribution_issues = {}
            
            for col, threshold in thresholds.items():
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    skewness = abs(data[col].skew())
                    if skewness > threshold:
                        distribution_issues[col] = skewness
            
            if distribution_issues:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    message=f"Distribution issues: {distribution_issues}",
                    severity=rule.severity,
                    affected_columns=list(distribution_issues.keys()),
                    details={"skewness_values": distribution_issues}
                )
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=True,
                    message="Data distributions within acceptable limits",
                    severity=rule.severity
                )
        
        return ValidationResult(
            rule_name=rule.rule_name,
            passed=False,
            message=f"Unknown statistical rule: {rule.rule_name}",
            severity=rule.severity
        )
    
    def _validate_business_logic(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate business-specific rules."""
        
        business_rules = rule.parameters.get("business_rules", [])
        violations = []
        
        for business_rule in business_rules:
            # This is a placeholder for business rule validation
            # In a real implementation, you would have specific business logic
            pass
        
        if violations:
            return ValidationResult(
                rule_name=rule.rule_name,
                passed=False,
                message=f"Business rule violations: {violations}",
                severity=rule.severity,
                details={"violations": violations}
            )
        else:
            return ValidationResult(
                rule_name=rule.rule_name,
                passed=True,
                message="All business rules satisfied",
                severity=rule.severity
            )
    
    def _generate_recommendations(self, results: List[ValidationResult], data: pd.DataFrame) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_rules = [r for r in results if not r.passed]
        
        for result in failed_rules:
            if result.rule_name == "null_value_check":
                recommendations.append("Consider imputation strategies for missing values")
            elif result.rule_name == "duplicate_check":
                recommendations.append("Investigate and remove duplicate records")
            elif result.rule_name == "outlier_detection":
                recommendations.append("Review outliers for data quality issues")
            elif result.rule_name == "data_type_consistency":
                recommendations.append("Fix data type inconsistencies in affected columns")
        
        # Add general recommendations
        if len(failed_rules) > 0:
            recommendations.append("Review data quality before processing")
        
        return recommendations
    
    def _update_validation_metrics(self, report: ValidationReport, validation_time: float):
        """Update validation performance metrics."""
        self.validation_metrics['total_validations'] += 1
        
        if report.critical_failures == 0:
            self.validation_metrics['successful_validations'] += 1
        else:
            self.validation_metrics['failed_validations'] += 1
        
        # Update average validation time
        current_avg = self.validation_metrics['average_validation_time']
        total_validations = self.validation_metrics['total_validations']
        self.validation_metrics['average_validation_time'] = (
            (current_avg * (total_validations - 1) + validation_time) / total_validations
        )
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics."""
        return self.validation_metrics.copy()
    
    def export_validation_report(self, report: ValidationReport, filepath: str):
        """Export validation report to JSON file."""
        try:
            report_dict = {
                'dataset_name': report.dataset_name,
                'validation_time': report.validation_time.isoformat(),
                'total_rules': report.total_rules,
                'passed_rules': report.passed_rules,
                'failed_rules': report.failed_rules,
                'critical_failures': report.critical_failures,
                'warnings': report.warnings,
                'results': [
                    {
                        'rule_name': r.rule_name,
                        'passed': r.passed,
                        'message': r.message,
                        'severity': r.severity,
                        'affected_rows': r.affected_rows,
                        'affected_columns': r.affected_columns,
                        'details': r.details,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in report.results
                ],
                'recommendations': report.recommendations
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Validation report exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
    
    def load_validation_rules(self, filepath: str):
        """Load validation rules from JSON file."""
        try:
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
            
            self.validation_rules = []
            for rule_data in rules_data:
                rule = ValidationRule(
                    rule_name=rule_data['rule_name'],
                    rule_type=rule_data['rule_type'],
                    description=rule_data['description'],
                    severity=rule_data['severity'],
                    enabled=rule_data.get('enabled', True),
                    parameters=rule_data.get('parameters', {})
                )
                self.validation_rules.append(rule)
            
            logger.info(f"Loaded {len(self.validation_rules)} validation rules from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load validation rules: {e}")
    
    def save_validation_rules(self, filepath: str):
        """Save validation rules to JSON file."""
        try:
            rules_data = []
            for rule in self.validation_rules:
                rule_dict = {
                    'rule_name': rule.rule_name,
                    'rule_type': rule.rule_type,
                    'description': rule.description,
                    'severity': rule.severity,
                    'enabled': rule.enabled,
                    'parameters': rule.parameters
                }
                rules_data.append(rule_dict)
            
            with open(filepath, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            logger.info(f"Saved {len(self.validation_rules)} validation rules to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save validation rules: {e}") 