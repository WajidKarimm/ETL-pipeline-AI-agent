"""
Drift Monitoring Module for ETL Pipeline

This module provides comprehensive drift detection capabilities:
- Data drift detection (statistical distribution changes)
- Concept drift detection (model performance degradation)
- Feature drift monitoring
- Drift alerts and reporting
- Baseline management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DriftMetric:
    """Represents a drift metric calculation."""
    metric_name: str
    value: float
    threshold: float
    is_drifted: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    dataset_name: str
    timestamp: datetime
    drift_type: str  # 'data', 'concept', 'feature'
    total_features: int
    drifted_features: int
    drift_severity: str  # 'low', 'medium', 'high', 'critical'
    metrics: List[DriftMetric]
    recommendations: List[str]
    baseline_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BaselineData:
    """Represents baseline data for drift comparison."""
    dataset_name: str
    timestamp: datetime
    data_statistics: Dict[str, Any]
    feature_distributions: Dict[str, Any]
    model_performance: Dict[str, float]
    sample_size: int

class DriftMonitor:
    """
    Comprehensive drift monitoring system for ML pipelines.
    """
    
    def __init__(self, monitoring_dir: str = "monitoring"):
        """
        Initialize the drift monitor.
        
        Args:
            monitoring_dir: Directory to store drift monitoring data
        """
        self.monitoring_dir = monitoring_dir
        os.makedirs(monitoring_dir, exist_ok=True)
        os.makedirs(os.path.join(monitoring_dir, "baselines"), exist_ok=True)
        os.makedirs(os.path.join(monitoring_dir, "reports"), exist_ok=True)
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'ks_test': 0.05,  # Kolmogorov-Smirnov test p-value threshold
            'psi_threshold': 0.25,  # Population Stability Index threshold
            'chi_square': 0.05,  # Chi-square test p-value threshold
            'isolation_forest': 0.1,  # Anomaly detection threshold
            'performance_drop': 0.1,  # Model performance degradation threshold
        }
        
        # Baseline data storage
        self.baselines = {}
        
        # Drift history
        self.drift_history = []
        
        # Performance tracking
        self.monitoring_metrics = {
            'total_checks': 0,
            'drift_detected': 0,
            'false_positives': 0,
            'average_drift_score': 0.0
        }
        
        logger.info("Drift Monitor initialized with comprehensive drift detection capabilities")
    
    def set_baseline(self, data: pd.DataFrame, dataset_name: str, 
                    model_performance: Optional[Dict[str, float]] = None) -> BaselineData:
        """
        Set baseline data for drift comparison.
        
        Args:
            data: Baseline dataset
            dataset_name: Name of the dataset
            model_performance: Optional model performance metrics
            
        Returns:
            BaselineData object
        """
        logger.info(f"Setting baseline for dataset: {dataset_name}")
        
        # Calculate data statistics
        data_statistics = self._calculate_data_statistics(data)
        
        # Calculate feature distributions
        feature_distributions = self._calculate_feature_distributions(data)
        
        # Create baseline data
        baseline = BaselineData(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            data_statistics=data_statistics,
            feature_distributions=feature_distributions,
            model_performance=model_performance or {},
            sample_size=len(data)
        )
        
        # Store baseline
        self.baselines[dataset_name] = baseline
        
        # Save baseline to file
        self._save_baseline(baseline)
        
        logger.info(f"Baseline set for {dataset_name} with {len(data)} samples")
        return baseline
    
    def detect_data_drift(self, data: pd.DataFrame, dataset_name: str, baseline_name: Optional[str] = None) -> DriftReport:
        """Detect data drift by comparing current data with baseline."""
        baseline_name = baseline_name or dataset_name
        
        if baseline_name not in self.baselines:
            raise ValueError(f"No baseline found for {baseline_name}")
        
        baseline = self.baselines[baseline_name]
        logger.info(f"Detecting data drift for {dataset_name} against baseline {baseline_name}")
        
        metrics = []
        drifted_features = 0
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in baseline.feature_distributions:
                try:
                    # Convert baseline values back to numpy array
                    baseline_values = np.array(baseline.feature_distributions[col]['values'])
                    current_values = data[col].dropna()
                    
                    if len(current_values) > 0 and len(baseline_values) > 0:
                        # Kolmogorov-Smirnov test for distribution drift
                        ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, current_values)
                        
                        # Population Stability Index
                        psi = self._calculate_psi(
                            np.array(baseline.feature_distributions[col]['histogram']),
                            current_values
                        )
                        
                        # Determine if drifted
                        is_drifted = (ks_pvalue < self.drift_thresholds['ks_test'] or 
                                     psi > self.drift_thresholds['psi_threshold'])
                        
                        if is_drifted:
                            drifted_features += 1
                        
                        metric = DriftMetric(
                            metric_name=f"drift_{col}",
                            value=min(ks_pvalue, psi),
                            threshold=self.drift_thresholds['ks_test'],
                            is_drifted=is_drifted,
                            confidence=1 - min(ks_pvalue, psi),
                            details={
                                'ks_statistic': float(ks_stat),
                                'ks_pvalue': float(ks_pvalue),
                                'psi': float(psi),
                                'feature': col
                            }
                        )
                        metrics.append(metric)
                except Exception as e:
                    logger.warning(f"Error calculating drift for feature {col}: {e}")
                    continue
        
        drift_severity = self._calculate_drift_severity(metrics)
        recommendations = self._generate_drift_recommendations(metrics, drift_severity)
        
        report = DriftReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            drift_type='data',
            total_features=len(numeric_cols),
            drifted_features=drifted_features,
            drift_severity=drift_severity,
            metrics=metrics,
            recommendations=recommendations,
            baseline_info={
                'baseline_name': baseline_name,
                'baseline_timestamp': baseline.timestamp.isoformat(),
                'baseline_sample_size': baseline.sample_size
            }
        )
        
        self._update_monitoring_metrics(report)
        self.drift_history.append(report)
        self._save_drift_report(report)
        
        logger.info(f"Data drift detection completed: {drifted_features}/{len(numeric_cols)} features drifted")
        return report
    
    def detect_concept_drift(self, data: pd.DataFrame, predictions: np.ndarray, 
                           true_labels: np.ndarray, dataset_name: str,
                           baseline_name: Optional[str] = None) -> DriftReport:
        """Detect concept drift by monitoring model performance changes."""
        baseline_name = baseline_name or dataset_name
        
        if baseline_name not in self.baselines:
            raise ValueError(f"No baseline found for {baseline_name}")
        
        baseline = self.baselines[baseline_name]
        logger.info(f"Detecting concept drift for {dataset_name}")
        
        metrics = []
        current_performance = {}
        
        try:
            if len(np.unique(true_labels)) > 1:  # Classification task
                current_performance['accuracy'] = float(accuracy_score(true_labels, predictions))
                try:
                    current_performance['auc'] = float(roc_auc_score(true_labels, predictions))
                except:
                    current_performance['auc'] = 0.5
            
            baseline_performance = baseline.model_performance
            
            for metric_name, current_value in current_performance.items():
                if metric_name in baseline_performance:
                    baseline_value = baseline_performance[metric_name]
                    performance_drop = baseline_value - current_value
                    
                    is_drifted = performance_drop > self.drift_thresholds['performance_drop']
                    
                    metric = DriftMetric(
                        metric_name=f"performance_{metric_name}",
                        value=current_value,
                        threshold=baseline_value * (1 - self.drift_thresholds['performance_drop']),
                        is_drifted=is_drifted,
                        confidence=1 - abs(performance_drop),
                        details={
                            'baseline_value': baseline_value,
                            'performance_drop': performance_drop,
                            'metric': metric_name
                        }
                    )
                    metrics.append(metric)
        except Exception as e:
            logger.error(f"Error in concept drift detection: {e}")
        
        drift_severity = self._calculate_drift_severity(metrics)
        recommendations = self._generate_drift_recommendations(metrics, drift_severity)
        
        report = DriftReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            drift_type='concept',
            total_features=len(metrics),
            drifted_features=len([m for m in metrics if m.is_drifted]),
            drift_severity=drift_severity,
            metrics=metrics,
            recommendations=recommendations,
            baseline_info={
                'baseline_name': baseline_name,
                'baseline_timestamp': baseline.timestamp.isoformat(),
                'baseline_performance': baseline_performance
            }
        )
        
        self._update_monitoring_metrics(report)
        self.drift_history.append(report)
        self._save_drift_report(report)
        
        logger.info(f"Concept drift detection completed: {len([m for m in metrics if m.is_drifted])} metrics drifted")
        return report
    
    def detect_feature_drift(self, data: pd.DataFrame, dataset_name: str,
                           baseline_name: Optional[str] = None) -> DriftReport:
        """Detect feature-level drift using anomaly detection."""
        baseline_name = baseline_name or dataset_name
        
        if baseline_name not in self.baselines:
            raise ValueError(f"No baseline found for {baseline_name}")
        
        baseline = self.baselines[baseline_name]
        logger.info(f"Detecting feature drift for {dataset_name}")
        
        metrics = []
        drifted_features = 0
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in baseline.feature_distributions:
                try:
                    baseline_values = np.array(baseline.feature_distributions[col]['values'])
                    current_values = data[col].dropna()
                    
                    if len(current_values) > 0 and len(baseline_values) > 0:
                        # Use Isolation Forest for anomaly detection
                        iso_forest = IsolationForest(
                            contamination=self.drift_thresholds['isolation_forest'], 
                            random_state=42
                        )
                        iso_forest.fit(baseline_values.reshape(-1, 1))
                        
                        # Predict anomalies in current data
                        current_values_array = current_values.values.reshape(-1, 1)
                        anomaly_scores = iso_forest.decision_function(current_values_array)
                        anomaly_ratio = float(np.mean(anomaly_scores < 0))
                        
                        is_drifted = anomaly_ratio > self.drift_thresholds['isolation_forest']
                        
                        if is_drifted:
                            drifted_features += 1
                        
                        metric = DriftMetric(
                            metric_name=f"anomaly_{col}",
                            value=anomaly_ratio,
                            threshold=self.drift_thresholds['isolation_forest'],
                            is_drifted=is_drifted,
                            confidence=anomaly_ratio,
                            details={
                                'anomaly_ratio': anomaly_ratio,
                                'feature': col,
                                'baseline_mean': float(np.mean(baseline_values)),
                                'current_mean': float(np.mean(current_values))
                            }
                        )
                        metrics.append(metric)
                except Exception as e:
                    logger.warning(f"Error calculating feature drift for {col}: {e}")
                    continue
        
        drift_severity = self._calculate_drift_severity(metrics)
        recommendations = self._generate_drift_recommendations(metrics, drift_severity)
        
        report = DriftReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            drift_type='feature',
            total_features=len(numeric_cols),
            drifted_features=drifted_features,
            drift_severity=drift_severity,
            metrics=metrics,
            recommendations=recommendations,
            baseline_info={
                'baseline_name': baseline_name,
                'baseline_timestamp': baseline.timestamp.isoformat(),
                'baseline_sample_size': baseline.sample_size
            }
        )
        
        self._update_monitoring_metrics(report)
        self.drift_history.append(report)
        self._save_drift_report(report)
        
        logger.info(f"Feature drift detection completed: {drifted_features}/{len(numeric_cols)} features drifted")
        return report
    
    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            'shape': list(data.shape),  # Convert tuple to list
            'memory_usage_mb': float(data.memory_usage(deep=True).sum() / 1024 / 1024),
            'null_counts': data.isnull().sum().to_dict(),
            'null_percentages': (data.isnull().sum() / len(data) * 100).to_dict(),
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},  # Convert to strings
            'numeric_summary': {}
        }
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats['numeric_summary'][col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'skewness': float(data[col].skew()),
                'kurtosis': float(data[col].kurtosis())
            }
        return stats
    
    def _calculate_feature_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        distributions = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 0:
                hist, bins = np.histogram(values, bins=min(20, len(values)//10))
                distributions[col] = {
                    'values': values.tolist(),
                    'histogram': hist.tolist(),
                    'bins': bins.tolist(),
                    'mean': values.mean(),
                    'std': values.std()
                }
        return distributions
    
    def _calculate_psi(self, baseline_hist: np.ndarray, current_values: pd.Series) -> float:
        """Calculate Population Stability Index."""
        try:
            current_hist, _ = np.histogram(current_values, bins=len(baseline_hist))
            baseline_norm = baseline_hist / np.sum(baseline_hist)
            current_norm = current_hist / np.sum(current_hist)
            
            psi = 0
            for i in range(len(baseline_norm)):
                if baseline_norm[i] > 0 and current_norm[i] > 0:
                    psi += (current_norm[i] - baseline_norm[i]) * np.log(current_norm[i] / baseline_norm[i])
            
            return abs(psi)
        except:
            return 0.0
    
    def _calculate_drift_severity(self, metrics: List[DriftMetric]) -> str:
        """Calculate overall drift severity based on metrics."""
        if not metrics:
            return 'low'
        
        drifted_metrics = [m for m in metrics if m.is_drifted]
        drift_ratio = len(drifted_metrics) / len(metrics)
        
        if drift_ratio >= 0.5:
            return 'critical'
        elif drift_ratio >= 0.3:
            return 'high'
        elif drift_ratio >= 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_drift_recommendations(self, metrics: List[DriftMetric], severity: str) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        drifted_metrics = [m for m in metrics if m.is_drifted]
        
        if severity == 'critical':
            recommendations.append("Immediate model retraining required")
            recommendations.append("Investigate data pipeline for issues")
        elif severity == 'high':
            recommendations.append("Consider model retraining")
            recommendations.append("Monitor drift trends closely")
        elif severity == 'medium':
            recommendations.append("Monitor affected features")
            recommendations.append("Consider feature engineering updates")
        
        if drifted_metrics:
            recommendations.append(f"Focus on {len(drifted_metrics)} drifted features")
        
        return recommendations
    
    def _update_monitoring_metrics(self, report: DriftReport):
        """Update monitoring performance metrics."""
        self.monitoring_metrics['total_checks'] += 1
        
        if report.drifted_features > 0:
            self.monitoring_metrics['drift_detected'] += 1
        
        if report.metrics:
            avg_drift_score = np.mean([m.value for m in report.metrics])
            current_avg = self.monitoring_metrics['average_drift_score']
            total_checks = self.monitoring_metrics['total_checks']
            self.monitoring_metrics['average_drift_score'] = (
                (current_avg * (total_checks - 1) + avg_drift_score) / total_checks
            )
    
    def _save_baseline(self, baseline: BaselineData):
        """Save baseline data to file."""
        try:
            baseline_file = os.path.join(self.monitoring_dir, "baselines", f"{baseline.dataset_name}_baseline.json")
            
            baseline_dict = {
                'dataset_name': baseline.dataset_name,
                'timestamp': baseline.timestamp.isoformat(),
                'data_statistics': baseline.data_statistics,
                'feature_distributions': baseline.feature_distributions,
                'model_performance': baseline.model_performance,
                'sample_size': baseline.sample_size
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_dict, f, indent=2)
            
            logger.info(f"Baseline saved to: {baseline_file}")
            
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def _save_drift_report(self, report: DriftReport):
        """Save drift report to file."""
        try:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.monitoring_dir, "reports", 
                                     f"drift_report_{report.dataset_name}_{timestamp}.json")
            
            report_dict = {
                'dataset_name': report.dataset_name,
                'timestamp': report.timestamp.isoformat(),
                'drift_type': report.drift_type,
                'total_features': report.total_features,
                'drifted_features': report.drifted_features,
                'drift_severity': report.drift_severity,
                'metrics': [
                    {
                        'metric_name': m.metric_name,
                        'value': float(m.value),
                        'threshold': float(m.threshold),
                        'is_drifted': bool(m.is_drifted),
                        'confidence': float(m.confidence),
                        'timestamp': m.timestamp.isoformat(),
                        'details': {k: (float(v) if isinstance(v, (np.floating, float)) else 
                                       bool(v) if isinstance(v, (np.bool_, bool)) else v) 
                                  for k, v in m.details.items()}
                    }
                    for m in report.metrics
                ],
                'recommendations': report.recommendations,
                'baseline_info': report.baseline_info
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Drift report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save drift report: {e}")
    
    def load_baseline(self, dataset_name: str) -> Optional[BaselineData]:
        """Load baseline data from file."""
        try:
            baseline_file = os.path.join(self.monitoring_dir, "baselines", f"{dataset_name}_baseline.json")
            
            if not os.path.exists(baseline_file):
                return None
            
            with open(baseline_file, 'r') as f:
                baseline_dict = json.load(f)
            
            baseline = BaselineData(
                dataset_name=baseline_dict['dataset_name'],
                timestamp=datetime.fromisoformat(baseline_dict['timestamp']),
                data_statistics=baseline_dict['data_statistics'],
                feature_distributions=baseline_dict['feature_distributions'],
                model_performance=baseline_dict['model_performance'],
                sample_size=baseline_dict['sample_size']
            )
            
            self.baselines[dataset_name] = baseline
            logger.info(f"Baseline loaded from: {baseline_file}")
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return None
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get monitoring performance metrics."""
        return self.monitoring_metrics.copy()
    
    def get_drift_history(self, dataset_name: Optional[str] = None) -> List[DriftReport]:
        """Get drift detection history."""
        if dataset_name:
            return [report for report in self.drift_history if report.dataset_name == dataset_name]
        return self.drift_history.copy() 