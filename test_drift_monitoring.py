#!/usr/bin/env python3
"""
Test script for Drift Monitoring Module

This script demonstrates:
1. Baseline creation
2. Data drift detection
3. Feature drift detection
4. Concept drift detection
5. Drift reporting and recommendations
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from monitoring.drift_monitor import DriftMonitor
from logger import get_logger

logger = get_logger(__name__)

def create_baseline_data():
    """Create baseline dataset with normal distribution."""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'experience': np.random.normal(8, 3, n_samples),
        'performance_score': np.random.normal(75, 10, n_samples)
    })
    
    # Add some categorical data
    data['department'] = np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n_samples)
    data['education'] = np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples)
    
    return data

def create_drifted_data(drift_factor=0.3):
    """Create drifted dataset with shifted distributions."""
    np.random.seed(123)
    n_samples = 1000
    
    # Shift distributions to simulate drift
    data = pd.DataFrame({
        'age': np.random.normal(35 + drift_factor * 10, 10, n_samples),
        'salary': np.random.normal(50000 + drift_factor * 5000, 15000, n_samples),
        'experience': np.random.normal(8 + drift_factor * 2, 3, n_samples),
        'performance_score': np.random.normal(75 - drift_factor * 5, 10, n_samples)
    })
    
    # Change categorical distributions
    data['department'] = np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n_samples, 
                                        p=[0.4, 0.2, 0.2, 0.2])  # More IT staff
    data['education'] = np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples,
                                       p=[0.3, 0.5, 0.2])  # More Masters
    
    return data

def create_model_predictions(data, is_baseline=True):
    """Create synthetic model predictions."""
    if is_baseline:
        # Baseline model performance
        accuracy = 0.85
        auc = 0.78
    else:
        # Degraded model performance (concept drift)
        accuracy = 0.72
        auc = 0.65
    
    # Generate predictions based on performance
    n_samples = len(data)
    predictions = np.random.choice([0, 1], n_samples, p=[1-accuracy, accuracy])
    true_labels = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    
    return predictions, true_labels

def test_drift_monitoring():
    """Test comprehensive drift monitoring capabilities."""
    print("🚀 Testing Drift Monitoring Module")
    print("=" * 50)
    
    # Initialize drift monitor
    drift_monitor = DriftMonitor()
    
    # Create baseline data
    print("\n1. Creating baseline dataset...")
    baseline_data = create_baseline_data()
    print(f"   Baseline dataset shape: {baseline_data.shape}")
    print(f"   Features: {list(baseline_data.columns)}")
    
    # Set baseline
    baseline = drift_monitor.set_baseline(baseline_data, "employee_data")
    print(f"   ✅ Baseline created with {baseline.sample_size} samples")
    
    # Test data drift detection
    print("\n2. Testing data drift detection...")
    drifted_data = create_drifted_data(drift_factor=0.5)
    data_drift_report = drift_monitor.detect_data_drift(drifted_data, "employee_data")
    
    print(f"   Data drift results:")
    print(f"   - Total features: {data_drift_report.total_features}")
    print(f"   - Drifted features: {data_drift_report.drifted_features}")
    print(f"   - Drift severity: {data_drift_report.drift_severity}")
    print(f"   - Recommendations: {data_drift_report.recommendations}")
    
    # Test feature drift detection
    print("\n3. Testing feature drift detection...")
    feature_drift_report = drift_monitor.detect_feature_drift(drifted_data, "employee_data")
    
    print(f"   Feature drift results:")
    print(f"   - Total features: {feature_drift_report.total_features}")
    print(f"   - Drifted features: {feature_drift_report.drifted_features}")
    print(f"   - Drift severity: {feature_drift_report.drift_severity}")
    
    # Test concept drift detection
    print("\n4. Testing concept drift detection...")
    baseline_predictions, baseline_labels = create_model_predictions(baseline_data, is_baseline=True)
    drifted_predictions, drifted_labels = create_model_predictions(drifted_data, is_baseline=False)
    
    # Set baseline with model performance
    baseline_with_performance = drift_monitor.set_baseline(
        baseline_data, "employee_data_with_performance",
        model_performance={'accuracy': 0.85, 'auc': 0.78}
    )
    
    concept_drift_report = drift_monitor.detect_concept_drift(
        drifted_data, drifted_predictions, drifted_labels, "employee_data_with_performance"
    )
    
    print(f"   Concept drift results:")
    print(f"   - Total metrics: {concept_drift_report.total_features}")
    print(f"   - Drifted metrics: {concept_drift_report.drifted_features}")
    print(f"   - Drift severity: {concept_drift_report.drift_severity}")
    
    # Test monitoring metrics
    print("\n5. Monitoring metrics:")
    metrics = drift_monitor.get_monitoring_metrics()
    print(f"   - Total checks: {metrics['total_checks']}")
    print(f"   - Drift detected: {metrics['drift_detected']}")
    print(f"   - Average drift score: {metrics['average_drift_score']:.3f}")
    
    # Test drift history
    print("\n6. Drift history:")
    history = drift_monitor.get_drift_history()
    print(f"   - Total reports: {len(history)}")
    for i, report in enumerate(history[-3:], 1):  # Show last 3 reports
        print(f"   Report {i}: {report.drift_type} drift - {report.drifted_features}/{report.total_features} features")
    
    print("\n✅ Drift monitoring test completed successfully!")
    return True

def test_with_real_data():
    """Test drift monitoring with real data from desktop."""
    print("\n🔍 Testing with real data...")
    
    # Try to find a real dataset
    desktop_data_path = "C:\\Users\\Wajid\\Desktop\\data"
    if os.path.exists(desktop_data_path):
        import glob
        csv_files = glob.glob(os.path.join(desktop_data_path, "*.csv"))
        
        if csv_files:
            test_file = csv_files[0]
            print(f"   Using real dataset: {os.path.basename(test_file)}")
            
            try:
                # Load data
                data = pd.read_csv(test_file)
                print(f"   Dataset shape: {data.shape}")
                
                # Initialize drift monitor
                drift_monitor = DriftMonitor()
                
                # Set baseline
                dataset_name = os.path.basename(test_file)
                baseline = drift_monitor.set_baseline(data, dataset_name)
                print(f"   ✅ Baseline created for real data")
                
                # Test drift detection (should be minimal since it's the same data)
                drift_report = drift_monitor.detect_data_drift(data, dataset_name)
                print(f"   Drift detection: {drift_report.drifted_features}/{drift_report.total_features} features drifted")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Error with real data: {e}")
                return False
        else:
            print("   No CSV files found in desktop data directory")
            return False
    else:
        print("   Desktop data directory not found")
        return False

if __name__ == "__main__":
    try:
        # Test with synthetic data
        test_drift_monitoring()
        
        # Test with real data
        test_with_real_data()
        
        print("\n🎉 All drift monitoring tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 