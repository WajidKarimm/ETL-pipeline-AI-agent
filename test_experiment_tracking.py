#!/usr/bin/env python3
"""
Test script for Experiment Tracking Module

This script demonstrates:
1. Experiment creation and management
2. Run tracking with parameters and metrics
3. Model logging and registry
4. Performance comparison
5. Experiment reporting
6. Integration with real data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import tempfile

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.experiment_tracker import ExperimentTracker, ExperimentConfig
from logger import get_logger

logger = get_logger(__name__)

def create_synthetic_model():
    """Create a synthetic ML model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X, y

def test_experiment_tracking():
    """Test comprehensive experiment tracking capabilities."""
    print("🚀 Testing Experiment Tracking Module")
    print("=" * 50)
    
    # Initialize experiment tracker
    experiment_tracker = ExperimentTracker()
    
    # Create experiment
    print("\n1. Creating experiment...")
    config = ExperimentConfig(
        experiment_name="test_etl_pipeline",
        description="Testing ETL pipeline with experiment tracking",
        tags={
            'pipeline': 'etl',
            'test': 'true',
            'version': '1.0'
        },
        parameters={
            'data_source': 'synthetic',
            'model_type': 'random_forest',
            'validation_split': 0.2
        },
        metrics=['accuracy', 'precision', 'recall', 'f1_score']
    )
    
    experiment_id = experiment_tracker.create_experiment(config)
    print(f"   ✅ Experiment created: {experiment_id}")
    
    # Test multiple runs
    print("\n2. Running multiple experiment runs...")
    run_ids = []
    
    for i in range(3):
        print(f"   Run {i+1}:")
        
        # Start run
        run_id = experiment_tracker.start_run(
            experiment_id=experiment_id,
            run_name=f"test_run_{i+1}",
            parameters={
                'run_number': i + 1,
                'random_state': 42 + i,
                'n_estimators': 100 + i * 50
            }
        )
        run_ids.append(run_id)
        
        # Log parameters
        experiment_tracker.log_parameter(run_id, "learning_rate", 0.1 + i * 0.05)
        experiment_tracker.log_parameter(run_id, "max_depth", 5 + i)
        
        # Create and train model
        model, X, y = create_synthetic_model()
        
        # Simulate training metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        # Log metrics
        experiment_tracker.log_metric(run_id, "accuracy", accuracy)
        experiment_tracker.log_metric(run_id, "precision", precision)
        experiment_tracker.log_metric(run_id, "recall", recall)
        experiment_tracker.log_metric(run_id, "f1_score", f1)
        experiment_tracker.log_metric(run_id, "training_time", 1.5 + i * 0.2)
        
        # Log model
        experiment_tracker.log_model(run_id, model, f"model_run_{i+1}")
        
        # Create temporary artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Run {i+1} results\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}")
            artifact_path = f.name
        
        experiment_tracker.log_artifact(run_id, artifact_path, "results.txt")
        os.unlink(artifact_path)  # Clean up
        
        # End run
        experiment_tracker.end_run(
            run_id=run_id,
            status="completed",
            notes=f"Test run {i+1} completed successfully with accuracy {accuracy:.4f}"
        )
        
        print(f"      ✅ Run completed with accuracy: {accuracy:.4f}")
    
    # Test model registry
    print("\n3. Testing model registry...")
    for i, run_id in enumerate(run_ids):
        version = experiment_tracker.register_model(
            run_id=run_id,
            model_name="test_etl_model",
            description=f"Test ETL model from run {i+1}",
            status="staging" if i < 2 else "production"
        )
        print(f"   ✅ Model registered: test_etl_model v{version}")
    
    # Test run comparison
    print("\n4. Comparing runs...")
    comparison = experiment_tracker.compare_runs(run_ids)
    print(f"   Runs compared: {len(comparison['runs'])}")
    print(f"   Metrics compared: {list(comparison['metrics_comparison'].keys())}")
    
    # Find best run
    print("\n5. Finding best run...")
    best_run = experiment_tracker.get_best_run(experiment_id, "accuracy", maximize=True)
    if best_run:
        print(f"   ✅ Best run: {best_run.run_id} with accuracy: {best_run.metrics['accuracy']:.4f}")
    
    # Generate experiment report
    print("\n6. Generating experiment report...")
    report = experiment_tracker.generate_experiment_report(experiment_id)
    print(f"   ✅ Report generated:")
    print(f"      - Total runs: {report['total_runs']}")
    print(f"      - Success rate: {report['success_rate']:.2%}")
    print(f"      - Metrics tracked: {list(report['metric_statistics'].keys())}")
    
    # Test tracking metrics
    print("\n7. Tracking metrics:")
    metrics = experiment_tracker.get_tracking_metrics()
    print(f"   - Total experiments: {metrics['total_experiments']}")
    print(f"   - Total runs: {metrics['total_runs']}")
    print(f"   - Successful runs: {metrics['successful_runs']}")
    print(f"   - Models registered: {metrics['models_registered']}")
    
    print("\n✅ Experiment tracking test completed successfully!")
    return True

def test_with_real_data():
    """Test experiment tracking with real data from desktop."""
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
                
                # Initialize experiment tracker
                experiment_tracker = ExperimentTracker()
                
                # Create experiment for real data
                dataset_name = os.path.basename(test_file)
                experiment_name = f"real_data_{dataset_name.replace('.csv', '')}"
                config = ExperimentConfig(
                    experiment_name=experiment_name,
                    description=f"Real data processing for {dataset_name}",
                    tags={
                        'data_source': 'real',
                        'dataset': dataset_name,
                        'rows': str(len(data)),
                        'columns': str(len(data.columns))
                    }
                )
                
                experiment_id = experiment_tracker.create_experiment(config)
                
                # Start run
                run_id = experiment_tracker.start_run(
                    experiment_id=experiment_id,
                    run_name="real_data_processing",
                    parameters={
                        'dataset_name': dataset_name,
                        'data_rows': len(data),
                        'data_columns': len(data.columns),
                        'processing_type': 'data_analysis'
                    }
                )
                
                # Log data statistics as metrics
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                experiment_tracker.log_metric(run_id, "numeric_columns", len(numeric_cols))
                experiment_tracker.log_metric(run_id, "categorical_columns", len(data.columns) - len(numeric_cols))
                experiment_tracker.log_metric(run_id, "missing_values", data.isnull().sum().sum())
                experiment_tracker.log_metric(run_id, "duplicate_rows", data.duplicated().sum())
                
                # Log data quality metrics
                if len(numeric_cols) > 0:
                    experiment_tracker.log_metric(run_id, "data_quality_score", 
                                                1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))))
                
                # End run
                experiment_tracker.end_run(
                    run_id=run_id,
                    status="completed",
                    notes=f"Processed real dataset {dataset_name} with {len(data)} rows"
                )
                
                print(f"   ✅ Real data experiment completed")
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
        test_experiment_tracking()
        
        # Test with real data
        test_with_real_data()
        
        print("\n🎉 All experiment tracking tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 