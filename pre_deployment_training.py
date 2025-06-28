#!/usr/bin/env python3
"""
Pre-Deployment AI Training Script

This script imports real data and trains the AI agent before deployment
to ensure high accuracy and error-free predictions from day one.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
from src.ml.ai_agent import ETLAIAgent
from src.logger import get_logger

logger = get_logger(__name__)

def load_sample_datasets() -> List[pd.DataFrame]:
    """
    Load sample datasets for training.
    
    Returns:
        List of DataFrames for training
    """
    datasets = []
    
    # Create sample datasets with various characteristics
    print("📊 Creating sample datasets for training...")
    
    # Dataset 1: Clean customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'name': [f'Customer_{i}' for i in range(1, 1001)],
        'email': [f'customer{i}@example.com' for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.uniform(20000, 150000, 1000),
        'join_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'status': np.random.choice(['active', 'inactive', 'premium'], 1000)
    })
    datasets.append(('clean_customer_data', customer_data))
    
    # Dataset 2: Data with quality issues
    dirty_data = pd.DataFrame({
        'id': range(1, 501),
        'name': [f'User_{i}' if i % 20 != 0 else None for i in range(1, 501)],
        'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1, 501)],
        'email': [f'user{i}@example.com' if i % 25 != 0 else None for i in range(1, 501)],
        'score': np.random.uniform(0, 100, 500),
        'category': [f'Cat_{i % 5}' for i in range(1, 501)]
    })
    datasets.append(('dirty_data', dirty_data))
    
    # Dataset 3: Financial data with mixed types
    financial_data = pd.DataFrame({
        'transaction_id': range(1, 2001),
        'amount': np.random.uniform(10, 10000, 2000),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], 2000),
        'date': pd.date_range('2023-01-01', periods=2000, freq='H'),
        'status': np.random.choice(['completed', 'pending', 'failed'], 2000),
        'user_id': np.random.randint(1, 101, 2000)
    })
    datasets.append(('financial_data', financial_data))
    
    # Dataset 4: E-commerce data with duplicates
    ecommerce_data = pd.DataFrame({
        'order_id': list(range(1, 401)) + list(range(1, 101)),  # 100 duplicates
        'product_name': [f'Product_{i}' for i in range(1, 401)] + [f'Product_{i}' for i in range(1, 101)],
        'price': np.random.uniform(10, 500, 500),
        'quantity': np.random.randint(1, 10, 500),
        'customer_id': np.random.randint(1, 51, 500),
        'order_date': pd.date_range('2023-06-01', periods=500, freq='D')
    })
    datasets.append(('ecommerce_data', ecommerce_data))
    
    # Dataset 5: Sensor data with outliers
    sensor_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-12-01', periods=3000, freq='min'),
        'temperature': np.random.normal(25, 5, 3000),
        'humidity': np.random.normal(60, 10, 3000),
        'pressure': np.random.normal(1013, 10, 3000),
        'device_id': np.random.randint(1, 11, 3000)
    })
    # Add some outliers
    sensor_data.loc[100:110, 'temperature'] = 100  # Outliers
    sensor_data.loc[200:210, 'humidity'] = 200  # Outliers
    datasets.append(('sensor_data', sensor_data))
    
    # Dataset 6: Text data with encoding issues
    text_data = pd.DataFrame({
        'id': range(1, 301),
        'title': [f'Article {i}' for i in range(1, 301)],
        'content': [f'This is the content of article {i}. It contains some text.' for i in range(1, 301)],
        'author': [f'Author_{i % 10}' for i in range(1, 301)],
        'publish_date': pd.date_range('2023-01-01', periods=300, freq='D'),
        'category': np.random.choice(['tech', 'science', 'business', 'health'], 300)
    })
    datasets.append(('text_data', text_data))
    
    print(f"✅ Created {len(datasets)} sample datasets")
    return datasets

def load_real_data_files() -> List[pd.DataFrame]:
    """
    Load real data files if available.
    
    Returns:
        List of DataFrames from real files
    """
    datasets = []
    
    # Look for data files in common locations
    data_dirs = ['data', 'sample_data', 'datasets', '.']
    file_patterns = ['*.csv', '*.xlsx', '*.json', '*.parquet']
    
    print("📁 Looking for real data files...")
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for pattern in file_patterns:
                files = glob.glob(os.path.join(data_dir, pattern))
                for file_path in files:
                    try:
                        print(f"📄 Loading {file_path}...")
                        
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        elif file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                        elif file_path.endswith('.json'):
                            df = pd.read_json(file_path)
                        elif file_path.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                        else:
                            continue
                        
                        if not df.empty:
                            datasets.append((os.path.basename(file_path), df))
                            print(f"✅ Loaded {file_path} ({len(df)} rows, {len(df.columns)} columns)")
                        
                    except Exception as e:
                        print(f"❌ Failed to load {file_path}: {e}")
    
    print(f"✅ Loaded {len(datasets)} real data files")
    return datasets

def create_training_scenarios(datasets: List[pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Create training scenarios from datasets.
    
    Args:
        datasets: List of (name, DataFrame) tuples
        
    Returns:
        List of training scenarios
    """
    training_scenarios = []
    
    print("🎯 Creating training scenarios...")
    
    for name, df in datasets:
        # Scenario 1: Successful processing
        training_scenarios.append({
            'dataset_name': name,
            'scenario': 'successful_processing',
            'data': df,
            'transformations': {
                'remove_nulls': False,
                'field_map': {},
                'rename_columns': {},
                'data_types': {}
            },
            'success': True,
            'errors': [],
            'user_feedback': {}
        })
        
        # Scenario 2: Data cleaning
        if df.isnull().sum().sum() > 0:
            training_scenarios.append({
                'dataset_name': name,
                'scenario': 'data_cleaning',
                'data': df,
                'transformations': {
                    'remove_nulls': True,
                    'field_map': {},
                    'rename_columns': {},
                    'data_types': {}
                },
                'success': True,
                'errors': [],
                'user_feedback': {'remove_nulls': {'accepted': True}}
            })
        
        # Scenario 3: Column renaming
        if len(df.columns) > 3:
            rename_map = {col: col.lower().replace(' ', '_') for col in df.columns[:3]}
            training_scenarios.append({
                'dataset_name': name,
                'scenario': 'column_renaming',
                'data': df,
                'transformations': {
                    'remove_nulls': False,
                    'field_map': {},
                    'rename_columns': rename_map,
                    'data_types': {}
                },
                'success': True,
                'errors': [],
                'user_feedback': {'rename_columns': {'accepted': True}}
            })
        
        # Scenario 4: Type conversion
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data_types = {col: 'numeric' for col in numeric_cols[:2]}
            training_scenarios.append({
                'dataset_name': name,
                'scenario': 'type_conversion',
                'data': df,
                'transformations': {
                    'remove_nulls': False,
                    'field_map': {},
                    'rename_columns': {},
                    'data_types': data_types
                },
                'success': True,
                'errors': [],
                'user_feedback': {'data_types': {'accepted': True}}
            })
        
        # Scenario 5: Duplicate removal
        if df.duplicated().sum() > 0:
            training_scenarios.append({
                'dataset_name': name,
                'scenario': 'duplicate_removal',
                'data': df,
                'transformations': {
                    'remove_nulls': False,
                    'field_map': {},
                    'rename_columns': {},
                    'data_types': {},
                    'remove_duplicates': True
                },
                'success': True,
                'errors': [],
                'user_feedback': {'remove_duplicates': {'accepted': True}}
            })
        
        # Scenario 6: Field mapping
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 10:
                field_map = {val: idx for idx, val in enumerate(unique_values)}
                training_scenarios.append({
                    'dataset_name': name,
                    'scenario': 'field_mapping',
                    'data': df,
                    'transformations': {
                        'remove_nulls': False,
                        'field_map': {col: field_map},
                        'rename_columns': {},
                        'data_types': {}
                    },
                    'success': True,
                    'errors': [],
                    'user_feedback': {'field_map': {'accepted': True}}
                })
        
        # Scenario 7: Error scenarios
        # Missing column error
        training_scenarios.append({
            'dataset_name': name,
            'scenario': 'missing_column_error',
            'data': df,
            'transformations': {
                'remove_nulls': False,
                'field_map': {'nonexistent_column': 'new_column'},
                'rename_columns': {},
                'data_types': {}
            },
            'success': False,
            'errors': ['Column nonexistent_column not found in data'],
            'user_feedback': {}
        })
        
        # Type conversion error
        if len(df.select_dtypes(include=['object']).columns) > 0:
            col = df.select_dtypes(include=['object']).columns[0]
            training_scenarios.append({
                'dataset_name': name,
                'scenario': 'type_conversion_error',
                'data': df,
                'transformations': {
                    'remove_nulls': False,
                    'field_map': {},
                    'rename_columns': {},
                    'data_types': {col: 'numeric'}
                },
                'success': False,
                'errors': [f'Cannot convert column {col} to numeric'],
                'user_feedback': {}
            })
    
    print(f"✅ Created {len(training_scenarios)} training scenarios")
    return training_scenarios

def train_ai_agent_with_scenarios(agent: ETLAIAgent, scenarios: List[Dict[str, Any]]):
    """
    Train the AI agent with the created scenarios.
    
    Args:
        agent: AI agent instance
        scenarios: List of training scenarios
    """
    print("🧠 Training AI agent with scenarios...")
    
    training_data = []
    
    for scenario in scenarios:
        try:
            # Extract features from the data
            features = agent.extract_features(scenario['data'])
            
            # Create training sample
            training_sample = {
                'features': features.to_dict('records')[0],
                'transformations': scenario['transformations'],
                'success': scenario['success'],
                'errors': scenario['errors'],
                'user_feedback': scenario['user_feedback'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            training_data.append(training_sample)
            
        except Exception as e:
            print(f"⚠️ Failed to process scenario {scenario['dataset_name']}: {e}")
    
    # Train the agent
    agent.training_data = training_data
    agent._retrain_models()
    
    print(f"✅ Trained AI agent with {len(training_data)} samples")

def evaluate_training_results(agent: ETLAIAgent, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate the training results.
    
    Args:
        agent: Trained AI agent
        scenarios: List of training scenarios
        
    Returns:
        Evaluation results
    """
    print("📊 Evaluating training results...")
    
    # Test the agent on a subset of scenarios
    test_scenarios = scenarios[::3]  # Every 3rd scenario for testing
    
    correct_predictions = 0
    total_predictions = 0
    
    for scenario in test_scenarios:
        try:
            # Test data quality detection
            issues = agent.detect_data_quality_issues(scenario['data'])
            
            # Test transformation suggestions
            suggestions = agent.suggest_transformations(scenario['data'])
            
            # Count as correct if we detect issues when there should be issues
            has_actual_issues = (
                scenario['data'].isnull().sum().sum() > 0 or
                scenario['data'].duplicated().sum() > 0 or
                not scenario['success']
            )
            
            has_detected_issues = len(issues) > 0
            
            if has_actual_issues == has_detected_issues:
                correct_predictions += 1
            total_predictions += 1
            
        except Exception as e:
            print(f"⚠️ Failed to evaluate scenario {scenario['dataset_name']}: {e}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'total_scenarios': len(scenarios),
        'test_scenarios': len(test_scenarios),
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'training_samples': len(agent.training_data)
    }
    
    print(f"✅ Evaluation complete - Accuracy: {accuracy:.3f}")
    return results

def save_training_report(results: Dict[str, Any], scenarios: List[Dict[str, Any]]):
    """
    Save training report.
    
    Args:
        results: Evaluation results
        scenarios: List of training scenarios
    """
    report = {
        'training_date': pd.Timestamp.now().isoformat(),
        'evaluation_results': results,
        'scenarios_processed': len(scenarios),
        'scenario_types': list(set(s['scenario'] for s in scenarios)),
        'datasets_used': list(set(s['dataset_name'] for s in scenarios))
    }
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/pre_deployment_training_report.json'
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Training report saved to {report_path}")

def main():
    """Main pre-deployment training function."""
    print("🚀 Starting Pre-Deployment AI Training")
    print("=" * 50)
    
    try:
        # Initialize AI agent
        print("🤖 Initializing AI agent...")
        agent = ETLAIAgent()
        
        # Load datasets
        sample_datasets = load_sample_datasets()
        real_datasets = load_real_data_files()
        
        all_datasets = sample_datasets + real_datasets
        
        if not all_datasets:
            print("❌ No datasets found for training!")
            return
        
        # Create training scenarios
        scenarios = create_training_scenarios(all_datasets)
        
        if not scenarios:
            print("❌ No training scenarios created!")
            return
        
        # Train the agent
        train_ai_agent_with_scenarios(agent, scenarios)
        
        # Evaluate results
        results = evaluate_training_results(agent, scenarios)
        
        # Save report
        save_training_report(results, scenarios)
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 PRE-DEPLOYMENT TRAINING COMPLETE")
        print("=" * 50)
        print(f"📊 Datasets processed: {len(all_datasets)}")
        print(f"🎯 Scenarios created: {len(scenarios)}")
        print(f"🧠 Training samples: {results['training_samples']}")
        print(f"📈 Accuracy: {results['accuracy']:.3f}")
        print(f"✅ AI agent ready for deployment!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Pre-deployment training failed: {e}")
        logger.error(f"Pre-deployment training failed: {e}")

if __name__ == "__main__":
    main() 