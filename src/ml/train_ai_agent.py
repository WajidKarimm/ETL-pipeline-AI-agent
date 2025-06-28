"""
AI Agent Training Script

This script trains the ETL AI Agent on sample data to improve its accuracy
and error prediction capabilities.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any
from src.ml.ai_agent import ETLAIAgent
from src.logger import get_logger

logger = get_logger(__name__)

def generate_training_data() -> List[Dict[str, Any]]:
    """
    Generate synthetic training data for the AI agent.
    
    Returns:
        List of training samples
    """
    training_data = []
    
    # Sample 1: Clean data (success)
    clean_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    training_data.append({
        'features': {
            'row_count': 100,
            'column_count': 5,
            'memory_usage_mb': clean_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': 3,
            'categorical_columns': 2,
            'datetime_columns': 0,
            'total_missing_values': 0,
            'missing_value_percentage': 0.0,
            'columns_with_missing': 0,
            'duplicate_rows': 0,
            'duplicate_percentage': 0.0
        },
        'transformations': {
            'remove_nulls': False,
            'field_map': {},
            'rename_columns': {},
            'data_types': {}
        },
        'success': True,
        'errors': [],
        'user_feedback': {},
        'timestamp': '2024-01-01T00:00:00'
    })
    
    # Sample 2: Data with missing values (success after cleaning)
    missing_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' if i % 10 != 0 else None for i in range(1, 101)],
        'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1, 101)],
        'email': [f'user{i}@example.com' if i % 20 != 0 else None for i in range(1, 101)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    training_data.append({
        'features': {
            'row_count': 100,
            'column_count': 4,
            'memory_usage_mb': missing_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': 2,
            'categorical_columns': 2,
            'datetime_columns': 0,
            'total_missing_values': missing_data.isnull().sum().sum(),
            'missing_value_percentage': (missing_data.isnull().sum().sum() / (len(missing_data) * len(missing_data.columns))) * 100,
            'columns_with_missing': (missing_data.isnull().sum() > 0).sum(),
            'duplicate_rows': 0,
            'duplicate_percentage': 0.0
        },
        'transformations': {
            'remove_nulls': True,
            'field_map': {},
            'rename_columns': {},
            'data_types': {}
        },
        'success': True,
        'errors': [],
        'user_feedback': {'remove_nulls': {'accepted': True}},
        'timestamp': '2024-01-01T01:00:00'
    })
    
    # Sample 3: Data with duplicates (success after cleaning)
    duplicate_data = pd.DataFrame({
        'id': list(range(1, 91)) + list(range(1, 11)),  # 10 duplicates
        'name': [f'User_{i}' for i in range(1, 91)] + [f'User_{i}' for i in range(1, 11)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'user{i}@example.com' for i in range(1, 91)] + [f'user{i}@example.com' for i in range(1, 11)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    training_data.append({
        'features': {
            'row_count': 100,
            'column_count': 4,
            'memory_usage_mb': duplicate_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': 2,
            'categorical_columns': 2,
            'datetime_columns': 0,
            'total_missing_values': 0,
            'missing_value_percentage': 0.0,
            'columns_with_missing': 0,
            'duplicate_rows': 10,
            'duplicate_percentage': 10.0
        },
        'transformations': {
            'remove_nulls': False,
            'field_map': {},
            'rename_columns': {},
            'data_types': {}
        },
        'success': True,
        'errors': [],
        'user_feedback': {'remove_duplicates': {'accepted': True}},
        'timestamp': '2024-01-01T02:00:00'
    })
    
    # Sample 4: Type conversion error (failure)
    type_error_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': [f'{np.random.randint(18, 80)}' if i % 5 != 0 else 'invalid' for i in range(1, 101)],
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    training_data.append({
        'features': {
            'row_count': 100,
            'column_count': 4,
            'memory_usage_mb': type_error_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': 1,
            'categorical_columns': 3,
            'datetime_columns': 0,
            'total_missing_values': 0,
            'missing_value_percentage': 0.0,
            'columns_with_missing': 0,
            'duplicate_rows': 0,
            'duplicate_percentage': 0.0
        },
        'transformations': {
            'remove_nulls': False,
            'field_map': {},
            'rename_columns': {},
            'data_types': {'age': 'numeric'}
        },
        'success': False,
        'errors': ['Type conversion error: Cannot convert column age to numeric'],
        'user_feedback': {},
        'timestamp': '2024-01-01T03:00:00'
    })
    
    # Sample 5: Missing column error (failure)
    missing_column_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'user{i}@example.com' for i in range(1, 101)]
    })
    
    training_data.append({
        'features': {
            'row_count': 100,
            'column_count': 4,
            'memory_usage_mb': missing_column_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': 2,
            'categorical_columns': 2,
            'datetime_columns': 0,
            'total_missing_values': 0,
            'missing_value_percentage': 0.0,
            'columns_with_missing': 0,
            'duplicate_rows': 0,
            'duplicate_percentage': 0.0
        },
        'transformations': {
            'remove_nulls': False,
            'field_map': {'nonexistent_column': 'new_column'},
            'rename_columns': {},
            'data_types': {}
        },
        'success': False,
        'errors': ['Missing column error: Column nonexistent_column not found'],
        'user_feedback': {},
        'timestamp': '2024-01-01T04:00:00'
    })
    
    # Add more diverse samples
    for i in range(5, 20):
        # Generate random data characteristics
        row_count = np.random.randint(50, 1000)
        col_count = np.random.randint(3, 10)
        missing_pct = np.random.uniform(0, 30)
        duplicate_pct = np.random.uniform(0, 20)
        
        # Simulate success/failure based on data quality
        success = missing_pct < 25 and duplicate_pct < 15
        
        training_data.append({
            'features': {
                'row_count': row_count,
                'column_count': col_count,
                'memory_usage_mb': np.random.uniform(0.1, 10.0),
                'numeric_columns': np.random.randint(1, col_count),
                'categorical_columns': np.random.randint(1, col_count),
                'datetime_columns': np.random.randint(0, 2),
                'total_missing_values': int(row_count * col_count * missing_pct / 100),
                'missing_value_percentage': missing_pct,
                'columns_with_missing': np.random.randint(0, col_count),
                'duplicate_rows': int(row_count * duplicate_pct / 100),
                'duplicate_percentage': duplicate_pct
            },
            'transformations': {
                'remove_nulls': missing_pct > 5,
                'field_map': {},
                'rename_columns': {},
                'data_types': {}
            },
            'success': success,
            'errors': [] if success else ['Data quality issues detected'],
            'user_feedback': {},
            'timestamp': f'2024-01-01T{i:02d}:00:00'
        })
    
    return training_data

def train_ai_agent(agent: ETLAIAgent, training_data: List[Dict[str, Any]]):
    """
    Train the AI agent with provided training data.
    
    Args:
        agent: AI agent instance
        training_data: List of training samples
    """
    # Silent training
    agent.training_data = training_data
    agent._retrain_models()

def evaluate_ai_agent(agent: ETLAIAgent, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate the trained AI agent.
    
    Args:
        agent: Trained AI agent
        test_data: Test samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not hasattr(agent.data_quality_classifier, 'classes_'):
        return {'error': 'Models not trained yet'}
    
    # Prepare test data
    X_test = []
    y_test = []
    
    for sample in test_data:
        features = pd.DataFrame([sample['features']])
        X = agent._prepare_features_for_prediction(features)
        has_issues = len(sample.get('errors', [])) > 0 or not sample['success']
        
        X_test.append(X.flatten())
        y_test.append(1 if has_issues else 0)
    
    if len(X_test) < 2:
        return {'error': 'Insufficient test data'}
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Make predictions
    y_pred = agent.data_quality_classifier.predict(X_test)
    y_proba = agent.data_quality_classifier.predict_proba(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'test_samples': len(test_data)
    }
    
    return metrics

def main():
    """Main training function."""
    # Silent training - only log errors
    try:
        # Initialize AI agent
        agent = ETLAIAgent()
        
        # Generate training data
        training_data = generate_training_data()
        
        # Split into train/test
        train_size = int(0.8 * len(training_data))
        train_data = training_data[:train_size]
        test_data = training_data[train_size:]
        
        # Train the agent silently
        train_ai_agent(agent, train_data)
        
        # Evaluate the agent silently
        metrics = evaluate_ai_agent(agent, test_data)
        
        # Save training report
        report = {
            'training_date': pd.Timestamp.now().isoformat(),
            'training_samples': len(train_data),
            'test_samples': len(test_data),
            'evaluation_metrics': metrics,
            'model_performance': agent.get_model_performance()
        }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        with open('reports/training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Only print summary if running directly
        if __name__ == "__main__":
            print("\n" + "="*50)
            print("AI AGENT TRAINING SUMMARY")
            print("="*50)
            print(f"Training samples: {len(train_data)}")
            print(f"Test samples: {len(test_data)}")
            print(f"Models trained: {sum(agent.get_model_performance()['models_trained'].values())}")
            
            if 'error' not in metrics:
                print(f"Accuracy: {metrics['accuracy']:.3f}")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1 Score: {metrics['f1_score']:.3f}")
            
            print("="*50)
        
    except Exception as e:
        # Only log errors
        logger.error(f"Training failed: {e}")
        if __name__ == "__main__":
            print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main() 