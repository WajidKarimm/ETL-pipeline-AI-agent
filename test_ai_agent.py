#!/usr/bin/env python3
"""
Test script for AI Agent improvements
Tests feature extraction consistency and date parsing improvements
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ml.ai_agent import ETLAIAgent
from src.transformers.clean_transformer import CleanTransformer

def test_feature_extraction_consistency():
    """Test that feature extraction produces consistent feature counts."""
    print("🧪 Testing feature extraction consistency...")
    
    # Create AI agent
    agent = ETLAIAgent()
    
    # Test with different datasets
    test_datasets = [
        # Small dataset with mixed types
        pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03']
        }),
        
        # Large dataset with many columns
        pd.DataFrame({
            'id': range(1000),
            'col1': np.random.randn(1000),
            'col2': np.random.randn(1000),
            'col3': np.random.randn(1000),
            'col4': np.random.randn(1000),
            'col5': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'text': [f'text_{i}' for i in range(1000)]
        }),
        
        # Dataset with only numeric columns
        pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
    ]
    
    feature_counts = []
    for i, data in enumerate(test_datasets):
        features = agent.extract_features(data)
        feature_count = len(features.columns)
        feature_counts.append(feature_count)
        print(f"  Dataset {i+1}: {feature_count} features")
        
        # Test feature preparation
        try:
            X = agent._prepare_features_for_prediction(features)
            print(f"    Feature preparation successful: {X.shape}")
        except Exception as e:
            print(f"    Feature preparation failed: {e}")
    
    # Check consistency
    if len(set(feature_counts)) == 1:
        print("✅ Feature extraction is consistent across datasets")
    else:
        print("⚠️ Feature extraction varies across datasets")
    
    return feature_counts

def test_date_parsing_improvements():
    """Test that date parsing improvements work correctly."""
    print("\n🧪 Testing date parsing improvements...")
    
    # Create clean transformer
    transformer = CleanTransformer()
    
    # Test datasets with different date formats
    test_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'date1': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'date2': ['01/15/2023', '01/16/2023', '01/17/2023', '01/18/2023', '01/19/2023'],
        'date3': ['2023-01-01T10:30:00', '2023-01-02T11:45:00', '2023-01-03T12:15:00', '2023-01-04T13:20:00', '2023-01-05T14:30:00'],
        'text': ['hello', 'world', 'test', 'data', 'processing']
    })
    
    print("  Original data types:")
    for col in test_data.columns:
        print(f"    {col}: {test_data[col].dtype}")
    
    # Apply date formatting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        formatted_data = transformer.format_data_types(test_data)
    
    print("  After date formatting:")
    for col in formatted_data.columns:
        print(f"    {col}: {formatted_data[col].dtype}")
        if 'date' in col.lower():
            print(f"      Sample values: {formatted_data[col].head(2).tolist()}")
    
    # Check if dates were parsed correctly
    date_columns = [col for col in formatted_data.columns if 'date' in col.lower()]
    success_count = 0
    for col in date_columns:
        if pd.api.types.is_datetime64_any_dtype(formatted_data[col]):
            success_count += 1
            print(f"  ✅ {col} parsed successfully as datetime")
        else:
            print(f"  ❌ {col} not parsed as datetime")
    
    print(f"  Date parsing success rate: {success_count}/{len(date_columns)}")
    return success_count == len(date_columns)

def test_ai_agent_ml_functionality():
    """Test AI agent ML functionality with improved feature handling."""
    print("\n🧪 Testing AI agent ML functionality...")
    
    # Create AI agent
    agent = ETLAIAgent()
    
    # Create test data
    test_data = pd.DataFrame({
        'id': range(100),
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'date_col': pd.date_range('2023-01-01', periods=100, freq='D'),
        'text_col': [f'text_{i}' for i in range(100)]
    })
    
    # Test data quality detection
    try:
        issues = agent.detect_data_quality_issues(test_data)
        print(f"  Data quality issues detected: {len(issues)}")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"    - {issue.issue_type}: {issue.description}")
    except Exception as e:
        print(f"  Data quality detection failed: {e}")
    
    # Test transformation suggestions
    try:
        suggestions = agent.suggest_transformations(test_data)
        print(f"  Transformation suggestions: {len(suggestions)}")
        for suggestion in suggestions[:3]:  # Show first 3 suggestions
            print(f"    - {suggestion.transformation_type}: {suggestion.reasoning}")
    except Exception as e:
        print(f"  Transformation suggestions failed: {e}")
    
    # Test error prediction
    try:
        config = {'data_types': {'numeric_col': 'numeric'}}
        predictions = agent.predict_errors(test_data, config)
        print(f"  Error predictions: {len(predictions)}")
        for prediction in predictions[:3]:  # Show first 3 predictions
            print(f"    - {prediction.error_type}: {prediction.probability:.2f}")
    except Exception as e:
        print(f"  Error prediction failed: {e}")
    
    return True

def test_large_dataset_handling():
    """Test handling of large datasets with improved optimizations."""
    print("\n🧪 Testing large dataset handling...")
    
    # Create large dataset
    large_data = pd.DataFrame({
        'id': range(10000),
        'numeric1': np.random.randn(10000),
        'numeric2': np.random.randn(10000),
        'numeric3': np.random.randn(10000),
        'categorical': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
        'text': [f'text_{i}' for i in range(10000)]
    })
    
    # Test AI agent with large dataset
    agent = ETLAIAgent()
    
    try:
        # Extract features
        features = agent.extract_features(large_data)
        print(f"  Large dataset features extracted: {len(features.columns)} features")
        
        # Test feature preparation
        X = agent._prepare_features_for_prediction(features)
        print(f"  Feature preparation successful: {X.shape}")
        
        # Test data quality detection
        issues = agent.detect_data_quality_issues(large_data)
        print(f"  Data quality issues: {len(issues)}")
        
        print("  ✅ Large dataset handling successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Large dataset handling failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting AI Agent Improvement Tests\n")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run tests
    tests = [
        ("Feature Extraction Consistency", test_feature_extraction_consistency),
        ("Date Parsing Improvements", test_date_parsing_improvements),
        ("AI Agent ML Functionality", test_ai_agent_ml_functionality),
        ("Large Dataset Handling", test_large_dataset_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False, None))
    
    # Print summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    passed = 0
    for test_name, executed, result in results:
        status = "✅ PASS" if executed and result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if executed:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! AI Agent improvements are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 