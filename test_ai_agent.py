"""
Test script to demonstrate AI Agent capabilities
"""

import pandas as pd
import numpy as np
from src.ml.ai_agent import ETLAIAgent

def test_ai_agent():
    """Test the AI agent with various data scenarios."""
    
    print("🤖 Testing ETL AI Agent")
    print("=" * 50)
    
    # Initialize AI agent
    ai_agent = ETLAIAgent()
    
    # Test 1: Clean data
    print("\n📊 Test 1: Clean Data")
    clean_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    issues = ai_agent.detect_data_quality_issues(clean_data)
    suggestions = ai_agent.suggest_transformations(clean_data)
    
    print(f"Data Quality Issues: {len(issues)}")
    print(f"Transformation Suggestions: {len(suggestions)}")
    
    if issues:
        for issue in issues:
            print(f"  - {issue.issue_type}: {issue.description}")
    
    if suggestions:
        for suggestion in suggestions:
            print(f"  - {suggestion.transformation_type}: {suggestion.reasoning}")
    
    # Test 2: Data with quality issues
    print("\n📊 Test 2: Data with Quality Issues")
    dirty_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' if i % 10 != 0 else None for i in range(1, 101)],
        'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1, 101)],
        'email': [f'user{i}@example.com' if i % 20 != 0 else None for i in range(1, 101)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    issues = ai_agent.detect_data_quality_issues(dirty_data)
    suggestions = ai_agent.suggest_transformations(dirty_data)
    
    print(f"Data Quality Issues: {len(issues)}")
    print(f"Transformation Suggestions: {len(suggestions)}")
    
    if issues:
        for issue in issues:
            print(f"  - {issue.severity.upper()}: {issue.description}")
            print(f"    Suggested fix: {issue.suggested_fix}")
    
    if suggestions:
        for suggestion in suggestions:
            print(f"  - {suggestion.transformation_type}: {suggestion.reasoning}")
            print(f"    Confidence: {suggestion.confidence:.2f}")
    
    # Test 3: Data with type conversion issues
    print("\n📊 Test 3: Data with Type Conversion Issues")
    type_issue_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': [f'{np.random.randint(18, 80)}' if i % 5 != 0 else 'invalid' for i in range(1, 101)],
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    issues = ai_agent.detect_data_quality_issues(type_issue_data)
    suggestions = ai_agent.suggest_transformations(type_issue_data)
    
    print(f"Data Quality Issues: {len(issues)}")
    print(f"Transformation Suggestions: {len(suggestions)}")
    
    if suggestions:
        for suggestion in suggestions:
            print(f"  - {suggestion.transformation_type}: {suggestion.reasoning}")
            print(f"    Target: {suggestion.target_column}")
    
    # Test 4: Model Performance
    print("\n📊 Test 4: Model Performance")
    performance = ai_agent.get_model_performance()
    print(f"Training Samples: {performance['training_samples']}")
    print(f"Models Trained: {performance['models_trained']}")
    
    # Test 5: Learning from operation
    print("\n📊 Test 5: Learning from Operation")
    
    # Simulate a successful operation
    ai_agent.learn_from_operation(
        data=clean_data,
        transformations={
            'remove_nulls': False,
            'field_map': {},
            'rename_columns': {},
            'data_types': {}
        },
        success=True,
        errors=[]
    )
    
    print("✅ Learned from successful operation")
    
    # Simulate a failed operation
    ai_agent.learn_from_operation(
        data=type_issue_data,
        transformations={
            'remove_nulls': False,
            'field_map': {},
            'rename_columns': {},
            'data_types': {'age': 'numeric'}
        },
        success=False,
        errors=['Type conversion error: Cannot convert column age to numeric']
    )
    
    print("✅ Learned from failed operation")
    
    # Check updated performance
    performance = ai_agent.get_model_performance()
    print(f"Updated Training Samples: {performance['training_samples']}")
    
    print("\n🎉 AI Agent testing completed!")
    print("The AI agent is now ready to assist with your ETL operations!")

if __name__ == "__main__":
    test_ai_agent() 