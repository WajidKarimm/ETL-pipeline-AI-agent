"""
Test script to demonstrate the ETL pipeline functionality.
This script creates sample data and runs the pipeline components.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.extractors.csv_extractor import CSVExtractor
from src.transformers.clean_transformer import CleanTransformer
from src.logger import get_logger

def create_sample_data():
    """Create sample CSV data for testing."""
    data = {
        'region': ['North', 'South', 'East', 'West', 'North', None],
        'value': [100, 200, 150, 300, 250, None],
        'old_column_name': ['A', 'B', 'C', 'D', 'E', 'F'],
        'category': ['High', 'Low', 'Medium', 'High', 'Low', None]
    }
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/sample_data.csv', index=False)
    print("Sample data created at data/sample_data.csv")
    return df

def test_etl_pipeline():
    """Test the ETL pipeline components."""
    logger = get_logger("test_pipeline")
    
    # Create sample data
    original_data = create_sample_data()
    print(f"Original data shape: {original_data.shape}")
    print(f"Original data:\n{original_data}")
    
    # 1. Extract
    print("\n=== EXTRACTING DATA ===")
    extractor_config = {'csv_options': {'encoding': 'utf-8'}}
    extractor = CSVExtractor(extractor_config)
    extracted_data = extractor.extract('data/sample_data.csv')
    print(f"Extracted data shape: {extracted_data.shape}")
    
    # 2. Transform
    print("\n=== TRANSFORMING DATA ===")
    transform_config = {
        'field_map': {
            'region': {'North': 1, 'South': 2, 'East': 3, 'West': 4}
        },
        'rename_map': {
            'old_column_name': 'new_column_name'
        },
        'dropna_axis': 0,
        'dropna_how': 'any'
    }
    
    transformer = CleanTransformer(transform_config)
    transformed_data = transformer.transform(extracted_data)
    print(f"Transformed data shape: {transformed_data.shape}")
    print(f"Transformed data:\n{transformed_data}")
    
    # 3. Show results
    print("\n=== RESULTS ===")
    print(f"Original rows: {len(original_data)}")
    print(f"After transformation: {len(transformed_data)}")
    print(f"Null values removed: {len(original_data) - len(transformed_data)}")
    print(f"Columns after renaming: {list(transformed_data.columns)}")
    
    # Show region mapping
    if 'region' in transformed_data.columns:
        print(f"Region mapping applied: {transformed_data['region'].unique()}")
    
    logger.info("ETL pipeline test completed successfully")
    return transformed_data

if __name__ == "__main__":
    print("Testing ETL Pipeline Components")
    print("=" * 40)
    
    try:
        result = test_etl_pipeline()
        print("\n✅ ETL pipeline test completed successfully!")
        print(f"Final data shape: {result.shape}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc() 