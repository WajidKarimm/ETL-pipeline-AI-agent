"""
Data cleaning and transformation logic.
- Remove nulls
- Map fields (e.g., 'region' to numeric ID)
- Rename columns to match target schema
Enhanced with robust error handling and data validation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List

from .base import BaseTransformer
from ..logger import get_logger


class CleanTransformer(BaseTransformer):
    """
    Transformer for cleaning and mapping data with robust error handling.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.field_map = (config or {}).get('field_map', {})
        self.rename_map = (config or {}).get('rename_map', {})
        self.dropna_axis = (config or {}).get('dropna_axis', 0)
        self.dropna_how = (config or {}).get('dropna_how', 'any')
        self.data_validation = (config or {}).get('data_validation', True)
        self.handle_duplicates = (config or {}).get('handle_duplicates', True)

    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data before transformation.
        
        Args:
            data: Input DataFrame
            
        Returns:
            bool: True if data is valid
        """
        if data is None or data.empty:
            self.logger.error("Input data is None or empty")
            return False
        
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Input data is not a pandas DataFrame")
            return False
        
        self.logger.info("Input data validation passed", 
                        rows=len(data), 
                        columns=list(data.columns))
        return True

    def clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing special characters and standardizing format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        cleaned_columns = []
        for col in data.columns:
            # Remove special characters and spaces
            cleaned = str(col).strip()
            cleaned = cleaned.replace(' ', '_').replace('-', '_').replace('.', '_')
            cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
            cleaned = cleaned.lower()
            
            # Ensure column name is not empty
            if not cleaned:
                cleaned = f"column_{len(cleaned_columns)}"
            
            # Ensure uniqueness
            if cleaned in cleaned_columns:
                cleaned = f"{cleaned}_{len(cleaned_columns)}"
            
            cleaned_columns.append(cleaned)
        
        data.columns = cleaned_columns
        self.logger.info("Column names cleaned", 
                        original_columns=list(data.columns),
                        cleaned_columns=cleaned_columns)
        
        return data

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on configuration with conservative approach.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        original_rows = len(data)
        
        if self.dropna_how != 'none':
            # More conservative approach: only remove rows that are mostly empty
            if self.dropna_how == 'any':
                # Instead of removing any row with a null, only remove rows that are completely empty
                # or have more than 50% null values
                null_percentage = data.isnull().sum(axis=1) / len(data.columns)
                data = data[null_percentage <= 0.5]  # Keep rows with <= 50% nulls
            else:
                # Use the original dropna logic for other cases
                data = data.dropna(axis=self.dropna_axis, how=self.dropna_how)
            
            rows_removed = original_rows - len(data)
            self.logger.info("Missing values handled conservatively", 
                           rows_removed=rows_removed,
                           remaining_rows=len(data),
                           removal_strategy="conservative (<=50% nulls)")
        
        return data

    def apply_field_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply field mapping with error handling.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with mapped fields
        """
        for col, mapping in self.field_map.items():
            if col in data.columns:
                try:
                    # Create a copy to avoid modifying original
                    data[col] = data[col].copy()
                    
                    # Apply mapping with error handling
                    mapped_values = []
                    unmapped_values = set()
                    
                    for value in data[col]:
                        if pd.isna(value):
                            mapped_values.append(np.nan)
                        elif value in mapping:
                            mapped_values.append(mapping[value])
                        else:
                            # Keep original value if not in mapping
                            mapped_values.append(value)
                            unmapped_values.add(value)
                    
                    data[col] = mapped_values
                    
                    # Log mapping results
                    if unmapped_values:
                        self.logger.warning(f"Unmapped values found in column {col}", 
                                          unmapped_values=list(unmapped_values)[:10])  # Show first 10
                    
                    self.logger.info(f"Field mapping applied to {col}", 
                                   unique_values=list(set(mapped_values) - {np.nan})[:10])
                    
                except Exception as e:
                    self.logger.error(f"Error mapping field {col}", error=str(e))
                    # Continue with other mappings
        
        return data

    def apply_column_renaming(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column renaming with validation.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with renamed columns
        """
        if self.rename_map:
            # Validate rename map
            valid_renames = {}
            for old_name, new_name in self.rename_map.items():
                if old_name in data.columns:
                    valid_renames[old_name] = new_name
                else:
                    self.logger.warning(f"Column {old_name} not found for renaming")
            
            if valid_renames:
                data = data.rename(columns=valid_renames)
                self.logger.info("Columns renamed", rename_map=valid_renames)
        
        return data

    def handle_duplicate_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate rows if configured.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with duplicates handled
        """
        if self.handle_duplicates:
            original_rows = len(data)
            data = data.drop_duplicates()
            duplicates_removed = original_rows - len(data)
            
            if duplicates_removed > 0:
                self.logger.info("Duplicate rows removed", 
                               duplicates_removed=duplicates_removed,
                               remaining_rows=len(data))
        
        return data

    def validate_output_data(self, data: pd.DataFrame) -> bool:
        """
        Validate output data after transformation.
        
        Args:
            data: Output DataFrame
            
        Returns:
            bool: True if data is valid
        """
        if data is None or data.empty:
            self.logger.warning("Output data is empty after transformation")
            return False
        
        # Check for completely empty columns
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            self.logger.warning("Empty columns detected", empty_columns=empty_columns)
        
        self.logger.info("Output data validation passed", 
                        rows=len(data), 
                        columns=list(data.columns))
        return True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame with comprehensive error handling.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        self.logger.info("Starting data cleaning and transformation",
                        rows=len(data), 
                        columns=list(data.columns))
        
        try:
            # Validate input
            if not self.validate_input_data(data):
                raise ValueError("Input data validation failed")
            
            # Create a copy to avoid modifying original
            df = data.copy()
            
            # Step 1: Clean column names
            df = self.clean_column_names(df)
            
            # Step 2: Handle missing values
            df = self.handle_missing_values(df)
            
            # Step 3: Apply field mapping
            df = self.apply_field_mapping(df)
            
            # Step 4: Apply column renaming
            df = self.apply_column_renaming(df)
            
            # Step 5: Handle duplicates
            df = self.handle_duplicate_rows(df)
            
            # Validate output
            if not self.validate_output_data(df):
                self.logger.warning("Output data validation failed, but continuing")
            
            self.logger.info("Transformation complete", 
                           original_rows=len(data),
                           transformed_rows=len(df),
                           columns=list(df.columns))
            
            return df
            
        except Exception as e:
            self.logger.error("Transformation failed", error=str(e))
            raise

    def get_transformation_summary(self, original_data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the transformation performed.
        
        Args:
            original_data: Original DataFrame
            transformed_data: Transformed DataFrame
            
        Returns:
            Dict: Summary of transformations
        """
        summary = {
            "original_rows": len(original_data),
            "transformed_rows": len(transformed_data),
            "rows_removed": len(original_data) - len(transformed_data),
            "original_columns": list(original_data.columns),
            "transformed_columns": list(transformed_data.columns),
            "columns_renamed": len(self.rename_map),
            "fields_mapped": len(self.field_map),
            "missing_values_handled": self.dropna_how != 'none',
            "duplicates_handled": self.handle_duplicates
        }
        
        return summary 