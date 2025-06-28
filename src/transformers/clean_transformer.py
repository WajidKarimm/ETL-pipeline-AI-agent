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
import warnings

from .base import BaseTransformer
from ..logger import get_logger

# Suppress dateutil warnings
warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually')


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
        self.organize_data = (config or {}).get('organize_data', True)

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
        Apply field mapping with error handling and large dataset optimization.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with mapped fields
        """
        if not self.field_map:
            return data
            
        # Check dataset size for optimization strategy
        dataset_size = len(data)
        is_large_dataset = dataset_size > 50000
        
        if is_large_dataset:
            self.logger.info(f"Large dataset detected ({dataset_size:,} rows), using optimized field mapping")
        
        for col, mapping in self.field_map.items():
            if col in data.columns:
                try:
                    # Create a copy to avoid modifying original
                    data[col] = data[col].copy()
                    
                    if is_large_dataset:
                        # Optimized mapping for large datasets using vectorized operations
                        # Create a mapping Series for efficient lookup
                        mapping_series = pd.Series(mapping)
                        
                        # Apply mapping using pandas map function (vectorized)
                        mapped_series = data[col].map(mapping_series)
                        
                        # Handle unmapped values (keep original)
                        unmapped_mask = mapped_series.isna() & data[col].notna()
                        if unmapped_mask.any():
                            mapped_series[unmapped_mask] = data[col][unmapped_mask]
                            unmapped_count = unmapped_mask.sum()
                            self.logger.warning(f"Found {unmapped_count:,} unmapped values in column {col}")
                        
                        data[col] = mapped_series
                        
                        # Log mapping results (limited for large datasets)
                        unique_mapped = mapped_series.dropna().unique()
                        self.logger.info(f"Field mapping applied to {col}", 
                                       unique_values_count=len(unique_mapped),
                                       sample_values=list(unique_mapped[:5]))
                    else:
                        # Original method for smaller datasets
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
            
            # Step 6: Organize data structure for better readability (if enabled)
            if self.organize_data:
                df = self.organize_data_structure(df)
                self.logger.info("Data organization applied")
            else:
                self.logger.info("Data organization skipped")
            
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

    def organize_data_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Organize data structure for better readability and analysis.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Well-organized DataFrame
        """
        try:
            # 1. Sort columns by type and importance
            data = self.sort_columns_by_type(data)
            
            # 2. Reorder rows for better organization
            data = self.sort_rows_logically(data)
            
            # 3. Format data types appropriately
            data = self.format_data_types(data)
            
            # 4. Add metadata columns if useful
            data = self.add_metadata_columns(data)
            
            # 5. Ensure consistent formatting
            data = self.ensure_consistent_formatting(data)
            
            self.logger.info("Data structure organized", 
                           final_columns=list(data.columns),
                           final_rows=len(data))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error organizing data structure: {str(e)}")
            return data
    
    def sort_columns_by_type(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sort columns by data type and importance.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with sorted columns
        """
        try:
            # Identify column types
            id_columns = []
            date_columns = []
            numeric_columns = []
            categorical_columns = []
            text_columns = []
            
            for col in data.columns:
                col_lower = col.lower()
                
                # ID columns (usually first)
                if any(keyword in col_lower for keyword in ['id', 'key', 'index', 'uuid']):
                    id_columns.append(col)
                # Date columns
                elif any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                    date_columns.append(col)
                # Numeric columns
                elif data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_columns.append(col)
                # Categorical columns (low cardinality)
                elif data[col].dtype == 'object' and data[col].nunique() <= 50:
                    categorical_columns.append(col)
                # Text columns (high cardinality)
                else:
                    text_columns.append(col)
            
            # Reorder columns: ID, Date, Categorical, Numeric, Text
            ordered_columns = id_columns + date_columns + categorical_columns + numeric_columns + text_columns
            
            # Ensure all columns are included
            remaining_columns = [col for col in data.columns if col not in ordered_columns]
            ordered_columns.extend(remaining_columns)
            
            return data[ordered_columns]
            
        except Exception as e:
            self.logger.warning(f"Error sorting columns: {str(e)}")
            return data
    
    def sort_rows_logically(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sort rows in a logical order for better readability.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with sorted rows
        """
        try:
            # Find the best column to sort by
            sort_columns = []
            
            # Priority 1: ID columns
            id_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['id', 'key', 'index'])]
            if id_cols:
                sort_columns.extend(id_cols)
            
            # Priority 2: Date columns
            date_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created'])]
            if date_cols:
                sort_columns.extend(date_cols)
            
            # Priority 3: First categorical column
            categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and data[col].nunique() <= 50]
            if categorical_cols:
                sort_columns.append(categorical_cols[0])
            
            # Sort by the identified columns
            if sort_columns:
                # Remove duplicates while preserving order
                unique_sort_cols = []
                for col in sort_columns:
                    if col not in unique_sort_cols:
                        unique_sort_cols.append(col)
                
                # Sort the data
                data = data.sort_values(by=unique_sort_cols, na_position='last')
                self.logger.info(f"Rows sorted by: {unique_sort_cols}")
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error sorting rows: {str(e)}")
            return data
    
    def format_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Format data types appropriately for better organization.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with formatted data types
        """
        try:
            for col in data.columns:
                col_lower = col.lower()
                
                # Convert date-like columns with better format detection
                if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated']):
                    try:
                        # Try common date formats first to avoid dateutil warnings
                        common_formats = [
                            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                            '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
                            '%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f'
                        ]
                        
                        # Try to parse with specific formats first
                        parsed = False
                        for fmt in common_formats:
                            try:
                                temp_data = pd.to_datetime(data[col], format=fmt, errors='coerce')
                                # Check if we got any valid dates
                                if temp_data.notna().sum() > 0:
                                    data[col] = temp_data
                                    parsed = True
                                    break
                            except:
                                continue
                        
                        # If no specific format worked, use infer_datetime_format with warning suppression
                        if not parsed:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                data[col] = pd.to_datetime(data[col], infer_datetime_format=True, errors='coerce')
                            
                    except Exception as e:
                        self.logger.debug(f"Could not parse date column {col}: {str(e)}")
                        pass
                
                # Convert numeric columns
                elif data[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        numeric_data = pd.to_numeric(data[col], errors='coerce')
                        # Only convert if more than 80% of values are numeric
                        if numeric_data.notna().sum() / len(data) > 0.8:
                            data[col] = numeric_data
                    except:
                        pass
                
                # Format boolean columns
                elif data[col].dtype == 'bool':
                    data[col] = data[col].astype('bool')
            
            self.logger.info("Data types formatted")
            return data
            
        except Exception as e:
            self.logger.warning(f"Error formatting data types: {str(e)}")
            return data
    
    def add_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful metadata columns for better organization.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with metadata columns
        """
        try:
            # Add row number if no ID column exists
            id_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['id', 'key', 'index'])]
            if not id_cols:
                data.insert(0, 'row_id', range(1, len(data) + 1))
                self.logger.info("Added row_id column")
            
            # Add data quality indicators
            if len(data.columns) > 2:
                # Calculate completeness percentage for each row
                completeness = (data.notna().sum(axis=1) / len(data.columns) * 100).round(2)
                data['completeness_pct'] = completeness
                
                # Add row length (for text columns)
                text_cols = [col for col in data.columns if data[col].dtype == 'object']
                if text_cols:
                    # Calculate average text length for each row
                    text_lengths = []
                    for col in text_cols:
                        text_lengths.append(data[col].astype(str).str.len())
                    if text_lengths:
                        avg_text_length = pd.concat(text_lengths, axis=1).mean(axis=1).round(1)
                        data['avg_text_length'] = avg_text_length
                
                self.logger.info("Added metadata columns: completeness_pct, avg_text_length")
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error adding metadata columns: {str(e)}")
            return data
    
    def ensure_consistent_formatting(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure consistent formatting across the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Consistently formatted DataFrame
        """
        try:
            # Standardize string formatting
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Remove leading/trailing whitespace
                    data[col] = data[col].astype(str).str.strip()
                    
                    # Standardize case for categorical-like columns
                    if data[col].nunique() <= 50:
                        # Convert to title case for consistency
                        data[col] = data[col].str.title()
            
            # Ensure numeric columns have consistent precision
            for col in data.columns:
                if data[col].dtype in ['float64', 'float32']:
                    # Round to 2 decimal places for display
                    data[col] = data[col].round(2)
            
            self.logger.info("Consistent formatting applied")
            return data
            
        except Exception as e:
            self.logger.warning(f"Error applying consistent formatting: {str(e)}")
            return data 