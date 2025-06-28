"""
Universal Data Extractor
Handles any file format with maximum robustness and error recovery.
Supports: CSV, ARFF, JSON, XML, Excel, TSV, and more.
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import chardet
import re
from typing import Any, Dict, Optional, List, Union
import warnings

from .base import BaseExtractor
from ..logger import get_logger


class UniversalExtractor(BaseExtractor):
    """
    Universal extractor that can handle any data file format with maximum robustness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_formats = ['csv', 'arff', 'json', 'xml', 'xlsx', 'xls', 'tsv', 'txt']
    
    def detect_file_format(self, file_path: str) -> str:
        """
        Detect the file format based on content and extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected format
        """
        try:
            # Read first few lines to detect format
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
            
            # Try to decode
            encoding = self.detect_encoding(raw_data)
            content = raw_data.decode(encoding, errors='ignore')
            lines = content.split('\n')[:20]  # First 20 lines
            
            # Check for ARFF format
            if any(line.strip().startswith('@relation') for line in lines):
                return 'arff'
            
            # Check for JSON format
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    json.loads(content)
                    return 'json'
                except:
                    pass
            
            # Check for XML format
            if content.strip().startswith('<'):
                try:
                    ET.fromstring(content)
                    return 'xml'
                except:
                    pass
            
            # Check for CSV/TSV format
            if any(',' in line for line in lines):
                return 'csv'
            elif any('\t' in line for line in lines):
                return 'tsv'
            
            # Default to CSV
            return 'csv'
            
        except Exception as e:
            self.logger.warning(f"Format detection failed, defaulting to CSV: {str(e)}")
            return 'csv'
    
    def detect_encoding(self, raw_data: bytes) -> str:
        """
        Detect encoding from raw bytes.
        
        Args:
            raw_data: Raw file bytes
            
        Returns:
            str: Detected encoding
        """
        try:
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if confidence > 0.7:
                return encoding
            else:
                return 'utf-8'
        except:
            return 'utf-8'
    
    def extract_arff(self, file_path: str) -> pd.DataFrame:
        """
        Extract data from ARFF (Weka) format files.
        
        Args:
            file_path: Path to ARFF file
            
        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            # Use pandas to read ARFF files
            df = pd.read_csv(file_path, comment='%', skip_blank_lines=True)
            
            # Find the @data section
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract data section
            data_start = content.find('@data')
            if data_start != -1:
                data_section = content[data_start:].split('\n')[1:]
                data_lines = [line.strip() for line in data_section if line.strip() and not line.startswith('%')]
                
                # Parse data lines
                data_rows = []
                for line in data_lines:
                    if line and not line.startswith('@'):
                        # Split by comma and handle quoted values
                        values = []
                        current_value = ""
                        in_quotes = False
                        
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                values.append(current_value.strip())
                                current_value = ""
                            else:
                                current_value += char
                        
                        values.append(current_value.strip())
                        data_rows.append(values)
                
                if data_rows:
                    # Create DataFrame
                    df = pd.DataFrame(data_rows)
                    
                    # Try to infer column names from @attribute lines
                    attribute_lines = [line for line in content.split('\n') if line.strip().startswith('@attribute')]
                    if attribute_lines:
                        column_names = []
                        for line in attribute_lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                col_name = parts[1].strip()
                                column_names.append(col_name)
                        
                        if len(column_names) == len(df.columns):
                            df.columns = column_names
            
            return df
            
        except Exception as e:
            self.logger.error(f"ARFF extraction failed: {str(e)}")
            # Fallback to CSV parsing
            return self.extract_csv_robust(file_path)
    
    def extract_csv_robust(self, file_path: str) -> pd.DataFrame:
        """
        Extract data from CSV with maximum robustness.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pd.DataFrame: Extracted data
        """
        # Check file size to determine reading strategy
        file_size = Path(file_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Use chunked reading for large files
        if file_size_mb > 100:
            self.logger.info(f"Large file detected ({file_size_mb:.1f} MB), using chunked reading")
            return self.extract_csv_chunked(file_path)
        
        # Try multiple parsing strategies for smaller files
        strategies = [
            # Strategy 1: Standard pandas
            lambda: pd.read_csv(file_path, engine='python', on_bad_lines='skip'),
            
            # Strategy 2: With different encoding
            lambda: pd.read_csv(file_path, encoding='latin-1', engine='python', on_bad_lines='skip'),
            
            # Strategy 3: With different separator
            lambda: pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip'),
            
            # Strategy 4: Manual parsing
            lambda: self.manual_csv_parse(file_path),
            
            # Strategy 5: Skip problematic lines
            lambda: pd.read_csv(file_path, engine='python', on_bad_lines='skip', error_bad_lines=False)
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                self.logger.info(f"CSV extraction successful with strategy {i}")
                return strategy()
            except Exception as e:
                self.logger.warning(f"Strategy {i} failed: {str(e)}")
                continue
        
        # If all strategies fail, return empty DataFrame
        self.logger.error("All CSV extraction strategies failed")
        return pd.DataFrame()
    
    def extract_csv_chunked(self, file_path: str) -> pd.DataFrame:
        """
        Extract large CSV files using chunked reading to manage memory.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            # Check file size and adjust chunk size accordingly
            file_size = Path(file_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Adjust chunk size based on file size
            if file_size_mb > 500:
                chunk_size = 5000  # Smaller chunks for very large files
            elif file_size_mb > 200:
                chunk_size = 8000  # Medium chunks for large files
            else:
                chunk_size = 10000  # Standard chunks for medium files
            
            chunks = []
            
            # First, read a small sample to determine structure
            try:
                sample_df = pd.read_csv(file_path, nrows=1000, engine='python', on_bad_lines='skip')
                self.logger.info(f"Sample read successful: {len(sample_df)} rows, {len(sample_df.columns)} columns")
            except Exception as e:
                self.logger.warning(f"Sample read failed, trying with different parameters: {str(e)}")
                # Try with different parameters for problematic files
                sample_df = pd.read_csv(file_path, nrows=1000, engine='python', on_bad_lines='skip', 
                                      encoding='latin-1', low_memory=False)
            
            # Read the full file in chunks with error handling
            chunk_count = 0
            total_rows = 0
            
            try:
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='python', 
                                       on_bad_lines='skip', low_memory=False):
                    chunks.append(chunk)
                    chunk_count += 1
                    total_rows += len(chunk)
                    
                    # Log progress for very large files
                    if chunk_count % 10 == 0:
                        self.logger.info(f"Processed {chunk_count} chunks, {total_rows:,} rows so far")
                    
                    # Safety check for extremely large files
                    if total_rows > 1000000:  # 1 million rows limit
                        self.logger.warning(f"File too large ({total_rows:,} rows), processing first 1M rows")
                        break
                        
            except Exception as e:
                self.logger.error(f"Chunked reading failed: {str(e)}")
                # Fallback to manual parsing for problematic files
                return self.manual_csv_parse_large(file_path)
            
            # Combine all chunks
            if chunks:
                result_df = pd.concat(chunks, ignore_index=True)
                self.logger.info(f"Chunked CSV extraction completed: {len(result_df):,} rows, {len(result_df.columns)} columns")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Chunked CSV extraction failed: {str(e)}")
            # Fallback to manual parsing
            return self.manual_csv_parse_large(file_path)
    
    def manual_csv_parse(self, file_path: str) -> pd.DataFrame:
        """
        Manual CSV parsing for problematic files.
        
        Args:
            file_path: Path to file
            
        Returns:
            pd.DataFrame: Parsed data
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find the first line that looks like data
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('@') and not line.startswith('%'):
                    data_start = i
                    break
            
            # Parse data lines
            data_rows = []
            for line in lines[data_start:]:
                line = line.strip()
                if line and not line.startswith('@') and not line.startswith('%'):
                    # Split by common separators
                    for sep in [',', '\t', ';', '|']:
                        if sep in line:
                            values = line.split(sep)
                            data_rows.append(values)
                            break
                    else:
                        # No separator found, treat as single column
                        data_rows.append([line])
            
            if data_rows:
                # Find the maximum number of columns
                max_cols = max(len(row) for row in data_rows)
                
                # Pad rows to have the same number of columns
                padded_rows = []
                for row in data_rows:
                    padded_row = row + [''] * (max_cols - len(row))
                    padded_rows.append(padded_row)
                
                df = pd.DataFrame(padded_rows)
                
                # Try to use first row as headers if it looks like headers
                if len(df) > 1:
                    first_row = df.iloc[0]
                    if all(isinstance(val, str) and len(val) < 50 for val in first_row):
                        df.columns = first_row
                        df = df.iloc[1:].reset_index(drop=True)
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Manual CSV parsing failed: {str(e)}")
            return pd.DataFrame()
    
    def manual_csv_parse_large(self, file_path: str) -> pd.DataFrame:
        """
        Manual CSV parsing for extremely large or problematic files.
        
        Args:
            file_path: Path to file
            
        Returns:
            pd.DataFrame: Parsed data
        """
        try:
            self.logger.info("Starting manual parsing for large file")
            
            # Read file in smaller chunks to avoid memory issues
            chunk_size = 10000
            all_data = []
            line_count = 0
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read header first
                header_line = f.readline().strip()
                if not header_line:
                    return pd.DataFrame({'error': ['Empty file']})
                
                # Parse header
                headers = header_line.split(',')
                self.logger.info(f"Detected {len(headers)} columns in header")
                
                # Read data in chunks
                chunk_data = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('@') and not line.startswith('%'):
                        # Split by comma and handle quoted values
                        values = self.parse_csv_line(line)
                        if len(values) == len(headers):
                            chunk_data.append(values)
                            line_count += 1
                        
                        # Process chunk when it reaches chunk_size
                        if len(chunk_data) >= chunk_size:
                            all_data.extend(chunk_data)
                            chunk_data = []
                            self.logger.info(f"Processed {line_count:,} lines")
                            
                            # Safety check
                            if line_count > 500000:  # 500k rows limit
                                self.logger.warning(f"File too large, processing first 500k rows")
                                break
                
                # Add remaining data
                if chunk_data:
                    all_data.extend(chunk_data)
            
            if all_data:
                df = pd.DataFrame(all_data, columns=headers)
                self.logger.info(f"Manual parsing completed: {len(df):,} rows, {len(df.columns)} columns")
                return df
            else:
                return pd.DataFrame({'error': ['No data could be parsed']})
                
        except Exception as e:
            self.logger.error(f"Manual CSV parsing failed: {str(e)}")
            return pd.DataFrame({'error': [f'Parsing failed: {str(e)}']})
    
    def parse_csv_line(self, line: str) -> List[str]:
        """
        Parse a single CSV line, handling quoted values.
        
        Args:
            line: CSV line to parse
            
        Returns:
            List[str]: Parsed values
        """
        values = []
        current_value = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current_value.strip())
                current_value = ""
            else:
                current_value += char
        
        values.append(current_value.strip())
        return values
    
    def extract(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from any file format with maximum robustness.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Extracted data
        """
        self.logger.info("Starting universal data extraction", file_path=file_path)
        
        try:
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Detect file format
            file_format = self.detect_file_format(file_path)
            self.logger.info(f"Detected file format: {file_format}")
            
            # Check for chunked reading configuration
            csv_options = self.config.get('csv_options', {})
            chunksize = csv_options.get('chunksize')
            
            # Extract based on format
            if file_format == 'arff':
                df = self.extract_arff(file_path)
            elif file_format in ['csv', 'tsv']:
                if chunksize:
                    # Use chunked reading for large files
                    df = self.extract_csv_chunked(file_path)
                else:
                    df = self.extract_csv_robust(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            elif file_format == 'xml':
                # Convert XML to DataFrame
                tree = ET.parse(file_path)
                root = tree.getroot()
                data = []
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        data.append({'tag': elem.tag, 'text': elem.text.strip()})
                df = pd.DataFrame(data)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                # Default to CSV
                if chunksize:
                    df = self.extract_csv_chunked(file_path)
                else:
                    df = self.extract_csv_robust(file_path)
            
            # Ensure we have a DataFrame
            if df is None or df.empty:
                self.logger.warning("Extraction resulted in empty DataFrame")
                df = pd.DataFrame({'message': ['No data could be extracted from this file']})
            
            # Clean up the DataFrame
            df = self.clean_extracted_data(df)
            
            self.logger.info("Universal extraction completed", 
                           rows=len(df),
                           columns=list(df.columns),
                           format=file_format)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Universal extraction failed: {str(e)}")
            # Return a minimal DataFrame instead of raising
            return pd.DataFrame({'error': [f'Extraction failed: {str(e)}']})
    
    def clean_extracted_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize extracted data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Clean column names
            df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Remove duplicate column names
            df.columns = pd.Index([f"{col}_{i}" if i > 0 else col for i, col in enumerate(df.columns)])
            
            # Try to convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Data cleaning failed: {str(e)}")
            return df 