"""
CSV data extractor for reading data from CSV files.
Enhanced with robust error handling and multiple parsing strategies.
"""

import pandas as pd
from typing import Any, Dict, Optional
from pathlib import Path
import chardet

from .base import BaseExtractor
from ..logger import get_logger


class CSVExtractor(BaseExtractor):
    """
    Extractor for reading data from CSV files.
    
    Supports various CSV formats and options with robust error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV extractor.
        
        Args:
            config: Configuration containing CSV settings
        """
        super().__init__(config)
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect the encoding of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected encoding
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.info("Encoding detected", 
                               encoding=encoding, 
                               confidence=confidence,
                               file_path=file_path)
                
                # Fallback to common encodings if confidence is low
                if confidence < 0.7:
                    return 'utf-8'
                return encoding
        except Exception as e:
            self.logger.warning("Encoding detection failed, using utf-8", 
                              error=str(e))
            return 'utf-8'
    
    def detect_separator(self, file_path: str, encoding: str) -> str:
        """
        Detect the separator used in the CSV file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            str: Detected separator
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
                
            separators = [',', ';', '\t', '|', ' ']
            max_fields = 0
            best_separator = ','
            
            for sep in separators:
                field_count = len(first_line.split(sep))
                if field_count > max_fields:
                    max_fields = field_count
                    best_separator = sep
            
            self.logger.info("Separator detected", 
                           separator=best_separator,
                           field_count=max_fields)
            
            return best_separator
        except Exception as e:
            self.logger.warning("Separator detection failed, using comma", 
                              error=str(e))
            return ','
    
    def extract(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from CSV file with robust error handling.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            pd.DataFrame: Extracted data
            
        Raises:
            Exception: If extraction fails
        """
        self.logger.info("Starting CSV data extraction", 
                        file_path=file_path)
        
        try:
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            # Detect encoding and separator
            encoding = self.detect_encoding(file_path)
            separator = self.detect_separator(file_path, encoding)
            
            # Try multiple parsing strategies
            df = self._try_parse_csv(file_path, encoding, separator, **kwargs)
            
            self.logger.info("CSV extraction completed", 
                           rows=len(df),
                           columns=list(df.columns),
                           file_size=Path(file_path).stat().st_size,
                           encoding=encoding,
                           separator=separator)
            
            return df
            
        except Exception as e:
            self.logger.error("CSV extraction failed", 
                            file_path=file_path,
                            error=str(e))
            raise
    
    def _try_parse_csv(self, file_path: str, encoding: str, separator: str, **kwargs) -> pd.DataFrame:
        """
        Try multiple strategies to parse CSV file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            separator: CSV separator
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Parsed data
        """
        # Strategy 1: Standard parsing
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=separator,
                engine='python',  # More flexible engine
                on_bad_lines='skip',  # Skip problematic lines
                **kwargs
            )
            return df
        except Exception as e:
            self.logger.warning("Standard parsing failed, trying alternative strategies", 
                              error=str(e))
        
        # Strategy 2: Try with different parameters
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=separator,
                engine='python',
                on_bad_lines='skip',
                error_bad_lines=False,
                warn_bad_lines=True,
                **kwargs
            )
            return df
        except Exception as e:
            self.logger.warning("Alternative parsing failed, trying manual parsing", 
                              error=str(e))
        
        # Strategy 3: Manual parsing for problematic files
        try:
            df = self._manual_parse_csv(file_path, encoding, separator)
            return df
        except Exception as e:
            self.logger.error("All parsing strategies failed", error=str(e))
            raise
    
    def _manual_parse_csv(self, file_path: str, encoding: str, separator: str) -> pd.DataFrame:
        """
        Manual CSV parsing for problematic files.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            separator: CSV separator
            
        Returns:
            pd.DataFrame: Parsed data
        """
        lines = []
        headers = None
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Split by separator and clean up
                fields = [field.strip().strip('"\'') for field in line.split(separator)]
                
                if line_num == 1:  # First non-empty line is header
                    headers = fields
                else:
                    # Pad or truncate fields to match header length
                    if len(fields) < len(headers):
                        fields.extend([''] * (len(headers) - len(fields)))
                    elif len(fields) > len(headers):
                        fields = fields[:len(headers)]
                    
                    lines.append(fields)
        
        if not headers:
            raise ValueError("No valid headers found in CSV file")
        
        # Create DataFrame
        df = pd.DataFrame(lines, columns=headers)
        
        # Try to infer data types
        for col in df.columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(df[col], errors='raise')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # Keep as string if conversion fails
                pass
        
        return df
    
    def extract_multiple(self, file_pattern: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from multiple CSV files matching a pattern.
        
        Args:
            file_pattern: Glob pattern for CSV files
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            pd.DataFrame: Combined extracted data
        """
        self.logger.info("Starting multiple CSV extraction", 
                        pattern=file_pattern)
        
        try:
            from pathlib import Path
            import glob
            
            files = glob.glob(file_pattern)
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
            
            all_dataframes = []
            for file_path in sorted(files):
                self.logger.info("Processing CSV file", file_path=file_path)
                df = self.extract(file_path, **kwargs)
                all_dataframes.append(df)
            
            # Combine all dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            self.logger.info("Multiple CSV extraction completed", 
                           total_files=len(files),
                           total_rows=len(combined_df))
            
            return combined_df
            
        except Exception as e:
            self.logger.error("Multiple CSV extraction failed", 
                            pattern=file_pattern,
                            error=str(e))
            raise 