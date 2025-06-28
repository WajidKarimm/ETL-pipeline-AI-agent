"""
CSV data extractor for reading data from CSV files.
"""

import pandas as pd
from typing import Any, Dict, Optional
from pathlib import Path

from .base import BaseExtractor
from ..logger import get_logger


class CSVExtractor(BaseExtractor):
    """
    Extractor for reading data from CSV files.
    
    Supports various CSV formats and options.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV extractor.
        
        Args:
            config: Configuration containing CSV settings
        """
        super().__init__(config)
    
    def extract(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from CSV file.
        
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
            
            # Default CSV reading parameters
            default_params = {
                'encoding': 'utf-8',
                'sep': ',',
                'header': 0,
                'na_values': ['', 'null', 'NULL', 'None', 'NaN'],
                'low_memory': False
            }
            
            # Override with config and kwargs
            read_params = {**default_params, **self.config.get('csv_options', {}), **kwargs}
            
            df = pd.read_csv(file_path, **read_params)
            
            self.logger.info("CSV extraction completed", 
                           rows=len(df),
                           columns=list(df.columns),
                           file_size=Path(file_path).stat().st_size)
            
            return df
            
        except Exception as e:
            self.logger.error("CSV extraction failed", 
                            file_path=file_path,
                            error=str(e))
            raise
    
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