"""
Base extractor class defining the interface for all data extractors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd

from ..logger import get_logger


class BaseExtractor(ABC):
    """
    Abstract base class for all data extractors.
    
    All extractors must implement the extract method and provide
    proper error handling and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the extractor with configuration.
        
        Args:
            config: Configuration dictionary for the extractor
        """
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def extract(self, **kwargs) -> pd.DataFrame:
        """
        Extract data from the source.
        
        Args:
            **kwargs: Additional arguments specific to the extractor
            
        Returns:
            pd.DataFrame: Extracted data
            
        Raises:
            Exception: If extraction fails
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate extracted data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.warning("Extracted data is empty or None")
            return False
        
        self.logger.info("Data validation passed", 
                        rows=len(data), 
                        columns=list(data.columns))
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the extraction process.
        
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        return {
            "extractor_type": self.__class__.__name__,
            "config": self.config
        } 