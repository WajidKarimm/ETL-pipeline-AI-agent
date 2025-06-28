"""
Base transformer class defining the interface for all data transformers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict

from ..logger import get_logger


class BaseTransformer(ABC):
    """
    Abstract base class for all data transformers.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(f"{self.__class__.__name__}")

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame and return the transformed DataFrame.
        """
        pass 