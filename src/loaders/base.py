"""
Base loader class defining the interface for all data loaders.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict

from ..logger import get_logger


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(f"{self.__class__.__name__}")

    @abstractmethod
    def load(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Load the DataFrame into the target destination.
        """
        pass 