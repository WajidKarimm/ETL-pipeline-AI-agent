"""
Data cleaning and transformation logic.
- Remove nulls
- Map fields (e.g., 'region' to numeric ID)
- Rename columns to match target schema
"""

import pandas as pd
from typing import Any, Dict, Optional

from .base import BaseTransformer
from ..logger import get_logger


class CleanTransformer(BaseTransformer):
    """
    Transformer for cleaning and mapping data.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.field_map = (config or {}).get('field_map', {})
        self.rename_map = (config or {}).get('rename_map', {})
        self.dropna_axis = (config or {}).get('dropna_axis', 0)
        self.dropna_how = (config or {}).get('dropna_how', 'any')

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting data cleaning and transformation",
                        rows=len(data), columns=list(data.columns))
        df = data.copy()

        # Remove nulls
        df = df.dropna(axis=self.dropna_axis, how=self.dropna_how)
        self.logger.info("Null values removed", rows=len(df))

        # Map fields (e.g., region to numeric ID)
        for col, mapping in self.field_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).astype('Int64')
                self.logger.info(f"Field mapped: {col}", unique_values=df[col].unique().tolist())

        # Rename columns
        if self.rename_map:
            df = df.rename(columns=self.rename_map)
            self.logger.info("Columns renamed", rename_map=self.rename_map)

        self.logger.info("Transformation complete", rows=len(df), columns=list(df.columns))
        return df 