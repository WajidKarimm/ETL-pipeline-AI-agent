"""
PostgreSQL data loader using SQLAlchemy.
"""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Dict

from .base import BaseLoader
from ..logger import get_logger


class PostgresLoader(BaseLoader):
    """
    Loader for loading data into PostgreSQL using SQLAlchemy.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine = create_engine(
            f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def load(self, data: pd.DataFrame, table_name: str, if_exists: str = 'append', index: bool = False, **kwargs) -> None:
        """
        Load the DataFrame into the specified PostgreSQL table.
        """
        self.logger.info("Starting data load to PostgreSQL", table=table_name, rows=len(data))
        try:
            data.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=index,
                method='multi',
                chunksize=self.config.get('batch_size', 1000),
                **kwargs
            )
            self.logger.info("Data loaded to PostgreSQL", table=table_name, rows=len(data))
        except SQLAlchemyError as e:
            self.logger.error("PostgreSQL load failed", table=table_name, error=str(e))
            raise 