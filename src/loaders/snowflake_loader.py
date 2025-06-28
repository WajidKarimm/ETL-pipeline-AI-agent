"""
Snowflake data loader using the Snowflake connector.
"""

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Dict

from .base import BaseLoader
from ..logger import get_logger


class SnowflakeLoader(BaseLoader):
    """
    Loader for loading data into Snowflake.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.conn_params = {
            'user': config['user'],
            'password': config['password'],
            'account': config['account'],
            'warehouse': config['warehouse'],
            'database': config['database'],
            'schema': config.get('schema', 'PUBLIC'),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def load(self, data: pd.DataFrame, table_name: str, **kwargs) -> None:
        """
        Load the DataFrame into the specified Snowflake table.
        """
        self.logger.info("Starting data load to Snowflake", table=table_name, rows=len(data))
        try:
            with snowflake.connector.connect(**self.conn_params) as conn:
                success, nchunks, nrows, _ = write_pandas(
                    conn,
                    data,
                    table_name,
                    schema=self.conn_params.get('schema', 'PUBLIC'),
                    chunk_size=self.config.get('batch_size', 1000),
                    **kwargs
                )
                if not success:
                    raise Exception(f"Failed to load data to Snowflake table {table_name}")
                self.logger.info("Data loaded to Snowflake", table=table_name, rows=nrows, chunks=nchunks)
        except Exception as e:
            self.logger.error("Snowflake load failed", table=table_name, error=str(e))
            raise 