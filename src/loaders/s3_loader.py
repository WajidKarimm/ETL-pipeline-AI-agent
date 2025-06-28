"""
S3 data loader using boto3.
"""

import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Dict
from io import StringIO, BytesIO

from .base import BaseLoader
from ..logger import get_logger


class S3Loader(BaseLoader):
    """
    Loader for uploading data to AWS S3 as CSV or Parquet.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=config['access_key_id'],
            aws_secret_access_key=config['secret_access_key'],
            region_name=config['region']
        )
        self.bucket = config['bucket']

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def load(self, data: pd.DataFrame, key: str, format: str = 'csv', **kwargs) -> None:
        """
        Upload the DataFrame to S3 as CSV or Parquet.
        """
        self.logger.info("Starting data upload to S3", bucket=self.bucket, key=key, rows=len(data), format=format)
        try:
            if format == 'csv':
                csv_buffer = StringIO()
                data.to_csv(csv_buffer, index=False, **kwargs)
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=csv_buffer.getvalue())
            elif format == 'parquet':
                parquet_buffer = BytesIO()
                data.to_parquet(parquet_buffer, index=False, **kwargs)
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=parquet_buffer.getvalue())
            else:
                raise ValueError(f"Unsupported format: {format}")
            self.logger.info("Data uploaded to S3", bucket=self.bucket, key=key, format=format)
        except (BotoCoreError, ClientError, ValueError) as e:
            self.logger.error("S3 upload failed", bucket=self.bucket, key=key, error=str(e))
            raise 