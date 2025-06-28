"""
Main ETL pipeline script.
- Extracts data from source
- Transforms data (cleaning, mapping, renaming)
- Loads data into destination
- Handles logging, retries, and error handling
"""

import sys
import traceback
import pandas as pd
from typing import Any, Dict
from pathlib import Path

from .config import get_config
from .logger import logger
from .extractors.api_extractor import APIExtractor
from .extractors.csv_extractor import CSVExtractor
from .transformers.clean_transformer import CleanTransformer
from .loaders.postgres_loader import PostgresLoader
from .loaders.snowflake_loader import SnowflakeLoader
from .loaders.s3_loader import S3Loader


def run_etl():
    config = get_config()
    logger.info("ETL pipeline started", source=config.etl.data_source_type, destination=config.etl.data_destination_type)
    try:
        # 1. Extract
        if config.etl.data_source_type == 'api':
            extractor = APIExtractor(config.api.dict())
            raw_data = extractor.extract()
        elif config.etl.data_source_type == 'csv':
            extractor = CSVExtractor(config.etl.dict())
            raw_data = extractor.extract(config.etl.get('csv_file_path', 'data/input.csv'))
        else:
            logger.error("Unsupported data source type", type=config.etl.data_source_type)
            sys.exit(1)

        if not extractor.validate_data(raw_data):
            logger.error("Extracted data is invalid or empty")
            sys.exit(1)

        # 2. Transform
        transformer = CleanTransformer({
            'field_map': config.etl.get('field_map', {}),
            'rename_map': config.etl.get('rename_map', {}),
            'dropna_axis': config.etl.get('dropna_axis', 0),
            'dropna_how': config.etl.get('dropna_how', 'any'),
        })
        clean_data = transformer.transform(raw_data)

        # 3. Load
        if config.etl.data_destination_type == 'postgresql':
            loader = PostgresLoader(config.database.dict())
            loader.load(clean_data, table_name=config.etl.get('target_table', 'etl_table'))
        elif config.etl.data_destination_type == 'snowflake':
            loader = SnowflakeLoader(config.snowflake.dict())
            loader.load(clean_data, table_name=config.etl.get('target_table', 'ETL_TABLE'))
        elif config.etl.data_destination_type == 's3':
            loader = S3Loader(config.s3.dict())
            loader.load(clean_data, key=config.etl.get('s3_key', 'etl/output.csv'), format=config.etl.get('s3_format', 'csv'))
        else:
            logger.error("Unsupported data destination type", type=config.etl.data_destination_type)
            sys.exit(1)

        logger.info("ETL pipeline completed successfully")
    except Exception as e:
        logger.error("ETL pipeline failed", error=str(e), traceback=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    run_etl() 