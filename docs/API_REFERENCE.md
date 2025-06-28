# API Reference - AI-Powered ETL Pipeline

This document provides comprehensive API documentation for the AI-Powered ETL Pipeline. It covers all classes, methods, configuration options, and usage examples.

## 📋 Table of Contents

1. [Core Classes](#core-classes)
2. [Configuration](#configuration)
3. [Extractors](#extractors)
4. [Transformers](#transformers)
5. [Loaders](#loaders)
6. [Utilities](#utilities)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## 🔧 Core Classes

### ETLPipeline

The main pipeline class that orchestrates the entire ETL process.

```python
from src.etl_pipeline import ETLPipeline
from src.config import ETLConfig

# Initialize pipeline
config = ETLConfig(
    source_type="csv",
    source_path="data/input.csv",
    destination_type="postgresql",
    destination_table="processed_data"
)
pipeline = ETLPipeline(config)

# Run the pipeline
result = pipeline.run()
```

#### Methods

##### `__init__(config: ETLConfig)`
Initialize the ETL pipeline with configuration.

**Parameters:**
- `config` (ETLConfig): Configuration object containing all pipeline settings

**Returns:**
- `ETLPipeline`: Initialized pipeline instance

##### `run() -> pd.DataFrame`
Execute the complete ETL pipeline.

**Returns:**
- `pd.DataFrame`: Processed data

**Raises:**
- `ExtractionError`: If data extraction fails
- `TransformationError`: If data transformation fails
- `LoadingError`: If data loading fails

##### `extract() -> pd.DataFrame`
Extract data from the source.

**Returns:**
- `pd.DataFrame`: Raw data from source

##### `transform(data: pd.DataFrame) -> pd.DataFrame`
Transform the extracted data.

**Parameters:**
- `data` (pd.DataFrame): Raw data to transform

**Returns:**
- `pd.DataFrame`: Transformed data

##### `load(data: pd.DataFrame) -> bool`
Load data to the destination.

**Parameters:**
- `data` (pd.DataFrame): Data to load

**Returns:**
- `bool`: True if successful, False otherwise

## ⚙️ Configuration

### ETLConfig

Configuration class using Pydantic for validation and type safety.

```python
from src.config import ETLConfig

config = ETLConfig(
    source_type="csv",
    source_path="data/input.csv",
    destination_type="postgresql",
    destination_table="processed_data",
    field_map={"old_name": "new_name"},
    remove_nulls=True
)
```

#### Configuration Options

##### Source Configuration
```python
source_config = {
    "type": "csv",  # "csv", "api", "universal"
    "path": "data/input.csv",  # File path for CSV
    "url": "https://api.example.com/data",  # API URL
    "headers": {"Authorization": "Bearer token"},  # API headers
    "params": {"limit": 1000},  # API parameters
    "encoding": "utf-8",  # File encoding
    "delimiter": ",",  # CSV delimiter
    "chunk_size": 10000  # Chunk size for large files
}
```

##### Transformation Configuration
```python
transform_config = {
    "remove_nulls": True,  # Remove rows with null values
    "remove_duplicates": True,  # Remove duplicate rows
    "field_map": {  # Map old column names to new ones
        "old_column": "new_column",
        "region": "location_id"
    },
    "rename_columns": {  # Rename specific columns
        "id": "record_id",
        "name": "full_name"
    },
    "data_types": {  # Convert data types
        "amount": "float",
        "date": "datetime",
        "category": "string"
    },
    "fill_missing": {  # Fill missing values
        "category": "unknown",
        "status": "pending"
    }
}
```

##### Destination Configuration
```python
destination_config = {
    "type": "postgresql",  # "postgresql", "snowflake", "s3"
    "table": "processed_data",  # Table name
    "schema": "public",  # Database schema
    "if_exists": "replace",  # "replace", "append", "fail"
    "batch_size": 1000,  # Batch size for inserts
    "index": False  # Create index on primary key
}
```

#### Environment Variables

```bash
# Database Connections
POSTGRESQL_HOST=localhost
POSTGRESQL_PORT=5432
POSTGRESQL_DATABASE=mydb
POSTGRESQL_USER=user
POSTGRESQL_PASSWORD=password

# Snowflake
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_WAREHOUSE=your-warehouse
SNOWFLAKE_DATABASE=your-database

# AWS S3
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_S3_BUCKET=your-bucket
AWS_REGION=us-east-1

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/etl.log

# Retry Configuration
MAX_RETRIES=3
RETRY_DELAY=5
```

## 📥 Extractors

### BaseExtractor

Abstract base class for all extractors.

```python
from src.extractors.base_extractor import BaseExtractor

class CustomExtractor(BaseExtractor):
    def extract(self) -> pd.DataFrame:
        # Implementation here
        pass
```

#### Methods

##### `extract() -> pd.DataFrame`
Extract data from the source. Must be implemented by subclasses.

**Returns:**
- `pd.DataFrame`: Extracted data

### CSVExtractor

Extract data from CSV files.

```python
from src.extractors.csv_extractor import CSVExtractor

config = {
    "path": "data/input.csv",
    "encoding": "utf-8",
    "delimiter": ",",
    "chunk_size": 10000
}
extractor = CSVExtractor(config)
data = extractor.extract()
```

#### Configuration Options
- `path` (str): Path to CSV file
- `encoding` (str): File encoding (default: "utf-8")
- `delimiter` (str): CSV delimiter (default: ",")
- `chunk_size` (int): Chunk size for large files
- `header` (int): Row number for header (default: 0)
- `skiprows` (int): Number of rows to skip

### APIExtractor

Extract data from REST APIs.

```python
from src.extractors.api_extractor import APIExtractor

config = {
    "url": "https://api.example.com/data",
    "headers": {"Authorization": "Bearer token"},
    "params": {"limit": 1000},
    "method": "GET",
    "timeout": 30
}
extractor = APIExtractor(config)
data = extractor.extract()
```

#### Configuration Options
- `url` (str): API endpoint URL
- `headers` (dict): HTTP headers
- `params` (dict): Query parameters
- `method` (str): HTTP method (default: "GET")
- `timeout` (int): Request timeout in seconds
- `retry_on_failure` (bool): Retry on failure (default: True)

### UniversalExtractor

Extract data from multiple file formats with automatic detection.

```python
from src.extractors.universal_extractor import UniversalExtractor

config = {
    "path": "data/input.file",
    "format": "auto",  # "auto", "csv", "json", "excel", "arff"
    "encoding": "utf-8"
}
extractor = UniversalExtractor(config)
data = extractor.extract()
```

#### Supported Formats
- CSV (comma, tab, semicolon separated)
- JSON (arrays, objects, newline-delimited)
- Excel (.xlsx, .xls)
- ARFF (Weka format)
- Parquet
- Feather
- Pickle

## 🔄 Transformers

### DataTransformer

Transform and clean data.

```python
from src.transformers.data_transformer import DataTransformer

transformer = DataTransformer()

# Basic transformation
cleaned_data = transformer.transform(
    data,
    remove_nulls=True,
    field_map={"old_name": "new_name"}
)

# Advanced transformation
cleaned_data = transformer.transform(
    data,
    remove_nulls=True,
    remove_duplicates=True,
    field_map={"region": "location_id"},
    rename_columns={"id": "record_id"},
    data_types={"amount": "float", "date": "datetime"},
    fill_missing={"category": "unknown"}
)
```

#### Methods

##### `transform(data: pd.DataFrame, **kwargs) -> pd.DataFrame`
Transform the input data according to specified rules.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `remove_nulls` (bool): Remove rows with null values
- `remove_duplicates` (bool): Remove duplicate rows
- `field_map` (dict): Map old column names to new ones
- `rename_columns` (dict): Rename specific columns
- `data_types` (dict): Convert data types
- `fill_missing` (dict): Fill missing values with specified values

**Returns:**
- `pd.DataFrame`: Transformed data

##### `clean_data(data: pd.DataFrame, **kwargs) -> pd.DataFrame`
Clean the data by removing nulls and duplicates.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `remove_nulls` (bool): Remove rows with null values
- `remove_duplicates` (bool): Remove duplicate rows
- `fill_missing` (dict): Fill missing values

**Returns:**
- `pd.DataFrame`: Cleaned data

##### `map_fields(data: pd.DataFrame, field_map: dict) -> pd.DataFrame`
Map column names according to the field map.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `field_map` (dict): Mapping of old column names to new ones

**Returns:**
- `pd.DataFrame`: Data with mapped column names

##### `convert_data_types(data: pd.DataFrame, data_types: dict) -> pd.DataFrame`
Convert data types of specified columns.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `data_types` (dict): Mapping of column names to target data types

**Returns:**
- `pd.DataFrame`: Data with converted data types

## 📤 Loaders

### BaseLoader

Abstract base class for all loaders.

```python
from src.loaders.base_loader import BaseLoader

class CustomLoader(BaseLoader):
    def load(self, data: pd.DataFrame) -> bool:
        # Implementation here
        pass
```

#### Methods

##### `load(data: pd.DataFrame) -> bool`
Load data to the destination. Must be implemented by subclasses.

**Parameters:**
- `data` (pd.DataFrame): Data to load

**Returns:**
- `bool`: True if successful, False otherwise

### PostgreSQLLoader

Load data to PostgreSQL database.

```python
from src.loaders.postgresql_loader import PostgreSQLLoader

config = {
    "table": "processed_data",
    "schema": "public",
    "if_exists": "replace",
    "batch_size": 1000,
    "index": False
}
loader = PostgreSQLLoader(config)
success = loader.load(data)
```

#### Configuration Options
- `table` (str): Target table name
- `schema` (str): Database schema (default: "public")
- `if_exists` (str): Action if table exists ("replace", "append", "fail")
- `batch_size` (int): Batch size for inserts
- `index` (bool): Create index on primary key
- `connection_pool_size` (int): Connection pool size

### SnowflakeLoader

Load data to Snowflake data warehouse.

```python
from src.loaders.snowflake_loader import SnowflakeLoader

config = {
    "table": "processed_data",
    "schema": "public",
    "warehouse": "my_warehouse",
    "database": "my_database",
    "if_exists": "replace"
}
loader = SnowflakeLoader(config)
success = loader.load(data)
```

#### Configuration Options
- `table` (str): Target table name
- `schema` (str): Database schema
- `warehouse` (str): Snowflake warehouse
- `database` (str): Snowflake database
- `if_exists` (str): Action if table exists
- `batch_size` (int): Batch size for inserts

### S3Loader

Load data to AWS S3.

```python
from src.loaders.s3_loader import S3Loader

config = {
    "bucket": "my-bucket",
    "key": "data/processed.csv",
    "format": "csv",
    "compression": "gzip"
}
loader = S3Loader(config)
success = loader.load(data)
```

#### Configuration Options
- `bucket` (str): S3 bucket name
- `key` (str): S3 object key
- `format` (str): Output format ("csv", "json", "parquet")
- `compression` (str): Compression type ("gzip", "bzip2", None)
- `region` (str): AWS region

## 🛠️ Utilities

### Logger

Structured logging utility.

```python
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("Processing started")
logger.error("An error occurred", extra={"error_code": 500})
```

#### Methods

##### `get_logger(name: str) -> Logger`
Get a logger instance with structured logging.

**Parameters:**
- `name` (str): Logger name (usually `__name__`)

**Returns:**
- `Logger`: Configured logger instance

##### `setup_logger(name: str, level: str, log_file: str) -> Logger`
Set up a custom logger with specific configuration.

**Parameters:**
- `name` (str): Logger name
- `level` (str): Log level ("DEBUG", "INFO", "WARNING", "ERROR")
- `log_file` (str): Log file path

**Returns:**
- `Logger`: Configured logger instance

### Retry Decorator

Retry mechanism for failed operations.

```python
from src.utils.retry import retry

@retry(max_attempts=3, delay=5)
def api_call():
    # API call that might fail
    pass
```

#### Parameters
- `max_attempts` (int): Maximum number of retry attempts
- `delay` (int): Delay between retries in seconds
- `backoff_factor` (float): Exponential backoff factor

## ❌ Error Handling

### Custom Exceptions

```python
from src.exceptions import (
    ExtractionError,
    TransformationError,
    LoadingError,
    ConfigurationError
)

try:
    pipeline.run()
except ExtractionError as e:
    logger.error(f"Extraction failed: {e}")
except TransformationError as e:
    logger.error(f"Transformation failed: {e}")
except LoadingError as e:
    logger.error(f"Loading failed: {e}")
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
```

### Error Types

#### ExtractionError
Raised when data extraction fails.

#### TransformationError
Raised when data transformation fails.

#### LoadingError
Raised when data loading fails.

#### ConfigurationError
Raised when configuration is invalid.

## 📝 Examples

### Complete ETL Pipeline

```python
from src.etl_pipeline import ETLPipeline
from src.config import ETLConfig

# Configure the pipeline
config = ETLConfig(
    source_type="csv",
    source_path="data/sales.csv",
    destination_type="postgresql",
    destination_table="sales_processed",
    field_map={
        "sales_amount": "revenue",
        "customer_id": "client_id",
        "transaction_date": "date"
    },
    remove_nulls=True,
    data_types={
        "revenue": "float",
        "date": "datetime",
        "client_id": "string"
    }
)

# Create and run pipeline
pipeline = ETLPipeline(config)
result = pipeline.run()

print(f"Processed {len(result)} records")
```

### Custom Extractor

```python
from src.extractors.base_extractor import BaseExtractor
import pandas as pd
import requests

class CustomAPIExtractor(BaseExtractor):
    def extract(self) -> pd.DataFrame:
        """Extract data from custom API."""
        try:
            response = requests.get(
                self.config["url"],
                headers=self.config.get("headers", {}),
                params=self.config.get("params", {}),
                timeout=self.config.get("timeout", 30)
            )
            response.raise_for_status()
            
            data = response.json()
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"API extraction failed: {e}")
            raise ExtractionError(f"Failed to extract data: {e}")

# Use custom extractor
config = {
    "url": "https://api.example.com/data",
    "headers": {"Authorization": "Bearer token"}
}
extractor = CustomAPIExtractor(config)
data = extractor.extract()
```

### Custom Transformer

```python
from src.transformers.data_transformer import DataTransformer
import pandas as pd

class CustomDataTransformer(DataTransformer):
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply custom transformations."""
        # Apply parent transformations
        data = super().transform(data, **kwargs)
        
        # Custom transformations
        if "custom_cleaning" in kwargs:
            data = self.custom_clean(data)
        
        if "custom_mapping" in kwargs:
            data = self.custom_map(data)
        
        return data
    
    def custom_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Custom data cleaning logic."""
        # Remove rows with invalid email addresses
        data = data[data['email'].str.contains('@', na=False)]
        
        # Standardize phone numbers
        data['phone'] = data['phone'].str.replace(r'[^\d]', '', regex=True)
        
        return data
    
    def custom_map(self, data: pd.DataFrame) -> pd.DataFrame:
        """Custom field mapping logic."""
        # Add derived fields
        data['full_name'] = data['first_name'] + ' ' + data['last_name']
        data['age_group'] = pd.cut(data['age'], bins=[0, 25, 50, 100], 
                                  labels=['Young', 'Adult', 'Senior'])
        
        return data

# Use custom transformer
transformer = CustomDataTransformer()
cleaned_data = transformer.transform(
    data,
    remove_nulls=True,
    custom_cleaning=True,
    custom_mapping=True
)
```

### Custom Loader

```python
from src.loaders.base_loader import BaseLoader
import pandas as pd

class CustomDatabaseLoader(BaseLoader):
    def load(self, data: pd.DataFrame) -> bool:
        """Load data to custom database."""
        try:
            # Custom database connection logic
            connection = self.get_connection()
            
            # Load data in batches
            batch_size = self.config.get("batch_size", 1000)
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                self.insert_batch(connection, batch)
            
            connection.close()
            self.logger.info(f"Successfully loaded {len(data)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Loading failed: {e}")
            return False
    
    def get_connection(self):
        """Get database connection."""
        # Custom connection logic
        pass
    
    def insert_batch(self, connection, batch: pd.DataFrame):
        """Insert a batch of records."""
        # Custom insert logic
        pass

# Use custom loader
config = {
    "table": "custom_table",
    "batch_size": 500
}
loader = CustomDatabaseLoader(config)
success = loader.load(data)
```

### Error Handling Example

```python
from src.etl_pipeline import ETLPipeline
from src.config import ETLConfig
from src.exceptions import ExtractionError, TransformationError, LoadingError
from src.logger import get_logger

logger = get_logger(__name__)

def run_etl_with_error_handling():
    """Run ETL pipeline with comprehensive error handling."""
    try:
        config = ETLConfig(
            source_type="csv",
            source_path="data/input.csv",
            destination_type="postgresql",
            destination_table="processed_data"
        )
        
        pipeline = ETLPipeline(config)
        result = pipeline.run()
        
        logger.info(f"ETL pipeline completed successfully. Processed {len(result)} records.")
        return result
        
    except ExtractionError as e:
        logger.error(f"Data extraction failed: {e}")
        # Handle extraction errors (e.g., retry with different source)
        raise
        
    except TransformationError as e:
        logger.error(f"Data transformation failed: {e}")
        # Handle transformation errors (e.g., skip problematic records)
        raise
        
    except LoadingError as e:
        logger.error(f"Data loading failed: {e}")
        # Handle loading errors (e.g., save to backup location)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Handle unexpected errors
        raise

# Run with error handling
try:
    result = run_etl_with_error_handling()
    print("ETL completed successfully!")
except Exception as e:
    print(f"ETL failed: {e}")
```

---

This API reference provides comprehensive documentation for all components of the AI-Powered ETL Pipeline. For more examples and use cases, see the [User Guide](USER_GUIDE.md) and [Examples](../examples/) directory. 