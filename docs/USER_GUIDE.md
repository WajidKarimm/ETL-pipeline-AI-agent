# User Guide - AI-Powered ETL Pipeline

Welcome to the comprehensive user guide for the AI-Powered ETL Pipeline! This guide will help you get started and make the most of our data processing capabilities.

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Web Interface Guide](#web-interface-guide)
3. [Command Line Usage](#command-line-usage)
4. [Common Use Cases](#common-use-cases)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)
7. [Advanced Features](#advanced-features)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Internet connection (for web interface)
- Data files or API endpoints to process

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Ai.etl.git
cd Ai.etl

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your credentials
```

### Quick Start
```bash
# Start the web interface
streamlit run app.py

# Or run from command line
python -m src.etl_pipeline
```

## 🌐 Web Interface Guide

### First Time Setup

1. **Launch the Application**
   - Run `streamlit run app.py`
   - Open your browser to `http://localhost:8501`

2. **Configure Your Environment**
   - Go to the "Configuration" section
   - Set up your database connections
   - Configure API credentials if needed

### Data Upload Process

#### Step 1: Upload Your Data
- **Drag & Drop**: Simply drag files into the upload area
- **Browse**: Click "Browse files" to select from your computer
- **Supported Formats**: CSV, JSON, Excel, ARFF, and more

#### Step 2: Preview Your Data
- Review the data structure and content
- Check for any obvious issues
- Note column names and data types

#### Step 3: Configure Transformations
- **Data Cleaning**: Remove nulls, duplicates, outliers
- **Field Mapping**: Rename columns or map values
- **Data Types**: Convert columns to appropriate types
- **Filtering**: Apply conditions to subset your data

#### Step 4: Select Destination
- **Database**: PostgreSQL, Snowflake
- **Cloud Storage**: AWS S3
- **Local File**: Download processed data

#### Step 5: Process and Monitor
- Click "Run ETL Pipeline"
- Monitor progress in real-time
- Review results and download if needed

### Web Interface Features

#### **Data Preview Panel**
- **Table View**: See your data in a sortable table
- **Statistics**: Basic statistics for each column
- **Data Types**: Automatic detection and suggestions
- **Missing Values**: Highlight missing data

#### **Configuration Panel**
- **Source Settings**: Configure data sources
- **Transformation Rules**: Define cleaning and mapping rules
- **Destination Settings**: Set up output locations
- **Validation**: Real-time validation of settings

#### **Processing Panel**
- **Progress Tracking**: Real-time progress bars
- **Log Output**: Live log messages
- **Error Handling**: Clear error messages with suggestions
- **Results Preview**: View processed data before download

## 💻 Command Line Usage

### Basic Commands

#### **Simple CSV Processing**
```bash
# Process a CSV file
python -m src.etl_pipeline \
  --source-type csv \
  --source-path data/input.csv \
  --destination-type postgresql \
  --destination-table processed_data
```

#### **API Data Extraction**
```bash
# Extract from API
python -m src.etl_pipeline \
  --source-type api \
  --source-url "https://api.example.com/data" \
  --destination-type s3 \
  --destination-bucket my-bucket
```

#### **With Custom Transformations**
```bash
# Apply custom transformations
python -m src.etl_pipeline \
  --source-type csv \
  --source-path data/input.csv \
  --field-map "old_name:new_name,region:location" \
  --remove-nulls \
  --destination-type postgresql
```

### Advanced Configuration

#### **Using Configuration Files**
```python
# config.yaml
source:
  type: csv
  path: data/input.csv
  encoding: utf-8
  delimiter: ','

transformations:
  remove_nulls: true
  field_map:
    old_column: new_column
    region: location_id
  data_types:
    amount: float
    date: datetime

destination:
  type: postgresql
  table: processed_data
  schema: public
  if_exists: replace
```

```bash
# Run with config file
python -m src.etl_pipeline --config config.yaml
```

## 📊 Common Use Cases

### **Business Intelligence Data Processing**

#### **Scenario**: Clean and load sales data
```python
# Configuration for sales data processing
config = {
    "source": {
        "type": "csv",
        "path": "sales_data.csv"
    },
    "transformations": {
        "remove_nulls": True,
        "field_map": {
            "sales_amount": "revenue",
            "customer_id": "client_id",
            "transaction_date": "date"
        },
        "data_types": {
            "revenue": "float",
            "date": "datetime",
            "client_id": "string"
        }
    },
    "destination": {
        "type": "postgresql",
        "table": "sales_processed",
        "schema": "analytics"
    }
}
```

#### **Steps**:
1. Upload your sales CSV file
2. Configure field mappings (e.g., `sales_amount` → `revenue`)
3. Set data types (ensure amounts are float, dates are datetime)
4. Remove any null values
5. Load to PostgreSQL analytics schema

### **Data Migration**

#### **Scenario**: Migrate legacy system data
```python
# Configuration for data migration
config = {
    "source": {
        "type": "api",
        "url": "https://legacy-system.com/api/data",
        "headers": {"Authorization": "Bearer token"}
    },
    "transformations": {
        "field_map": {
            "legacy_id": "new_id",
            "old_status": "status",
            "created_at": "timestamp"
        },
        "data_cleaning": {
            "remove_duplicates": True,
            "fill_missing": {"status": "unknown"}
        }
    },
    "destination": {
        "type": "snowflake",
        "table": "migrated_data",
        "warehouse": "migration_wh"
    }
}
```

### **Real-time Data Processing**

#### **Scenario**: Process streaming API data
```python
# Configuration for real-time processing
config = {
    "source": {
        "type": "api",
        "url": "https://streaming-api.com/events",
        "params": {"limit": 1000}
    },
    "transformations": {
        "data_validation": {
            "required_fields": ["id", "timestamp", "value"],
            "data_types": {"value": "float"}
        }
    },
    "destination": {
        "type": "postgresql",
        "table": "real_time_events",
        "if_exists": "append"
    }
}
```

### **Data Quality Management**

#### **Scenario**: Clean and validate customer data
```python
# Configuration for data quality
config = {
    "source": {
        "type": "csv",
        "path": "customer_data.csv"
    },
    "transformations": {
        "data_cleaning": {
            "remove_duplicates": True,
            "standardize_emails": True,
            "validate_phone_numbers": True
        },
        "field_mapping": {
            "email_address": "email",
            "phone": "phone_number",
            "full_name": "name"
        }
    },
    "destination": {
        "type": "s3",
        "bucket": "clean-data",
        "key": "customers/validated.csv"
    }
}
```

## 🔧 Troubleshooting

### **Common Issues and Solutions**

#### **File Upload Problems**

**Problem**: "File format not supported"
- **Solution**: Use the Universal Extractor which supports multiple formats
- **Alternative**: Convert your file to CSV or JSON format

**Problem**: "File too large"
- **Solution**: 
  - Use chunked processing
  - Increase system memory
  - Split large files into smaller chunks

#### **Database Connection Issues**

**Problem**: "Connection timeout"
- **Solution**:
  - Check network connectivity
  - Verify database credentials
  - Ensure database is running and accessible

**Problem**: "Authentication failed"
- **Solution**:
  - Verify username and password
  - Check if user has required permissions
  - Ensure SSL settings are correct

#### **Data Processing Issues**

**Problem**: "Column not found"
- **Solution**:
  - Check column names in your data
  - Use the data preview to verify column names
  - Update field mappings to match actual column names

**Problem**: "Data type conversion failed"
- **Solution**:
  - Review data types in preview
  - Clean data before conversion
  - Use appropriate data type mappings

#### **API Issues**

**Problem**: "Rate limit exceeded"
- **Solution**:
  - Implement retry logic with exponential backoff
  - Reduce request frequency
  - Contact API provider for higher limits

**Problem**: "Authentication error"
- **Solution**:
  - Verify API keys and tokens
  - Check authentication headers
  - Ensure API credentials are valid

### **Debug Mode**

Enable debug logging for detailed troubleshooting:

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.etl_pipeline --verbose
```

## 📈 Best Practices

### **Data Preparation**

1. **Understand Your Data**
   - Preview data before processing
   - Identify data types and formats
   - Check for missing values and outliers

2. **Plan Your Transformations**
   - Map out required field changes
   - Define data cleaning rules
   - Plan data type conversions

3. **Test with Sample Data**
   - Use small datasets for testing
   - Verify transformations work as expected
   - Check output format and quality

### **Performance Optimization**

1. **File Size Management**
   - Use appropriate chunk sizes for large files
   - Compress data when possible
   - Consider streaming for very large datasets

2. **Database Optimization**
   - Use appropriate indexes
   - Batch insert operations
   - Optimize connection pooling

3. **Memory Management**
   - Monitor memory usage
   - Use generators for large datasets
   - Clean up temporary data

### **Error Handling**

1. **Validate Input Data**
   - Check data quality before processing
   - Handle missing or invalid data
   - Log data quality issues

2. **Implement Retry Logic**
   - Use exponential backoff
   - Set appropriate retry limits
   - Handle transient failures

3. **Monitor and Alert**
   - Set up monitoring for pipeline health
   - Configure alerts for failures
   - Track processing metrics

## 🚀 Advanced Features

### **Custom Transformations**

#### **Custom Data Cleaning Functions**
```python
def custom_cleaner(data):
    """Custom data cleaning function."""
    # Remove rows with invalid email addresses
    data = data[data['email'].str.contains('@', na=False)]
    
    # Standardize phone numbers
    data['phone'] = data['phone'].str.replace(r'[^\d]', '', regex=True)
    
    return data

# Use in pipeline
config = {
    "transformations": {
        "custom_functions": [custom_cleaner]
    }
}
```

#### **Data Validation Rules**
```python
validation_rules = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "phone": r"^\d{10}$",
    "age": lambda x: 0 <= x <= 120
}

config = {
    "transformations": {
        "validation_rules": validation_rules
    }
}
```

### **Scheduling and Automation**

#### **Using Cron Jobs**
```bash
# Daily data processing
0 2 * * * cd /path/to/etl && python -m src.etl_pipeline --config daily_config.yaml

# Hourly API data extraction
0 * * * * cd /path/to/etl && python -m src.etl_pipeline --config hourly_config.yaml
```

#### **Using Airflow**
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

def run_etl_pipeline():
    from src.etl_pipeline import ETLPipeline
    from src.config import ETLConfig
    
    config = ETLConfig.from_file("config.yaml")
    pipeline = ETLPipeline(config)
    pipeline.run()

dag = DAG(
    'etl_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(hours=1)
)

etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl_pipeline,
    dag=dag
)
```

### **Monitoring and Logging**

#### **Custom Logging Configuration**
```python
import logging
from src.logger import setup_logger

# Set up custom logging
logger = setup_logger(
    name="custom_etl",
    level=logging.INFO,
    log_file="custom_etl.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

#### **Metrics Collection**
```python
import time
from src.metrics import MetricsCollector

metrics = MetricsCollector()

def process_with_metrics():
    start_time = time.time()
    
    try:
        # Your ETL processing here
        result = pipeline.run()
        
        # Record success metrics
        metrics.record_success(
            duration=time.time() - start_time,
            records_processed=len(result)
        )
        
    except Exception as e:
        # Record failure metrics
        metrics.record_failure(
            duration=time.time() - start_time,
            error=str(e)
        )
        raise
```

## 📞 Getting Help

### **Documentation Resources**
- **README.md**: Project overview and quick start
- **API Documentation**: Technical reference
- **Examples**: Code examples and templates
- **Tutorials**: Step-by-step guides

### **Community Support**
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Contributing Guide**: Learn how to contribute

### **Professional Support**
- **Enterprise Support**: For business users
- **Consulting Services**: Custom implementations
- **Training**: Workshops and training sessions

---

**Ready to transform your data?** 🚀

Start with the [Quick Start](#getting-started) section and explore the [Common Use Cases](#common-use-cases) to see how the ETL pipeline can help you! 