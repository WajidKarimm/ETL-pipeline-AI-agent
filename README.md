# AI-Powered ETL Pipeline

A production-ready, modular ETL (Extract, Transform, Load) pipeline with a user-friendly web interface that automates data processing workflows.

## 🎯 What You're Looking For

### If you need to:
- **Extract data** from REST APIs, CSV files, or other sources
- **Transform data** by cleaning, mapping fields, removing nulls, or renaming columns
- **Load data** into databases (PostgreSQL, Snowflake) or cloud storage (AWS S3)
- **Process data** through a web interface without coding
- **Automate data workflows** with retry logic and error handling
- **Scale data operations** with a modular, extensible architecture

**This is exactly what you need!**

## 🚀 Key Features

### 🔧 **Modular Architecture**
- **Extractors**: REST API, CSV, Universal (supports multiple formats)
- **Transformers**: Data cleaning, field mapping, null removal, column renaming
- **Loaders**: PostgreSQL, Snowflake, AWS S3
- **Easy to extend** with new data sources and destinations

### 🌐 **Web Interface**
- **Streamlit-based UI** for non-technical users
- **File upload** and drag-and-drop support
- **Data preview** and visualization
- **Configuration management** through the interface
- **Real-time processing** with progress indicators

### 🛡️ **Production-Ready**
- **Structured logging** with file and console output
- **Retry mechanisms** with exponential backoff
- **Error handling** and validation
- **Type safety** with Pydantic configuration
- **Environment-based configuration**

### 📊 **Data Processing Capabilities**
- **Universal file support**: CSV, JSON, Excel, ARFF, and more
- **Smart parsing**: Automatic format detection and robust error handling
- **Data transformation**: Cleaning, mapping, filtering, aggregation
- **Schema management**: Automatic column detection and type inference

## 📁 Project Structure

```
├── app.py                 # Streamlit web interface
├── src/
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup
│   ├── etl_pipeline.py    # Main ETL orchestration
│   ├── extractors/        # Data extraction modules
│   │   ├── api_extractor.py
│   │   ├── csv_extractor.py
│   │   └── universal_extractor.py
│   ├── transformers/      # Data transformation modules
│   │   └── data_transformer.py
│   └── loaders/           # Data loading modules
│       ├── postgresql_loader.py
│       ├── snowflake_loader.py
│       └── s3_loader.py
├── data/                  # Sample data and uploads
├── requirements.txt       # Python dependencies
├── test_pipeline.py       # Unit tests
└── env_example.txt        # Environment variables template
```

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd Ai.etl

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**
```bash
# Copy environment template
cp env_example.txt .env

# Edit .env with your credentials
# - Database connections (PostgreSQL, Snowflake)
# - AWS credentials for S3
# - API endpoints and keys
```

### 3. **Run the Web Interface**
```bash
streamlit run app.py
```

### 4. **Use the Pipeline**
1. **Upload your data** (CSV, JSON, Excel, etc.)
2. **Configure transformations** (cleaning, mapping, filtering)
3. **Select destination** (database or cloud storage)
4. **Process and monitor** in real-time

## 📖 Detailed Usage Guide

### **Web Interface Features**

#### **Data Upload**
- **Drag & Drop**: Simply drag files into the upload area
- **Multiple Formats**: Supports CSV, JSON, Excel, ARFF, and more
- **Large Files**: Handles files up to 200MB with progress tracking
- **Preview**: See your data before processing

#### **Configuration Panel**
- **Source Settings**: Configure API endpoints, file paths, authentication
- **Transformation Rules**: Define cleaning, mapping, and filtering rules
- **Destination Settings**: Set up database connections and table schemas
- **Validation**: Real-time validation of all settings

#### **Processing & Monitoring**
- **Real-time Progress**: See processing status and progress bars
- **Error Handling**: Clear error messages with suggestions
- **Data Preview**: View transformed data before loading
- **Download Results**: Export processed data in various formats

### **Programmatic Usage**

#### **Basic ETL Pipeline**
```python
from src.etl_pipeline import ETLPipeline
from src.config import ETLConfig

# Configure your pipeline
config = ETLConfig(
    source_type="csv",
    source_path="data/input.csv",
    destination_type="postgresql",
    destination_table="processed_data"
)

# Run the pipeline
pipeline = ETLPipeline(config)
result = pipeline.run()
```

#### **Custom Transformations**
```python
# Define field mappings
field_map = {
    "old_column": "new_column",
    "region": "location_id"
}

# Define cleaning rules
cleaning_config = {
    "remove_nulls": True,
    "remove_duplicates": True,
    "fill_missing": {"category": "unknown"}
}

# Apply transformations
transformer = DataTransformer()
cleaned_data = transformer.transform(
    data, 
    field_map=field_map,
    cleaning_config=cleaning_config
)
```

## 🔧 Configuration Options

### **Environment Variables**
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
```

### **Pipeline Configuration**
```python
# Source configuration
source_config = {
    "type": "api",  # or "csv", "universal"
    "url": "https://api.example.com/data",
    "headers": {"Authorization": "Bearer token"},
    "params": {"limit": 1000}
}

# Transformation configuration
transform_config = {
    "remove_nulls": True,
    "field_map": {"old_name": "new_name"},
    "rename_columns": {"id": "record_id"},
    "data_types": {"amount": "float", "date": "datetime"}
}

# Destination configuration
destination_config = {
    "type": "postgresql",  # or "snowflake", "s3"
    "table": "processed_data",
    "schema": "public",
    "if_exists": "replace"  # or "append", "fail"
}
```

## 🧪 Testing

### **Run Tests**
```bash
# Run all tests
python -m pytest test_pipeline.py -v

# Run specific test
python -m pytest test_pipeline.py::test_csv_extraction -v
```

### **Test Data**
- Sample CSV files in `data/` directory
- Test configurations in `test_pipeline.py`
- Mock API responses for testing

## 🔍 Troubleshooting

### **Common Issues**

#### **CSV Parsing Errors**
- **Problem**: Complex CSV formats (ARFF, custom delimiters)
- **Solution**: Use the Universal Extractor which handles multiple formats

#### **Database Connection Issues**
- **Problem**: Connection timeouts or authentication errors
- **Solution**: Check environment variables and network connectivity

#### **Memory Issues with Large Files**
- **Problem**: Out of memory when processing large datasets
- **Solution**: Use chunked processing or increase system memory

#### **API Rate Limiting**
- **Problem**: API requests being throttled
- **Solution**: Configure retry logic with exponential backoff

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.etl_pipeline --verbose
```

## 🚀 Deployment

### **Local Development**
```bash
# Development setup
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

### **Production Deployment**
```bash
# Using Docker
docker build -t etl-pipeline .
docker run -p 8501:8501 etl-pipeline

# Using cloud platforms
# - Heroku: Add Procfile and requirements.txt
# - AWS: Use ECS or Lambda
# - Google Cloud: Use Cloud Run
```

### **Environment Setup**
```bash
# Production environment
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export MAX_RETRIES=5
export RETRY_DELAY=60
```

## 📈 Use Cases

### **Business Intelligence**
- Extract data from multiple sources
- Clean and standardize data formats
- Load into data warehouses for analysis

### **Data Migration**
- Move data between different systems
- Transform legacy data formats
- Validate data integrity during migration

### **Real-time Data Processing**
- Process streaming data from APIs
- Apply real-time transformations
- Load into operational databases

### **Data Quality Management**
- Identify and clean dirty data
- Standardize data formats
- Monitor data quality metrics

## 🤝 Contributing

### **Development Setup**
```bash
# Fork the repository
git clone <your-fork-url>
cd Ai.etl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest
```

### **Code Style**
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Write unit tests for new features

### **Pull Request Process**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Getting Help**
- **Documentation**: Check this README and inline code comments
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

### **Community**
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)
- **Roadmap**: See [ROADMAP.md](ROADMAP.md)

---

**Ready to transform your data workflow?** 🚀

Start with the [Quick Start](#-quick-start) guide and explore the [Web Interface](#web-interface-features) to see how easy data processing can be!
