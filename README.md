# ETL-pipeline-AI-agent

An AI-powered ETL agent that builds and maintains scalable data pipelines from natural language instructions, automating development and enabling seamless team collaboration.

## Overview
This project provides a modular, production-ready ETL (Extract, Transform, Load) pipeline in Python. It supports extracting data from REST APIs or CSV files, transforming the data (cleaning, mapping, renaming), and loading it into Snowflake, PostgreSQL, or AWS S3.

## Features
- **Modular architecture**: Easily extend extractors, transformers, and loaders.
- **Configurable**: All settings via environment variables or `.env` file.
- **Logging**: Structured logging with file and console output.
- **Retry & Failure Handling**: Retries for extraction and loading steps.
- **Type Safety**: Pydantic-based configuration.
- **Production-Ready**: Error handling, validation, and monitoring hooks.

## Directory Structure
```
src/
  config.py           # Configuration management
  logger.py           # Logging setup
  etl_pipeline.py     # Main ETL script
  extractors/         # Data extractors (API, CSV, ...)
  transformers/       # Data transformers (cleaning, mapping, ...)
  loaders/            # Data loaders (Postgres, Snowflake, S3, ...)
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   - Copy `env_example.txt` to `.env` and fill in your values.
4. **Run the ETL pipeline**
   ```bash
   python -m src.etl_pipeline
   ```

## Environment Variables
See `env_example.txt` for all required variables (API keys, DB credentials, etc).

## Customization
- **Source**: Set `DATA_SOURCE_TYPE` to `api` or `csv`.
- **Destination**: Set `DATA_DESTINATION_TYPE` to `snowflake`, `postgresql`, or `s3`.
- **Field Mapping**: Use `field_map` and `rename_map` in ETL config for custom transformations.

## Logging
- Logs are written to both console and file (see `LOG_FILE_PATH`).
- Log level is configurable via `LOG_LEVEL`.

## Retry & Failure Handling
- Extraction and loading steps use exponential backoff and retry (configurable via `MAX_RETRIES`, `RETRY_DELAY`).
- All errors are logged with stack traces.

## Collaboration Notes
- **Schema Decisions**: Target schema is defined by `rename_map` and loader table/column names. Review these with your data team.
- **Assumptions**:
  - API returns paginated JSON with `data` or `results` key.
  - Field mapping (e.g., `region` to ID) is provided in config.
  - Null removal is axis 0 (rows) and `how=any` by default.
- **Extensibility**: Add new extractors, transformers, or loaders by subclassing the base classes.
- **Testing**: Use `pytest` for unit tests. Add tests for each module.

## Next Steps
- Add more extractors/loaders as needed (e.g., MySQL, Google Cloud Storage).
- Integrate with orchestration tools (Airflow, Prefect) for scheduling.
- Add monitoring/metrics endpoints if required.

---
For questions or contributions, please open an issue or pull request.
