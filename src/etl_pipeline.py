"""
Main ETL Pipeline with AI Agent Integration

This module provides the main ETL pipeline with AI-powered features for:
- Automatic data quality detection
- Intelligent transformation suggestions
- Error prediction and prevention
- Learning from operations to improve accuracy
"""

import sys
import traceback
import pandas as pd
from typing import Any, Dict, Optional, List
from pathlib import Path
import time

from .config import get_config
from .logger import logger
from .extractors.api_extractor import APIExtractor
from .extractors.csv_extractor import CSVExtractor
from .transformers.clean_transformer import CleanTransformer
from .loaders.postgres_loader import PostgresLoader
from .loaders.snowflake_loader import SnowflakeLoader
from .loaders.s3_loader import S3Loader
from .ml.ai_agent import ETLAIAgent, DataQualityIssue, TransformationSuggestion


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


class ETLPipeline:
    """
    Enhanced ETL Pipeline with AI Agent integration.
    """
    
    def __init__(self, config):
        """
        Initialize the ETL pipeline with AI agent.
        
        Args:
            config: ETL configuration
        """
        self.config = config
        self.ai_agent = ETLAIAgent()
        self.logger = logger
        
        # Initialize components
        self.extractor = get_extractor(config.source_type, config.source_config)
        self.transformer = get_transformer(config.transform_config)
        self.loader = get_loader(config.destination_type, config.destination_config)
        
        logger.info("ETL Pipeline initialized with AI Agent")
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete ETL pipeline with AI assistance.
        
        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract data
            logger.info("Starting data extraction...")
            data = self.extract()
            
            # Step 2: AI-powered data quality analysis
            logger.info("Performing AI-powered data quality analysis...")
            quality_issues = self.analyze_data_quality(data)
            
            # Step 3: AI-powered transformation suggestions
            logger.info("Generating AI-powered transformation suggestions...")
            suggestions = self.get_transformation_suggestions(data)
            
            # Step 4: Apply transformations
            logger.info("Applying transformations...")
            transformed_data = self.transform(data)
            
            # Step 5: Load data
            logger.info("Loading data to destination...")
            success = self.load(transformed_data)
            
            # Step 6: Learn from operation
            logger.info("Learning from operation...")
            self.learn_from_operation(data, transformed_data, success)
            
            # Log completion
            duration = time.time() - start_time
            logger.info(f"ETL pipeline completed successfully in {duration:.2f} seconds")
            
            return transformed_data
            
        except Exception as e:
            # Learn from failure
            self.learn_from_operation(data if 'data' in locals() else None, 
                                    transformed_data if 'transformed_data' in locals() else None, 
                                    False, [str(e)])
            raise
    
    def extract(self) -> pd.DataFrame:
        """
        Extract data from source with AI assistance.
        
        Returns:
            Raw data DataFrame
        """
        try:
            data = self.extractor.extract()
            
            # AI analysis of extracted data
            if not data.empty:
                logger.info(f"Extracted {len(data)} rows and {len(data.columns)} columns")
                
                # Quick quality check
                quality_issues = self.ai_agent.detect_data_quality_issues(data)
                if quality_issues:
                    logger.warning(f"Detected {len(quality_issues)} data quality issues during extraction")
                    for issue in quality_issues:
                        logger.warning(f"- {issue.issue_type}: {issue.description}")
            
            return data
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise
    
    def analyze_data_quality(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """
        Analyze data quality using AI agent.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of detected quality issues
        """
        try:
            issues = self.ai_agent.detect_data_quality_issues(data)
            
            if issues:
                logger.info(f"AI Agent detected {len(issues)} data quality issues:")
                for issue in issues:
                    logger.info(f"- {issue.severity.upper()}: {issue.description}")
                    logger.info(f"  Suggested fix: {issue.suggested_fix}")
            
            return issues
            
        except Exception as e:
            logger.warning(f"Data quality analysis failed: {e}")
            return []
    
    def get_transformation_suggestions(self, data: pd.DataFrame) -> List[TransformationSuggestion]:
        """
        Get AI-powered transformation suggestions.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of transformation suggestions
        """
        try:
            suggestions = self.ai_agent.suggest_transformations(data)
            
            if suggestions:
                logger.info(f"AI Agent suggested {len(suggestions)} transformations:")
                for suggestion in suggestions:
                    logger.info(f"- {suggestion.transformation_type}: {suggestion.reasoning}")
                    logger.info(f"  Target: {suggestion.target_column}, Confidence: {suggestion.confidence:.2f}")
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Transformation suggestions failed: {e}")
            return []
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with AI assistance.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Get AI suggestions
            suggestions = self.ai_agent.suggest_transformations(data)
            
            # Apply user-configured transformations
            transformed_data = self.transformer.transform(data)
            
            # Log transformation results
            logger.info(f"Transformed data: {len(transformed_data)} rows, {len(transformed_data.columns)} columns")
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise
    
    def load(self, data: pd.DataFrame) -> bool:
        """
        Load data to destination with AI assistance.
        
        Args:
            data: Data to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.loader.load(data)
            
            if success:
                logger.info(f"Successfully loaded {len(data)} records to destination")
            else:
                logger.error("Data loading failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False
    
    def learn_from_operation(self, original_data: Optional[pd.DataFrame], 
                           transformed_data: Optional[pd.DataFrame], 
                           success: bool, errors: List[str] = None):
        """
        Learn from the ETL operation to improve future performance.
        
        Args:
            original_data: Original data before transformation
            transformed_data: Data after transformation
            success: Whether the operation was successful
            errors: List of errors encountered
        """
        try:
            if original_data is not None:
                # Prepare transformation config for learning
                transform_config = {
                    'remove_nulls': getattr(self.config, 'remove_nulls', False),
                    'field_map': getattr(self.config, 'field_map', {}),
                    'rename_columns': getattr(self.config, 'rename_columns', {}),
                    'data_types': getattr(self.config, 'data_types', {})
                }
                
                # Learn from the operation
                self.ai_agent.learn_from_operation(
                    data=original_data,
                    transformations=transform_config,
                    success=success,
                    errors=errors or []
                )
                
                logger.info("Successfully learned from ETL operation")
            
        except Exception as e:
            logger.warning(f"Failed to learn from operation: {e}")
    
    def get_ai_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive AI insights about the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing AI insights
        """
        try:
            insights = {
                'data_quality_issues': self.analyze_data_quality(data),
                'transformation_suggestions': self.ai_agent.suggest_transformations(data),
                'model_performance': self.ai_agent.get_model_performance(),
                'data_statistics': {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                    'missing_values': data.isnull().sum().sum(),
                    'duplicate_rows': data.duplicated().sum()
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get AI insights: {e}")
            return {}
    
    def predict_errors(self, data: pd.DataFrame, transformations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict potential errors before applying transformations.
        
        Args:
            data: Input DataFrame
            transformations: Planned transformations
            
        Returns:
            List of predicted errors
        """
        try:
            # This would use the AI agent's error prediction capabilities
            # For now, return basic predictions
            predictions = []
            
            # Check for common issues
            if 'field_map' in transformations:
                for old_col, new_col in transformations['field_map'].items():
                    if old_col not in data.columns:
                        predictions.append({
                            'error_type': 'missing_column',
                            'probability': 1.0,
                            'description': f"Column '{old_col}' not found in data",
                            'suggestion': f"Check column names or remove mapping for '{old_col}'"
                        })
            
            if 'data_types' in transformations:
                for col, target_type in transformations['data_types'].items():
                    if col in data.columns and target_type == 'numeric' and data[col].dtype == 'object':
                        try:
                            pd.to_numeric(data[col], errors='raise')
                        except:
                            predictions.append({
                                'error_type': 'type_conversion',
                                'probability': 0.9,
                                'description': f"Cannot convert column '{col}' to numeric",
                                'suggestion': f"Clean non-numeric values in column '{col}'"
                            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error prediction failed: {e}")
            return []
    
    def get_ai_recommendations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get AI recommendations for optimal ETL configuration.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing AI recommendations
        """
        try:
            recommendations = {
                'data_quality_actions': [],
                'transformation_actions': [],
                'configuration_suggestions': []
            }
            
            # Analyze data and provide recommendations
            quality_issues = self.analyze_data_quality(data)
            for issue in quality_issues:
                recommendations['data_quality_actions'].append({
                    'action': issue.suggested_fix,
                    'priority': issue.severity,
                    'confidence': issue.confidence
                })
            
            suggestions = self.ai_agent.suggest_transformations(data)
            for suggestion in suggestions:
                recommendations['transformation_actions'].append({
                    'action': suggestion.transformation_type,
                    'target': suggestion.target_column,
                    'parameters': suggestion.parameters,
                    'confidence': suggestion.confidence,
                    'reasoning': suggestion.reasoning
                })
            
            # Configuration suggestions based on data characteristics
            if data.isnull().sum().sum() > 0:
                recommendations['configuration_suggestions'].append({
                    'setting': 'remove_nulls',
                    'value': True,
                    'reason': 'Data contains missing values'
                })
            
            if data.duplicated().sum() > 0:
                recommendations['configuration_suggestions'].append({
                    'setting': 'remove_duplicates',
                    'value': True,
                    'reason': 'Data contains duplicate rows'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get AI recommendations: {e}")
            return {}


if __name__ == "__main__":
    run_etl() 