"""
Model Deployment and Preprocessing System

This module provides comprehensive model deployment capabilities for the AI-powered ETL pipeline:
- Model preprocessing and feature engineering
- Model serialization and versioning
- Deployment to various platforms
- Model serving and API endpoints
- Performance monitoring and A/B testing
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import pickle
import yaml
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import hashlib
import zipfile
import shutil

# ML libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# API and serving
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.logger import get_logger

logger = get_logger(__name__)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Custom categorical encoder that handles missing values."""
    
    def __init__(self):
        self.label_encoders = {}
        self.column_names = None
        
    def fit(self, X, y=None):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        self.column_names = X.columns.tolist()
        for col in X.columns:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(X[col].fillna('MISSING'))
        return self
        
    def transform(self, X):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.column_names)
        
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.label_encoders:
                X_transformed[col] = self.label_encoders[col].transform(X[col].fillna('MISSING'))
        return X_transformed.values  # Return numpy array

@dataclass
class ModelMetadata:
    """Model metadata for versioning and tracking."""
    model_id: str
    model_name: str
    version: str
    created_at: str
    model_type: str
    features: List[str]
    target_column: str
    performance_metrics: Dict[str, float]
    training_data_size: int
    preprocessing_steps: List[str]
    dependencies: Dict[str, str]
    author: str
    description: str

@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    deployment_name: str
    model_path: str
    api_port: int = 5000
    batch_size: int = 1000
    enable_monitoring: bool = True
    enable_a_b_testing: bool = False
    deployment_platform: str = "local"  # local, cloud, kubernetes
    scaling_config: Dict[str, Any] = None

class ModelPreprocessor:
    """Handles data preprocessing for model training and inference."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preprocessing_pipeline = None
        self.feature_columns = []
        self.target_column = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.datetime_columns = []
        
    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure and suggest preprocessing steps."""
        analysis = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'data_types': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'unique_counts': {col: data[col].nunique() for col in data.columns},
            'suggested_preprocessing': []
        }
        
        # Identify column types
        for col in data.columns:
            if data[col].dtype in ['object', 'category']:
                if data[col].nunique() <= 50:  # Low cardinality categorical
                    analysis['suggested_preprocessing'].append({
                        'column': col,
                        'action': 'label_encode',
                        'reason': f'Categorical column with {data[col].nunique()} unique values'
                    })
                else:  # High cardinality categorical
                    analysis['suggested_preprocessing'].append({
                        'column': col,
                        'reason': f'High cardinality categorical column with {data[col].nunique()} unique values'
                    })
            elif data[col].dtype in ['int64', 'float64']:
                analysis['suggested_preprocessing'].append({
                    'column': col,
                    'action': 'scale',
                    'reason': 'Numerical column requiring scaling'
                })
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                analysis['suggested_preprocessing'].append({
                    'column': col,
                    'action': 'extract_features',
                    'reason': 'Datetime column - extract year, month, day, etc.'
                })
        
        return analysis
    
    def build_preprocessing_pipeline(self, data: pd.DataFrame, target_column: str, 
                                   preprocessing_config: Dict[str, Any] = None) -> Pipeline:
        """Build a preprocessing pipeline based on data analysis."""
        
        self.target_column = target_column
        self.feature_columns = [col for col in data.columns if col != target_column]
        
        # Analyze data types
        self.categorical_columns = data[self.feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = data[self.feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.datetime_columns = data[self.feature_columns].select_dtypes(include=['datetime']).columns.tolist()
        
        # Build preprocessing transformers
        transformers = []
        
        # Numerical preprocessing
        if self.numerical_columns:
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numerical', numerical_transformer, self.numerical_columns))
        
        # Categorical preprocessing
        if self.categorical_columns:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
                ('encoder', CategoricalEncoder())
            ])
            transformers.append(('categorical', categorical_transformer, self.categorical_columns))
        
        # Create preprocessing pipeline
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        return self.preprocessing_pipeline
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the preprocessing pipeline and transform the data."""
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not built. Call build_preprocessing_pipeline first.")
        
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        X_transformed = self.preprocessing_pipeline.fit_transform(X)
        return X_transformed, y
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessing pipeline."""
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not fitted.")
        
        X = data[self.feature_columns]
        return self.preprocessing_pipeline.transform(X)
    
    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline."""
        if self.preprocessing_pipeline is None:
            raise ValueError("No preprocessing pipeline to save.")
        
        pipeline_data = {
            'pipeline': self.preprocessing_pipeline,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'datetime_columns': self.datetime_columns
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Preprocessing pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a saved preprocessing pipeline."""
        pipeline_data = joblib.load(filepath)
        
        self.preprocessing_pipeline = pipeline_data['pipeline']
        self.feature_columns = pipeline_data['feature_columns']
        self.target_column = pipeline_data['target_column']
        self.categorical_columns = pipeline_data['categorical_columns']
        self.numerical_columns = pipeline_data['numerical_columns']
        self.datetime_columns = pipeline_data['datetime_columns']
        
        logger.info(f"Preprocessing pipeline loaded from {filepath}")

class ModelTrainer:
    """Handles model training with various algorithms."""
    
    def __init__(self):
        self.models = {}
        self.training_history = []
        
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest',
                   model_params: Dict[str, Any] = None) -> Any:
        """Train a model with specified type and parameters."""
        
        if model_params is None:
            model_params = {}
        
        # Select model type
        if model_type == 'random_forest':
            if self._is_classification(y):
                model = RandomForestClassifier(**model_params)
            else:
                model = RandomForestRegressor(**model_params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**model_params)
        elif model_type == 'linear_regression':
            model = LinearRegression(**model_params)
        elif model_type == 'svm':
            if self._is_classification(y):
                model = SVC(**model_params)
            else:
                model = SVR(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X, y)
        
        # Store model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = model
        
        # Record training
        self.training_history.append({
            'model_id': model_id,
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'data_shape': X.shape,
            'parameters': model_params
        })
        
        logger.info(f"Model {model_id} trained successfully")
        return model, model_id
    
    def _is_classification(self, y: np.ndarray) -> bool:
        """Determine if the problem is classification or regression."""
        unique_values = np.unique(y)
        return len(unique_values) <= 20  # Heuristic for classification
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        
        if self._is_classification(y_test):
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2
            }
        
        return metrics

class ModelDeployer:
    """Handles model deployment and serving."""
    
    def __init__(self, deployment_dir: str = "deployments"):
        self.deployment_dir = deployment_dir
        os.makedirs(deployment_dir, exist_ok=True)
        self.deployments = {}
        
    def deploy_model(self, model: Any, preprocessor: ModelPreprocessor, 
                    metadata: ModelMetadata, config: DeploymentConfig) -> str:
        """Deploy a model with preprocessing pipeline."""
        
        # Create deployment directory
        deployment_path = os.path.join(self.deployment_dir, config.deployment_name)
        os.makedirs(deployment_path, exist_ok=True)
        
        # Save model and preprocessor
        model_path = os.path.join(deployment_path, "model.pkl")
        preprocessor_path = os.path.join(deployment_path, "preprocessor.pkl")
        metadata_path = os.path.join(deployment_path, "metadata.json")
        config_path = os.path.join(deployment_path, "config.yaml")
        
        # Save components
        joblib.dump(model, model_path)
        preprocessor.save_pipeline(preprocessor_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        # Create deployment package
        package_path = self._create_deployment_package(deployment_path, config.deployment_name)
        
        # Store deployment info
        self.deployments[config.deployment_name] = {
            'path': deployment_path,
            'package_path': package_path,
            'metadata': metadata,
            'config': config,
            'deployed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model deployed successfully: {config.deployment_name}")
        return deployment_path
    
    def _create_deployment_package(self, deployment_path: str, deployment_name: str) -> str:
        """Create a deployment package (ZIP file)."""
        package_path = os.path.join(self.deployment_dir, f"{deployment_name}.zip")
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(deployment_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, deployment_path)
                    zipf.write(file_path, arcname)
        
        return package_path
    
    def load_deployment(self, deployment_name: str) -> Tuple[Any, ModelPreprocessor, ModelMetadata]:
        """Load a deployed model."""
        if deployment_name not in self.deployments:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        deployment = self.deployments[deployment_name]
        deployment_path = deployment['path']
        
        # Load components
        model = joblib.load(os.path.join(deployment_path, "model.pkl"))
        preprocessor = ModelPreprocessor()
        preprocessor.load_pipeline(os.path.join(deployment_path, "preprocessor.pkl"))
        
        with open(os.path.join(deployment_path, "metadata.json"), 'r') as f:
            metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)
        
        return model, preprocessor, metadata
    
    def create_api_server(self, deployment_name: str) -> Optional[Flask]:
        """Create a Flask API server for model serving."""
        if not FLASK_AVAILABLE:
            logger.warning("Flask not available. Cannot create API server.")
            return None
        
        try:
            model, preprocessor, metadata = self.load_deployment(deployment_name)
            
            app = Flask(__name__)
            
            @app.route('/predict', methods=['POST'])
            def predict():
                try:
                    data = request.get_json()
                    input_data = pd.DataFrame(data['data'])
                    
                    # Preprocess data
                    X_transformed = preprocessor.transform(input_data)
                    
                    # Make prediction
                    predictions = model.predict(X_transformed)
                    
                    return jsonify({
                        'predictions': predictions.tolist(),
                        'model_id': metadata.model_id,
                        'timestamp': datetime.now().isoformat()
                    })
                
                except Exception as e:
                    return jsonify({'error': str(e)}), 400
            
            @app.route('/health', methods=['GET'])
            def health():
                return jsonify({'status': 'healthy', 'model_id': metadata.model_id})
            
            @app.route('/metadata', methods=['GET'])
            def get_metadata():
                return jsonify(asdict(metadata))
            
            logger.info(f"API server created for deployment: {deployment_name}")
            return app
            
        except Exception as e:
            logger.error(f"Failed to create API server: {e}")
            return None
    
    def deploy_to_mlflow(self, model: Any, preprocessor: ModelPreprocessor, 
                        metadata: ModelMetadata, experiment_name: str = "etl_ai_models"):
        """Deploy model to MLflow (if available)."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Cannot deploy to MLflow.")
            return None
        
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log preprocessor
                mlflow.sklearn.log_model(preprocessor.preprocessing_pipeline, "preprocessor")
                
                # Log metadata
                mlflow.log_params({
                    'model_id': metadata.model_id,
                    'model_type': metadata.model_type,
                    'target_column': metadata.target_column,
                    'feature_count': len(metadata.features)
                })
                
                # Log metrics
                for metric_name, metric_value in metadata.performance_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log artifacts
                mlflow.log_artifact("models/")
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Model deployed to MLflow with run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to deploy to MLflow: {e}")
            return None

class ModelManager:
    """High-level model management interface."""
    
    def __init__(self, models_dir: str = "models", deployments_dir: str = "deployments"):
        self.models_dir = models_dir
        self.deployments_dir = deployments_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(deployments_dir, exist_ok=True)
        
        self.preprocessor = ModelPreprocessor()
        self.trainer = ModelTrainer()
        self.deployer = ModelDeployer(deployments_dir)
        
    def train_and_deploy(self, data: pd.DataFrame, target_column: str, 
                        model_type: str = 'random_forest', deployment_name: str = None,
                        model_params: Dict[str, Any] = None) -> str:
        """Complete pipeline: train and deploy a model."""
        
        # Analyze data
        analysis = self.preprocessor.analyze_data(data)
        logger.info(f"Data analysis completed: {len(analysis['suggested_preprocessing'])} preprocessing steps suggested")
        
        # Build preprocessing pipeline
        self.preprocessor.build_preprocessing_pipeline(data, target_column)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_transformed, y = self.preprocessor.fit_transform(data)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
        
        # Train model
        model, model_id = self.trainer.train_model(X_train, y_train, model_type, model_params)
        
        # Evaluate model
        metrics = self.trainer.evaluate_model(model, X_test, y_test)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=f"{model_type}_model",
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            features=self.preprocessor.feature_columns,
            target_column=target_column,
            performance_metrics=metrics,
            training_data_size=len(data),
            preprocessing_steps=[step['action'] for step in analysis['suggested_preprocessing']],
            dependencies={
                'scikit-learn': '1.0.0',
                'pandas': '1.3.0',
                'numpy': '1.21.0'
            },
            author="ETL AI Agent",
            description=f"Auto-generated {model_type} model for {target_column} prediction"
        )
        
        # Deploy model
        if deployment_name is None:
            deployment_name = f"{model_type}_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = DeploymentConfig(
            deployment_name=deployment_name,
            model_path=os.path.join(self.models_dir, f"{model_id}.pkl"),
            api_port=5000,
            enable_monitoring=True
        )
        
        deployment_path = self.deployer.deploy_model(model, self.preprocessor, metadata, config)
        
        logger.info(f"Model training and deployment completed: {deployment_path}")
        return deployment_path
    
    def predict(self, deployment_name: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a deployed model."""
        model, preprocessor, metadata = self.deployer.load_deployment(deployment_name)
        
        X_transformed = preprocessor.transform(data)
        predictions = model.predict(X_transformed)
        
        return predictions
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        return [
            {
                'name': name,
                'metadata': deployment['metadata'],
                'deployed_at': deployment['deployed_at'],
                'path': deployment['path']
            }
            for name, deployment in self.deployer.deployments.items()
        ]
    
    def get_deployment_info(self, deployment_name: str) -> Dict[str, Any]:
        """Get detailed information about a deployment."""
        if deployment_name not in self.deployer.deployments:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        deployment = self.deployer.deployments[deployment_name]
        return {
            'name': deployment_name,
            'metadata': deployment['metadata'],
            'config': deployment['config'],
            'deployed_at': deployment['deployed_at'],
            'path': deployment['path']
        }

# Convenience functions
def create_model_manager() -> ModelManager:
    """Create a model manager instance."""
    return ModelManager()

def quick_train_and_deploy(data: pd.DataFrame, target_column: str, 
                          model_type: str = 'random_forest') -> str:
    """Quick training and deployment function."""
    manager = create_model_manager()
    return manager.train_and_deploy(data, target_column, model_type)

def load_deployed_model(deployment_name: str) -> Tuple[Any, ModelPreprocessor, ModelMetadata]:
    """Load a deployed model."""
    manager = create_model_manager()
    return manager.deployer.load_deployment(deployment_name) 