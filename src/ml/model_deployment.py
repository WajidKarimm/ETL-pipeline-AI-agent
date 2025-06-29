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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

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

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """Transform datetime columns into numerical features."""
    
    def __init__(self):
        self.datetime_columns = []
        
    def fit(self, X, y=None):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Identify datetime columns
        self.datetime_columns = X.select_dtypes(include=['datetime']).columns.tolist()
        return self
    
    def transform(self, X):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        for col in self.datetime_columns:
            if col in X_transformed.columns:
                # Extract datetime features
                X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                X_transformed[f'{col}_dayofweek'] = X_transformed[col].dt.dayofweek
                X_transformed[f'{col}_quarter'] = X_transformed[col].dt.quarter
                
                # Remove original datetime column
                X_transformed = X_transformed.drop(columns=[col])
        
        return X_transformed

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Custom categorical encoder that handles missing values and mixed data types."""
    
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
            # Convert all values to strings and handle missing values
            col_data = X[col].astype(str).fillna('MISSING')
            # Replace empty strings with MISSING
            col_data = col_data.replace('', 'MISSING')
            self.label_encoders[col].fit(col_data)
        return self
        
    def transform(self, X):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.column_names)
        
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.label_encoders:
                # Convert all values to strings and handle missing values
                col_data = X[col].astype(str).fillna('MISSING')
                # Replace empty strings with MISSING
                col_data = col_data.replace('', 'MISSING')
                # Handle unseen categories by using a default value
                try:
                    X_transformed[col] = self.label_encoders[col].transform(col_data)
                except ValueError:
                    # If there are unseen categories, use a fallback approach
                    X_transformed[col] = col_data.map(lambda x: self.label_encoders[col].transform([x])[0] 
                                                    if x in self.label_encoders[col].classes_ else 0)
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
        
        # Data analysis
        if show_progress:
            print(f"📊 Data analysis:")
            print(f"   Total samples: {len(data):,}")
            print(f"   Features: {len(data.columns) - 1}")
            print(f"   Target column: {target_column}")
            print(f"   Target type: {'Classification' if self.trainer._is_classification(data[target_column]) else 'Regression'}")
        
        # Build preprocessing pipeline
        if show_progress:
            print("🔧 Building preprocessing pipeline...")
        
        # Analyze data types more carefully
        feature_columns = [col for col in data.columns if col != target_column]
        categorical_columns = []
        numerical_columns = []
        datetime_columns = []
        
        for col in feature_columns:
            # Check if column is datetime
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                datetime_columns.append(col)
            # Check if column contains mixed types or non-numeric data
            elif pd.api.types.is_numeric_dtype(data[col]):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)
        
        if show_progress:
            print(f"   Numerical columns: {numerical_columns}")
            print(f"   Categorical columns: {categorical_columns}")
            print(f"   Datetime columns: {datetime_columns}")
        
        # Build preprocessing pipeline
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('numerical', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_columns),
                ('categorical', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
                    ('encoder', CategoricalEncoder())
                ]), categorical_columns),
                ('datetime', Pipeline([
                    ('datetime_features', DateTimeTransformer()),
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), datetime_columns)
            ],
            remainder='drop'  # Drop any remaining columns to avoid data type conflicts
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
    """Enhanced model trainer with proper splits, epochs, and automatic model selection."""
    
    def __init__(self):
        self.models = {}
        self.training_history = []
        self.best_model = None
        self.best_score = 0
        
    def suggest_model_type(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze data and suggest the best model type."""
        y = data[target_column]
        is_classification = self._is_classification(y)
        
        # Analyze data characteristics
        n_samples = len(data)
        n_features = len(data.columns) - 1  # Exclude target
        n_classes = len(np.unique(y)) if is_classification else None
        
        # Data size categories
        if n_samples < 1000:
            size_category = "small"
        elif n_samples < 10000:
            size_category = "medium"
        else:
            size_category = "large"
        
        # Feature complexity
        if n_features < 10:
            complexity = "low"
        elif n_features < 50:
            complexity = "medium"
        else:
            complexity = "high"
        
        # Suggest models based on characteristics
        if is_classification:
            if n_classes == 2:
                # Binary classification
                if size_category == "small":
                    suggestions = ["logistic_regression", "decision_tree", "random_forest"]
                elif size_category == "medium":
                    suggestions = ["random_forest", "gradient_boosting", "svm"]
                else:
                    suggestions = ["random_forest", "gradient_boosting", "neural_network"]
            else:
                # Multi-class classification
                if size_category == "small":
                    suggestions = ["decision_tree", "random_forest", "logistic_regression"]
                elif size_category == "medium":
                    suggestions = ["random_forest", "gradient_boosting", "svm"]
                else:
                    suggestions = ["random_forest", "gradient_boosting", "neural_network"]
        else:
            # Regression
            if size_category == "small":
                suggestions = ["linear_regression", "decision_tree", "random_forest"]
            elif size_category == "medium":
                suggestions = ["random_forest", "gradient_boosting", "svm"]
            else:
                suggestions = ["random_forest", "gradient_boosting", "neural_network"]
        
        return {
            "problem_type": "classification" if is_classification else "regression",
            "n_classes": n_classes,
            "data_size": size_category,
            "feature_complexity": complexity,
            "suggested_models": suggestions,
            "recommended_model": suggestions[0],
            "reasoning": f"Based on {n_samples} samples, {n_features} features, and {size_category} dataset size"
        }
    
    def train_model_with_splits(self, data: pd.DataFrame, target_column: str, 
                               model_type: str = 'auto', model_params: Dict[str, Any] = None,
                               test_size: float = 0.2, val_size: float = 0.2, 
                               random_state: int = 42, show_progress: bool = True,
                               preprocessor: ModelPreprocessor = None) -> Dict[str, Any]:
        """Train model with proper train/validation/test splits and progress tracking."""
        
        if model_params is None:
            model_params = {}
        
        # Auto-select model if requested
        if model_type == 'auto':
            suggestion = self.suggest_model_type(data, target_column)
            model_type = suggestion['recommended_model']
            if show_progress:
                print(f"🤖 Auto-selected model: {model_type}")
                print(f"📊 Reasoning: {suggestion['reasoning']}")
        
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data: train -> temp, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self._is_classification(y) else None
        )
        
        # Split temp data: train, validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, 
            stratify=y_temp if self._is_classification(y_temp) else None
        )
        
        if show_progress:
            print(f"📊 Data splits:")
            print(f"   Training: {len(X_train):,} samples ({len(X_train)/len(data)*100:.1f}%)")
            print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(data)*100:.1f}%)")
            print(f"   Test: {len(X_test):,} samples ({len(X_test)/len(data)*100:.1f}%)")
            print(f"🎯 Target: {target_column} ({'Classification' if self._is_classification(y) else 'Regression'})")
        
        # Transform data using preprocessor if provided
        if preprocessor is not None:
            if show_progress:
                print("🔧 Preprocessing training data...")
            
            # Fit preprocessor on training data and transform all splits
            X_train_transformed = preprocessor.preprocessing_pipeline.fit_transform(X_train)
            X_val_transformed = preprocessor.preprocessing_pipeline.transform(X_val)
            X_test_transformed = preprocessor.preprocessing_pipeline.transform(X_test)
        else:
            # Use data as-is (fallback)
            X_train_transformed = X_train
            X_val_transformed = X_val
            X_test_transformed = X_test
        
        # Initialize model
        model = self._create_model(model_type, model_params, y)
        
        # Train model with progress tracking
        if show_progress:
            print(f"🚀 Training {model_type} model...")
        
        # For models that support epochs/progress, show training progress
        if hasattr(model, 'n_estimators') and model_type in ['random_forest', 'gradient_boosting']:
            # Show progress for ensemble methods
            n_estimators = model_params.get('n_estimators', 100)
            if show_progress:
                print(f"   Training {n_estimators} estimators...")
                for i in range(0, n_estimators, max(1, n_estimators // 10)):
                    if i > 0:
                        print(f"   Progress: {i}/{n_estimators} estimators trained")
        
        # Train the model
        start_time = datetime.now()
        model.fit(X_train_transformed, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        if show_progress:
            print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        if show_progress:
            print("📈 Evaluating on validation set...")
        
        val_metrics = self.evaluate_model(model, X_val_transformed, y_val)
        
        # Evaluate on test set
        if show_progress:
            print("📊 Evaluating on test set...")
        
        test_metrics = self.evaluate_model(model, X_test_transformed, y_test)
        
        # Cross-validation for robustness
        if show_progress:
            print("🔄 Performing cross-validation...")
        
        cv_scores = cross_val_score(model, X_train_transformed, y_train, cv=5, scoring='accuracy' if self._is_classification(y) else 'r2')
        
        # Store model and results
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = {
            'model': model,
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'data_shape': X.shape,
            'parameters': model_params,
            'training_time': training_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Update best model
        val_score = val_metrics.get('accuracy', val_metrics.get('r2_score', 0))
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_model = model_id
        
        # Record training history
        self.training_history.append({
            'model_id': model_id,
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'data_shape': X.shape,
            'parameters': model_params,
            'training_time': training_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        if show_progress:
            print(f"🎉 Model {model_id} trained and evaluated successfully!")
            print(f"📊 Validation Score: {val_score:.4f}")
            print(f"📊 Test Score: {test_metrics.get('accuracy', test_metrics.get('r2_score', 0)):.4f}")
            print(f"🔄 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return {
            'model_id': model_id,
            'model': model,
            'model_type': model_type,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'training_time': training_time,
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
    
    def _create_model(self, model_type: str, model_params: Dict[str, Any], y: np.ndarray) -> Any:
        """Create model instance based on type."""
        is_classification = self._is_classification(y)
        
        if model_type == 'random_forest':
            if is_classification:
                return RandomForestClassifier(random_state=42, **model_params)
            else:
                return RandomForestRegressor(random_state=42, **model_params)
        elif model_type == 'gradient_boosting':
            if is_classification:
                return GradientBoostingClassifier(random_state=42, **model_params)
            else:
                return GradientBoostingRegressor(random_state=42, **model_params)
        elif model_type == 'decision_tree':
            if is_classification:
                return DecisionTreeClassifier(random_state=42, **model_params)
            else:
                return DecisionTreeRegressor(random_state=42, **model_params)
        elif model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000, **model_params)
        elif model_type == 'linear_regression':
            return LinearRegression(**model_params)
        elif model_type == 'svm':
            if is_classification:
                return SVC(random_state=42, **model_params)
            else:
                return SVR(**model_params)
        elif model_type == 'neural_network':
            if is_classification:
                return MLPClassifier(random_state=42, max_iter=500, **model_params)
            else:
                return MLPRegressor(random_state=42, max_iter=500, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _is_classification(self, y: np.ndarray) -> bool:
        """Determine if the problem is classification or regression."""
        unique_values = np.unique(y)
        return len(unique_values) <= 20  # Heuristic for classification
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        y_pred = model.predict(X)
        
        if self._is_classification(y):
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Add per-class metrics for multi-class
            if len(np.unique(y)) > 2:
                class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                metrics['per_class_metrics'] = class_report
        else:
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mae = np.mean(np.abs(y - y_pred))
            
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2_score': r2
            }
        
        return metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training runs."""
        if not self.training_history:
            return {"message": "No models trained yet"}
        
        summary = {
            'total_models': len(self.training_history),
            'best_model_id': self.best_model,
            'best_score': self.best_score,
            'models_trained': []
        }
        
        for history in self.training_history:
            model_info = {
                'model_id': history['model_id'],
                'model_type': history['model_type'],
                'training_date': history['training_date'],
                'training_time': history['training_time'],
                'val_score': history['val_metrics'].get('accuracy', history['val_metrics'].get('r2_score', 0)),
                'test_score': history['test_metrics'].get('accuracy', history['test_metrics'].get('r2_score', 0)),
                'cv_mean': history['cv_mean'],
                'cv_std': history['cv_std']
            }
            summary['models_trained'].append(model_info)
        
        return summary

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
        self.deployments = {}  # Store deployment info
        
    def train_and_deploy(self, data: pd.DataFrame, target_column: str, 
                        model_type: str = 'auto', deployment_name: str = None,
                        model_params: Dict[str, Any] = None, show_progress: bool = True) -> str:
        """Train and deploy a model with enhanced capabilities."""
        
        if deployment_name is None:
            deployment_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if model_params is None:
            model_params = {}
        
        # Analyze data and suggest model if auto-selection is requested
        if model_type == 'auto':
            suggestion = self.trainer.suggest_model_type(data, target_column)
            model_type = suggestion['recommended_model']
            if show_progress:
                print(f"🤖 Auto-selected model: {model_type}")
                print(f"📊 Problem type: {suggestion['problem_type']}")
                print(f"📊 Data size: {suggestion['data_size']}")
                print(f"📊 Feature complexity: {suggestion['feature_complexity']}")
                print(f"📊 Reasoning: {suggestion['reasoning']}")
        
        # Data analysis
        if show_progress:
            print(f"📊 Data analysis:")
            print(f"   Total samples: {len(data):,}")
            print(f"   Features: {len(data.columns) - 1}")
            print(f"   Target column: {target_column}")
            print(f"   Target type: {'Classification' if self.trainer._is_classification(data[target_column]) else 'Regression'}")
        
        # Build preprocessing pipeline
        if show_progress:
            print("🔧 Building preprocessing pipeline...")
        
        # Analyze data types more carefully
        feature_columns = [col for col in data.columns if col != target_column]
        categorical_columns = []
        numerical_columns = []
        datetime_columns = []
        
        for col in feature_columns:
            # Check if column is datetime
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                datetime_columns.append(col)
            # Check if column contains mixed types or non-numeric data
            elif pd.api.types.is_numeric_dtype(data[col]):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)
        
        if show_progress:
            print(f"   Numerical columns: {numerical_columns}")
            print(f"   Categorical columns: {categorical_columns}")
            print(f"   Datetime columns: {datetime_columns}")
        
        # Build preprocessing pipeline
        self.preprocessor.build_preprocessing_pipeline(data, target_column)
        
        # Train model with proper splits
        training_result = self.trainer.train_model_with_splits(
            data=data,
            target_column=target_column,
            model_type=model_type,
            model_params=model_params,
            show_progress=show_progress,
            preprocessor=self.preprocessor
        )
        
        model = training_result['model']
        model_id = training_result['model_id']
        
        # Create metadata
        # Prepare performance metrics including cv_mean
        performance_metrics = training_result['test_metrics'].copy()
        if 'cv_mean' in training_result:
            performance_metrics['cv_mean'] = training_result['cv_mean']
        if 'cv_std' in training_result:
            performance_metrics['cv_std'] = training_result['cv_std']
        
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=deployment_name,
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            features=[col for col in data.columns if col != target_column],
            target_column=target_column,
            performance_metrics=performance_metrics,
            training_data_size=len(data),
            preprocessing_steps=["StandardScaler", "LabelEncoder", "Feature Engineering"],
            dependencies={"scikit-learn": "1.0.0", "pandas": "1.3.0", "numpy": "1.21.0"},
            author="AI ETL Pipeline",
            description=f"Auto-trained {model_type} model for {target_column} prediction"
        )
        
        # Create deployment config
        config = DeploymentConfig(
            deployment_name=deployment_name,
            model_path=os.path.join(self.deployments_dir, deployment_name),
            api_port=5000,
            batch_size=1000,
            enable_monitoring=True,
            enable_a_b_testing=False,
            deployment_platform="local"
        )
        
        # Deploy model
        deployment_path = self.deployer.deploy_model(model, self.preprocessor, metadata, config)
        
        # Store deployment info
        self.deployments[deployment_name] = {
            'path': deployment_path,
            'metadata': metadata,
            'config': config,
            'training_result': training_result,
            'deployed_at': datetime.now().isoformat()
        }
        
        if show_progress:
            print(f"🚀 Model deployed successfully: {deployment_name}")
            print(f"📁 Deployment path: {deployment_path}")
            print(f"📊 Final test score: {training_result['test_metrics'].get('accuracy', training_result['test_metrics'].get('r2_score', 0)):.4f}")
        
        logger.info(f"Model {deployment_name} trained and deployed successfully")
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
        # First check if we have the deployment info stored locally
        if deployment_name in self.deployments:
            return self.deployments[deployment_name]
        
        # Otherwise, try to get it from the deployer
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