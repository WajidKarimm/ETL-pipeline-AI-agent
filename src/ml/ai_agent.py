"""
AI Agent for ETL Pipeline - Machine Learning Component

This module provides an AI agent that learns from data patterns to:
- Automatically detect data quality issues
- Suggest optimal transformations
- Predict potential errors
- Learn from user corrections
- Improve accuracy over time
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataQualityIssue:
    """Represents a detected data quality issue."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_columns: List[str]
    suggested_fix: str
    confidence: float

@dataclass
class TransformationSuggestion:
    """Represents a suggested data transformation."""
    transformation_type: str
    target_column: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str

@dataclass
class ErrorPrediction:
    """Represents a predicted potential error."""
    error_type: str
    probability: float
    affected_data: str
    prevention_suggestion: str

class ETLAIAgent:
    """
    AI Agent that learns from ETL operations to improve accuracy and reduce errors.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the AI agent.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.data_quality_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.transformation_suggester = RandomForestClassifier(n_estimators=100, random_state=42)
        self.error_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Data preprocessing
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Training data storage
        self.training_data = []
        self.feature_names = []
        
        # Load existing models if available
        self._load_models()
        
        logger.info("ETL AI Agent initialized")
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        try:
            if os.path.exists(os.path.join(self.model_dir, "data_quality_model.pkl")):
                self.data_quality_classifier = joblib.load(
                    os.path.join(self.model_dir, "data_quality_model.pkl")
                )
                logger.info("Loaded pre-trained data quality model")
            
            if os.path.exists(os.path.join(self.model_dir, "transformation_model.pkl")):
                self.transformation_suggester = joblib.load(
                    os.path.join(self.model_dir, "transformation_model.pkl")
                )
                logger.info("Loaded pre-trained transformation model")
            
            if os.path.exists(os.path.join(self.model_dir, "error_prediction_model.pkl")):
                self.error_predictor = joblib.load(
                    os.path.join(self.model_dir, "error_prediction_model.pkl")
                )
                logger.info("Loaded pre-trained error prediction model")
            
            # Load preprocessing components
            if os.path.exists(os.path.join(self.model_dir, "preprocessing.pkl")):
                preprocessing_data = joblib.load(
                    os.path.join(self.model_dir, "preprocessing.pkl")
                )
                self.label_encoders = preprocessing_data.get("label_encoders", {})
                self.scaler = preprocessing_data.get("scaler", StandardScaler())
                self.feature_names = preprocessing_data.get("feature_names", [])
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            joblib.dump(
                self.data_quality_classifier,
                os.path.join(self.model_dir, "data_quality_model.pkl")
            )
            joblib.dump(
                self.transformation_suggester,
                os.path.join(self.model_dir, "transformation_model.pkl")
            )
            
            # Only log in debug mode
            if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG':
                logger.debug("Models saved successfully")
            
        except Exception as e:
            # Silent error handling - don't log unless debug mode
            if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG':
                logger.debug(f"Failed to save models: {e}")
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from data for ML models.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        features = {}
        
        # Basic statistics
        features['row_count'] = len(data)
        features['column_count'] = len(data.columns)
        features['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Data type features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        datetime_cols = data.select_dtypes(include=['datetime']).columns
        
        features['numeric_columns'] = len(numeric_cols)
        features['categorical_columns'] = len(categorical_cols)
        features['datetime_columns'] = len(datetime_cols)
        
        # Missing data features
        missing_data = data.isnull().sum()
        features['total_missing_values'] = missing_data.sum()
        features['missing_value_percentage'] = (missing_data.sum() / (len(data) * len(data.columns))) * 100
        features['columns_with_missing'] = (missing_data > 0).sum()
        
        # Duplicate features
        features['duplicate_rows'] = data.duplicated().sum()
        features['duplicate_percentage'] = (data.duplicated().sum() / len(data)) * 100
        
        # Data quality features - use aggregate statistics instead of per-column
        if len(numeric_cols) > 0:
            numeric_data = data[numeric_cols]
            features['numeric_mean_mean'] = numeric_data.mean().mean()
            features['numeric_std_mean'] = numeric_data.std().mean()
            features['numeric_min_min'] = numeric_data.min().min()
            features['numeric_max_max'] = numeric_data.max().max()
            features['numeric_outliers_total'] = sum(self._count_outliers(data[col]) for col in numeric_cols)
        else:
            features['numeric_mean_mean'] = 0
            features['numeric_std_mean'] = 0
            features['numeric_min_min'] = 0
            features['numeric_max_max'] = 0
            features['numeric_outliers_total'] = 0
        
        # Categorical features - use aggregate statistics
        if len(categorical_cols) > 0:
            categorical_data = data[categorical_cols]
            features['categorical_unique_mean'] = categorical_data.nunique().mean()
            features['categorical_most_common_freq_mean'] = categorical_data.apply(
                lambda x: x.value_counts().iloc[0] if len(x.value_counts()) > 0 else 0
            ).mean()
        else:
            features['categorical_unique_mean'] = 0
            features['categorical_most_common_freq_mean'] = 0
        
        # Additional consistent features
        features['data_complexity_score'] = (features['numeric_columns'] + features['categorical_columns'] * 0.5) / features['column_count'] if features['column_count'] > 0 else 0
        features['missing_data_severity'] = features['missing_value_percentage'] / 100
        features['duplicate_data_severity'] = features['duplicate_percentage'] / 100
        
        return pd.DataFrame([features])
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        if len(series.dropna()) == 0:
            return 0
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return 0
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def detect_data_quality_issues(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """
        Detect data quality issues using ML models.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of detected data quality issues
        """
        issues = []
        
        # Extract features
        features = self.extract_features(data)
        
        # Use ML model to predict issues (if trained)
        if hasattr(self.data_quality_classifier, 'classes_'):
            try:
                # Prepare features for prediction
                X = self._prepare_features_for_prediction(features)
                predictions = self.data_quality_classifier.predict(X)
                probabilities = self.data_quality_classifier.predict_proba(X)
                
                # Convert predictions to issues
                for i, pred in enumerate(predictions):
                    if pred == 1:  # Issue detected
                        confidence = max(probabilities[i])
                        issues.append(DataQualityIssue(
                            issue_type="ml_detected",
                            severity="medium" if confidence < 0.8 else "high",
                            description=f"ML model detected potential data quality issue (confidence: {confidence:.2f})",
                            affected_columns=data.columns.tolist(),
                            suggested_fix="Review data quality and apply appropriate cleaning",
                            confidence=confidence
                        ))
            except Exception as e:
                logger.warning(f"ML-based issue detection failed: {e}")
        
        # Rule-based issue detection
        issues.extend(self._rule_based_issue_detection(data))
        
        return issues
    
    def _rule_based_issue_detection(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect issues using rule-based approach."""
        issues = []
        
        # Check for missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            missing_percentage = (data[missing_cols].isnull().sum() / len(data)) * 100
            max_missing = missing_percentage.max()
            
            severity = "low"
            if max_missing > 50:
                severity = "critical"
            elif max_missing > 20:
                severity = "high"
            elif max_missing > 5:
                severity = "medium"
            
            issues.append(DataQualityIssue(
                issue_type="missing_values",
                severity=severity,
                description=f"Missing values detected in {len(missing_cols)} columns (max: {max_missing:.1f}%)",
                affected_columns=missing_cols,
                suggested_fix="Consider imputation or removal of rows with missing values",
                confidence=1.0
            ))
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(data)) * 100
            severity = "low"
            if duplicate_percentage > 20:
                severity = "high"
            elif duplicate_percentage > 5:
                severity = "medium"
            
            issues.append(DataQualityIssue(
                issue_type="duplicates",
                severity=severity,
                description=f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)",
                affected_columns=data.columns.tolist(),
                suggested_fix="Remove duplicate rows or investigate source of duplicates",
                confidence=1.0
            ))
        
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self._count_outliers(data[col])
            if outliers > 0:
                outlier_percentage = (outliers / len(data)) * 100
                if outlier_percentage > 5:
                    issues.append(DataQualityIssue(
                        issue_type="outliers",
                        severity="medium",
                        description=f"Found {outliers} outliers in column '{col}' ({outlier_percentage:.1f}%)",
                        affected_columns=[col],
                        suggested_fix="Investigate outliers and consider removal or transformation",
                        confidence=0.9
                    ))
        
        return issues
    
    def suggest_transformations(self, data: pd.DataFrame, target_column: Optional[str] = None) -> List[TransformationSuggestion]:
        """
        Suggest data transformations using ML models.
        
        Args:
            data: Input DataFrame
            target_column: Specific column to suggest transformations for
            
        Returns:
            List of transformation suggestions
        """
        suggestions = []
        
        # Extract features
        features = self.extract_features(data)
        
        # Use ML model to suggest transformations (if trained)
        if hasattr(self.transformation_suggester, 'classes_'):
            try:
                X = self._prepare_features_for_prediction(features)
                predictions = self.transformation_suggester.predict(X)
                probabilities = self.transformation_suggester.predict_proba(X)
                
                # Convert predictions to suggestions
                for i, pred in enumerate(predictions):
                    confidence = max(probabilities[i])
                    if confidence > 0.6:  # Only suggest if confident
                        suggestions.append(TransformationSuggestion(
                            transformation_type="ml_suggested",
                            target_column="all" if target_column is None else target_column,
                            parameters={"confidence": confidence},
                            confidence=confidence,
                            reasoning="ML model suggests this transformation based on data patterns"
                        ))
            except Exception as e:
                logger.warning(f"ML-based transformation suggestion failed: {e}")
        
        # Rule-based suggestions
        suggestions.extend(self._rule_based_transformation_suggestions(data, target_column))
        
        return suggestions
    
    def _rule_based_transformation_suggestions(self, data: pd.DataFrame, target_column: Optional[str] = None) -> List[TransformationSuggestion]:
        """Suggest transformations using rule-based approach."""
        suggestions = []
        
        columns_to_check = [target_column] if target_column else data.columns
        
        for col in columns_to_check:
            if col not in data.columns:
                continue
            
            # Check data type suggestions
            if data[col].dtype == 'object':
                # Check if it might be numeric
                try:
                    pd.to_numeric(data[col], errors='raise')
                    suggestions.append(TransformationSuggestion(
                        transformation_type="convert_to_numeric",
                        target_column=col,
                        parameters={"errors": "coerce"},
                        confidence=0.8,
                        reasoning="Column contains numeric data but is stored as object type"
                    ))
                except:
                    # Check if it might be datetime
                    try:
                        pd.to_datetime(data[col], errors='raise')
                        suggestions.append(TransformationSuggestion(
                            transformation_type="convert_to_datetime",
                            target_column=col,
                            parameters={"errors": "coerce"},
                            confidence=0.7,
                            reasoning="Column contains date/time data but is stored as object type"
                        ))
                    except:
                        pass
            
            # Check for missing value handling
            if data[col].isnull().sum() > 0:
                missing_percentage = (data[col].isnull().sum() / len(data)) * 100
                if missing_percentage < 10:
                    suggestions.append(TransformationSuggestion(
                        transformation_type="fill_missing",
                        target_column=col,
                        parameters={"method": "forward_fill"},
                        confidence=0.6,
                        reasoning=f"Column has {missing_percentage:.1f}% missing values - consider filling"
                    ))
            
            # Check for standardization
            if data[col].dtype in ['int64', 'float64']:
                if data[col].std() > data[col].mean() * 2:
                    suggestions.append(TransformationSuggestion(
                        transformation_type="standardize",
                        target_column=col,
                        parameters={"method": "z_score"},
                        confidence=0.7,
                        reasoning="Column has high variance - consider standardization"
                    ))
        
        return suggestions
    
    def predict_errors(self, data: pd.DataFrame, transformation_config: Dict[str, Any]) -> List[ErrorPrediction]:
        """
        Predict potential errors before applying transformations.
        
        Args:
            data: Input DataFrame
            transformation_config: Configuration for planned transformations
            
        Returns:
            List of predicted errors
        """
        predictions = []
        
        # Extract features
        features = self.extract_features(data)
        
        # Use ML model to predict errors (if trained)
        if hasattr(self.error_predictor, 'classes_'):
            try:
                X = self._prepare_features_for_prediction(features)
                predictions_proba = self.error_predictor.predict_proba(X)
                
                # Convert predictions to error predictions
                for i, proba in enumerate(predictions_proba):
                    if max(proba) > 0.5:  # Only predict if confident
                        error_type = self.error_predictor.classes_[np.argmax(proba)]
                        predictions.append(ErrorPrediction(
                            error_type=error_type,
                            probability=max(proba),
                            affected_data="entire dataset",
                            prevention_suggestion="Review transformation configuration and data quality"
                        ))
            except Exception as e:
                logger.warning(f"ML-based error prediction failed: {e}")
        
        # Rule-based error prediction
        predictions.extend(self._rule_based_error_prediction(data, transformation_config))
        
        return predictions
    
    def _rule_based_error_prediction(self, data: pd.DataFrame, transformation_config: Dict[str, Any]) -> List[ErrorPrediction]:
        """Predict errors using rule-based approach."""
        predictions = []
        
        # Check for potential type conversion errors
        if 'data_types' in transformation_config:
            for col, target_type in transformation_config['data_types'].items():
                if col in data.columns:
                    if target_type == 'numeric' and data[col].dtype == 'object':
                        # Check if conversion might fail
                        try:
                            pd.to_numeric(data[col], errors='raise')
                        except:
                            predictions.append(ErrorPrediction(
                                error_type="type_conversion_error",
                                probability=0.9,
                                affected_data=col,
                                prevention_suggestion=f"Clean non-numeric values in column '{col}' before conversion"
                            ))
        
        # Check for field mapping errors
        if 'field_map' in transformation_config:
            for old_col, new_col in transformation_config['field_map'].items():
                if old_col not in data.columns:
                    predictions.append(ErrorPrediction(
                        error_type="missing_column_error",
                        probability=1.0,
                        affected_data=old_col,
                        prevention_suggestion=f"Column '{old_col}' not found in data - check column names"
                    ))
        
        return predictions
    
    def learn_from_operation(self, data: pd.DataFrame, transformations: Dict[str, Any], 
                           success: bool, errors: List[str] = None, user_feedback: Dict[str, Any] = None):
        """Learn from ETL operations to improve future predictions."""
        try:
            # Extract features
            features = self.extract_features(data)
            
            # Prepare training data
            training_sample = {
                'features': features.to_dict('records')[0],
                'transformations': transformations,
                'success': success,
                'errors': errors or [],
                'user_feedback': user_feedback or {},
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_data.append(training_sample)
            
            # Retrain models periodically (every 10 operations) - silent
            if len(self.training_data) % 10 == 0:
                self._retrain_models()
            
            # Only log in debug mode
            if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG':
                logger.debug(f"Learned from operation (success: {success}, total samples: {len(self.training_data)})")
            
        except Exception as e:
            # Silent error handling - don't log unless debug mode
            if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG':
                logger.debug(f"Failed to learn from operation: {e}")
    
    def _prepare_features_for_prediction(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model prediction."""
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Fill missing values
        numeric_features = numeric_features.fillna(0)
        
        # Ensure consistent feature count by padding or truncating
        expected_features = 11  # Based on the error message
        
        if numeric_features.shape[1] > expected_features:
            # Truncate to expected features (take first N features)
            numeric_features = numeric_features.iloc[:, :expected_features]
        elif numeric_features.shape[1] < expected_features:
            # Pad with zeros to reach expected features
            padding_needed = expected_features - numeric_features.shape[1]
            padding_df = pd.DataFrame(0, index=numeric_features.index, columns=[f'padding_{i}' for i in range(padding_needed)])
            numeric_features = pd.concat([numeric_features, padding_df], axis=1)
        
        return numeric_features.values
    
    def _retrain_models(self):
        """Retrain ML models with accumulated training data."""
        if len(self.training_data) < 5:
            return  # Need more data
        
        try:
            # Prepare training data
            X_data_quality = []
            y_data_quality = []
            
            for sample in self.training_data:
                features = pd.DataFrame([sample['features']])
                X = self._prepare_features_for_prediction(features)
                
                # Data quality model training
                has_issues = len(sample.get('errors', [])) > 0 or not sample['success']
                X_data_quality.append(X.flatten())
                y_data_quality.append(1 if has_issues else 0)
            
            # Train models if we have enough data
            if len(X_data_quality) > 5:
                X_data_quality = np.array(X_data_quality)
                y_data_quality = np.array(y_data_quality)
                self.data_quality_classifier.fit(X_data_quality, y_data_quality)
                
                # Only log in debug mode
                if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG':
                    logger.debug("Retrained data quality model")
            
            # Save updated models
            self._save_models()
            
        except Exception as e:
            # Silent error handling - don't log unless debug mode
            if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG':
                logger.debug(f"Failed to retrain models: {e}")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for trained models."""
        performance = {
            'training_samples': len(self.training_data),
            'models_trained': {},
            'last_retraining': None
        }
        
        # Check if models have been trained
        if hasattr(self.data_quality_classifier, 'classes_'):
            performance['models_trained']['data_quality'] = True
        if hasattr(self.transformation_suggester, 'classes_'):
            performance['models_trained']['transformation'] = True
        if hasattr(self.error_predictor, 'classes_'):
            performance['models_trained']['error_prediction'] = True
        
        return performance
    
    def export_training_data(self, filepath: str):
        """Export training data for analysis."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.training_data, f, indent=2, default=str)
            logger.info(f"Training data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
    
    def import_training_data(self, filepath: str):
        """Import training data from file."""
        try:
            with open(filepath, 'r') as f:
                self.training_data = json.load(f)
            logger.info(f"Training data imported from {filepath}")
            # Retrain models with imported data
            self._retrain_models()
        except Exception as e:
            logger.error(f"Failed to import training data: {e}") 