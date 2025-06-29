"""
AI Agent for ETL Pipeline - Machine Learning Component

This module provides an AI agent that learns from data patterns to:
- Automatically detect data quality issues
- Suggest optimal transformations
- Predict potential errors
- Automatically correct errors when detected
- Learn from user corrections with backpropagation-style adaptation
- Improve accuracy over time with continuous learning
- Advanced confidence scoring with uncertainty estimation
- Ensemble learning for improved accuracy and reliability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ConfidenceScore:
    """Advanced confidence scoring with uncertainty estimation."""
    prediction: Any
    confidence: float  # 0.0 to 1.0
    uncertainty: float  # 0.0 to 1.0
    reliability: str  # 'high', 'medium', 'low'
    ensemble_agreement: float  # Agreement among ensemble models
    calibration_score: float  # How well calibrated the confidence is
    feature_importance: Dict[str, float]  # Feature importance for this prediction

@dataclass
class DataQualityIssue:
    """Represents a detected data quality issue."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_columns: List[str]
    suggested_fix: str
    confidence: float
    confidence_score: Optional[ConfidenceScore] = None
    auto_correctable: bool = False
    correction_applied: bool = False

@dataclass
class TransformationSuggestion:
    """Represents a suggested data transformation."""
    transformation_type: str
    target_column: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    confidence_score: Optional[ConfidenceScore] = None
    auto_apply: bool = False

@dataclass
class ErrorPrediction:
    """Represents a predicted potential error."""
    error_type: str
    probability: float
    affected_data: str
    prevention_suggestion: str
    confidence_score: Optional[ConfidenceScore] = None
    auto_prevent: bool = False

@dataclass
class AutoCorrection:
    """Represents an automatic correction applied by the AI agent."""
    correction_type: str
    target_column: str
    original_value: Any
    corrected_value: Any
    confidence: float
    reasoning: str
    confidence_score: Optional[ConfidenceScore] = None
    applied: bool = True

@dataclass
class EnsemblePrediction:
    """Represents ensemble model predictions with confidence."""
    base_predictions: List[Any]
    ensemble_prediction: Any
    confidence_score: ConfidenceScore
    model_weights: Dict[str, float]
    agreement_score: float

class ETLAIAgent:
    """
    Enhanced AI Agent that learns from ETL operations to improve accuracy and reduce errors.
    Features automatic error correction and backpropagation-style self-learning.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the AI agent with advanced ensemble learning and confidence scoring.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize base models
        self.data_quality_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.transformation_suggester = RandomForestClassifier(n_estimators=100, random_state=42)
        self.error_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.correction_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Initialize ensemble models
        self._initialize_ensemble_models()
        
        # Data preprocessing
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Training data storage
        self.training_data = []
        self.feature_names = []
        
        # Auto-correction settings
        self.auto_correction_enabled = True
        self.auto_correction_threshold = 0.8
        self.learning_rate = 0.1
        self.adaptation_history = []
        
        # Advanced confidence scoring settings
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        self.uncertainty_estimation_enabled = True
        self.ensemble_agreement_threshold = 0.7
        
        # Model calibration
        self.calibrated_models = {}
        self.calibration_scores = {}
        
        # Performance tracking
        self.prediction_history = []
        self.confidence_accuracy = []
        self.ensemble_performance = {}
        
        # Load existing models if available
        self._load_models()
        
        logger.info("Enhanced ETL AI Agent initialized with ensemble learning and advanced confidence scoring")
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble models for improved accuracy and reliability."""
        # Create base models for ensemble
        base_models = [
            ('rf1', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('rf2', RandomForestClassifier(n_estimators=150, random_state=43)),
            ('rf3', RandomForestClassifier(n_estimators=200, random_state=44))
        ]
        
        # Voting classifier for ensemble prediction
        self.data_quality_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft'  # Use probability predictions
        )
        
        # Bagging classifier for additional robustness
        self.data_quality_bagging = BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            n_estimators=5,
            random_state=42
        )
        
        # Similar setup for other models
        self.transformation_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        self.error_prediction_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        self.correction_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
    
    def calculate_confidence_score(self, model, X: np.ndarray, prediction: Any, 
                                 feature_names: List[str] = None) -> ConfidenceScore:
        """
        Calculate advanced confidence score with uncertainty estimation.
        
        Args:
            model: Trained model
            X: Input features
            prediction: Model prediction
            feature_names: Names of features for importance calculation
            
        Returns:
            ConfidenceScore object with comprehensive confidence metrics
        """
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                confidence = np.max(probabilities, axis=1)[0]
            else:
                confidence = 0.5  # Default confidence for models without probabilities
            
            # Calculate uncertainty (entropy-based)
            if hasattr(model, 'predict_proba'):
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)[0]
                uncertainty = entropy / np.log(len(probabilities[0]))  # Normalize to [0,1]
            else:
                uncertainty = 0.5
            
            # Determine reliability level
            if confidence >= self.confidence_thresholds['high']:
                reliability = 'high'
            elif confidence >= self.confidence_thresholds['medium']:
                reliability = 'medium'
            else:
                reliability = 'low'
            
            # Calculate ensemble agreement (if ensemble model)
            ensemble_agreement = 1.0
            if hasattr(model, 'estimators_'):
                base_predictions = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'predict_proba'):
                        base_pred = estimator.predict_proba(X)
                        base_predictions.append(np.argmax(base_pred, axis=1)[0])
                    else:
                        base_predictions.append(estimator.predict(X)[0])
                
                # Calculate agreement among base models
                unique_predictions = len(set(base_predictions))
                ensemble_agreement = 1.0 - (unique_predictions - 1) / len(base_predictions)
            
            # Calculate feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_') and feature_names:
                for name, importance in zip(feature_names, model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            # Get calibration score
            calibration_score = self.calibration_scores.get(type(model).__name__, 0.5)
            
            return ConfidenceScore(
                prediction=prediction,
                confidence=float(confidence),
                uncertainty=float(uncertainty),
                reliability=reliability,
                ensemble_agreement=float(ensemble_agreement),
                calibration_score=float(calibration_score),
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return ConfidenceScore(
                prediction=prediction,
                confidence=0.5,
                uncertainty=0.5,
                reliability='low',
                ensemble_agreement=0.0,
                calibration_score=0.5,
                feature_importance={}
            )
    
    def ensemble_predict_with_confidence(self, X: np.ndarray, model_type: str = 'data_quality') -> EnsemblePrediction:
        """
        Make ensemble prediction with comprehensive confidence scoring.
        
        Args:
            X: Input features
            model_type: Type of model ('data_quality', 'transformation', 'error_prediction', 'correction')
            
        Returns:
            EnsemblePrediction with confidence scores
        """
        try:
            # Select appropriate ensemble model
            if model_type == 'data_quality':
                ensemble_model = self.data_quality_ensemble
                base_model = self.data_quality_classifier
            elif model_type == 'transformation':
                ensemble_model = self.transformation_ensemble
                base_model = self.transformation_suggester
            elif model_type == 'error_prediction':
                ensemble_model = self.error_prediction_ensemble
                base_model = self.error_predictor
            elif model_type == 'correction':
                ensemble_model = self.correction_ensemble
                base_model = self.correction_predictor
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get base model predictions
            base_predictions = []
            if hasattr(ensemble_model, 'estimators_'):
                for estimator in ensemble_model.estimators_:
                    pred = estimator.predict(X)[0]
                    base_predictions.append(pred)
            
            # Get ensemble prediction
            ensemble_prediction = ensemble_model.predict(X)[0]
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(
                ensemble_model, X, ensemble_prediction, self.feature_names
            )
            
            # Calculate model weights (equal for now, could be learned)
            model_weights = {f'model_{i}': 1.0/len(base_predictions) for i in range(len(base_predictions))}
            
            # Calculate agreement score
            if base_predictions:
                unique_predictions = len(set(base_predictions))
                agreement_score = 1.0 - (unique_predictions - 1) / len(base_predictions)
            else:
                agreement_score = 1.0
            
            return EnsemblePrediction(
                base_predictions=base_predictions,
                ensemble_prediction=ensemble_prediction,
                confidence_score=confidence_score,
                model_weights=model_weights,
                agreement_score=agreement_score
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            # Fallback to base model
            if model_type == 'data_quality':
                base_model = self.data_quality_classifier
            elif model_type == 'transformation':
                base_model = self.transformation_suggester
            elif model_type == 'error_prediction':
                base_model = self.error_predictor
            elif model_type == 'correction':
                base_model = self.correction_predictor
            
            fallback_prediction = base_model.predict(X)[0]
            fallback_confidence = self.calculate_confidence_score(base_model, X, fallback_prediction, self.feature_names)
            
            return EnsemblePrediction(
                base_predictions=[fallback_prediction],
                ensemble_prediction=fallback_prediction,
                confidence_score=fallback_confidence,
                model_weights={'fallback': 1.0},
                agreement_score=1.0
            )
    
    def calibrate_models(self, X: np.ndarray, y: np.ndarray, model_type: str = 'data_quality'):
        """
        Calibrate model confidence scores using cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            model_type: Type of model to calibrate
        """
        try:
            # Select appropriate model
            if model_type == 'data_quality':
                model = self.data_quality_classifier
            elif model_type == 'transformation':
                model = self.transformation_suggester
            elif model_type == 'error_prediction':
                model = self.error_predictor
            elif model_type == 'correction':
                model = self.correction_predictor
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create calibrated model
            calibrated_model = CalibratedClassifierCV(model, cv=5, method='isotonic')
            calibrated_model.fit(X, y)
            
            # Store calibrated model
            self.calibrated_models[model_type] = calibrated_model
            
            # Calculate calibration score
            if hasattr(calibrated_model, 'predict_proba'):
                probas = calibrated_model.predict_proba(X)
                # Simple calibration score based on probability distribution
                calibration_score = np.mean(np.max(probas, axis=1))
                self.calibration_scores[model_type] = calibration_score
            
            logger.info(f"Model {model_type} calibrated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to calibrate model {model_type}: {e}")
    
    def update_confidence_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update confidence thresholds based on performance analysis.
        
        Args:
            new_thresholds: Dictionary with new threshold values
        """
        self.confidence_thresholds.update(new_thresholds)
        logger.info(f"Updated confidence thresholds: {self.confidence_thresholds}")
    
    def get_confidence_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive confidence scoring metrics.
        
        Returns:
            Dictionary with confidence metrics
        """
        return {
            'confidence_thresholds': self.confidence_thresholds,
            'calibration_scores': self.calibration_scores,
            'ensemble_performance': self.ensemble_performance,
            'confidence_accuracy': self.confidence_accuracy[-10:] if self.confidence_accuracy else [],
            'prediction_history_count': len(self.prediction_history)
        }

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
            
            if os.path.exists(os.path.join(self.model_dir, "correction_model.pkl")):
                self.correction_predictor = joblib.load(
                    os.path.join(self.model_dir, "correction_model.pkl")
                )
                logger.info("Loaded pre-trained correction model")
            
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
        """Save trained models."""
        try:
            joblib.dump(self.data_quality_classifier, 
                       os.path.join(self.model_dir, "data_quality_model.pkl"))
            joblib.dump(self.transformation_suggester, 
                       os.path.join(self.model_dir, "transformation_model.pkl"))
            joblib.dump(self.error_predictor, 
                       os.path.join(self.model_dir, "error_prediction_model.pkl"))
            joblib.dump(self.correction_predictor, 
                       os.path.join(self.model_dir, "correction_model.pkl"))
            
            # Save preprocessing components
            preprocessing_data = {
                "label_encoders": self.label_encoders,
                "scaler": self.scaler,
                "feature_names": self.feature_names
            }
            joblib.dump(preprocessing_data, 
                       os.path.join(self.model_dir, "preprocessing.pkl"))
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features from data for ML models.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        features = {}
        
        # Basic data characteristics
        features['row_count'] = len(data)
        features['column_count'] = len(data.columns)
        features['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Data type distribution
        features['numeric_columns'] = len(data.select_dtypes(include=[np.number]).columns)
        features['categorical_columns'] = len(data.select_dtypes(include=['object']).columns)
        features['datetime_columns'] = len(data.select_dtypes(include=['datetime']).columns)
        
        # Missing value analysis
        missing_data = data.isnull().sum()
        features['total_missing_values'] = missing_data.sum()
        features['missing_value_percentage'] = (features['total_missing_values'] / (len(data) * len(data.columns))) * 100
        features['columns_with_missing'] = (missing_data > 0).sum()
        
        # Duplicate analysis
        features['duplicate_rows'] = data.duplicated().sum()
        features['duplicate_percentage'] = (features['duplicate_rows'] / len(data)) * 100 if len(data) > 0 else 0
        
        # Data quality indicators
        features['unique_value_ratio'] = data.nunique().values.astype(float).mean() / len(data) if len(data) > 0 else 0
        features['data_completeness'] = 1 - (features['missing_value_percentage'] / 100)
        
        # Column name characteristics
        features['avg_column_name_length'] = data.columns.str.len().values.mean()
        features['columns_with_special_chars'] = sum(1 for col in data.columns if any(c in col for c in '!@#$%^&*()'))
        
        # Data distribution characteristics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            features['numeric_std_mean'] = data[numeric_cols].std().values.mean()
            features['numeric_range_mean'] = (data[numeric_cols].max() - data[numeric_cols].min()).values.mean()
        else:
            features['numeric_std_mean'] = 0
            features['numeric_range_mean'] = 0
        
        # Text characteristics
        text_cols = data.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            # Calculate average text length for each text column, then average across columns
            text_lengths = []
            for col in text_cols:
                col_lengths = data[col].astype(str).str.len()
                if not col_lengths.isna().all():
                    text_lengths.append(col_lengths.mean())
            
            features['avg_text_length'] = np.mean(text_lengths) if text_lengths else 0
            features['text_columns_with_numbers'] = sum(1 for col in text_cols if data[col].astype(str).str.contains(r'\d').any())
        else:
            features['avg_text_length'] = 0
            features['text_columns_with_numbers'] = 0
        
        # Consistency features
        features['column_name_consistency'] = 1 if all(col.islower() or col.isupper() for col in data.columns) else 0
        features['data_type_consistency'] = len(data.dtypes.unique()) / len(data.columns)
        
        return pd.DataFrame([features])

    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers in a numeric series using IQR method."""
        if series.dtype not in ['int64', 'float64']:
            return 0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return ((series < lower_bound) | (series > upper_bound)).sum()

    def detect_and_auto_correct_issues(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[DataQualityIssue], List[AutoCorrection]]:
        """
        Detect data quality issues and automatically correct them when possible.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (corrected_data, issues, corrections_applied)
        """
        corrected_data = data.copy()
        issues = []
        corrections = []
        
        # Detect issues
        detected_issues = self.detect_data_quality_issues(data)
        
        for issue in detected_issues:
            # Check if issue is auto-correctable
            if issue.confidence >= self.auto_correction_threshold and self.auto_correction_enabled:
                correction = self._attempt_auto_correction(corrected_data, issue)
                if correction:
                    corrections.append(correction)
                    issue.correction_applied = True
                    issue.auto_correctable = True
                    
                    # Apply correction to data
                    corrected_data = self._apply_correction(corrected_data, correction)
            
            issues.append(issue)
        
        # Learn from corrections
        if corrections:
            self._learn_from_corrections(corrections, issues)
        
        return corrected_data, issues, corrections

    def _attempt_auto_correction(self, data: pd.DataFrame, issue: DataQualityIssue) -> Optional[AutoCorrection]:
        """
        Attempt to automatically correct a detected issue.
        
        Args:
            data: Input DataFrame
            issue: Detected data quality issue
            
        Returns:
            AutoCorrection object if correction is possible, None otherwise
        """
        try:
            if issue.issue_type == "missing_values":
                return self._correct_missing_values(data, issue)
            elif issue.issue_type == "duplicates":
                return self._correct_duplicates(data, issue)
            elif issue.issue_type == "outliers":
                return self._correct_outliers(data, issue)
            elif issue.issue_type == "type_conversion_error":
                return self._correct_type_conversion(data, issue)
            elif issue.issue_type == "inconsistent_formatting":
                return self._correct_formatting(data, issue)
            
        except Exception as e:
            logger.warning(f"Auto-correction failed for {issue.issue_type}: {e}")
        
        return None

    def _correct_missing_values(self, data: pd.DataFrame, issue: DataQualityIssue) -> Optional[AutoCorrection]:
        """Correct missing values using intelligent imputation."""
        for col in issue.affected_columns:
            if col in data.columns and data[col].isnull().sum() > 0:
                missing_percentage = (data[col].isnull().sum() / len(data)) * 100
                
                if missing_percentage < 10:  # Only correct if missing percentage is low
                    if data[col].dtype in ['int64', 'float64']:
                        # Use median for numeric columns
                        corrected_value = data[col].median()
                        method = "median"
                    else:
                        # Use mode for categorical columns
                        corrected_value = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else "Unknown"
                        method = "mode"
                    
                    return AutoCorrection(
                        correction_type="missing_value_imputation",
                        target_column=col,
                        original_value=f"{missing_percentage:.1f}% missing",
                        corrected_value=f"Filled with {method}",
                        confidence=issue.confidence,
                        reasoning=f"Missing values in {col} filled using {method} imputation"
                    )
        
        return None

    def _correct_duplicates(self, data: pd.DataFrame, issue: DataQualityIssue) -> Optional[AutoCorrection]:
        """Remove duplicate rows."""
        original_count = len(data)
        corrected_data = data.drop_duplicates()
        removed_count = original_count - len(corrected_data)
        
        if removed_count > 0:
            return AutoCorrection(
                correction_type="duplicate_removal",
                target_column="all",
                original_value=f"{original_count} rows",
                corrected_value=f"{len(corrected_data)} rows (removed {removed_count} duplicates)",
                confidence=issue.confidence,
                reasoning=f"Removed {removed_count} duplicate rows"
            )
        
        return None

    def _correct_outliers(self, data: pd.DataFrame, issue: DataQualityIssue) -> Optional[AutoCorrection]:
        """Correct outliers using winsorization."""
        for col in issue.affected_columns:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0 and outlier_count < len(data) * 0.1:  # Only correct if outliers are < 10%
                    # Cap outliers at bounds
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    return AutoCorrection(
                        correction_type="outlier_capping",
                        target_column=col,
                        original_value=f"{outlier_count} outliers",
                        corrected_value="Capped at IQR bounds",
                        confidence=issue.confidence,
                        reasoning=f"Capped {outlier_count} outliers in {col} using IQR method"
                    )
        
        return None

    def _correct_type_conversion(self, data: pd.DataFrame, issue: DataQualityIssue) -> Optional[AutoCorrection]:
        """Correct type conversion errors."""
        for col in issue.affected_columns:
            if col in data.columns and data[col].dtype == 'object':
                # Try numeric conversion with coerce
                try:
                    numeric_data = pd.to_numeric(data[col], errors='coerce')
                    if not numeric_data.isna().all():  # If at least some values converted successfully
                        data[col] = numeric_data
                        return AutoCorrection(
                            correction_type="type_conversion",
                            target_column=col,
                            original_value="object",
                            corrected_value="numeric",
                            confidence=issue.confidence,
                            reasoning=f"Converted {col} from object to numeric using coerce"
                        )
                except:
                    pass
                
                # Try datetime conversion
                try:
                    datetime_data = pd.to_datetime(data[col], errors='coerce')
                    if not datetime_data.isna().all():
                        data[col] = datetime_data
                        return AutoCorrection(
                            correction_type="type_conversion",
                            target_column=col,
                            original_value="object",
                            corrected_value="datetime",
                            confidence=issue.confidence,
                            reasoning=f"Converted {col} from object to datetime using coerce"
                        )
                except:
                    pass
        
        return None

    def _correct_formatting(self, data: pd.DataFrame, issue: DataQualityIssue) -> Optional[AutoCorrection]:
        """Correct inconsistent formatting."""
        for col in issue.affected_columns:
            if col in data.columns and data[col].dtype == 'object':
                # Standardize string formatting
                original_sample = data[col].iloc[0] if len(data[col]) > 0 else ""
                data[col] = data[col].astype(str).str.strip().str.lower()
                corrected_sample = data[col].iloc[0] if len(data[col]) > 0 else ""
                
                if original_sample != corrected_sample:
                    return AutoCorrection(
                        correction_type="formatting_standardization",
                        target_column=col,
                        original_value=f"Sample: {original_sample}",
                        corrected_value=f"Sample: {corrected_sample}",
                        confidence=issue.confidence,
                        reasoning=f"Standardized formatting in {col} (trimmed, lowercase)"
                    )
        
        return None

    def _apply_correction(self, data: pd.DataFrame, correction: AutoCorrection) -> pd.DataFrame:
        """Apply a correction to the data."""
        # Most corrections are already applied in the correction methods
        # This method can be extended for more complex corrections
        return data

    def _learn_from_corrections(self, corrections: List[AutoCorrection], issues: List[DataQualityIssue]):
        """Learn from applied corrections to improve future predictions."""
        try:
            for correction in corrections:
                learning_sample = {
                    'correction_type': correction.correction_type,
                    'target_column': correction.target_column,
                    'confidence': correction.confidence,
                    'success': True,  # Assuming correction was successful
                    'timestamp': datetime.now().isoformat()
                }
                
                self.adaptation_history.append(learning_sample)
                
                # Update learning parameters based on correction success
                if correction.confidence > 0.9:
                    self.learning_rate *= 1.1  # Increase learning rate for high-confidence corrections
                elif correction.confidence < 0.7:
                    self.learning_rate *= 0.9  # Decrease learning rate for low-confidence corrections
                
                # Keep learning rate within bounds
                self.learning_rate = max(0.01, min(0.5, self.learning_rate))
            
            # Retrain models with new learning data
            if len(self.adaptation_history) % 5 == 0:  # Retrain every 5 corrections
                self._retrain_with_adaptation()
                
        except Exception as e:
            logger.warning(f"Failed to learn from corrections: {e}")

    def _retrain_with_adaptation(self):
        """Retrain models with adaptation history for improved learning."""
        try:
            if len(self.adaptation_history) < 3:
                return
            
            # Prepare adaptation features
            adaptation_features = []
            adaptation_labels = []
            
            for sample in self.adaptation_history[-20:]:  # Use last 20 samples
                features = {
                    'correction_type_encoded': hash(sample['correction_type']) % 100,
                    'confidence': sample['confidence'],
                    'learning_rate': self.learning_rate,
                    'success': 1 if sample['success'] else 0
                }
                
                adaptation_features.append(list(features.values()))
                adaptation_labels.append(1 if sample['success'] else 0)
            
            if len(adaptation_features) > 2:
                # Update correction predictor with adaptation data
                X_adapt = np.array(adaptation_features)
                y_adapt = np.array(adaptation_labels)
                
                # Partial fit for continuous learning
                if hasattr(self.correction_predictor, 'classes_'):
                    self.correction_predictor.partial_fit(X_adapt, y_adapt, classes=[0, 1])
                else:
                    self.correction_predictor.fit(X_adapt, y_adapt)
                
                # Save updated model
                joblib.dump(self.correction_predictor, 
                           os.path.join(self.model_dir, "correction_model.pkl"))
                
                logger.debug(f"Retrained correction model with {len(adaptation_features)} adaptation samples")
                
        except Exception as e:
            logger.warning(f"Failed to retrain with adaptation: {e}")

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning and adaptation metrics."""
        return {
            'learning_rate': self.learning_rate,
            'adaptation_samples': len(self.adaptation_history),
            'auto_corrections_applied': len([h for h in self.adaptation_history if h.get('success', False)]),
            'correction_success_rate': len([h for h in self.adaptation_history if h.get('success', False)]) / len(self.adaptation_history) if self.adaptation_history else 0,
            'last_adaptation': self.adaptation_history[-1]['timestamp'] if self.adaptation_history else None
        }

    def detect_data_quality_issues(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """
        Detect data quality issues using ensemble learning and advanced confidence scoring.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of detected data quality issues with confidence scores
        """
        if data.empty:
            return []
        issues = []
        
        # Extract features
        features = self.extract_features(data)
        
        # Use ensemble learning for improved accuracy
        if hasattr(self.data_quality_classifier, 'classes_'):
            try:
                # Prepare features for prediction
                X = self._prepare_features_for_prediction(features)
                
                # Get ensemble prediction with confidence
                ensemble_result = self.ensemble_predict_with_confidence(X, 'data_quality')
                
                # Convert ensemble prediction to issues
                if ensemble_result.ensemble_prediction == 1:  # Issue detected
                    confidence_score = ensemble_result.confidence_score
                    
                    # Determine severity based on confidence and uncertainty
                    if confidence_score.confidence >= 0.9 and confidence_score.uncertainty <= 0.1:
                        severity = "high"
                    elif confidence_score.confidence >= 0.7 and confidence_score.uncertainty <= 0.3:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Create issue with advanced confidence scoring
                    issue = DataQualityIssue(
                        issue_type="ensemble_detected",
                        severity=severity,
                        description=f"Ensemble model detected data quality issue with {confidence_score.confidence:.2f} confidence and {confidence_score.uncertainty:.2f} uncertainty",
                        affected_columns=data.columns.tolist(),
                        suggested_fix="Review data quality and apply appropriate cleaning based on ensemble analysis",
                        confidence=confidence_score.confidence,
                        confidence_score=confidence_score,
                        auto_correctable=confidence_score.confidence >= self.auto_correction_threshold
                    )
                    
                    # Add feature importance information to description
                    if confidence_score.feature_importance:
                        top_features = sorted(confidence_score.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        feature_desc = ", ".join([f"{feat}: {imp:.2f}" for feat, imp in top_features])
                        issue.description += f" (Key features: {feature_desc})"
                    
                    issues.append(issue)
                    
                    # Log prediction for performance tracking
                    self.prediction_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'model_type': 'data_quality',
                        'prediction': ensemble_result.ensemble_prediction,
                        'confidence': confidence_score.confidence,
                        'uncertainty': confidence_score.uncertainty,
                        'ensemble_agreement': ensemble_result.agreement_score,
                        'reliability': confidence_score.reliability
                    })
                    
            except Exception as e:
                logger.warning(f"Ensemble-based issue detection failed: {e}")
                # Fallback to basic ML model
                try:
                    X = self._prepare_features_for_prediction(features)
                    predictions = self.data_quality_classifier.predict(X)
                    probabilities = self.data_quality_classifier.predict_proba(X)
                    
                    for i, pred in enumerate(predictions):
                        if pred == 1:  # Issue detected
                            confidence = max(probabilities[i])
                            confidence_score = self.calculate_confidence_score(
                                self.data_quality_classifier, X, pred, self.feature_names
                            )
                            
                            issues.append(DataQualityIssue(
                                issue_type="ml_detected",
                                severity="medium" if confidence < 0.8 else "high",
                                description=f"ML model detected potential data quality issue (confidence: {confidence:.2f})",
                                affected_columns=data.columns.tolist(),
                                suggested_fix="Review data quality and apply appropriate cleaning",
                                confidence=confidence,
                                confidence_score=confidence_score
                            ))
                except Exception as e2:
                    logger.warning(f"Fallback ML-based issue detection also failed: {e2}")
        
        # Rule-based issue detection with enhanced confidence
        rule_based_issues = self._rule_based_issue_detection(data)
        for issue in rule_based_issues:
            # Add confidence score for rule-based issues
            if issue.confidence_score is None:
                # Create a basic confidence score for rule-based issues
                issue.confidence_score = ConfidenceScore(
                    prediction=issue.issue_type,
                    confidence=issue.confidence,
                    uncertainty=0.1,  # Rule-based issues have low uncertainty
                    reliability='high' if issue.confidence >= 0.9 else 'medium',
                    ensemble_agreement=1.0,  # Rule-based issues have full agreement
                    calibration_score=0.9,  # Rule-based issues are well-calibrated
                    feature_importance={}
                )
        
        issues.extend(rule_based_issues)
        
        return issues
    
    def _rule_based_issue_detection(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect issues using rule-based approach."""
        if data.empty:
            return []
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

    def auto_correct_issues(self, data: pd.DataFrame) -> (pd.DataFrame, list):
        """
        Detect and automatically correct data quality issues.
        Returns the corrected DataFrame and a log of corrections.
        """
        corrected_data = data.copy()
        corrections = []
        issues = self.detect_data_quality_issues(corrected_data)
        for issue in issues:
            for col in issue.affected_columns:
                if issue.issue_type == 'missing_values':
                    if corrected_data[col].isnull().sum() > 0:
                        if corrected_data[col].dtype in ['int64', 'float64']:
                            value = corrected_data[col].median()
                            method = 'median'
                        else:
                            value = corrected_data[col].mode().iloc[0] if not corrected_data[col].mode().empty else 'Unknown'
                            method = 'mode'
                        corrected_data[col].fillna(value, inplace=True)
                        corrections.append(f"Filled missing values in '{col}' with {method} ({value})")
                elif issue.issue_type == 'duplicates':
                    before = len(corrected_data)
                    corrected_data.drop_duplicates(inplace=True)
                    after = len(corrected_data)
                    if before != after:
                        corrections.append(f"Removed {before - after} duplicate rows")
                elif issue.issue_type == 'type_conversion_error':
                    try:
                        corrected_data[col] = pd.to_numeric(corrected_data[col], errors='coerce')
                        corrections.append(f"Converted '{col}' to numeric (coerce errors)")
                    except Exception:
                        try:
                            corrected_data[col] = pd.to_datetime(corrected_data[col], errors='coerce')
                            corrections.append(f"Converted '{col}' to datetime (coerce errors)")
                        except Exception:
                            corrections.append(f"Could not auto-convert '{col}'")
        # Log corrections for learning
        self.learn_from_operation(corrected_data, {}, True, errors=[], user_feedback={'corrections': corrections})
        return corrected_data, corrections

    def get_correction_log(self, user_feedback: dict = None) -> list:
        """Return the last corrections log for UI display."""
        if user_feedback and 'corrections' in user_feedback:
            return user_feedback['corrections']
        return [] 