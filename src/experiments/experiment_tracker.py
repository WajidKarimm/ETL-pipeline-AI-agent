"""
Experiment Tracking Module for ETL Pipeline

This module provides comprehensive experiment tracking capabilities:
- MLflow integration for experiment management
- Model registry and versioning
- Performance tracking and comparison
- Hyperparameter optimization tracking
- Deployment management
- Experiment visualization and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import uuid
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_name: str
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    run_id: str
    experiment_name: str
    status: str  # 'running', 'completed', 'failed'
    start_time: datetime
    end_time: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    model_version: Optional[str] = None
    notes: str = ""

@dataclass
class ModelRegistryEntry:
    """Entry in the model registry."""
    model_name: str
    version: str
    run_id: str
    model_path: str
    registered_time: datetime
    status: str  # 'staging', 'production', 'archived'
    metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""

class ExperimentTracker:
    """
    Comprehensive experiment tracking system with MLflow integration.
    """
    
    def __init__(self, tracking_uri: str = "sqlite:///experiments.db", 
                 registry_uri: str = "sqlite:///model_registry.db",
                 experiments_dir: str = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow model registry URI
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        os.makedirs(os.path.join(experiments_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(experiments_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(experiments_dir, "artifacts"), exist_ok=True)
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(registry_uri)
            self.mlflow_available = True
            logger.info("MLflow integration enabled")
        else:
            self.mlflow_available = False
            logger.warning("MLflow not available, using local tracking only")
        
        # Local experiment tracking
        self.experiments = {}
        self.runs = {}
        self.model_registry = {}
        
        # Performance tracking
        self.tracking_metrics = {
            'total_experiments': 0,
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'models_registered': 0
        }
        
        logger.info("Experiment Tracker initialized with comprehensive tracking capabilities")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        if self.mlflow_available:
            try:
                mlflow.create_experiment(
                    name=config.experiment_name,
                    tags=config.tags
                )
                logger.info(f"MLflow experiment created: {config.experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to create MLflow experiment: {e}")
        
        # Store experiment locally
        self.experiments[experiment_id] = {
            'id': experiment_id,
            'config': config,
            'created_time': datetime.now(),
            'runs': []
        }
        
        self.tracking_metrics['total_experiments'] += 1
        logger.info(f"Experiment created: {config.experiment_name} (ID: {experiment_id})")
        
        return experiment_id
    
    def start_run(self, experiment_id: str, run_name: str = None, 
                  parameters: Dict[str, Any] = None) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_id: ID of the experiment
            run_name: Name for this run
            parameters: Initial parameters for the run
            
        Returns:
            Run ID
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        run_id = str(uuid.uuid4())
        run_name = run_name or f"run_{len(self.experiments[experiment_id]['runs']) + 1}"
        
        # Start MLflow run if available
        if self.mlflow_available:
            try:
                experiment = self.experiments[experiment_id]
                mlflow.set_experiment(experiment['config'].experiment_name)
                mlflow.start_run(run_name=run_name)
                if parameters:
                    mlflow.log_params(parameters)
                logger.info(f"MLflow run started: {run_name}")
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        
        # Create local run record
        run = ExperimentResult(
            run_id=run_id,
            experiment_name=self.experiments[experiment_id]['config'].experiment_name,
            status='running',
            start_time=datetime.now(),
            parameters=parameters or {}
        )
        
        self.runs[run_id] = run
        self.experiments[experiment_id]['runs'].append(run_id)
        self.tracking_metrics['total_runs'] += 1
        
        logger.info(f"Run started: {run_name} (ID: {run_id})")
        return run_id
    
    def log_parameter(self, run_id: str, key: str, value: Any):
        """Log a parameter for the current run."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        self.runs[run_id].parameters[key] = value
        
        if self.mlflow_available:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter to MLflow: {e}")
    
    def log_metric(self, run_id: str, key: str, value: float, step: int = None):
        """Log a metric for the current run."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        self.runs[run_id].metrics[key] = value
        
        if self.mlflow_available:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric to MLflow: {e}")
    
    def log_artifact(self, run_id: str, local_path: str, artifact_path: str = None):
        """Log an artifact for the current run."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Copy artifact to experiments directory
        artifact_dir = os.path.join(self.experiments_dir, "artifacts", run_id)
        os.makedirs(artifact_dir, exist_ok=True)
        
        if os.path.isfile(local_path):
            import shutil
            dest_path = os.path.join(artifact_dir, os.path.basename(local_path))
            shutil.copy2(local_path, dest_path)
            self.runs[run_id].artifacts.append(dest_path)
        
        if self.mlflow_available:
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")
    
    def log_model(self, run_id: str, model, model_name: str = "model"):
        """Log a model for the current run."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Save model locally
        model_dir = os.path.join(self.experiments_dir, "models", run_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        try:
            import joblib
            joblib.dump(model, model_path)
            self.runs[run_id].model_path = model_path
            logger.info(f"Model saved locally: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model locally: {e}")
        
        # Log to MLflow if available
        if self.mlflow_available:
            try:
                mlflow.sklearn.log_model(model, model_name)
                logger.info(f"Model logged to MLflow: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")
    
    def end_run(self, run_id: str, status: str = "completed", notes: str = ""):
        """End an experiment run."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self.runs[run_id]
        run.status = status
        run.end_time = datetime.now()
        run.notes = notes
        
        if status == "completed":
            self.tracking_metrics['successful_runs'] += 1
        else:
            self.tracking_metrics['failed_runs'] += 1
        
        if self.mlflow_available:
            try:
                mlflow.end_run()
                logger.info(f"MLflow run ended: {run_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
        
        # Save run results
        self._save_run_results(run)
        
        logger.info(f"Run ended: {run_id} (Status: {status})")
    
    def register_model(self, run_id: str, model_name: str, 
                      description: str = "", status: str = "staging") -> str:
        """
        Register a model from a run.
        
        Args:
            run_id: ID of the run containing the model
            model_name: Name for the registered model
            description: Description of the model
            status: Model status (staging, production, archived)
            
        Returns:
            Model version
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self.runs[run_id]
        if not run.model_path:
            raise ValueError(f"No model found in run {run_id}")
        
        # Generate version
        if model_name not in self.model_registry:
            version = "1.0.0"
        else:
            versions = [entry.version for entry in self.model_registry[model_name]]
            latest_version = max(versions, key=lambda v: [int(x) for x in v.split('.')])
            major, minor, patch = map(int, latest_version.split('.'))
            version = f"{major}.{minor}.{patch + 1}"
        
        # Create registry entry
        entry = ModelRegistryEntry(
            model_name=model_name,
            version=version,
            run_id=run_id,
            model_path=run.model_path,
            registered_time=datetime.now(),
            status=status,
            metrics=run.metrics.copy(),
            description=description
        )
        
        if model_name not in self.model_registry:
            self.model_registry[model_name] = []
        
        self.model_registry[model_name].append(entry)
        self.tracking_metrics['models_registered'] += 1
        
        # Register with MLflow if available
        if self.mlflow_available:
            try:
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model",
                    name=model_name
                )
                logger.info(f"Model registered with MLflow: {model_name} v{version}")
            except Exception as e:
                logger.warning(f"Failed to register model with MLflow: {e}")
        
        logger.info(f"Model registered: {model_name} v{version}")
        return version
    
    def get_experiment_runs(self, experiment_id: str) -> List[ExperimentResult]:
        """Get all runs for an experiment."""
        if experiment_id not in self.experiments:
            return []
        
        run_ids = self.experiments[experiment_id]['runs']
        return [self.runs[run_id] for run_id in run_ids if run_id in self.runs]
    
    def get_best_run(self, experiment_id: str, metric: str, 
                    maximize: bool = True) -> Optional[ExperimentResult]:
        """Get the best run based on a metric."""
        runs = self.get_experiment_runs(experiment_id)
        if not runs:
            return None
        
        # Filter runs with the metric
        runs_with_metric = [run for run in runs if metric in run.metrics]
        if not runs_with_metric:
            return None
        
        if maximize:
            return max(runs_with_metric, key=lambda r: r.metrics[metric])
        else:
            return min(runs_with_metric, key=lambda r: r.metrics[metric])
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        comparison = {
            'runs': {},
            'metrics_comparison': {},
            'parameters_comparison': {}
        }
        
        for run_id in run_ids:
            if run_id in self.runs:
                run = self.runs[run_id]
                comparison['runs'][run_id] = {
                    'experiment_name': run.experiment_name,
                    'status': run.status,
                    'start_time': run.start_time.isoformat(),
                    'end_time': run.end_time.isoformat() if run.end_time else None,
                    'duration': (run.end_time - run.start_time).total_seconds() if run.end_time else None
                }
                
                # Compare metrics
                for metric, value in run.metrics.items():
                    if metric not in comparison['metrics_comparison']:
                        comparison['metrics_comparison'][metric] = {}
                    comparison['metrics_comparison'][metric][run_id] = value
                
                # Compare parameters
                for param, value in run.parameters.items():
                    if param not in comparison['parameters_comparison']:
                        comparison['parameters_comparison'][param] = {}
                    comparison['parameters_comparison'][param][run_id] = value
        
        return comparison
    
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate a comprehensive experiment report."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        runs = self.get_experiment_runs(experiment_id)
        
        # Calculate statistics
        completed_runs = [run for run in runs if run.status == "completed"]
        failed_runs = [run for run in runs if run.status == "failed"]
        
        # Aggregate metrics
        all_metrics = {}
        for run in completed_runs:
            for metric, value in run.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate metric statistics
        metric_stats = {}
        for metric, values in all_metrics.items():
            metric_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        
        report = {
            'experiment_id': experiment_id,
            'experiment_name': experiment['config'].experiment_name,
            'description': experiment['config'].description,
            'created_time': experiment['created_time'].isoformat(),
            'total_runs': len(runs),
            'completed_runs': len(completed_runs),
            'failed_runs': len(failed_runs),
            'success_rate': len(completed_runs) / len(runs) if runs else 0,
            'metric_statistics': metric_stats,
            'runs_summary': [
                {
                    'run_id': run.run_id,
                    'status': run.status,
                    'start_time': run.start_time.isoformat(),
                    'end_time': run.end_time.isoformat() if run.end_time else None,
                    'duration': (run.end_time - run.start_time).total_seconds() if run.end_time else None,
                    'metrics': run.metrics,
                    'parameters': run.parameters
                }
                for run in runs
            ]
        }
        
        return report
    
    def _save_run_results(self, run: ExperimentResult):
        """Save run results to file."""
        try:
            run_file = os.path.join(self.experiments_dir, "runs", f"{run.run_id}.json")
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            run_dict = {
                'run_id': run.run_id,
                'experiment_name': run.experiment_name,
                'status': run.status,
                'start_time': run.start_time.isoformat(),
                'end_time': run.end_time.isoformat() if run.end_time else None,
                'parameters': convert_numpy_types(run.parameters),
                'metrics': convert_numpy_types(run.metrics),
                'artifacts': run.artifacts,
                'model_path': run.model_path,
                'model_version': run.model_version,
                'notes': run.notes
            }
            
            with open(run_file, 'w') as f:
                json.dump(run_dict, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save run results: {e}")
    
    def get_tracking_metrics(self) -> Dict[str, Any]:
        """Get tracking performance metrics."""
        return self.tracking_metrics.copy()
    
    def export_experiment_data(self, experiment_id: str, filepath: str):
        """Export experiment data to JSON file."""
        try:
            report = self.generate_experiment_report(experiment_id)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Experiment data exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export experiment data: {e}") 