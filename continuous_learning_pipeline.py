#!/usr/bin/env python3
"""
Continuous Learning Pipeline for AI Agent Enhancement
Automatically processes large datasets and continuously improves AI agent capabilities
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule

# Add src to path
sys.path.append('.')

from src.ml.ai_agent import ETLAIAgent
from src.transformers.clean_transformer import CleanTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLearningPipeline:
    """
    Automated pipeline for continuous AI agent learning and improvement
    """
    
    def __init__(self, data_directories=None, model_update_interval=24):
        """
        Initialize the continuous learning pipeline
        
        Args:
            data_directories: List of directories containing datasets to process
            model_update_interval: Hours between model retraining
        """
        self.ai_agent = ETLAIAgent()
        self.transformer = CleanTransformer()
        self.data_directories = data_directories or ['data', 'demo_data']
        self.model_update_interval = model_update_interval
        
        # Learning metrics tracking
        self.learning_metrics = {
            'total_datasets_processed': 0,
            'total_rows_processed': 0,
            'total_issues_detected': 0,
            'total_corrections_applied': 0,
            'accuracy_improvements': [],
            'processing_times': [],
            'last_model_update': None,
            'learning_rate_history': []
        }
        
        # Performance tracking
        self.performance_history = []
        
        # Create directories for learning data
        os.makedirs('learning_data', exist_ok=True)
        os.makedirs('learning_reports', exist_ok=True)
        
        logger.info("Continuous Learning Pipeline initialized")
    
    def discover_datasets(self):
        """Discover all available datasets in the configured directories"""
        datasets = []
        
        for directory in self.data_directories:
            if os.path.exists(directory):
                for file_path in Path(directory).rglob('*'):
                    if file_path.suffix.lower() in ['.csv', '.json', '.xlsx', '.parquet']:
                        datasets.append(str(file_path))
        
        logger.info(f"Discovered {len(datasets)} datasets")
        return datasets
    
    def process_dataset(self, file_path):
        """
        Process a single dataset and learn from it
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            dict: Processing results and learning metrics
        """
        start_time = time.time()
        results = {
            'file_path': file_path,
            'success': False,
            'rows_processed': 0,
            'issues_detected': 0,
            'corrections_applied': 0,
            'processing_time': 0,
            'error': None,
            'learning_gained': False
        }
        
        try:
            logger.info(f"Processing dataset: {file_path}")
            
            # Load dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            results['rows_processed'] = len(df)
            
            # AI Agent Analysis
            issues = self.ai_agent.detect_data_quality_issues(df)
            results['issues_detected'] = len(issues)
            
            # Auto-correction
            corrected_data, detected_issues, corrections = self.ai_agent.detect_and_auto_correct_issues(df)
            results['corrections_applied'] = len(corrections)
            
            # Transformation suggestions
            suggestions = self.ai_agent.suggest_transformations(df)
            
            # Error predictions
            config = {'data_types': {col: 'auto' for col in df.columns}}
            predictions = self.ai_agent.predict_errors(df, config)
            
            # Learn from this operation
            self.ai_agent.learn_from_operation(
                data=df,
                transformations={'transform': 'continuous_learning'},
                success=True,
                errors=[],
                user_feedback={'source': 'continuous_learning', 'file': file_path}
            )
            
            # Test transformation
            try:
                transformed = self.transformer.transform(df)
                transformation_success = True
            except Exception as e:
                transformation_success = False
                logger.warning(f"Transformation failed for {file_path}: {e}")
            
            results['success'] = True
            results['learning_gained'] = True
            
            # Store learning data
            self._store_learning_data(file_path, df, issues, corrections, suggestions, predictions)
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error processing {file_path}: {e}")
        
        results['processing_time'] = time.time() - start_time
        return results
    
    def _store_learning_data(self, file_path, df, issues, corrections, suggestions, predictions):
        """Store learning data for analysis and improvement"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_data/learning_{timestamp}_{os.path.basename(file_path)}.json"
        
        learning_data = {
            'timestamp': timestamp,
            'file_path': file_path,
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'data_types': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'issues_detected': [{'type': i.issue_type, 'description': i.description, 'confidence': i.confidence} for i in issues],
            'corrections_applied': [{'type': c.correction_type, 'reasoning': c.reasoning, 'confidence': c.confidence} for c in corrections],
            'suggestions': [{'type': s.transformation_type, 'reasoning': s.reasoning, 'confidence': s.confidence} for s in suggestions],
            'predictions': [{'type': p.error_type, 'probability': p.probability} for p in predictions]
        }
        
        with open(filename, 'w') as f:
            json.dump(learning_data, f, indent=2, default=str)
    
    def process_datasets_batch(self, max_workers=4):
        """Process multiple datasets in parallel"""
        datasets = self.discover_datasets()
        
        if not datasets:
            logger.warning("No datasets found to process")
            return
        
        logger.info(f"Processing {len(datasets)} datasets with {max_workers} workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {executor.submit(self.process_dataset, dataset): dataset for dataset in datasets}
            
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update learning metrics
                    self._update_learning_metrics(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {dataset}: {e}")
        
        return results
    
    def _update_learning_metrics(self, result):
        """Update learning metrics with processing results"""
        if result['success']:
            self.learning_metrics['total_datasets_processed'] += 1
            self.learning_metrics['total_rows_processed'] += result['rows_processed']
            self.learning_metrics['total_issues_detected'] += result['issues_detected']
            self.learning_metrics['total_corrections_applied'] += result['corrections_applied']
            self.learning_metrics['processing_times'].append(result['processing_time'])
            
            # Calculate accuracy improvement
            if result['corrections_applied'] > 0:
                accuracy = result['corrections_applied'] / max(result['issues_detected'], 1)
                self.learning_metrics['accuracy_improvements'].append(accuracy)
    
    def retrain_models(self):
        """Retrain AI agent models with accumulated learning data"""
        logger.info("Retraining AI agent models...")
        
        try:
            # Get current performance
            current_performance = self.ai_agent.get_model_performance()
            
            # Retrain models
            self.ai_agent._retrain_models()
            
            # Get new performance
            new_performance = self.ai_agent.get_model_performance()
            
            # Track performance improvement
            performance_improvement = {
                'timestamp': datetime.now().isoformat(),
                'training_samples_before': current_performance.get('training_samples', 0),
                'training_samples_after': new_performance.get('training_samples', 0),
                'models_trained': new_performance.get('models_trained', {})
            }
            
            self.performance_history.append(performance_improvement)
            self.learning_metrics['last_model_update'] = datetime.now().isoformat()
            
            # Save updated models
            self.ai_agent._save_models()
            
            logger.info(f"Models retrained successfully. Training samples: {performance_improvement['training_samples_before']} -> {performance_improvement['training_samples_after']}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def generate_learning_report(self):
        """Generate a comprehensive learning report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"learning_reports/learning_report_{timestamp}.json"
        
        # Get current AI agent performance
        current_performance = self.ai_agent.get_model_performance()
        learning_metrics = self.ai_agent.get_learning_metrics()
        
        report = {
            'timestamp': timestamp,
            'pipeline_metrics': self.learning_metrics,
            'ai_agent_performance': current_performance,
            'ai_agent_learning': learning_metrics,
            'performance_history': self.performance_history,
            'summary': {
                'total_datasets_processed': self.learning_metrics['total_datasets_processed'],
                'total_rows_processed': self.learning_metrics['total_rows_processed'],
                'average_processing_time': np.mean(self.learning_metrics['processing_times']) if self.learning_metrics['processing_times'] else 0,
                'average_accuracy': np.mean(self.learning_metrics['accuracy_improvements']) if self.learning_metrics['accuracy_improvements'] else 0,
                'models_trained': current_performance.get('models_trained', {}),
                'learning_rate': learning_metrics.get('learning_rate', 0)
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Learning report generated: {report_file}")
        return report
    
    def run_continuous_learning_cycle(self):
        """Run one complete learning cycle"""
        logger.info("Starting continuous learning cycle...")
        
        # Process datasets
        results = self.process_datasets_batch()
        
        # Generate report
        report = self.generate_learning_report()
        
        # Check if models need retraining
        if self.learning_metrics['total_datasets_processed'] % 10 == 0:  # Retrain every 10 datasets
            self.retrain_models()
        
        logger.info(f"Learning cycle completed. Processed {len(results)} datasets")
        return results, report
    
    def start_continuous_learning(self, interval_hours=6):
        """
        Start continuous learning with scheduled intervals
        
        Args:
            interval_hours: Hours between learning cycles
        """
        logger.info(f"Starting continuous learning with {interval_hours}-hour intervals")
        
        # Schedule learning cycles
        schedule.every(interval_hours).hours.do(self.run_continuous_learning_cycle)
        
        # Run initial cycle
        self.run_continuous_learning_cycle()
        
        # Keep running scheduled cycles
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_single_learning_session(self, max_datasets=None):
        """
        Run a single learning session
        
        Args:
            max_datasets: Maximum number of datasets to process (None for all)
        """
        logger.info("Running single learning session...")
        
        datasets = self.discover_datasets()
        if max_datasets:
            datasets = datasets[:max_datasets]
        
        results = []
        for dataset in datasets:
            result = self.process_dataset(dataset)
            results.append(result)
            self._update_learning_metrics(result)
            
            logger.info(f"Processed {dataset}: {result['rows_processed']} rows, {result['issues_detected']} issues, {result['corrections_applied']} corrections")
        
        # Generate report
        report = self.generate_learning_report()
        
        # Retrain models if enough data processed
        if len(results) >= 5:
            self.retrain_models()
        
        return results, report

def main():
    """Main function to run the continuous learning pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Learning Pipeline for AI Agent')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                       help='Learning mode: single session or continuous')
    parser.add_argument('--interval', type=int, default=6,
                       help='Hours between learning cycles (continuous mode)')
    parser.add_argument('--max-datasets', type=int, default=None,
                       help='Maximum datasets to process (single mode)')
    parser.add_argument('--data-dirs', nargs='+', default=['data', 'demo_data'],
                       help='Directories containing datasets')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ContinuousLearningPipeline(data_directories=args.data_dirs)
    
    if args.mode == 'continuous':
        try:
            pipeline.start_continuous_learning(interval_hours=args.interval)
        except KeyboardInterrupt:
            logger.info("Continuous learning stopped by user")
    else:
        results, report = pipeline.run_single_learning_session(max_datasets=args.max_datasets)
        
        # Print summary
        print("\n" + "="*60)
        print("LEARNING SESSION SUMMARY")
        print("="*60)
        print(f"Datasets processed: {len(results)}")
        print(f"Total rows processed: {report['summary']['total_rows_processed']:,}")
        print(f"Average processing time: {report['summary']['average_processing_time']:.2f}s")
        print(f"Average accuracy: {report['summary']['average_accuracy']:.2f}")
        print(f"Models trained: {report['summary']['models_trained']}")
        print(f"Learning rate: {report['summary']['learning_rate']:.3f}")
        print("="*60)

if __name__ == "__main__":
    main() 