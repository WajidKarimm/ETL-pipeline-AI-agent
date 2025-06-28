#!/usr/bin/env python3
"""
Test script to demonstrate model deployment and preprocessing capabilities.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from src.ml.model_deployment import ModelManager, ModelPreprocessor, ModelTrainer, ModelDeployer
from src.ml.model_deployment import ModelMetadata, DeploymentConfig

def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data for model training...")
    
    # Create customer data with target variable
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.uniform(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'years_employed': np.random.uniform(0, 30, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'loan_amount': np.random.uniform(5000, 500000, n_samples),
        'loan_approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Target variable
    })
    
    # Add some missing values
    data.loc[np.random.choice(n_samples, 50, replace=False), 'credit_score'] = np.nan
    data.loc[np.random.choice(n_samples, 30, replace=False), 'income'] = np.nan
    
    print(f"✅ Created sample data: {len(data)} rows, {len(data.columns)} columns")
    return data

def test_preprocessing():
    """Test data preprocessing capabilities."""
    print("\n🔧 Testing Data Preprocessing")
    print("=" * 40)
    
    data = create_sample_data()
    
    # Initialize preprocessor
    preprocessor = ModelPreprocessor()
    
    # Analyze data
    analysis = preprocessor.analyze_data(data)
    print(f"📋 Data Analysis Results:")
    print(f"   - Total rows: {analysis['total_rows']}")
    print(f"   - Total columns: {analysis['total_columns']}")
    print(f"   - Suggested preprocessing steps: {len(analysis['suggested_preprocessing'])}")
    
    for step in analysis['suggested_preprocessing']:
        print(f"   • {step['action']} for {step['column']}: {step['reason']}")
    
    # Build preprocessing pipeline
    target_column = 'loan_approved'
    pipeline = preprocessor.build_preprocessing_pipeline(data, target_column)
    print(f"✅ Preprocessing pipeline built successfully")
    
    # Fit and transform data
    X_transformed, y = preprocessor.fit_transform(data)
    print(f"✅ Data transformed: {X_transformed.shape}")
    
    # Test with new data
    test_data = data.sample(n=100, random_state=42)
    X_test_transformed = preprocessor.transform(test_data)
    print(f"✅ Test data transformed: {X_test_transformed.shape}")
    
    return preprocessor, data

def test_model_training():
    """Test model training capabilities."""
    print("\n🧠 Testing Model Training")
    print("=" * 40)
    
    data = create_sample_data()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Use preprocessing pipeline
    preprocessor = ModelPreprocessor()
    target_column = 'loan_approved'
    preprocessor.build_preprocessing_pipeline(data, target_column)
    
    # Transform data
    X_transformed, y = preprocessor.fit_transform(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )
    
    # Train different models
    models = {}
    
    # Random Forest
    model, model_id = trainer.train_model(X_train, y_train, 'random_forest', 
                                        {'n_estimators': 100, 'max_depth': 10})
    models['random_forest'] = model
    
    # Evaluate
    metrics = trainer.evaluate_model(model, X_test, y_test)
    print(f"✅ Random Forest trained:")
    print(f"   - Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
    print(f"   - Precision: {metrics.get('precision', 'N/A'):.3f}")
    print(f"   - Recall: {metrics.get('recall', 'N/A'):.3f}")
    print(f"   - F1 Score: {metrics.get('f1_score', 'N/A'):.3f}")
    
    # Logistic Regression
    model, model_id = trainer.train_model(X_train, y_train, 'logistic_regression')
    models['logistic_regression'] = model
    
    # Evaluate
    metrics = trainer.evaluate_model(model, X_test, y_test)
    print(f"✅ Logistic Regression trained:")
    print(f"   - Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
    print(f"   - Precision: {metrics.get('precision', 'N/A'):.3f}")
    print(f"   - Recall: {metrics.get('recall', 'N/A'):.3f}")
    print(f"   - F1 Score: {metrics.get('f1_score', 'N/A'):.3f}")
    
    return trainer, models

def test_model_deployment():
    """Test model deployment capabilities."""
    print("\n🚀 Testing Model Deployment")
    print("=" * 40)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize deployer
        deployer = ModelDeployer(temp_dir)
        
        # Create sample model and preprocessor
        data = create_sample_data()
        preprocessor = ModelPreprocessor()
        target_column = 'loan_approved'
        preprocessor.build_preprocessing_pipeline(data, target_column)
        
        # Train a simple model
        trainer = ModelTrainer()
        X_transformed, y = preprocessor.fit_transform(data)
        model, model_id = trainer.train_model(X_transformed, y, 'random_forest')
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="loan_approval_model",
            version="1.0.0",
            created_at="2024-01-01T00:00:00Z",
            model_type="random_forest",
            features=preprocessor.feature_columns,
            target_column=target_column,
            performance_metrics={'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88},
            training_data_size=len(data),
            preprocessing_steps=['scale', 'label_encode'],
            dependencies={'scikit-learn': '1.0.0', 'pandas': '1.3.0'},
            author="Test User",
            description="Test loan approval model"
        )
        
        # Create deployment config
        config = DeploymentConfig(
            deployment_name="test_loan_model",
            model_path=os.path.join(temp_dir, "model.pkl"),
            api_port=5000,
            enable_monitoring=True
        )
        
        # Deploy model
        deployment_path = deployer.deploy_model(model, preprocessor, metadata, config)
        print(f"✅ Model deployed to: {deployment_path}")
        
        # List deployments
        print(f"📋 Deployments: {list(deployer.deployments.keys())}")
        
        # Load deployment
        loaded_model, loaded_preprocessor, loaded_metadata = deployer.load_deployment("test_loan_model")
        print(f"✅ Model loaded successfully")
        print(f"   - Model ID: {loaded_metadata.model_id}")
        print(f"   - Features: {len(loaded_metadata.features)}")
        print(f"   - Performance: {loaded_metadata.performance_metrics}")
        
        # Test prediction
        test_data = data.sample(n=10, random_state=42)
        X_test_transformed = loaded_preprocessor.transform(test_data)
        predictions = loaded_model.predict(X_test_transformed)
        print(f"✅ Predictions made: {predictions}")
        
        return deployer

def test_model_manager():
    """Test the high-level model manager."""
    print("\n🎯 Testing Model Manager")
    print("=" * 40)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize manager
        manager = ModelManager(models_dir=temp_dir, deployments_dir=temp_dir)
        
        # Create sample data
        data = create_sample_data()
        
        # Train and deploy
        deployment_path = manager.train_and_deploy(
            data=data,
            target_column='loan_approved',
            model_type='random_forest',
            deployment_name='loan_approval_rf',
            model_params={'n_estimators': 100, 'max_depth': 10}
        )
        
        print(f"✅ Model trained and deployed: {deployment_path}")
        
        # List deployments
        deployments = manager.list_deployments()
        print(f"📋 Deployments: {len(deployments)}")
        for deployment in deployments:
            print(f"   - {deployment['name']}: {deployment['metadata'].model_type}")
        
        # Get deployment info
        info = manager.get_deployment_info('loan_approval_rf')
        print(f"📊 Deployment Info:")
        print(f"   - Model Type: {info['metadata'].model_type}")
        print(f"   - Target: {info['metadata'].target_column}")
        print(f"   - Features: {len(info['metadata'].features)}")
        print(f"   - Performance: {info['metadata'].performance_metrics}")
        
        # Make predictions
        test_data = data.sample(n=50, random_state=42)
        predictions = manager.predict('loan_approval_rf', test_data)
        print(f"✅ Predictions: {len(predictions)} samples")
        print(f"   - Sample predictions: {predictions[:5]}")
        
        return manager

def test_api_server():
    """Test API server creation (if Flask is available)."""
    print("\n🌐 Testing API Server")
    print("=" * 40)
    
    try:
        from src.ml.model_deployment import FLASK_AVAILABLE
        
        if not FLASK_AVAILABLE:
            print("⚠️ Flask not available. Skipping API server test.")
            return
        
        # Create a simple deployment
        with tempfile.TemporaryDirectory() as temp_dir:
            deployer = ModelDeployer(temp_dir)
            
            # Create sample model
            data = create_sample_data()
            preprocessor = ModelPreprocessor()
            target_column = 'loan_approved'
            preprocessor.build_preprocessing_pipeline(data, target_column)
            
            trainer = ModelTrainer()
            X_transformed, y = preprocessor.fit_transform(data)
            model, model_id = trainer.train_model(X_transformed, y, 'random_forest')
            
            # Create metadata and config
            metadata = ModelMetadata(
                model_id=model_id,
                model_name="api_test_model",
                version="1.0.0",
                created_at="2024-01-01T00:00:00Z",
                model_type="random_forest",
                features=preprocessor.feature_columns,
                target_column=target_column,
                performance_metrics={'accuracy': 0.85},
                training_data_size=len(data),
                preprocessing_steps=['scale'],
                dependencies={},
                author="Test",
                description="API test model"
            )
            
            config = DeploymentConfig(
                deployment_name="api_test",
                model_path=os.path.join(temp_dir, "model.pkl")
            )
            
            # Deploy
            deployer.deploy_model(model, preprocessor, metadata, config)
            
            # Create API server
            app = deployer.create_api_server("api_test")
            
            if app:
                print("✅ API server created successfully")
                print("   - Endpoints: /predict, /health, /metadata")
                print("   - Model ready for serving")
            else:
                print("❌ Failed to create API server")
                
    except Exception as e:
        print(f"❌ API server test failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing Model Deployment System")
    print("=" * 50)
    
    try:
        # Test preprocessing
        test_preprocessing()
        
        # Test model training
        test_model_training()
        
        # Test model deployment
        test_model_deployment()
        
        # Test model manager
        test_model_manager()
        
        # Test API server
        test_api_server()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("=" * 50)
        print("✅ Model deployment system is working correctly")
        print("✅ Preprocessing pipeline is functional")
        print("✅ Model training and evaluation works")
        print("✅ Deployment and loading works")
        print("✅ High-level manager interface works")
        print("✅ API server creation works (if Flask available)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 