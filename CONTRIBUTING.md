# Contributing to AI-Powered ETL Pipeline

Thank you for your interest in contributing to our ETL pipeline project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### **Ways to Contribute**
- 🐛 **Report bugs** and issues
- 💡 **Suggest new features** and improvements
- 📝 **Improve documentation**
- 🔧 **Fix bugs** and implement features
- 🧪 **Add tests** and improve test coverage
- 🌐 **Translate** documentation to other languages

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8 or higher
- Git
- pip or conda for package management

### **Development Setup**
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork locally
git clone https://github.com/YOUR_USERNAME/Ai.etl.git
cd Ai.etl

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install development dependencies
pip install pytest black flake8 mypy pre-commit

# 6. Set up pre-commit hooks
pre-commit install
```

### **Project Structure for Contributors**
```
src/
├── extractors/          # Data extraction modules
│   ├── __init__.py
│   ├── base_extractor.py
│   ├── api_extractor.py
│   ├── csv_extractor.py
│   └── universal_extractor.py
├── transformers/        # Data transformation modules
│   ├── __init__.py
│   ├── base_transformer.py
│   └── data_transformer.py
├── loaders/            # Data loading modules
│   ├── __init__.py
│   ├── base_loader.py
│   ├── postgresql_loader.py
│   ├── snowflake_loader.py
│   └── s3_loader.py
├── config.py           # Configuration management
├── logger.py           # Logging setup
└── etl_pipeline.py     # Main ETL orchestration
```

## 📋 Development Guidelines

### **Code Style**
We follow PEP 8 guidelines and use several tools to maintain code quality:

```bash
# Format code with black
black src/ tests/ app.py

# Check code style with flake8
flake8 src/ tests/ app.py

# Type checking with mypy
mypy src/

# Run all checks
pre-commit run --all-files
```

### **Code Standards**
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Add docstrings to all functions and classes
- **Error Handling**: Use appropriate exception handling
- **Logging**: Use structured logging with appropriate log levels

### **Example Code Structure**
```python
from typing import Dict, Any, Optional
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

class ExampleExtractor:
    """Example extractor for demonstration purposes."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the extractor with configuration.
        
        Args:
            config: Configuration dictionary containing extractor settings
        """
        self.config = config
        self.logger = logger
    
    def extract(self) -> pd.DataFrame:
        """Extract data from the source.
        
        Returns:
            DataFrame containing extracted data
            
        Raises:
            ExtractionError: If data extraction fails
        """
        try:
            # Implementation here
            pass
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise ExtractionError(f"Failed to extract data: {e}")
```

## 🧪 Testing

### **Running Tests**
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_extractors.py

# Run specific test function
pytest tests/test_extractors.py::test_api_extractor

# Run tests in parallel
pytest -n auto
```

### **Writing Tests**
- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Test both success and failure scenarios

### **Test Example**
```python
import pytest
from unittest.mock import Mock, patch
from src.extractors.api_extractor import APIExtractor

class TestAPIExtractor:
    """Test cases for APIExtractor."""
    
    def test_extract_success(self):
        """Test successful API data extraction."""
        # Arrange
        config = {"url": "https://api.example.com/data"}
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}
        
        # Act
        with patch('requests.get', return_value=mock_response):
            extractor = APIExtractor(config)
            result = extractor.extract()
        
        # Assert
        assert len(result) == 1
        assert result.iloc[0]['id'] == 1
        assert result.iloc[0]['name'] == 'test'
    
    def test_extract_failure(self):
        """Test API extraction failure handling."""
        # Arrange
        config = {"url": "https://api.example.com/data"}
        
        # Act & Assert
        with patch('requests.get', side_effect=Exception("API Error")):
            extractor = APIExtractor(config)
            with pytest.raises(Exception):
                extractor.extract()
```

## 🔧 Adding New Features

### **Adding a New Extractor**
1. Create a new file in `src/extractors/`
2. Inherit from `BaseExtractor`
3. Implement the required methods
4. Add tests in `tests/test_extractors.py`
5. Update documentation

### **Adding a New Loader**
1. Create a new file in `src/loaders/`
2. Inherit from `BaseLoader`
3. Implement the required methods
4. Add tests in `tests/test_loaders.py`
5. Update configuration options

### **Adding a New Transformer**
1. Create a new file in `src/transformers/`
2. Inherit from `BaseTransformer`
3. Implement the required methods
4. Add tests in `tests/test_transformers.py`
5. Update documentation

## 📝 Documentation

### **Documentation Standards**
- Keep documentation up to date with code changes
- Use clear, concise language
- Include code examples
- Add type hints in docstrings
- Update README.md for user-facing changes

### **Documentation Structure**
```
docs/
├── api/                 # API documentation
├── user-guide/          # User guides
├── developer-guide/     # Developer documentation
└── examples/           # Code examples
```

## 🔄 Pull Request Process

### **Before Submitting a PR**
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Test** your changes thoroughly
5. **Commit** with clear messages: `git commit -m "Add amazing feature"`
6. **Push** to your fork: `git push origin feature/amazing-feature`
7. **Create** a Pull Request

### **Pull Request Guidelines**
- **Title**: Use clear, descriptive titles
- **Description**: Explain what the PR does and why
- **Tests**: Ensure all tests pass
- **Documentation**: Update relevant documentation
- **Screenshots**: Include screenshots for UI changes

### **Commit Message Format**
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(extractors): add MongoDB extractor

- Add MongoDB connection support
- Implement data extraction from collections
- Add configuration options for authentication

Closes #123
```

## 🐛 Reporting Issues

### **Bug Reports**
When reporting bugs, please include:
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Step-by-step instructions
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, dependencies
- **Screenshots**: If applicable

### **Feature Requests**
When requesting features, please include:
- **Description**: Clear description of the feature
- **Use Case**: Why this feature is needed
- **Proposed Solution**: How you think it should work
- **Alternatives**: Any alternative solutions considered

## 🏷️ Issue Labels

We use the following labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority issues
- `priority: low`: Low priority issues

## 📞 Getting Help

### **Communication Channels**
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### **Code of Conduct**
We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## 🎉 Recognition

### **Contributors**
All contributors will be recognized in:
- The project's README.md
- Release notes
- Contributor hall of fame

### **Contributor Types**
- **Code Contributors**: Those who contribute code
- **Documentation Contributors**: Those who improve documentation
- **Bug Reporters**: Those who report valuable bugs
- **Feature Requesters**: Those who suggest useful features

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to the AI-Powered ETL Pipeline! 🚀 