# Contributing to Land Degradation ML Project

Thank you for your interest in contributing! This project is a B.Tech Major Project focused on predicting land degradation using machine learning and Google Earth Engine data.

## How to Contribute

### Before You Start
- This is an academic project with specific methodologies and data pipelines
- The core datasets (in `data/`) are pre-built and should not be regenerated
- Phases 1-3 are complete; Phase 4 (ML pipeline) is the focus for continued development

### Contribution Guidelines

1. **Fork the repository** and create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up your environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   pip install -r Land_Degradation_App/requirements.txt
   ```

3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add docstrings to new functions and classes
   - Keep commits focused and descriptive

4. **Test your changes**
   - Run the application tests:
     ```bash
     python Land_Degradation_App/tests/run_app_tests.py
     ```
   - If adding new code, include corresponding tests

5. **Push and create a Pull Request**
   - Provide a clear description of what changed and why
   - Link any related issues

## Code Style

- Use type hints where possible (Python 3.10+)
- Document complex logic with comments
- Use relative imports within modules
- Follow the existing project structure

## Areas for Contribution

- **Model Improvements**: Hyperparameter tuning, new algorithms, ensemble methods
- **App Features**: New dashboards, visualization improvements, export formats
- **Documentation**: Clarifications, tutorials, examples
- **Testing**: Additional test coverage, edge cases
- **Performance**: Optimization of data loading, inference, or visualization

## Questions?

Please open an issue or check the main README.md for project context and methodology.
