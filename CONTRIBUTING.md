# Contributing to OSHFS

Thank you for considering contributing to the Open-Source Hybrid Forecasting System! This document outlines the process and guidelines for contributing to the project.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template to create a new issue
- Include detailed steps to reproduce the bug
- Include system information and package versions

### Suggesting Features

- Check if the feature has already been suggested in the Issues section
- Use the feature request template to create a new issue
- Clearly describe the feature and its benefits
- Consider how the feature fits into the existing architecture

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Update documentation if needed
6. Commit your changes (`git commit -m 'Add your feature'`)
7. Push to the branch (`git push origin feature/your-feature`)
8. Open a Pull Request

## Development Guidelines

### Code Style

This project uses:
- [Black](https://black.readthedocs.io/) for code formatting
- [Flake8](https://flake8.pycqa.org/) for linting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](https://mypy.readthedocs.io/) for type checking

Run the following before committing:

```bash
black oshfs tests
flake8 oshfs tests
isort oshfs tests
mypy oshfs
```

### Testing

- Write tests for all new features and bug fixes
- Maintain or improve test coverage
- Run the test suite before submitting a PR:

```bash
pytest --cov=oshfs tests/
```

### Documentation

- Update documentation for all new features and changes
- Write clear docstrings following Google style
- Include examples in docstrings
- Update the relevant guides if needed

## Development Setup

1. Fork and clone the repository
2. Install Poetry: `pip install poetry`
3. Install dependencies: `poetry install`
4. Install pre-commit hooks: `poetry run pre-commit install`

## Release Process

1. Update version in `pyproject.toml` and `oshfs/__init__.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. CI will automatically publish to PyPI

Thank you for contributing to OSHFS!
