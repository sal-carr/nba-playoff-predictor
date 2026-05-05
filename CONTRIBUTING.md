# Contributing

Thank you for your interest in contributing to the NBA Playoff Predictor!

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── src/                 # Core library code
├── flows/               # Metaflow pipelines
├── scripts/             # CLI utilities
├── data/                # Data files (gitignored except samples)
├── results/             # Prediction outputs
├── docs/                # Documentation
└── tests/               # Unit tests
```

## Running the Pipeline

1. **Acquire data:**
   ```bash
   python src/data_acquisition.py
   ```

2. **Train the model:**
   ```bash
   python flows/train_model.py run
   ```

3. **Generate predictions:**
   ```bash
   python scripts/predict_series.py
   ```

## Code Style

- Use Python type hints where practical
- Follow PEP 8 conventions
- Keep functions focused and well-documented

## Testing

```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation if needed
5. Submit a PR with a clear description

## Questions?

Open an issue for questions or feature requests.
