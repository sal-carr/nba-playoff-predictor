.PHONY: install data train predict-series predict-games test clean help

# Default target
help:
	@echo "NBA Playoff Predictor - Available commands:"
	@echo ""
	@echo "  make install        Install dependencies"
	@echo "  make data           Fetch NBA game data"
	@echo "  make train          Train the prediction model"
	@echo "  make predict-series Predict playoff series outcomes"
	@echo "  make predict-games  Predict individual upcoming games"
	@echo "  make test           Run tests"
	@echo "  make clean          Remove generated files"
	@echo "  make card           View model evaluation card"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Fetch data from NBA API
data:
	python src/data_acquisition.py

# Train the model
train:
	python flows/train_model.py run

# Train with extended date range
train-current:
	python flows/train_model.py run --end-date 2026-12-31

# Predict playoff series
predict-series:
	python scripts/predict_series.py

# Predict next games
predict-games:
	python scripts/predict_next_games.py

# Run tests
test:
	pytest tests/ -v

# View model evaluation card
card:
	python flows/train_model.py card view join_tuning

# Clean generated files
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist build

# Full pipeline: data -> train -> predict
pipeline: data train predict-series
	@echo "Pipeline complete! Check results/"
