.PHONY: help install bootstrap fetch-data derive-features train serve test-api validate visualize clean

# Default target
help:
	@echo "Portfolio Anomaly Detection - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make bootstrap     Full setup: fetch data, derive features, train models"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make fetch-data    Download market data from Yahoo Finance"
	@echo "  make derive-features  Calculate technical indicators"
	@echo "  make train         Train Autoencoder and Isolation Forest models"
	@echo ""
	@echo "Run:"
	@echo "  make serve         Start the FastAPI server (port 8000)"
	@echo "  make test-api      Test the API with sample portfolios"
	@echo ""
	@echo "Analysis:"
	@echo "  make validate      Run model validation suite"
	@echo "  make visualize     Generate diagnostic visualizations"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove generated data and models"

# Installation
install:
	pip install -e .

# Full bootstrap (one command to rule them all)
bootstrap: fetch-data derive-features train
	@echo ""
	@echo "✓ Bootstrap complete! Run 'make serve' to start the API."

# Data pipeline
fetch-data:
	@echo "Fetching market data..."
	python scripts/fetch_data.py

derive-features:
	@echo "Calculating technical indicators..."
	python scripts/derive_features.py

train:
	@echo "Training models..."
	python scripts/train_models.py

# API
serve:
	@echo "Starting API server on http://localhost:8000"
	@echo "API docs available at http://localhost:8000/docs"
	python -m uvicorn api.service:app --reload --host 0.0.0.0 --port 8000

test-api:
	@echo "Testing API with sample portfolios..."
	python api/test_client.py

# Analysis
validate:
	@echo "Running model validation..."
	python scripts/validate_models.py

visualize:
	@echo "Generating visualizations..."
	python scripts/visualize_model.py

# Evaluate a portfolio (usage: make evaluate PORTFOLIO="AAPL:100 MSFT:50")
evaluate:
	python scripts/evaluate_portfolio.py $(PORTFOLIO)

# Cleanup
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*.sqlite
	rm -rf models/market_universe/
	rm -rf *.png
	rm -rf model_validation_report_*.txt
	@echo "✓ Clean complete"

# Development helpers
lint:
	@echo "Running linter..."
	ruff check src/ api/ scripts/

format:
	@echo "Formatting code..."
	ruff format src/ api/ scripts/
