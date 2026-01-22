.PHONY: help install bootstrap fetch-data fetch-custom train serve test-api validate visualize clean score-universe create-dashboard test-portfolios

# Default target
help:
	@echo "Portfolio Anomaly Detection - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make bootstrap     Full setup: fetch data and train models"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make fetch-data    Download S&P 500 market data (5 years, ~500 symbols)"
	@echo "  make fetch-custom  Download custom predefined universe (10 years, ~60 symbols)"
	@echo "  make train         Train models (automatically extracts features)"
	@echo ""
	@echo "Run:"
	@echo "  make serve         Start the FastAPI server (port 8000)"
	@echo "  make test-api      Test the API with sample portfolios"
	@echo ""
	@echo "Analysis:"
	@echo "  make validate      Run model validation suite"
	@echo "  make visualize     Generate diagnostic visualizations"
	@echo "  make score-universe Score all securities in universe"
	@echo "  make create-dashboard Create validation dashboard from results"
	@echo ""
	@echo "Testing:"
	@echo "  make test-portfolios Test API with various portfolio scenarios"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove generated data and models"

# Installation
install:
	pip install -e .

# Full bootstrap (one command to rule them all)
bootstrap: fetch-data train
	@echo ""
	@echo "✓ Bootstrap complete! Run 'make serve' to start the API."

# Data pipeline
fetch-data:
	@echo "Fetching S&P 500 market data (5 years, ~500 symbols)..."
	python scripts/fetch_data.py --sp500

fetch-custom:
	@echo "Fetching custom predefined universe (10 years, ~60 symbols)..."
	python scripts/fetch_data.py --custom --years 10

train:
	@echo "Training models (features are extracted automatically)..."
	@echo "Usage: python scripts/train_model.py <model_type> [options]"
	@echo "  Example: python scripts/train_model.py individual --train-end 2024-06-30"
	@echo "  Example: python scripts/train_model.py cross_sectional --train-end 2024-06-30"

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
	@echo "Usage: python scripts/validate_model.py <model_type> [options]"
	@echo "  Example: python scripts/validate_model.py individual"
	@echo "  Example: python scripts/validate_model.py cross_sectional --synthetic"

visualize:
	@echo "Generating visualizations..."
	python scripts/visualize_model.py

# Score universe (score all securities)
score-universe:
	@echo "Scoring universe..."
	@echo "Usage: python scripts/score_universe.py <model_type>"
	@echo "  Example: python scripts/score_universe.py individual"
	@echo "  Example: python scripts/score_universe.py cross_sectional"

# Create validation dashboard from existing results
create-dashboard:
	@echo "Creating validation dashboard..."
	@echo "Usage: python scripts/create_validation_dashboard.py <model_type> [--timestamp TIMESTAMP]"
	@echo "  Example: python scripts/create_validation_dashboard.py individual"
	@echo "  Example: python scripts/create_validation_dashboard.py cross_sectional --timestamp 20250113_143022"

# Test portfolios (requires API server running)
test-portfolios:
	@echo "Testing portfolios against API..."
	@echo "Note: API server must be running (make serve)"
	python scripts/test_portfolios.py

# Analyze a portfolio (usage: make analyze PORTFOLIO="AAPL:100 MSFT:50")
analyze:
	python scripts/analyze_portfolio.py --shares $(PORTFOLIO)

# Cleanup
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*.sqlite
	rm -rf models/model_types/
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
