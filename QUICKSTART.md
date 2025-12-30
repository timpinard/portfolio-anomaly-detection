# Portfolio Anomaly Detection - Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- Internet connection (for downloading market data from Yahoo Finance)
- Optional: Anthropic API key for LLM features

## Step-by-Step Setup

### 1. Create Virtual Environment (Recommended)

```bash
cd portfolio-anomaly-detection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -e .
```

This installs all required packages including:
- `yfinance` - Yahoo Finance market data (free, no API key required)
- `torch` - Neural networks
- `scikit-learn` - Machine learning
- `fastapi` - API framework
- `pandas`, `numpy` - Data processing

### 3. Set Optional API Keys

```bash
# Optional: Add Anthropic key for LLM features
export ANTHROPIC_API_KEY="your_key_here" or add to .env file
```

**Note:** Market data is fetched from Yahoo Finance (free, no API key required). Only set `ANTHROPIC_API_KEY` if you want LLM-powered analysis features.

### 4. Bootstrap System

```bash
make bootstrap
```

This runs three scripts:
1. **fetch_data.py** - Downloads ~5 years of market data from Yahoo Finance (free, no API key)
2. **derive_features.py** - Calculates technical indicators and features
3. **train_models.py** - Trains autoencoder and isolation forest models

**Time:** ~5-10 minutes depending on internet speed

### 5. Start API

```bash
make serve
```

API will be available at http://localhost:8000

### 6. Test It!

Open a new terminal:

```bash
make test-api
```

You should see output analyzing 3 different portfolio configurations.

## What Just Happened?

1. **Market Data**: Downloaded historical prices for 40 diverse stocks (tech, finance, healthcare, etc.)
2. **Features**: Calculated returns, volatility, RSI, MACD, Bollinger Bands for each stock
3. **Training**: Models learned "normal" market behavior from the data
4. **API**: Service ready to analyze any portfolio against these learned patterns

## Quick API Test

Using `curl`:

```bash
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100, "cost_basis": 150},
      "MSFT": {"shares": 50, "cost_basis": 300},
      "GOOGL": {"shares": 30, "cost_basis": 2500}
    },
    "benchmark": "SPY"
  }'
```

Using Python:

```python
import requests

portfolio = {
    "positions": {
        "AAPL": {"shares": 100, "cost_basis": 150},
        "MSFT": {"shares": 50, "cost_basis": 300},
        "GOOGL": {"shares": 30, "cost_basis": 2500}
    },
    "benchmark": "SPY"
}

response = requests.post(
    "http://localhost:8000/portfolio/analyze",
    json=portfolio
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Anomaly: {result['model_results']['autoencoder']['is_anomaly']}")
```

## Interactive Documentation

Visit http://localhost:8000/docs for:
- Interactive API testing
- Request/response schemas
- Example payloads
- Try different portfolios

## What the Models Do

### Autoencoder
- Learns patterns in market data (correlations, volatility, returns)
- Reconstructs portfolio features
- High reconstruction error = unusual/anomalous portfolio

### Isolation Forest
- Identifies statistical outliers
- Flags portfolios that don't fit normal market distributions

### Consensus
- Both models agree → High confidence
- Models disagree → Medium confidence (investigate further)

## Common Issues

### "Database not found"
Make sure you've run the data fetch script:
```bash
python scripts/fetch_data.py
```

### "Models not found"
```bash
make bootstrap
```

### "No features calculated"
Make sure you've run all steps in order:
```bash
python scripts/fetch_data.py
python scripts/derive_features.py
```

### API connection error
Make sure API is running:
```bash
python api/service.py
```

## Model Diagnostics

After training models, you can visualize their behavior:

```bash
# Generate visual diagnostics
python scripts/visualize_model.py
```

This creates diagnostic plots in `results/diagnostics/`:
- **score_distributions.png** - Score distributions with thresholds
- **anomalies_over_time.png** - Anomaly patterns over time
- **feature_comparison.png** - Feature distributions for normal vs anomalies
- **top_anomalies.png** - Detailed analysis of top anomalies
- **model_agreement.png** - Where models agree/disagree
- **diagnostic_summary.txt** - Text summary of findings

Use these visualizations to:
- Understand how models score data points
- Identify patterns in flagged anomalies
- Validate model behavior
- Detect potential issues (e.g., too many/few anomalies)

## Evaluate Your Portfolio

After models are trained, you can evaluate any portfolio with z-score analysis:

```bash
# Evaluate portfolio with equal weights
python scripts/evaluate_portfolio.py AAPL MSFT GOOGL

# Evaluate with specific shares (weights calculated automatically)
python scripts/evaluate_portfolio.py AAPL:100 MSFT:50 GOOGL:25

# Save results to JSON
python scripts/evaluate_portfolio.py AAPL MSFT GOOGL --output results/my_portfolio.json
```

The script provides:
- **Per-security anomaly scores** from both models
- **Z-scores** relative to the market basket (market-relative risk)
- **Weighted portfolio aggregates** for overall risk assessment
- **Comparison** to training set baseline

**Understanding Z-Scores:**
- `z ≈ 0`: Portfolio behaves like the market today
- `z > 1`: More anomalous than the market (elevated risk)
- `z > 2`: Much more anomalous than the market (high risk)
- `z < -1`: Calmer/more normal than the market

See README.md for detailed z-score interpretation guide.

## Next Steps

1. **Validate Models**: Run `python scripts/validate_models.py` to check model performance
2. **Visual Diagnostics**: Run `python scripts/visualize_model.py` to understand model behavior
3. **Evaluate Portfolio**: Run `python scripts/evaluate_portfolio.py` with your holdings
4. **Customize Universe**: Edit `config/model_config.yaml` to use different stocks
5. **Add LLM Analysis**: Set `ANTHROPIC_API_KEY` for natural language reports
6. **Extend Features**: Modify `src/data/feature_extractor.py` to add indicators
7. **Build Frontend**: Create dashboard using `requests` library

## File Structure Reference

```
portfolio-anomaly-detection/
├── data/
│   └── processed/
│       └── market_data.sqlite    # Market prices and features
├── models/
│   └── market_universe/
│       ├── autoencoder/          # Trained autoencoder
│       └── isolation_forest/     # Trained isolation forest
├── scripts/
│   ├── fetch_data.py              # Step 1: Download data from Yahoo Finance
│   ├── derive_features.py        # Step 2: Calculate features
│   ├── train_models.py           # Step 3: Train models
│   ├── validate_models.py        # Validate model performance
│   └── visualize_model.py        # Visual diagnostics and model behavior
└── api/
    ├── service.py                # FastAPI application
    └── test_client.py            # Test script
```

## Makefile Commands

```bash
make setup      # Install dependencies
make bootstrap  # Fetch data + derive features + train models
make serve      # Start API
make test-api   # Run tests
make clean      # Remove generated files
```

## Support

- **Documentation**: See README.md for full documentation
- **API Docs**: http://localhost:8000/docs when running
- **Issues**: Check that all dependencies are installed and API keys are set

Enjoy analyzing portfolios! 🚀
