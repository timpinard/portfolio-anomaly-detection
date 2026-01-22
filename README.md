# Portfolio Anomaly Detection System

A dual-model machine learning system for detecting portfolio anomalies and attributing sources of divergence.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This system provides two complementary analytical approaches for comprehensive portfolio analysis:

1. **Risk Assessment** (Individual Model): Detects unusual portfolio characteristics
   - Uses individual stock features (returns, volatility, technical indicators)
   - Answers: "Is this portfolio exhibiting unusual risk patterns?"
   - Output: Risk level (low/medium/high), anomaly flags, confidence scores

2. **Attribution Analysis** (Cross-Sectional Model): Identifies divergence drivers  
   - Uses market-relative features (relative returns, correlations, beta)
   - Answers: "Which stocks are moving differently from the market?"
   - Output: Divergence contributors, health scores, structural z-scores

### Key Features

- **Dual Autoencoder Architecture** - Two specialized models with complementary feature sets
- **Isolation Forest Ensemble** - Statistical outlier detection for risk assessment
- **Portfolio-Weighted Aggregation** - Captures portfolio "personality" 
- **Market-Relative Normalization** - Z-scores provide context-aware risk levels
- **Comprehensive Visualization Suite** - Automated diagnostic plots and model validation
- **FastAPI REST Interface** - Production-ready API with interactive documentation
- **Optional LLM Integration** - Natural language explanations via Anthropic Claude

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

```bash
# Install dependencies
pip install -e .

# Bootstrap system (fetch data + train both models)
make bootstrap

# Start API server
make serve

# Analyze a portfolio (uses both models)
python scripts/analyze_portfolio.py AAPL MSFT GOOGL
```

The bootstrap process:
1. Downloads 5 years of S&P 500 market data from Yahoo Finance (free, no API key)
2. Calculates technical features for all stocks
3. Trains both individual and cross-sectional models
4. Takes ~15-30 minutes depending on internet speed

## Architecture

### Individual Model Type (Risk Assessment)

**Purpose:** Detects portfolios with unusual risk characteristics

**Features:** 30 individual stock indicators organized into categories:
- **Momentum:** Returns across multiple timeframes (1d, 5d, 20d, 60d)
- **Volatility:** Rolling volatility (20d, 60d annualized)
- **Technical Indicators:** RSI, MACD, Bollinger Bands
- **Volume:** Volume ratios and standard deviations
- **Price Position:** Relative to moving averages and 52-week highs/lows

**Models:**
- **Autoencoder:** Neural network (128→64→32→15 encoding) learns normal market patterns
- **Isolation Forest:** Statistical outlier detection with 200 trees
- **Ensemble:** Weighted combination (85% autoencoder, 15% isolation forest)

**Output:**
```python
{
  "risk_level": "medium",
  "is_anomaly": true,
  "autoencoder_score": 0.0234,
  "isolation_forest_score": -0.15,
  "consensus": true,
  "recommendations": ["Monitor momentum divergence", "Review sector concentration"]
}
```

**Use Cases:**
- High volatility with low returns (market churn)
- Momentum extremes (overbought/oversold)
- Technical breakdowns (MACD divergence, RSI exhaustion)
- Volume anomalies (liquidity concerns)

### Cross-Sectional Model Type (Attribution)

**Purpose:** Identifies which stocks are driving divergence from market behavior

**Features:** 15 market-relative indicators:
- **Relative Returns:** Stock return minus market return (1d, 5d, 20d)
- **Correlations:** Rolling correlation with market proxy (20d, 60d)
- **Beta Dynamics:** Market sensitivity and changes over time
- **Direction Agreement:** Percentage of days moving with market
- **Universe Z-Scores:** How different from average stock today
- **Percentile Ranks:** Relative performance vs all stocks

**Model:**
- **Autoencoder Only:** Same architecture (128→64→32→15 encoding) trained on S&P 500 universe
- Reconstruction error measures divergence from typical market relationships
- No isolation forest (cross-sectional patterns are different)

**Output:**
```python
{
  "portfolio_score": 0.0156,
  "structural_z": 1.23,
  "health_score": -1.45,
  "contributors": [
    {"symbol": "TSLA", "weight": 0.30, "ae_score": 0.0234, "contribution": 0.0070},
    {"symbol": "AAPL", "weight": 0.25, "ae_score": 0.0189, "contribution": 0.0047}
  ],
  "market_baseline": {"ae_mean": 0.0123, "ae_std": 0.0045}
}
```

**Use Cases:**
- Market up 2%, portfolio down 5% - why?
- Identifying correlation breakdowns
- Detecting stocks with shifting beta
- Finding relative underperformance/outperformance

### Unified Analysis Interface

The `PortfolioAnalyzer` class provides a single entry point for both approaches:

```python
from portfolio.portfolio_analyzer import PortfolioAnalyzer

# Initialize analyzer
analyzer = PortfolioAnalyzer()

# Define portfolio
holdings = [
    {"symbol": "AAPL", "weight": 0.40},
    {"symbol": "MSFT", "weight": 0.35},
    {"symbol": "GOOGL", "weight": 0.25}
]

# Run comprehensive analysis (both models)
result = analyzer.analyze(holdings)

# Access risk assessment (individual model)
print(f"Risk Level: {result['risk_assessment']['risk_level']}")
print(f"Is Anomaly: {result['risk_assessment']['is_anomaly']}")

# Access attribution (cross-sectional model)
print(f"Health Score: {result['attribution']['health_score']}")
print(f"Top Contributor: {result['attribution']['contributors'][0]['symbol']}")

# Get summary
for item in result['summary']:
    print(item)
```

You can also run each approach independently:

```python
# Risk assessment only (faster)
risk = analyzer.analyze_risk(holdings)

# Attribution only
attribution = analyzer.analyze_attribution(holdings, analysis_date='2024-01-15')
```

## API Endpoints

The FastAPI service exposes both analysis types with interactive documentation at `http://localhost:8000/docs`

### Main Endpoints

**POST /portfolio/analyze** - Risk assessment (individual model)
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

**POST /portfolio/health** - Health analysis with attribution (cross-sectional or individual model)
```bash
curl -X POST http://localhost:8000/portfolio/health \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "AAPL", "weight": 0.40},
      {"symbol": "MSFT", "weight": 0.40},
      {"symbol": "GOOGL", "weight": 0.20}
    ],
    "analysis_date": "2024-01-15",
    "experiment": "cross_sectional",
    "market_proxy": "SPY"
  }'
```

**POST /portfolio/explain** - LLM-powered explanation (requires ANTHROPIC_API_KEY)
```bash
curl -X POST http://localhost:8000/portfolio/explain \
  -H "Content-Type: application/json" \
  -d '{
    "recommendations": ["Monitor momentum divergence"],
    "risk_level": "medium",
    "portfolio_metrics": {"total_value": 50000}
  }'
```

See [API_ENDPOINTS.md](API_ENDPOINTS.md) for complete API documentation.

## Command Line Interface

The `scripts/analyze_portfolio.py` script provides flexible command-line analysis:

```bash
# Comprehensive analysis (both models)
python scripts/analyze_portfolio.py AAPL MSFT GOOGL

# With custom weights
python scripts/analyze_portfolio.py AAPL:0.4 MSFT:0.35 GOOGL:0.25

# With share counts (weights calculated automatically)
python scripts/analyze_portfolio.py --shares AAPL:100 MSFT:50 GOOGL:30

# Risk assessment only (faster)
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --risk-only

# Attribution only
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --attribution-only

# Specific analysis date
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --date 2024-01-15

# JSON output
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --json --output results.json

# From file
python scripts/analyze_portfolio.py --file my_portfolio.json
```

## Interactive Tutorial Notebook

For those who want to understand how the cross-sectional model works under the hood, we provide `portfolio_cross_sectional_model_microcosm.ipynb` - a self-contained Jupyter notebook that demonstrates the complete pipeline using synthetic data. The notebook generates a realistic market universe of 30 stocks, computes cross-sectional features, trains a small autoencoder, and analyzes example portfolios (tech-heavy, defensive, and erratic) with full attribution. It runs in under a minute and requires no API keys or data downloads.

The synthetic data approach makes every step transparent and reproducible, allowing you to experiment freely with features, architecture, and portfolio compositions. While simplified compared to the production system (6 features vs 15, smaller network), the notebook captures the essential concepts and provides a sandbox for learning before diving into the full codebase. See [MICROCOSM_GUIDE.md](docs/MICROCOSM_GUIDE.md) for detailed usage and examples.

## Project Structure

```
portfolio-anomaly-detection/
├── data/
│   └── processed/
│       └── market_data.sqlite          # Market prices, features, and scores
├── models/
│   └── model_types/
│       ├── individual/                 # Risk assessment models
│       │   ├── autoencoder/
│       │   └── isolation_forest/
│       └── cross_sectional/            # Attribution models
│           └── autoencoder/
├── src/
│   ├── features/
│   │   ├── base.py                     # Base feature extractor
│   │   ├── individual.py               # Individual stock features (30)
│   │   ├── cross_sectional.py          # Cross-sectional features (15)
│   │   ├── factory.py                  # Feature extractor factory
│   │   └── storage.py                  # Feature caching
│   ├── models/
│   │   ├── autoencoder.py              # PyTorch autoencoder
│   │   └── isolation_forest.py        # Scikit-learn wrapper
│   └── portfolio/
│       ├── portfolio_analyzer.py       # Unified analysis interface
│       └── analyze.py                  # Attribution logic
├── scripts/
│   ├── analyze_portfolio.py            # CLI analysis tool
│   ├── train_model.py                  # Model training script
│   ├── score_universe.py               # Universe scoring script
│   ├── validate_model.py              # Model validation
│   ├── visualize_model.py              # Diagnostic visualizations
│   ├── fetch_data.py                   # Data fetching from Yahoo Finance
│   └── test_portfolios.py              # Portfolio testing
├── api/
│   ├── service.py                      # FastAPI application
│   ├── business_logic.py               # Portfolio scoring logic
│   ├── schemas.py                      # Pydantic models
│   └── API_ENDPOINTS.md                # API documentation
├── experimental/
│   ├── llm_service.py                  # LLM client wrapper
│   └── llm_agents.py                   # Agentic LLM integration
├── config/
│   └── model_config.yaml               # Configuration
├── docs/
│   └── Portfolio_Anomaly_Detection_Whitepaper.pdf
└── pyproject.toml                      # Dependencies
```

## Advanced Usage

### Training Models

Both model types must be trained separately:

```bash
# Train individual model (risk assessment)
python scripts/train_model.py individual --train-end 2024-06-30

# Train cross-sectional model (attribution)
python scripts/train_model.py cross_sectional --train-end 2024-06-30

# Custom training window
python scripts/train_model.py individual --train-start 2020-01-01 --train-end 2024-06-30

# Specify device
python scripts/train_model.py individual --device cuda
```

Training generates:
- Model checkpoints in `models/model_types/{model_type}/`
- Comprehensive visualizations in `results/training/{model_type}/`
- Training logs with loss curves and metrics

### Validating Models

Run validation to assess model performance:

```bash
# Validate individual model
python scripts/validate_model.py individual

# Validate cross-sectional model
python scripts/validate_model.py cross_sectional

# Generate validation dashboard
python scripts/create_validation_dashboard.py
```

Validation produces:
- Performance metrics (precision, recall, F1)
- Confusion matrices
- Score distribution plots
- Example anomalies with explanations

### Scoring the Universe

After training, score the full S&P 500 universe to populate the database:

```bash
# Score with individual model (for risk assessment)
python scripts/score_universe.py individual

# Score with cross-sectional model (for attribution)
python scripts/score_universe.py cross_sectional

# Score specific date range
python scripts/score_universe.py individual --start-date 2024-01-01 --end-date 2024-06-30
```

This creates daily anomaly scores for all stocks in the database, enabling:
- Fast portfolio analysis (no need to recompute features)
- Historical lookback and trend analysis
- Market-wide anomaly patterns

### Model Diagnostics

Generate visual diagnostics to understand model behavior:

```bash
# Visualize individual model
python scripts/visualize_model.py individual

# Visualize cross-sectional model
python scripts/visualize_model.py cross_sectional
```

Creates diagnostic plots in `results/diagnostics/`:
- **score_distributions.png** - Score distributions with thresholds
- **anomalies_over_time.png** - Anomaly patterns over time
- **feature_comparison.png** - Feature distributions (normal vs anomalies)
- **top_anomalies.png** - Detailed analysis of top anomalies
- **model_agreement.png** - Where models agree/disagree (individual only)

### Configuration

Edit `model_config.yaml` to customize:

```yaml
# Global settings
data:
  universe: sp500_sample  # or sp500, custom
  lookback_years: 3
  
features:
  returns_windows: [1, 5, 20, 60]
  volatility_windows: [20, 60]
  technical_indicators: [rsi, macd, bollinger_bands]

# Model-specific settings
model_types:
  individual:
    models:
      autoencoder:
        encoding_dim: 15
        hidden_dims: [128, 64, 32]
        threshold_percentile: 97
      isolation_forest:
        contamination: 0.05
        n_estimators: 200
  
  cross_sectional:
    features:
      config:
        market_proxy: SPY
        correlation_windows: [20, 60]
    models:
      autoencoder:
        encoding_dim: 15
        threshold_percentile: 97
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step setup guide
- **[api/API_ENDPOINTS.md](api/API_ENDPOINTS.md)** - Complete API reference
- **[docs/Portfolio_Anomaly_Detection_Whitepaper.pdf](docs/Portfolio_Anomaly_Detection_Whitepaper.pdf)** - Original whitepaper covering individual model methodology and validation

**Coming Soon:**
- Cross-Sectional Model Methodology Paper - Detailed explanation of attribution approach

## Performance Metrics

### Individual Model (from validation study)

| Metric | Value |
|--------|-------|
| F1 Score (Synthetic Anomalies) | 0.63 |
| Recall | 93% |
| Model Agreement | >94% |
| Ensemble Weight | 85% AE / 15% IF |

The 93% recall ensures strong capability to catch true anomalies, critical for a risk detection system where false negatives are more costly than false positives.

### Cross-Sectional Model

| Specification | Value |
|---------------|-------|
| Training Data | 10 years S&P 500 daily |
| Features | 15 cross-sectional metrics |
| Architecture | 128→64→32→15 encoding |
| Threshold | 97th percentile |

*Formal validation study in progress*

## When to Use Which Model

### Use Individual Model (Risk Assessment) When:
- Evaluating portfolio for unusual risk characteristics
- Detecting technical breakdowns or momentum extremes
- Assessing absolute risk levels
- Monitoring for volatility spikes or volume anomalies
- **Question:** "Is this portfolio unusual?"

### Use Cross-Sectional Model (Attribution) When:
- Portfolio performance diverges from market
- Need to identify which holdings drive divergence
- Investigating correlation breakdowns
- Analyzing relative underperformance/outperformance
- **Question:** "Which stocks are moving differently from the market?"

### Use Both (Comprehensive Analysis) When:
- Need complete portfolio assessment
- Want both risk levels and attribution
- Investigating complex portfolio issues
- Preparing detailed portfolio reports

**Example Scenario:**
```
Market: +2.5% today
Your Portfolio: -4.0% today

Individual Model: "Risk level: MEDIUM (high volatility, momentum divergence)"
Cross-Sectional Model: "Top divergence contributors: TSLA (-8%, 30% weight), 
                        NVDA (-6%, 25% weight)"

Insight: Two large positions moved sharply against the market, driving 
         the portfolio divergence.
```

## LLM Integration (Experimental)

Optional integration with Anthropic Claude for natural language explanations of anomaly patterns.

### Setup

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Or add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Usage

```bash
# Via API endpoint
curl -X POST http://localhost:8000/portfolio/explain \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100},
      "MSFT": {"shares": 50}
    }
  }'
```

### Example Output

```json
{
  "explanation": "Your portfolio shows elevated risk signals that warrant attention. 
  The autoencoder detected unusual patterns in the momentum indicators — specifically, 
  short-term returns are diverging from longer-term trends, which historically 
  precedes increased volatility. Combined with the 85% tech sector concentration, 
  I'd recommend reviewing your diversification strategy...",
  "risk_level": "medium",
  "key_concerns": [
    "Momentum divergence across timeframes",
    "High sector concentration"
  ]
}
```

### Architecture

The LLM integration uses an agentic flow:

```
Portfolio Analysis Results
         │
         ▼
┌─────────────────────────┐
│    Agentic Flow         │
│  ┌───────────────────┐  │
│  │ Context Agent     │──┼──► Gathers portfolio context
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │ Interpreter Agent │──┼──► Translates scores to insights  
│  └───────────────────┘  │
└─────────────────────────┘
         │
         ▼
  Natural Language Report
```

**Status:** 🧪 Experimental - Works but not production-hardened. Use core ML system for reliable scoring; use LLM for enhanced interpretability.

## Dependencies

Core requirements (installed via `pip install -e .`):

```
# Data & ML
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0

# Market Data
yfinance>=0.2.28

# API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Database
sqlite3 (built-in)

# Optional: LLM Integration
anthropic>=0.3.0
python-dotenv>=1.0.0
```

No API keys required for core functionality - market data comes from Yahoo Finance (free).

## Troubleshooting

### "Database not found"
```bash
python scripts/fetch_data.py
```

### "Models not found"
```bash
# Train both models
python scripts/train_model.py individual --train-end 2024-06-30
python scripts/train_model.py cross_sectional --train-end 2024-06-30
```

### "No features calculated"
Make sure you've run data fetch before training:
```bash
python scripts/fetch_data.py
python scripts/train_model.py individual --train-end 2024-06-30
```

### API connection refused
```bash
# Make sure API is running
python -m uvicorn api.service:app --reload
# Or use make command
make serve
```

### GPU/CUDA errors
```bash
# Force CPU training
python scripts/train_model.py individual --device cpu
```

## Future Enhancements

1. **Cross-Sectional Model Validation** - Formal validation study matching individual model rigor
2. **Real-Time Monitoring** - WebSocket streaming for live portfolio surveillance  
3. **Historical Backtesting** - Validation against known market events (crashes, bubbles)
4. **Sector-Specific Models** - Industry-tuned anomaly detection
5. **Portfolio-Specific Models** - Personalized detection as user history accumulates
6. **Multi-Asset Extension** - Bonds, commodities, crypto support
7. **LLM Enhancement** - Production-ready natural language explanations

## Contributing

This is a research/demonstration project. For questions or suggestions:

1. Open an issue on GitHub
2. Email directly (see contact below)
3. Connect on LinkedIn

## References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. ICDM.
2. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. ICLR.
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. ACM Computing Surveys.
4. de Prado, M. L. (2018). Advances in Financial Machine Learning. Wiley.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

For detailed methodology and validation of the individual model approach, see the [whitepaper](docs/Portfolio_Anomaly_Detection_Whitepaper.pdf).

## Author

**Tim Pinard**  
Software Engineer | Machine Learning & Financial Applications

- **GitHub:** [timpinard/portfolio-anomaly-detection](https://github.com/timpinard/portfolio-anomaly-detection)
- **Email:** t.pinard@gmail.com
- **LinkedIn:** [linkedin.com/in/timpinard](https://linkedin.com/in/timpinard)

---

## License

[Your chosen license]

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{pinard2025portfolio,
  author = {Pinard, Tim},
  title = {Portfolio Anomaly Detection: A Dual-Model Approach},
  year = {2025},
  url = {https://github.com/timpinard/portfolio-anomaly-detection}
}
```
