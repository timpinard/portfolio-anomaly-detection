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

This runs three steps:
1. **fetch_data.py** - Downloads 5 years of market data for complete S&P 500 universe (~500 symbols) from Yahoo Finance (free, no API key)
2. **train_model.py individual** - Trains the individual model (risk assessment)
3. **train_model.py cross_sectional** - Trains the cross-sectional model (attribution)

**Time:** ~15-30 minutes depending on internet speed (S&P 500 dataset is comprehensive)

**Note:** By default, the system fetches the complete S&P 500 universe. For a smaller custom universe (~60 symbols), use `make fetch-custom` instead.

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

1. **Market Data**: Downloaded historical prices for complete S&P 500 universe (~500 symbols)
2. **Features**: Calculated returns, volatility, RSI, MACD, Bollinger Bands for each stock
3. **Training**: Two models learned different aspects of market behavior:
   - **Individual model**: Learned "normal" individual stock patterns for risk assessment
   - **Cross-sectional model**: Learned typical stock-vs-market relationships for attribution
4. **API**: Service ready to analyze any portfolio using both approaches

## Understanding the Two Model Types

This system uses two complementary approaches to provide comprehensive portfolio analysis:

### Individual Model (Risk Assessment)

**Question it answers:** "Is this portfolio unusual?"

**What it does:**
- Analyzes individual stock characteristics (returns, volatility, RSI, MACD, etc.)
- Compares portfolio features to historical patterns of normal market behavior
- Detects portfolios with unusual risk characteristics

**Use cases:**
- Detecting high volatility or momentum extremes
- Identifying technical breakdowns (oversold/overbought conditions)
- Spotting volume anomalies or liquidity concerns
- General portfolio risk assessment

**Example scenario:**
```
Portfolio has extremely high short-term returns with very low RSI
→ Individual model flags this as unusual (momentum exhaustion pattern)
→ Risk level: HIGH
```

### Cross-Sectional Model (Attribution)

**Question it answers:** "Which stocks are driving divergence from the market?"

**What it does:**
- Analyzes market-relative characteristics (relative returns, correlations, beta)
- Compares how each stock behaves relative to the market (SPY)
- Identifies which holdings are moving differently from expected market relationships

**Use cases:**
- Market is up but your portfolio is down - why?
- Identifying correlation breakdowns
- Finding stocks with shifting beta (becoming defensive or aggressive)
- Detecting relative underperformance or outperformance

**Example scenario:**
```
Market: +2% today, Your Portfolio: -4% today
→ Cross-sectional model identifies:
  - TSLA (-8%, 30% weight) - top divergence contributor
  - NVDA (-6%, 25% weight) - second contributor
→ Insight: Two large positions moved sharply against market
```

### When to Use Which Model

| Scenario | Use Individual | Use Cross-Sectional | Use Both |
|----------|---------------|---------------------|----------|
| "Is my portfolio risky?" | ✅ | | ✅ |
| "Why am I underperforming?" | | ✅ | ✅ |
| "Which stocks are problematic?" | | ✅ | ✅ |
| "General health check" | ✅ | ✅ | ✅ |
| "High volatility concerns" | ✅ | | ✅ |
| "Correlation breakdown" | | ✅ | ✅ |

**Best practice:** Use both models together for comprehensive analysis. They provide complementary insights.

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

# Risk assessment (individual model)
print(f"Risk Level: {result['risk_level']}")
print(f"Anomaly: {result['model_results']['autoencoder']['is_anomaly']}")

# Attribution (cross-sectional model)
if 'attribution' in result:
    print(f"Health Score: {result['attribution']['health_score']}")
    print(f"Top Contributor: {result['attribution']['contributors'][0]['symbol']}")
```

## Interactive Documentation

Visit http://localhost:8000/docs for:
- Interactive API testing
- Request/response schemas
- Example payloads
- Try different portfolios

## What the Models Do

### Two Approaches, Complementary Insights

#### Individual Model Approach (Risk Assessment)

Uses these algorithms to assess portfolio risk:

**Autoencoder**
- Learns patterns in individual stock features (correlations, volatility, returns)
- Reconstructs portfolio features using a compressed representation
- High reconstruction error = unusual/anomalous portfolio
- Architecture: 30 features → 128 → 64 → 32 → 15 (latent) → 32 → 64 → 128 → 30 features

**Isolation Forest**
- Identifies statistical outliers using decision tree ensemble
- Flags portfolios that don't fit normal market distributions
- Works by isolating anomalous points with fewer random partitions

**Ensemble Consensus**
- Both models agree → High confidence
- Models disagree → Medium confidence (investigate further)
- Weighted combination: 85% autoencoder, 15% isolation forest

#### Cross-Sectional Model Approach (Attribution)

Uses an autoencoder trained on market-relative features:

**Autoencoder (Cross-Sectional)**
- Learns how stocks typically behave relative to the market (SPY)
- Trained on 15 cross-sectional features:
  - Relative returns (stock return - market return)
  - Rolling correlations with market
  - Beta (market sensitivity)
  - Direction agreement (% days moving with market)
  - Universe z-scores (how different from average stock)
- Measures divergence from expected market relationships
- High reconstruction error = stock moving unusually vs market
- **Output:** Attribution scores showing which holdings drive divergence

**Why no Isolation Forest for cross-sectional?**
Cross-sectional patterns are more nuanced and relational. The autoencoder's ability to learn complex market-relative relationships is more valuable than simple outlier detection.

### Feature Types Comparison

| Feature Category | Individual Model | Cross-Sectional Model |
|------------------|------------------|----------------------|
| Returns | ✅ Absolute returns (1d, 5d, 20d, 60d) | ✅ Relative returns (stock - market) |
| Volatility | ✅ Rolling volatility | ✅ Correlation with market |
| Technical | ✅ RSI, MACD, Bollinger Bands | ❌ (not market-relative) |
| Market Relationship | ❌ (stock in isolation) | ✅ Beta, correlation, direction agreement |
| Volume | ✅ Volume ratios | ❌ (not used for attribution) |
| Price Position | ✅ MA, 52-week highs | ✅ Percentile rank vs universe |

## Common Issues

### "Database not found"
Make sure you've run the data fetch script:
```bash
python scripts/fetch_data.py
```

### "Models not found"
Train both models:
```bash
make bootstrap
# Or manually:
python scripts/train_model.py individual --train-end 2024-06-30
python scripts/train_model.py cross_sectional --train-end 2024-06-30
```

### "No features calculated"
Make sure you've run all steps in order:
```bash
python scripts/fetch_data.py
python scripts/train_model.py individual --train-end 2024-06-30
python scripts/train_model.py cross_sectional --train-end 2024-06-30
```

### API connection error
Make sure API is running:
```bash
python -m uvicorn api.service:app --reload
# Or use make command:
make serve
```

### GPU/CUDA errors
Force CPU training if GPU is unavailable:
```bash
python scripts/train_model.py individual --device cpu
python scripts/train_model.py cross_sectional --device cpu
```

## Model Diagnostics

After training models, you can visualize their behavior:

```bash
# Generate visual diagnostics for individual model
python scripts/visualize_model.py individual

# Generate visual diagnostics for cross-sectional model
python scripts/visualize_model.py cross_sectional
```

This creates diagnostic plots in `results/diagnostics/{model_type}/`:
- **score_distributions.png** - Score distributions with thresholds
- **anomalies_over_time.png** - Anomaly patterns over time
- **feature_comparison.png** - Feature distributions for normal vs anomalies
- **top_anomalies.png** - Detailed analysis of top anomalies
- **model_agreement.png** - Where models agree/disagree (individual model only)
- **diagnostic_summary.txt** - Text summary of findings

Use these visualizations to:
- Understand how models score data points
- Identify patterns in flagged anomalies
- Validate model behavior
- Detect potential issues (e.g., too many/few anomalies)

## Analyze Your Portfolio

After models are trained, you can analyze any portfolio using both individual and cross-sectional models:

### Basic Usage

```bash
# Comprehensive analysis (both models) with equal weights
python scripts/analyze_portfolio.py AAPL MSFT GOOGL

# Analyze with specific shares (weights calculated automatically)
python scripts/analyze_portfolio.py --shares AAPL:100 MSFT:50 GOOGL:25

# Analyze with custom weights
python scripts/analyze_portfolio.py AAPL:0.4 MSFT:0.35 GOOGL:0.25

# From JSON file
python scripts/analyze_portfolio.py --file my_portfolio.json
```

### Selective Analysis

```bash
# Risk assessment only (individual model, faster)
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --risk-only

# Attribution analysis only (cross-sectional model)
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --attribution-only

# Attribution for specific date
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --date 2024-01-15
```

### Output Options

```bash
# JSON output
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --json

# Save to file
python scripts/analyze_portfolio.py AAPL MSFT GOOGL --json --output results/my_portfolio.json

# Human-readable report (default)
python scripts/analyze_portfolio.py AAPL MSFT GOOGL
```

### Understanding the Output

#### Risk Assessment (Individual Model)

```
RISK ASSESSMENT (Individual Model Approach)
--------------------------------------------------------------------------------
Risk Level: MEDIUM
Anomaly Detected: Yes

Model Results:
  Autoencoder: score=0.0234, anomaly=True, threshold=0.0189
  Isolation Forest: score=-0.15, anomaly=False
  Consensus: False (confidence: medium)

Recommendations:
  • Monitor momentum divergence between short and long-term returns
  • Review technical indicators for oversold conditions
  • Consider sector concentration - 85% in Technology
```

**What this means:**
- Portfolio has unusual characteristics compared to historical patterns
- Autoencoder detected the anomaly, but Isolation Forest did not (lower confidence)
- Specific concerns are listed in recommendations

**Risk Levels:**
- `low`: Portfolio within normal risk parameters
- `medium`: Some anomalous characteristics detected, monitor closely
- `high`: Significant anomalies detected, investigate immediately

#### Attribution Analysis (Cross-Sectional Model)

```
ATTRIBUTION ANALYSIS (Cross-Sectional Model Approach)
--------------------------------------------------------------------------------
Analysis Date: 2024-01-15
Model Type: cross_sectional

Portfolio Metrics:
  Structural Score: 0.0156
  Structural Z-Score: +1.23
  Health Score: -1.45

Market Baseline:
  Mean: 0.0123
  Std: 0.0045
  95th percentile: 0.0201

Top Contributors (by divergence):
  TSLA  : weight=30.00%, score= 0.0234, contribution= 0.0070
  AAPL  : weight=25.00%, score= 0.0189, contribution= 0.0047
  GOOGL : weight=20.00%, score= 0.0145, contribution= 0.0029
```

**What this means:**
- **Structural Score**: Overall portfolio divergence from market baseline
- **Structural Z-Score**: How many standard deviations above/below market average
  - +1.23 = portfolio slightly more divergent than typical
  - Values > +2 indicate significant divergence
- **Health Score**: Directional + structural combination
  - Negative values = portfolio moving against market trends
  - -1.45 = moderate contra-market behavior
- **Top Contributors**: Stocks driving the most divergence
  - TSLA has highest individual divergence score (0.0234)
  - Combined with 30% weight, contributes most to portfolio divergence

**Health Score Interpretation:**
- `< -2.0`: Significant divergence, investigate immediately
- `-2.0 to -1.0`: Moderate divergence, monitor closely
- `-1.0 to +1.0`: Normal range
- `+1.0 to +2.0`: Moderate divergence (different pattern)
- `> +2.0`: Significant divergence

## Model Validation

Validate model performance with test scenarios:

```bash
# Validate individual model (risk assessment)
python scripts/validate_model.py individual

# Validate cross-sectional model (attribution)
python scripts/validate_model.py cross_sectional
```

This runs the model against validation scenarios and generates:
- Performance metrics (precision, recall, F1)
- Confusion matrices
- Score distributions
- Visual diagnostics

**When to validate:**
- After training new models
- When changing hyperparameters
- To verify model behavior before production use
- Periodically to check for model drift

## Next Steps

1. **Validate Both Models**: 
   ```bash
   python scripts/validate_model.py individual
   python scripts/validate_model.py cross_sectional
   ```

2. **Score the Universe**: Populate database with daily scores for all stocks
   ```bash
   python scripts/score_universe.py individual
   python scripts/score_universe.py cross_sectional
   ```

3. **Visual Diagnostics**: Generate visualizations to understand model behavior
   ```bash
   python scripts/visualize_model.py individual
   python scripts/visualize_model.py cross_sectional
   ```

4. **Analyze Your Portfolio**: Run comprehensive analysis
   ```bash
   python scripts/analyze_portfolio.py AAPL MSFT GOOGL AMZN
   ```

5. **Customize Universe**: Edit `config/model_config.yaml` to use different stocks
   ```yaml
   data:
     universe: custom
     custom_symbols: [AAPL, MSFT, GOOGL, AMZN, TSLA, ...]
   ```

6. **Add LLM Analysis**: Set `ANTHROPIC_API_KEY` for natural language reports
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   curl -X POST http://localhost:8000/portfolio/explain -d '{"positions": {...}}'
   ```

7. **Extend Features**: Modify feature extractors to add new indicators
   - Edit `src/data/individual.py` for individual features
   - Edit `src/data/cross_sectional.py` for cross-sectional features

8. **Build Frontend**: Create dashboard using the API
   - Use the FastAPI endpoints
   - Interactive docs at http://localhost:8000/docs
   - Example: `requests.post("http://localhost:8000/portfolio/analyze", json=portfolio)`

## Understanding Model Training

### What Gets Trained

Both models use autoencoders (neural networks) that learn to compress and reconstruct features:

**Individual Model Training:**
```bash
python scripts/train_model.py individual --train-end 2024-06-30
```
- Learns patterns in 30 individual stock features
- Trains autoencoder (128→64→32→15 encoding)
- Trains isolation forest on same features
- Saves models to `models/model_types/individual/`
- Generates visualizations in `results/training/individual/`

**Cross-Sectional Model Training:**
```bash
python scripts/train_model.py cross_sectional --train-end 2024-06-30
```
- Learns patterns in 15 market-relative features
- Trains autoencoder (same architecture)
- No isolation forest (not applicable to cross-sectional)
- Saves models to `models/model_types/cross_sectional/`
- Generates visualizations in `results/training/cross_sectional/`

### Training Data

- **Source**: S&P 500 universe daily data from Yahoo Finance
- **Default window**: 10 years of historical data
- **Split**: 80% training, 20% validation
- **Data cleaning**: Mahalanobis distance outlier removal (top 5%)

### What to Expect

During training you'll see:
```
Epoch 10/100: Loss = 0.0234 (training), 0.0245 (validation)
Epoch 20/100: Loss = 0.0189 (training), 0.0201 (validation)
...
Training complete! Best validation loss: 0.0178
Models saved to: models/model_types/individual/
```

Training generates multiple plot types:
- **Training plots**: Loss curves, learning progress
- **Validation plots**: Score distributions, threshold analysis
- **Summary plots**: Performance overview, example anomalies

## File Structure Reference

```
portfolio-anomaly-detection/
├── data/
│   └── processed/
│       └── market_data.sqlite    # Market prices, features, and scores
├── models/
│   └── model_types/
│       ├── individual/           # Risk assessment models
│       │   ├── autoencoder/
│       │   │   ├── model.pt
│       │   │   └── scaler.joblib
│       │   └── isolation_forest/
│       │       └── model.joblib
│       └── cross_sectional/      # Attribution models
│           └── autoencoder/
│               ├── model.pt
│               └── scaler.joblib
├── results/
│   ├── training/                 # Training visualizations
│   │   ├── individual/
│   │   └── cross_sectional/
│   ├── validation/               # Validation results
│   │   ├── individual/
│   │   └── cross_sectional/
│   └── diagnostics/              # Model diagnostics
│       ├── individual/
│       └── cross_sectional/
├── src/
│   ├── features/
│   │   ├── base.py              # Base feature extractor
│   │   ├── individual.py         # Individual features (30)
│   │   ├── cross_sectional.py    # Cross-sectional features (15)
│   │   ├── factory.py            # Feature extractor factory
│   │   └── storage.py            # Feature caching
│   └── models/
│       ├── autoencoder.py        # PyTorch autoencoder
│       └── isolation_forest.py   # Scikit-learn wrapper
├── scripts/
│   ├── fetch_data.py             # Download data from Yahoo Finance
│   ├── train_model.py            # Train models
│   ├── score_universe.py         # Score all stocks
│   ├── validate_model.py         # Validate model performance
│   ├── visualize_model.py        # Generate diagnostics
│   └── analyze_portfolio.py      # CLI analysis tool
├── api/
│   └── service.py                # FastAPI application
└── config/
    └── model_config.yaml          # Configuration
```

## Makefile Commands

```bash
make setup       # Install dependencies
make bootstrap   # Fetch data + train both models
make serve       # Start API
make test-api    # Run API tests
make clean       # Remove generated files
```

## Advanced Configuration

Edit `model_config.yaml` to customize:

### Data Settings
```yaml
data:
  universe: sp500_sample  # sp500, sp500_sample, or custom
  lookback_years: 10      # Years of historical data
  interval: "1day"        # Data frequency
```

### Feature Windows
```yaml
features:
  returns_windows: [1, 5, 20, 60]      # Days for return calculations
  volatility_windows: [20, 60]          # Days for volatility
  ma_windows: [50, 200]                 # Moving average periods
```

### Model Hyperparameters
```yaml
model_types:
  individual:
    models:
      autoencoder:
        encoding_dim: 15              # Latent space dimensions
        hidden_dims: [128, 64, 32]    # Hidden layer sizes
        epochs: 100
        batch_size: 64
        learning_rate: 0.001
        threshold_percentile: 97      # 97th percentile = top 3% flagged
```

### Cross-Sectional Features
```yaml
model_types:
  cross_sectional:
    features:
      config:
        market_proxy: SPY             # Market benchmark
        correlation_windows: [20, 60] # Correlation periods
        beta_window: 60               # Beta calculation window
```

## Support

- **Documentation**: See [README.md](README.md) for complete documentation
- **API Docs**: http://localhost:8000/docs when API is running
- **Issues**: Check that all dependencies are installed and API keys are set correctly
- **Whitepaper**: See [docs/Portfolio_Anomaly_Detection_Whitepaper.pdf](docs/Portfolio_Anomaly_Detection_Whitepaper.pdf) for detailed methodology

## Key Takeaways

✅ **Two Models = Comprehensive Analysis**
- Individual model for risk assessment: "Is this portfolio unusual?"
- Cross-sectional model for attribution: "Which stocks drive divergence?"

✅ **Easy to Use**
- Simple CLI: `python analyze_portfolio.py AAPL MSFT GOOGL`
- REST API: `POST /portfolio/analyze`
- Unified Python interface: `PortfolioAnalyzer().analyze(holdings)`

✅ **Well-Validated**
- Individual model: 93% recall, validated against synthetic scenarios
- Comprehensive visualization suite for model diagnostics
- Continuous validation workflow

✅ **Extensible**
- Add custom features to `feature_extractor.py`
- Modify model architectures in config
- Integrate with existing portfolio management systems

Enjoy analyzing portfolios! 🚀