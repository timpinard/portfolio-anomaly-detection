# Portfolio Anomaly Detection

A machine learning system for detecting unusual patterns in investment portfolios using a dual-model ensemble approach.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Traditional portfolio risk metrics (VaR, Sharpe ratio, beta) measure *known* risks based on historical patterns and assumptions. This system answers a different question:

> **“Is there something unusual about this portfolio that might indicate elevated risk?”**

The system detects anomalies that traditional metrics do not explicitly target:

* Unusual behavior relative to normal market patterns
* Internal indicator conflicts (e.g., momentum vs. volatility disagreement)
* Hidden concentration risk
* Subtle structural shifts that may precede larger problems

---

## How It Works

### Dual-Model Ensemble

Two complementary unsupervised algorithms with different strengths:

| Model                | Type           | Strength                                    |
| -------------------- | -------------- | ------------------------------------------- |
| **Autoencoder**      | Neural Network | Complex non-linear pattern deviations       |
| **Isolation Forest** | Tree-Based     | Isolation of observations in sparse regions |

Models are weighted **85% Autoencoder / 15% Isolation Forest** based on validation performance. When both models agree on an anomaly, confidence is high.

---

### 30 Engineered Features

Each portfolio is represented using 30 technical indicators:

* **Momentum:** Returns across 1, 5, 20, and 60-day windows
* **Volatility:** 20 and 60-day annualized volatility
* **RSI:** Momentum exhaustion (overbought / oversold)
* **MACD:** Trend direction, signal, and histogram
* **Bollinger Bands:** Price position within volatility envelopes
* **Volume:** Trading activity ratios and variability
* **Price Position:** Relationship to moving averages and 52-week highs/lows

---

### Portfolio-Weighted Aggregation

Rather than scoring securities individually, features are aggregated into a single portfolio-weighted vector. This captures overall portfolio *personality* and detects when that personality is internally inconsistent or unusual relative to the broader market.

---

## Performance

| Metric                         | Autoencoder | Isolation Forest |
| ------------------------------ | ----------- | ---------------- |
| F1 Score (Synthetic Anomalies) | **0.63**    | 0.19             |
| Recall                         | **93%**     | Lower            |
| Model Agreement                | **>94%**    | —                |

*Validation uses domain-realistic synthetic anomalies (market crashes, momentum bubbles, volatility spikes, correlation breakdowns, liquidity crises), not purely statistical extremes.*

---

## Quick Start

### Installation

```bash
git clone https://github.com/timpinard/portfolio-anomaly-detection.git
cd portfolio-anomaly-detection
pip install -e .
```

### Bootstrap (Fetch Data, Engineer Features, Train Models)

```bash
make bootstrap
```

This will:

1. Download 5 years of market data for 60+ securities
2. Compute 30 technical indicators per security
3. Train Autoencoder and Isolation Forest models

---

### Start the API

```bash
make serve
```

---

### Analyze a Portfolio

```bash
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100},
      "MSFT": {"shares": 50},
      "GOOGL": {"shares": 30}
    }
  }'
```

---

### Example Response

```json
{
  "risk_level": "medium",
  "concentration_risk": "low",
  "model_results": {
    "autoencoder": {
      "score": 0.0234,
      "is_anomaly": false
    },
    "isolation_forest": {
      "is_anomaly": false
    },
    "consensus": {
      "models_agree": true,
      "confidence": "high"
    }
  },
  "recommendations": [
    "Portfolio within normal risk parameters",
    "Tech concentration at 85% - consider diversification"
  ]
}
```

---

## API Endpoints

| Endpoint             | Method | Description                                  |
| -------------------- | ------ | -------------------------------------------- |
| `/portfolio/analyze` | POST   | Analyze portfolio and return risk assessment |
| `/portfolio/explain` | POST   | Generate feature-level explanation           |
| `/health`            | GET    | Service health check                         |

---

## Configuration

Edit `config/model_config.yaml` to customize behavior.

---

## Documentation

* **Technical Whitepaper** — Full architecture, feature engineering, and validation methodology
* **LinkedIn Article** — High-level overview and lessons learned

---

## Key Design Decisions

* **Universal Training:** No portfolio-specific history required
* **Performance-Based Ensemble Weighting:** Autoencoder primary, Isolation Forest confirmatory
* **Realistic Validation:** Domain-driven anomaly scenarios
* **Market-Relative Scoring:** Z-score normalization against current market conditions

---

## Experimental: LLM-Powered Explanations

The `/experimental` folder contains a **non-core, optional** agentic LLM integration that translates anomaly scores into natural language insights using Claude.

---

## License

MIT License

---

## Author

**Tim Pinard** — Software Engineer

* Website: [https://timpinard.com](https://timpinard.com)
* GitHub: [https://github.com/timpinard](https://github.com/timpinard)
* LinkedIn: [https://linkedin.com/in/timpinard](https://linkedin.com/in/timpinard)
