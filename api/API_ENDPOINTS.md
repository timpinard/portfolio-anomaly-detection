# API Endpoints Reference

## Base URL
```
http://localhost:8000
```

## Available Endpoints

### 1. Health Check
**GET** `/health`

Check API health and model availability.

```bash
curl -X GET http://localhost:8000/health \
  -H "Content-Type: application/json"
```

---

### 2. Root (List Endpoints)
**GET** `/`

List all available endpoints.

```bash
curl -X GET http://localhost:8000/ \
  -H "Content-Type: application/json"
```

---

### 3. Portfolio Health Analysis
**POST** `/portfolio/health`

Analyze portfolio health using cross-sectional or individual models. This endpoint uses pre-computed anomaly scores to assess portfolio health and attribution.

**Request Body:**
- `holdings`: Array of objects with `symbol` and `weight` (0-1)
- `analysis_date`: Date string in YYYY-MM-DD format
- `experiment`: Model type - `"cross_sectional"` or `"individual"` (default: `"cross_sectional"`)
- `market_proxy`: Market proxy symbol (default: `"SPY"`)
- `contra_horizon`: Number of days for contra return calculation (default: `5`)

**Example - Cross-sectional Model:**
```bash
curl -X POST http://localhost:8000/portfolio/health \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "AAPL", "weight": 0.40},
      {"symbol": "MSFT", "weight": 0.40},
      {"symbol": "TSLA", "weight": 0.05},
      {"symbol": "NVDA", "weight": 0.05},
      {"symbol": "META", "weight": 0.10}
    ],
    "analysis_date": "2025-01-10",
    "experiment": "cross_sectional",
    "market_proxy": "SPY",
    "contra_horizon": 5
  }'
```

**Example - Individual Model:**
```bash
curl -X POST http://localhost:8000/portfolio/health \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "AAPL", "weight": 0.30},
      {"symbol": "MSFT", "weight": 0.30},
      {"symbol": "GOOGL", "weight": 0.20},
      {"symbol": "AMZN", "weight": 0.20}
    ],
    "analysis_date": "2025-01-10",
    "experiment": "individual",
    "market_proxy": "SPY",
    "contra_horizon": 5
  }'
```

**Response:**
- `date`: Analysis date
- `portfolio_score`: Weighted aggregate anomaly score
- `contra_return`: Portfolio return vs market over horizon
- `health_score`: Combined structural + contrarian health metric
- `contributors`: Per-holding score breakdown

---

### 4. Portfolio Analyze
**POST** `/portfolio/analyze`

Analyze portfolio for anomalies and risk using individual models. Returns anomaly detection results, risk assessment, concentration metrics, and recommendations.

**Request Body:**
- `positions`: Dictionary mapping ticker symbols to position objects
  - `shares`: Number of shares (required)
  - `cost_basis`: Cost basis per share (optional)
- `benchmark`: Benchmark symbol for comparison (optional, default: `"SPY"`)

**Query Parameters:**
- `extended`: Set to `true` for extended metrics with per-security detail (optional)

**Example - Basic Analysis:**
```bash
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100, "cost_basis": 150.0},
      "MSFT": {"shares": 50, "cost_basis": 300.0},
      "GOOGL": {"shares": 25, "cost_basis": 120.0}
    },
    "benchmark": "SPY"
  }'
```

**Example - Extended Analysis (with per-security detail):**
```bash
curl -X POST "http://localhost:8000/portfolio/analyze?extended=true" \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100},
      "MSFT": {"shares": 50},
      "TSLA": {"shares": 10}
    },
    "benchmark": "SPY"
  }'
```

**Response:**
- `model_results`: Autoencoder and Isolation Forest scores
- `risk_level`: Overall risk assessment (low/medium/high)
- `concentration_risk`: Concentration risk level
- `portfolio_metrics`: Portfolio metrics (extended if requested)
- `recommendations`: Actionable recommendations
- `message`: Human-readable summary

---

### 5. Analyze and Explain
**POST** `/portfolio/analyze-and-explain`

Orchestrated endpoint that combines portfolio analysis with LLM explanation (if LLM is available).

**Request Body:**
- `portfolio`: Portfolio object (same structure as `/portfolio/analyze`)
- `include_explanation`: Whether to include LLM explanation (default: `true`)
- `include_portfolio_context`: Whether to include full portfolio context (default: `true`)
- `include_security_detail`: Whether to include per-security scoring detail (default: `true`)

**Example:**
```bash
curl -X POST http://localhost:8000/portfolio/analyze-and-explain \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "positions": {
        "AAPL": {"shares": 100},
        "MSFT": {"shares": 50},
        "GOOGL": {"shares": 25}
      },
      "benchmark": "SPY"
    },
    "include_explanation": true,
    "include_portfolio_context": true,
    "include_security_detail": true
  }'
```

**Response:**
- `analysis`: Full portfolio analysis (same as `/portfolio/analyze`)
- `explanation`: LLM-generated explanation (if available and requested)
- `explanation_requested`: Whether explanation was requested
- `explanation_available`: Whether LLM explanation was successfully generated

---

### 6. Explain Recommendations
**POST** `/portfolio/explain`

Generate natural language explanation of portfolio recommendations using LLM. Requires LLM to be configured (ANTHROPIC_API_KEY).

**Request Body:**
- `recommendations`: List of recommendation strings
- `risk_level`: Portfolio risk level (low/medium/high)
- `portfolio_metrics`: Portfolio metrics dictionary
- `include_portfolio`: Whether to include full portfolio context (optional)
- `portfolio`: Full portfolio data (optional)
- `analysis`: Full analysis results (optional)
- `feature_conclusions`: Feature interpretation conclusions (optional)
- `security_scores`: Per-security scores (optional)

**Example:**
```bash
curl -X POST http://localhost:8000/portfolio/explain \
  -H "Content-Type: application/json" \
  -d '{
    "recommendations": [
      "Consider reducing concentration in top 3 positions",
      "Monitor TSLA for increased volatility"
    ],
    "risk_level": "medium",
    "portfolio_metrics": {
      "total_value": 50000,
      "n_positions": 3,
      "concentration_hhi": 0.45
    },
    "include_portfolio": false
  }'
```

**Response:**
- `success`: Whether explanation was generated
- `explanation`: LLM-generated explanation text
- `agents_executed`: List of agents that executed
- `errors`: Any errors encountered
- `timestamp`: Request timestamp

---

## Quick Reference

| Endpoint | Method | Purpose | Model Type |
|----------|--------|---------|------------|
| `/health` | GET | Health check | N/A |
| `/` | GET | List endpoints | N/A |
| `/portfolio/health` | POST | Health analysis with attribution | `cross_sectional` or `individual` |
| `/portfolio/analyze` | POST | Risk assessment | `individual` |
| `/portfolio/analyze-and-explain` | POST | Analysis + LLM explanation | `individual` |
| `/portfolio/explain` | POST | LLM explanation only | N/A |

## Notes

1. **Model Types:**
   - `individual`: Analyzes each security independently (risk assessment)
   - `cross_sectional`: Analyzes relationships across securities (attribution/divergence)

2. **Portfolio Format:**
   - `/portfolio/health` uses `holdings` with `weight` (0-1)
   - `/portfolio/analyze` uses `positions` with `shares` (number of shares)

3. **LLM Features:**
   - LLM explanation requires `ANTHROPIC_API_KEY` environment variable
   - If LLM is unavailable, endpoints will still work but skip explanation

4. **Date Format:**
   - All dates should be in `YYYY-MM-DD` format

5. **Testing:**
   - Run `./api/test_endpoints.sh` to test all endpoints
   - Or use the interactive docs at `http://localhost:8000/docs`
