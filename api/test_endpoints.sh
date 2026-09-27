#!/bin/bash
# Test curl commands for Portfolio Anomaly Detection API
# Make sure the API server is running: make serve

BASE_URL="http://localhost:8000"

echo "=== Testing Portfolio Anomaly Detection API ==="
echo ""

# 1. Health Check
echo "1. Health Check"
echo "GET $BASE_URL/health"
curl -X GET "$BASE_URL/health" \
  -H "Content-Type: application/json" | jq '.'
echo ""
echo ""

# 2. Root endpoint (list all endpoints)
echo "2. Root Endpoint (List All Endpoints)"
echo "GET $BASE_URL/"
curl -X GET "$BASE_URL/" \
  -H "Content-Type: application/json" | jq '.'
echo ""
echo ""

# 3. Portfolio Health Analysis (Cross-sectional model)
# This is the endpoint your original curl was targeting
echo "3. Portfolio Health Analysis (Cross-sectional Model)"
echo "POST $BASE_URL/portfolio/health"
curl -X POST "$BASE_URL/portfolio/health" \
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
  }' | jq '.'
echo ""
echo ""

# 4. Portfolio Health Analysis (Individual model)
echo "4. Portfolio Health Analysis (Individual Model)"
echo "POST $BASE_URL/portfolio/health"
curl -X POST "$BASE_URL/portfolio/health" \
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
  }' | jq '.'
echo ""
echo ""

# 5. Portfolio Analyze (uses positions with shares, not weights)
echo "5. Portfolio Analyze (Individual Model - Risk Assessment)"
echo "POST $BASE_URL/portfolio/analyze"
curl -X POST "$BASE_URL/portfolio/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100, "cost_basis": 150.0},
      "MSFT": {"shares": 50, "cost_basis": 300.0},
      "GOOGL": {"shares": 25, "cost_basis": 120.0}
    },
    "benchmark": "SPY"
  }' | jq '.'
echo ""
echo ""

# 6. Portfolio Analyze with Extended Metrics
echo "6. Portfolio Analyze with Extended Metrics"
echo "POST $BASE_URL/portfolio/analyze?extended=true"
curl -X POST "$BASE_URL/portfolio/analyze?extended=true" \
  -H "Content-Type: application/json" \
  -d '{
    "positions": {
      "AAPL": {"shares": 100},
      "MSFT": {"shares": 50},
      "TSLA": {"shares": 10}
    },
    "benchmark": "SPY"
  }' | jq '.'
echo ""
echo ""

# 7. Analyze and Explain (Orchestrated endpoint)
echo "7. Analyze and Explain (Orchestrated - includes LLM if available)"
echo "POST $BASE_URL/portfolio/analyze-and-explain"
curl -X POST "$BASE_URL/portfolio/analyze-and-explain" \
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
  }' | jq '.'
echo ""
echo ""

# 8. Explain Recommendations (requires previous analysis)
echo "8. Explain Recommendations (standalone)"
echo "POST $BASE_URL/portfolio/explain"
curl -X POST "$BASE_URL/portfolio/explain" \
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
  }' | jq '.'
echo ""
echo ""

echo "=== All tests complete ==="
