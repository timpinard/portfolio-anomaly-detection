# Experimental: LLM-Powered Explanations

This folder contains an agentic LLM integration that transforms ML anomaly scores into natural language insights and recommendations.

## Overview

While the core system produces numerical scores and structured risk assessments, these outputs require interpretation. The LLM integration bridges this gap by:

1. **Interpreting Recommendations** — Converting structured risk findings into plain English
2. **Providing Context** — Explaining *why* a portfolio might be flagged as anomalous
3. **Suggesting Actions** — Translating risk signals into actionable next steps

## Architecture

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

## Files

| File | Description |
|------|-------------|
| `llm_service.py` | Low-level LLM client wrapper (Anthropic Claude) |
| `llm_agents.py` | Modular agent framework for multi-step reasoning |

## Usage

### Setup

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Or add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### API Endpoint

With the API running (`make serve`), use the explain endpoint:

```bash
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


## Requirements

```bash
pip install anthropic python-dotenv
```

## Status

🧪 **Experimental** — This integration works but is not yet production-hardened. Use the core ML system for reliable scoring; use this for enhanced interpretability.

## Coming Soon

A detailed article on this LLM integration architecture will be published on LinkedIn. Follow [Tim Pinard](https://linkedin.com/in/timpinard) for updates.
