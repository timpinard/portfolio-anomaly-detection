#!/usr/bin/env python3
"""Test client for portfolio anomaly detection API."""

import requests
import json

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("="*80)
    print("HEALTH CHECK")
    print("="*80)
    
    response = requests.get(f"{API_URL}/health")
    print(json.dumps(response.json(), indent=2))
    print()


def test_normal_portfolio():
    """Test with a balanced portfolio."""
    print("="*80)
    print("TEST 1: Balanced Portfolio")
    print("="*80)
    
    portfolio = {
        "positions": {
            "AAPL": {"shares": 100, "cost_basis": 150},
            "MSFT": {"shares": 80, "cost_basis": 300},
            "GOOGL": {"shares": 30, "cost_basis": 2500},
            "JPM": {"shares": 50, "cost_basis": 140},
            "JNJ": {"shares": 60, "cost_basis": 160}
        },
        "benchmark": "SPY"
    }
    
    response = requests.post(f"{API_URL}/portfolio/analyze", json=portfolio)
    result = response.json()
    
    print(f"Risk Level: {result['risk_level']}")
    print(f"Concentration Risk: {result['concentration_risk']}")
    print(f"Volatility Risk: {result['volatility_risk']}")
    print(f"\nModel Results:")
    print(f"  Autoencoder:")
    print(f"    Score: {result['model_results']['autoencoder']['score']:.6f}")
    print(f"    Threshold: {result['model_results']['autoencoder']['threshold']:.6f}")
    print(f"    Anomaly: {result['model_results']['autoencoder']['is_anomaly']}")
    print(f"  Isolation Forest:")
    print(f"    Anomaly: {result['model_results']['isolation_forest']['is_anomaly']}")
    print(f"  Consensus:")
    print(f"    Models Agree: {result['model_results']['consensus']['models_agree']}")
    print(f"    Confidence: {result['model_results']['consensus']['confidence']}")
    print(f"\nPortfolio Metrics:")
    print(f"  Total Value: ${result['portfolio_metrics']['total_value']:,.2f}")
    print(f"  Positions: {result['portfolio_metrics']['n_positions']}")
    print(f"  Max Position: {result['portfolio_metrics']['max_position_weight']*100:.1f}%")
    print(f"  Top 3 Concentration: {result['portfolio_metrics']['top_3_concentration']*100:.1f}%")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    print(f"\n{result['message']}")
    print()


def test_concentrated_portfolio():
    """Test with highly concentrated portfolio."""
    print("="*80)
    print("TEST 2: Concentrated Portfolio (High Risk)")
    print("="*80)
    
    portfolio = {
        "positions": {
            "NVDA": {"shares": 500, "cost_basis": 400},  # 50%+ of portfolio
            "AAPL": {"shares": 50, "cost_basis": 150},
            "MSFT": {"shares": 30, "cost_basis": 300}
        },
        "benchmark": "SPY"
    }
    
    response = requests.post(f"{API_URL}/portfolio/analyze", json=portfolio)
    result = response.json()

    print(f"Risk Level: {result['risk_level']}")
    print(f"Concentration Risk: {result['concentration_risk']}")
    print(f"\nPortfolio Metrics:")
    print(f"  Positions: {result['portfolio_metrics']['n_positions']}")
    print(f"  Max Position: {result['portfolio_metrics']['max_position_weight']*100:.1f}%")
    print(f"  HHI: {result['portfolio_metrics']['concentration_hhi']:.3f}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    print()


def test_diversified_portfolio():
    """Test with well-diversified portfolio."""
    print("="*80)
    print("TEST 3: Diversified Portfolio (Low Risk)")
    print("="*80)
    
    portfolio = {
        "positions": {
            # Tech
            "AAPL": {"shares": 50, "cost_basis": 150},
            "MSFT": {"shares": 40, "cost_basis": 300},
            # Finance
            "JPM": {"shares": 60, "cost_basis": 140},
            "BAC": {"shares": 100, "cost_basis": 30},
            # Healthcare
            "JNJ": {"shares": 50, "cost_basis": 160},
            "UNH": {"shares": 20, "cost_basis": 450},
            # Consumer
            "HD": {"shares": 30, "cost_basis": 300},
            "MCD": {"shares": 40, "cost_basis": 270},
            # Energy
            "XOM": {"shares": 80, "cost_basis": 100},
            "CVX": {"shares": 60, "cost_basis": 150},
            # ETF for market exposure
            "SPY": {"shares": 50, "cost_basis": 400}
        },
        "benchmark": "SPY"
    }
    
    response = requests.post(f"{API_URL}/portfolio/analyze", json=portfolio)
    result = response.json()
    
    print(f"Risk Level: {result['risk_level']}")
    print(f"Concentration Risk: {result['concentration_risk']}")
    print(f"\nPortfolio Metrics:")
    print(f"  Positions: {result['portfolio_metrics']['n_positions']}")
    print(f"  Max Position: {result['portfolio_metrics']['max_position_weight']*100:.1f}%")
    print(f"  Top 3 Concentration: {result['portfolio_metrics']['top_3_concentration']*100:.1f}%")
    print(f"  HHI: {result['portfolio_metrics']['concentration_hhi']:.3f}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    print()



if __name__ == "__main__":
    print("\n" + "="*80)
    print("PORTFOLIO ANOMALY DETECTION API - TEST CLIENT")
    print("="*80 + "\n")
    
    try:
        test_health()
        test_normal_portfolio()
        test_concentrated_portfolio()
        test_diversified_portfolio()
        
        print("="*80)
        print("✓ ALL TESTS COMPLETE")
        print("="*80)
        print("\nNote: LLM explanation tests require ANTHROPIC_API_KEY environment variable")
        print("      Set it with: export ANTHROPIC_API_KEY='your_key_here'")
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        print("  Make sure the API is running: python api/service.py")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
