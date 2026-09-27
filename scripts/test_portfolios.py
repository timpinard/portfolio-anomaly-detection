#!/usr/bin/env python3
"""
Test Portfolios for LLM Response Testing

This file contains a variety of portfolio configurations designed to test
different aspects of the anomaly detection and LLM explanation system.

Usage:
    # Run all tests
    python test_portfolios.py
    
    # Run specific test
    python test_portfolios.py --test concentrated_winner
    
    # Save results to file
    python test_portfolios.py --output results.json
"""

import requests
import json
import sys
import argparse
from typing import Dict, Any

API_URL = "http://localhost:8000"


# =============================================================================
# TEST PORTFOLIO DEFINITIONS
# =============================================================================

TEST_PORTFOLIOS = {
    
    # -------------------------------------------------------------------------
    # SCENARIO 1: Concentrated Winner (like your MU example)
    # One stock has run up significantly, creating dangerous concentration
    # -------------------------------------------------------------------------
    "concentrated_winner": {
        "description": "Big winner creating dangerous concentration - similar to MU scenario",
        "expected_risk": "critical",
        "expected_issues": ["severe concentration", "single stock >50%", "anomaly from momentum"],
        "portfolio": {
            "positions": {
                "NVDA": {"shares": 200, "cost_basis": 150},   # Bought at $150, now ~$500+ = huge gain
                "AAPL": {"shares": 50, "cost_basis": 170},
                "MSFT": {"shares": 30, "cost_basis": 280},
                "JNJ": {"shares": 20, "cost_basis": 160}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 2: Well-Diversified Portfolio
    # Healthy allocation across sectors with reasonable weights
    # -------------------------------------------------------------------------
    "well_diversified": {
        "description": "Well-balanced portfolio across multiple sectors",
        "expected_risk": "low",
        "expected_issues": [],
        "portfolio": {
            "positions": {
                "AAPL": {"shares": 50, "cost_basis": 150},    # Tech
                "MSFT": {"shares": 40, "cost_basis": 280},    # Tech
                "JNJ": {"shares": 60, "cost_basis": 155},     # Healthcare
                "UNH": {"shares": 15, "cost_basis": 450},     # Healthcare
                "JPM": {"shares": 45, "cost_basis": 140},     # Financials
                "PG": {"shares": 50, "cost_basis": 145},      # Consumer Staples
                "XOM": {"shares": 70, "cost_basis": 95},      # Energy
                "HD": {"shares": 25, "cost_basis": 300}       # Consumer Discretionary
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 3: Tech-Heavy Portfolio
    # Over-concentrated in technology sector
    # -------------------------------------------------------------------------
    "tech_heavy": {
        "description": "Heavily concentrated in technology sector",
        "expected_risk": "high",
        "expected_issues": ["sector concentration", "correlated holdings", "high volatility exposure"],
        "portfolio": {
            "positions": {
                "AAPL": {"shares": 100, "cost_basis": 150},
                "MSFT": {"shares": 80, "cost_basis": 280},
                "GOOGL": {"shares": 40, "cost_basis": 130},
                "META": {"shares": 50, "cost_basis": 300},
                "NVDA": {"shares": 30, "cost_basis": 400},
                "AMD": {"shares": 100, "cost_basis": 100}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 4: Upside Down Portfolio
    # Multiple significant losers that investor is holding
    # -------------------------------------------------------------------------
    "bag_holder": {
        "description": "Portfolio with multiple significant losing positions",
        "expected_risk": "medium",
        "expected_issues": ["unrealized losses", "potential value traps", "need to evaluate thesis"],
        "portfolio": {
            "positions": {
                "INTC": {"shares": 200, "cost_basis": 55},    # Intel - struggling
                "BA": {"shares": 50, "cost_basis": 250},      # Boeing - issues
                "PYPL": {"shares": 100, "cost_basis": 180},   # PayPal - fallen
                "DIS": {"shares": 75, "cost_basis": 140},     # Disney - challenged
                "AAPL": {"shares": 30, "cost_basis": 120}     # Apple - winner to offset
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 5: Dividend Income Portfolio
    # Conservative, income-focused holdings
    # -------------------------------------------------------------------------
    "dividend_income": {
        "description": "Conservative dividend-focused portfolio",
        "expected_risk": "low",
        "expected_issues": [],
        "portfolio": {
            "positions": {
                "JNJ": {"shares": 100, "cost_basis": 150},    # Healthcare dividend
                "PG": {"shares": 80, "cost_basis": 140},      # Consumer staples
                "KO": {"shares": 150, "cost_basis": 55},      # Coca-Cola
                "PEP": {"shares": 60, "cost_basis": 160},     # PepsiCo
                "VZ": {"shares": 120, "cost_basis": 45},      # Verizon
                "T": {"shares": 200, "cost_basis": 25}        # AT&T
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 6: Small Account Starter
    # New investor with limited positions
    # -------------------------------------------------------------------------
    "small_starter": {
        "description": "New investor with few positions",
        "expected_risk": "medium",
        "expected_issues": ["limited diversification", "few positions"],
        "portfolio": {
            "positions": {
                "AAPL": {"shares": 10, "cost_basis": 175},
                "VOO": {"shares": 5, "cost_basis": 400}       # S&P 500 ETF
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 7: Volatile Growth Portfolio
    # High-beta, momentum stocks
    # -------------------------------------------------------------------------
    "volatile_growth": {
        "description": "High-volatility growth stocks",
        "expected_risk": "high",
        "expected_issues": ["high volatility", "momentum exposure", "potential anomalies"],
        "portfolio": {
            "positions": {
                "TSLA": {"shares": 50, "cost_basis": 200},
                "NVDA": {"shares": 40, "cost_basis": 300},
                "AMD": {"shares": 100, "cost_basis": 80},
                "PLTR": {"shares": 200, "cost_basis": 15},
                "COIN": {"shares": 30, "cost_basis": 100}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 8: Single Stock (Extreme Concentration)
    # Investor with only one holding
    # -------------------------------------------------------------------------
    "single_stock": {
        "description": "Only one stock - maximum concentration risk",
        "expected_risk": "critical",
        "expected_issues": ["no diversification", "100% concentration", "maximum risk"],
        "portfolio": {
            "positions": {
                "AAPL": {"shares": 200, "cost_basis": 150}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 9: Healthcare Focus
    # Sector bet on healthcare
    # -------------------------------------------------------------------------
    "healthcare_focus": {
        "description": "Healthcare sector concentration",
        "expected_risk": "medium",
        "expected_issues": ["sector concentration", "regulatory risk exposure"],
        "portfolio": {
            "positions": {
                "JNJ": {"shares": 80, "cost_basis": 155},
                "UNH": {"shares": 25, "cost_basis": 400},
                "PFE": {"shares": 150, "cost_basis": 35},
                "ABBV": {"shares": 60, "cost_basis": 140},
                "MRK": {"shares": 70, "cost_basis": 95},
                "LLY": {"shares": 15, "cost_basis": 500}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 10: Mixed Winners and Losers
    # Portfolio with both big gains and losses
    # -------------------------------------------------------------------------
    "mixed_performance": {
        "description": "Mix of winners and losers",
        "expected_risk": "medium",
        "expected_issues": ["mixed signals", "tax loss harvesting opportunity"],
        "portfolio": {
            "positions": {
                "NVDA": {"shares": 30, "cost_basis": 150},    # Big winner
                "AAPL": {"shares": 50, "cost_basis": 120},    # Winner
                "INTC": {"shares": 100, "cost_basis": 50},    # Loser
                "PYPL": {"shares": 40, "cost_basis": 200},    # Big loser
                "MSFT": {"shares": 25, "cost_basis": 250},    # Modest winner
                "BA": {"shares": 20, "cost_basis": 220}       # Loser
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 11: Blue Chip Conservative
    # Large-cap, stable companies
    # -------------------------------------------------------------------------
    "blue_chip": {
        "description": "Conservative large-cap blue chips",
        "expected_risk": "low",
        "expected_issues": [],
        "portfolio": {
            "positions": {
                "AAPL": {"shares": 40, "cost_basis": 140},
                "MSFT": {"shares": 35, "cost_basis": 260},
                "JNJ": {"shares": 50, "cost_basis": 150},
                "PG": {"shares": 45, "cost_basis": 140},
                "WMT": {"shares": 40, "cost_basis": 140},
                "JPM": {"shares": 35, "cost_basis": 130}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 12: Tiny Anomaly (Immaterial Problem)
    # Anomalous stock but tiny position
    # -------------------------------------------------------------------------
    "tiny_anomaly": {
        "description": "Anomalous stock but position too small to matter",
        "expected_risk": "low",
        "expected_issues": ["immaterial anomaly"],
        "portfolio": {
            "positions": {
                "AAPL": {"shares": 100, "cost_basis": 150},
                "MSFT": {"shares": 80, "cost_basis": 280},
                "JNJ": {"shares": 60, "cost_basis": 155},
                "FMC": {"shares": 5, "cost_basis": 100}       # Tiny position, potentially anomalous
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 13: Recent IPO Heavy
    # Newer, potentially volatile stocks
    # -------------------------------------------------------------------------
    "recent_growth": {
        "description": "Newer growth companies",
        "expected_risk": "high",
        "expected_issues": ["volatility", "limited history", "growth premium"],
        "portfolio": {
            "positions": {
                "PLTR": {"shares": 150, "cost_basis": 12},
                "SNOW": {"shares": 25, "cost_basis": 180},
                "CRWD": {"shares": 30, "cost_basis": 150},
                "NET": {"shares": 50, "cost_basis": 60},
                "DDOG": {"shares": 40, "cost_basis": 90}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 14: Financial Sector Bet
    # Banking and financial services concentration
    # -------------------------------------------------------------------------
    "financials_focus": {
        "description": "Financial sector concentration",
        "expected_risk": "medium",
        "expected_issues": ["sector concentration", "interest rate sensitivity"],
        "portfolio": {
            "positions": {
                "JPM": {"shares": 80, "cost_basis": 130},
                "BAC": {"shares": 150, "cost_basis": 32},
                "WFC": {"shares": 100, "cost_basis": 45},
                "GS": {"shares": 20, "cost_basis": 350},
                "MS": {"shares": 50, "cost_basis": 85},
                "BLK": {"shares": 10, "cost_basis": 700}
            }
        }
    },
    
    # -------------------------------------------------------------------------
    # SCENARIO 15: Energy Sector
    # Oil and gas concentration
    # -------------------------------------------------------------------------
    "energy_focus": {
        "description": "Energy sector concentration",
        "expected_risk": "medium",
        "expected_issues": ["sector concentration", "commodity price exposure", "volatility"],
        "portfolio": {
            "positions": {
                "XOM": {"shares": 100, "cost_basis": 85},
                "CVX": {"shares": 60, "cost_basis": 140},
                "COP": {"shares": 50, "cost_basis": 100},
                "SLB": {"shares": 80, "cost_basis": 45},
                "OXY": {"shares": 100, "cost_basis": 55}
            }
        }
    }
}


# =============================================================================
# TEST RUNNER
# =============================================================================

def test_portfolio(name: str, config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Test a single portfolio against the API.
    
    Args:
        name: Test name
        config: Portfolio configuration
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"Description: {config['description']}")
    print(f"Expected Risk: {config['expected_risk']}")
    print(f"{'='*80}")
    
    request_body = {
        "portfolio": config["portfolio"],
        "include_explanation": True,
        "include_portfolio_context": True,
        "include_security_detail": True
    }
    
    try:
        response = requests.post(
            f"{API_URL}/portfolio/analyze-and-explain",
            json=request_body,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract key metrics
        analysis = result.get("analysis", {})
        risk_level = analysis.get("risk_level", "unknown")
        metrics = analysis.get("portfolio_metrics", {})
        
        print(f"\nRESULTS:")
        print(f"  Risk Level: {risk_level.upper()}")
        print(f"  Total Value: ${metrics.get('total_value', 0):,.2f}")
        print(f"  Positions: {metrics.get('n_positions', 0)}")
        print(f"  Max Position: {metrics.get('max_position_weight', 0)*100:.1f}%")
        print(f"  Top 3 Concentration: {metrics.get('top_3_concentration', 0)*100:.1f}%")
        
        # Anomaly info
        anomaly_rate = metrics.get('anomaly_rate_ae', 0) * 100
        anomaly_count = metrics.get('anomaly_count_ae', 0)
        print(f"  Anomaly Rate: {anomaly_rate:.0f}% ({anomaly_count} holdings)")
        
        # Z-score
        ae_z = metrics.get('ae_weighted_z_score')
        if ae_z is not None:
            print(f"  Portfolio Z-Score: {ae_z:+.2f}")
        
        # Security breakdown
        security_scores = metrics.get('security_scores', [])
        if security_scores and verbose:
            print(f"\n  HOLDINGS:")
            for s in security_scores[:5]:  # Top 5
                status_icon = "🚨" if s.get('consensus_anomaly') else "⚡" if s.get('status') == 'warning' else "✓"
                gain_loss = s.get('gain_loss_percent')
                gain_str = f" ({gain_loss:+.0f}%)" if gain_loss is not None else ""
                print(f"    {status_icon} {s['symbol']}: {s['weight']*100:.1f}%{gain_str} - {s.get('status', 'unknown')}")
        
        # Explanation preview
        explanation = result.get("explanation", {})
        if explanation.get("success") and verbose:
            exp_text = explanation.get("explanation", "")
            preview = exp_text[:300] + "..." if len(exp_text) > 300 else exp_text
            print(f"\n  LLM EXPLANATION PREVIEW:")
            print(f"  {preview}")
        
        # Compare to expected
        expected_risk = config['expected_risk']
        risk_match = risk_level.lower() == expected_risk.lower()
        
        print(f"\n  VALIDATION:")
        print(f"    Expected Risk: {expected_risk} | Actual: {risk_level} | {'✓ MATCH' if risk_match else '✗ MISMATCH'}")
        
        return {
            "name": name,
            "success": True,
            "risk_level": risk_level,
            "expected_risk": expected_risk,
            "risk_match": risk_match,
            "metrics": metrics,
            "full_response": result
        }
        
    except requests.exceptions.RequestException as e:
        print(f"\n  ERROR: {e}")
        return {
            "name": name,
            "success": False,
            "error": str(e)
        }


def run_all_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run all test portfolios."""
    results = {}
    
    for name, config in TEST_PORTFOLIOS.items():
        results[name] = test_portfolio(name, config, verbose)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    successful = sum(1 for r in results.values() if r.get('success'))
    risk_matches = sum(1 for r in results.values() if r.get('risk_match'))
    
    print(f"Total Tests: {total}")
    print(f"Successful API Calls: {successful}/{total}")
    print(f"Risk Level Matches: {risk_matches}/{total}")
    
    if successful < total:
        print("\nFailed Tests:")
        for name, result in results.items():
            if not result.get('success'):
                print(f"  - {name}: {result.get('error', 'Unknown error')}")
    
    if risk_matches < successful:
        print("\nRisk Mismatches:")
        for name, result in results.items():
            if result.get('success') and not result.get('risk_match'):
                print(f"  - {name}: Expected {result.get('expected_risk')}, Got {result.get('risk_level')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test portfolio analysis API")
    parser.add_argument("--test", type=str, help="Run specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available Test Portfolios:")
        print("-" * 60)
        for name, config in TEST_PORTFOLIOS.items():
            print(f"  {name:25s} - {config['description']}")
        return
    
    verbose = not args.quiet
    
    if args.test:
        if args.test not in TEST_PORTFOLIOS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(TEST_PORTFOLIOS.keys())}")
            sys.exit(1)
        results = {args.test: test_portfolio(args.test, TEST_PORTFOLIOS[args.test], verbose)}
    else:
        results = run_all_tests(verbose)
    
    if args.output:
        # Remove full_response for cleaner output
        clean_results = {}
        for name, result in results.items():
            clean_results[name] = {k: v for k, v in result.items() if k != 'full_response'}
        
        with open(args.output, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()