#!/usr/bin/env python3
"""
Analyze a portfolio using both individual and cross-sectional model approaches.

This script provides a unified interface for comprehensive portfolio analysis:
1. Risk Assessment (individual model): Detects unusual portfolio characteristics
2. Attribution (cross-sectional model): Identifies which stocks drive market divergence

This script provides comprehensive portfolio analysis using individual and cross-sectional models.

Usage:
    # Analyze with equal weights
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL
    
    # Analyze with custom weights
    python scripts/analyze_portfolio.py AAPL:0.4 MSFT:0.35 GOOGL:0.25
    
    # Analyze with share counts
    python scripts/analyze_portfolio.py --shares AAPL:100 MSFT:50 GOOGL:30
    
    # Risk assessment only (faster)
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --risk-only
    
    # Attribution only
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --attribution-only
    
    # Specify analysis date for attribution
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --date 2024-01-15
    
    # JSON output
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --json
    
    # Read from file
    python scripts/analyze_portfolio.py --file portfolio.json
"""

import sys
import argparse
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.portfolio_analyzer import PortfolioAnalyzer, format_analysis_report


def analyze_portfolio_for_api(
    holdings: List[Dict],
    individual_model_type: str = 'individual',
    cross_sectional_model_type: str = 'cross_sectional',
    include_risk: bool = True,
    include_attribution: bool = True,
    analysis_date: Optional[str] = None
) -> Dict:
    """
    API-compatible function to analyze a portfolio.
    
    This function can be imported and used by the API or other Python code.
    
    Args:
        holdings: List of dicts with 'symbol' and either 'shares' or 'weight'
        individual_model_type: Name of individual model type
        cross_sectional_model_type: Name of cross-sectional model type
        include_risk: Include risk assessment
        include_attribution: Include attribution analysis
        analysis_date: Date for attribution analysis (defaults to latest)
    
    Returns:
        Dict with analysis results (same format as PortfolioAnalyzer.analyze())
    """
    analyzer = PortfolioAnalyzer(
        individual_model_type=individual_model_type,
        cross_sectional_model_type=cross_sectional_model_type
    )
    
    return analyzer.analyze(
        holdings=holdings,
        analysis_date=analysis_date,
        include_risk=include_risk,
        include_attribution=include_attribution
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_holdings(holdings_args: list, use_shares: bool = False) -> list:
    """
    Parse holdings from command line arguments.
    
    Args:
        holdings_args: List of strings like "AAPL" or "AAPL:0.4" or "AAPL:100"
        use_shares: If True, parse as shares; if False, parse as weights
    
    Returns:
        List of dicts with 'symbol' and either 'weight' or 'shares'
    """
    holdings = []
    
    for arg in holdings_args:
        if ':' in arg:
            symbol, value = arg.split(':', 1)
            value = float(value)
        else:
            symbol = arg
            value = 1.0  # Default equal weight or 1 share
        
        holding = {'symbol': symbol.upper()}
        
        if use_shares:
            holding['shares'] = value
        else:
            holding['weight'] = value
        
        holdings.append(holding)
    
    # Normalize weights if needed
    if not use_shares:
        total_weight = sum(h['weight'] for h in holdings)
        for h in holdings:
            h['weight'] = h['weight'] / total_weight
    
    return holdings


def load_holdings_from_file(filepath: Path) -> list:
    """
    Load holdings from a JSON file.
    
    Expected format:
    {
        "holdings": [
            {"symbol": "AAPL", "weight": 0.4},
            {"symbol": "MSFT", "weight": 0.35},
            {"symbol": "GOOGL", "weight": 0.25}
        ]
    }
    
    Or with shares:
    {
        "holdings": [
            {"symbol": "AAPL", "shares": 100},
            {"symbol": "MSFT", "shares": 50}
        ]
    }
    """
    with open(filepath) as f:
        data = json.load(f)
    
    if 'holdings' not in data:
        raise ValueError("JSON file must contain 'holdings' key")
    
    return data['holdings']


def main():
    parser = argparse.ArgumentParser(
        description='Analyze portfolio using individual and cross-sectional model approaches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Equal weights
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL
    
    # Custom weights
    python scripts/analyze_portfolio.py AAPL:0.4 MSFT:0.35 GOOGL:0.25
    
    # Share counts
    python scripts/analyze_portfolio.py --shares AAPL:100 MSFT:50 GOOGL:30
    
    # Risk assessment only
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --risk-only
    
    # Attribution only
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --attribution-only
    
    # Specific date
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --date 2024-01-15
    
    # JSON output
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --json
    
    # From file
    python scripts/analyze_portfolio.py --file my_portfolio.json
    
    # Custom model types
    python scripts/analyze_portfolio.py AAPL MSFT GOOGL --individual individual --cross-sectional cross_sectional
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'holdings',
        nargs='*',
        help='Holdings as SYMBOL or SYMBOL:WEIGHT (e.g., AAPL:0.4)'
    )
    input_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Load holdings from JSON file'
    )
    
    # Analysis options
    parser.add_argument(
        '--shares',
        action='store_true',
        help='Interpret values as share counts instead of weights'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Analysis date for attribution (YYYY-MM-DD), defaults to latest'
    )
    parser.add_argument(
        '--risk-only',
        action='store_true',
        help='Only run risk assessment (individual model approach)'
    )
    parser.add_argument(
        '--attribution-only',
        action='store_true',
        help='Only run attribution analysis (cross-sectional model approach)'
    )
    
    # Model options
    parser.add_argument(
        '--individual',
        '--individual-model-type',
        type=str,
        default='individual',
        dest='individual_model_type',
        help='Name of individual model type (default: individual)'
    )
    parser.add_argument(
        '--cross-sectional',
        '--cross-sectional-model-type',
        type=str,
        default='cross_sectional',
        dest='cross_sectional_model_type',
        help='Name of cross-sectional model type (default: cross_sectional)'
    )
    
    # Output options
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output result as JSON'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Save output to file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.risk_only and args.attribution_only:
        parser.error("Cannot specify both --risk-only and --attribution-only")
    
    # Load holdings
    if args.file:
        logger.info(f"Loading holdings from {args.file}")
        holdings = load_holdings_from_file(args.file)
    else:
        if not args.holdings:
            parser.error("Must provide holdings or --file")
        holdings = parse_holdings(args.holdings, args.shares)
    
    logger.info(f"Analyzing portfolio with {len(holdings)} holdings:")
    for h in holdings:
        if 'weight' in h:
            logger.info(f"  {h['symbol']}: {h['weight']*100:.2f}%")
        else:
            logger.info(f"  {h['symbol']}: {h['shares']} shares")
    
    # Initialize analyzer
    try:
        analyzer = PortfolioAnalyzer(
            individual_model_type=args.individual_model_type,
            cross_sectional_model_type=args.cross_sectional_model_type
        )
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Determine what to run
    include_risk = not args.attribution_only
    include_attribution = not args.risk_only
    
    # Run analysis
    try:
        logger.info("\nRunning analysis...")
        result = analyzer.analyze(
            holdings=holdings,
            analysis_date=args.date,
            include_risk=include_risk,
            include_attribution=include_attribution
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Output results
    if args.json:
        output = json.dumps(result, indent=2, default=str)
    else:
        output = format_analysis_report(result)
    
    if args.output:
        logger.info(f"\nSaving results to {args.output}")
        with open(args.output, 'w') as f:
            f.write(output)
        logger.info("✓ Results saved")
    else:
        print("\n" + output)
    
    # Exit code based on risk level
    if include_risk and 'risk_assessment' in result:
        risk = result['risk_assessment']
        if 'error' not in risk:
            if risk['risk_level'] == 'high':
                sys.exit(2)
            elif risk['risk_level'] == 'medium':
                sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
