"""
Universe definitions for market data and model training.

This module provides reusable universe definitions that can be used
across different scripts for consistency.
"""


def get_custom_universe() -> list[str]:
    """
    Get predefined custom universe symbols.
    
    Returns a curated list of ~60 symbols across multiple sectors:
    - Market indices (SPY, QQQ, IWM, DIA, VTI)
    - Technology (AAPL, MSFT, GOOGL, NVDA, META, etc.)
    - Finance (JPM, BAC, WFC, GS, etc.)
    - Healthcare (JNJ, UNH, PFE, ABBV, etc.)
    - Consumer (AMZN, TSLA, HD, MCD, etc.)
    - Energy (XOM, CVX, COP, SLB, etc.)
    - Industrials (BA, CAT, HON, UPS, etc.)
    
    Returns:
        List of ticker symbols (strings)
    
    Example:
        >>> symbols = get_custom_universe()
        >>> len(symbols)
        60
        >>> 'AAPL' in symbols
        True
    """
    universe = {
        # Market indices
        'indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
        
        # Tech
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ORCL', 'CSCO', 'INTC'],
        
        # Finance  
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'C', 'MS', 'SCHW', 'BLK'],
        
        # Healthcare
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'ABT'],
        
        # Consumer
        'consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT'],
        
        # Energy
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO'],
        
        # Industrials
        'industrials': ['BA', 'CAT', 'HON', 'UPS', 'GE', 'MMM', 'LMT', 'RTX']
    }
    
    all_symbols = []
    for sector, symbols in universe.items():
        all_symbols.extend(symbols)
    
    return all_symbols
