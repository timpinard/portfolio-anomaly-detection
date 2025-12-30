#!/usr/bin/env python3
"""FastAPI service for portfolio anomaly detection - MVP version."""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import sqlite3
import json
import yfinance as yf

# Add paths
api_dir = Path(__file__).parent
project_root = api_dir.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(api_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    Portfolio, 
    PortfolioAnalysisResponse, 
    HealthResponse,
    LLMExplanationRequest,
    LLMExplanationResponse,
    OrchestratedAnalysisRequest,
    OrchestratedAnalysisResponse,
    SecurityScore,
    BasketStatistics,
    ExtendedPortfolioMetrics
)
from business_logic import (
    calculate_portfolio_weights,
    calculate_concentration_metrics,
    assess_concentration_risk,
    get_risk_level,
    generate_recommendations,
    generate_message,
    interpret_portfolio_features
)
from experimental.llm_service import is_llm_available
from experimental.llm_agents import AgenticFlow, RecommendationInterpreterAgent, PortfolioContextAgent

# Import models
from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector
from data.feature_extractor import MarketFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Portfolio Anomaly Detection API",
    description="AI-powered portfolio risk analysis using dual-model anomaly detection",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODELS = {}
BASKET_STATS = None
MODEL_DIR = project_root / 'models' / 'market_universe'
BASKET_STATS_FILE = MODEL_DIR / 'basket_stats.json'


def load_models():
    """Load trained models."""
    if 'autoencoder' not in MODELS:
        logger.info("Loading autoencoder model...")
        ae_model = AutoencoderAnomalyDetector(
            sector='autoencoder',
            model_dir=MODEL_DIR
        )
        ae_model.load()
        MODELS['autoencoder'] = ae_model
    
    if 'isolation_forest' not in MODELS:
        logger.info("Loading isolation forest model...")
        if_model = IsolationForestAnomalyDetector(
            sector='isolation_forest',
            model_dir=MODEL_DIR
        )
        if_model.load()
        MODELS['isolation_forest'] = if_model
    
    return MODELS['autoencoder'], MODELS['isolation_forest']


def load_basket_stats() -> Optional[BasketStatistics]:
    """Load pre-computed basket statistics for z-score calculation."""
    global BASKET_STATS
    
    if BASKET_STATS is not None:
        return BASKET_STATS
    
    if BASKET_STATS_FILE.exists():
        try:
            with open(BASKET_STATS_FILE, 'r') as f:
                data = json.load(f)
            BASKET_STATS = BasketStatistics(**data)
            logger.info(f"Loaded basket statistics: {BASKET_STATS.n_securities} securities")
            return BASKET_STATS
        except Exception as e:
            logger.warning(f"Could not load basket stats: {e}")
    
    # If no pre-computed stats, try to compute them on demand
    logger.info("No pre-computed basket stats found, will compute on demand if needed")
    return None


def compute_basket_stats_on_demand(
    ae_model: AutoencoderAnomalyDetector,
    if_model: IsolationForestAnomalyDetector
) -> Optional[BasketStatistics]:
    """Compute basket statistics from training data on demand."""
    global BASKET_STATS
    
    if BASKET_STATS is not None:
        return BASKET_STATS
    
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    if not db_path.exists():
        logger.warning("Database not found, cannot compute basket stats")
        return None
    
    try:
        extractor = MarketFeatureExtractor(str(db_path))
        
        # Get training set symbols
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_features ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if len(symbols) < 5:
            logger.warning(f"Insufficient symbols for basket stats: {len(symbols)}")
            return None
        
        logger.info(f"Computing basket statistics from {len(symbols)} symbols...")
        
        ae_scores = []
        if_scores = []
        
        for symbol in symbols:
            try:
                df = extractor.load_price_data(symbol)
                if df.empty or len(df) < 60:
                    continue
                
                features = extractor.calculate_all_features(df)
                if features.empty:
                    continue
                
                latest = features.iloc[-1]
                feature_cols = extractor.get_feature_columns(features)
                X = np.array([[latest[col] for col in feature_cols]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
                ae_score = float(ae_model.score(X)[0])
                if_score = float(if_model.score(X)[0])
                
                ae_scores.append(ae_score)
                if_scores.append(if_score)
                
            except Exception as e:
                logger.debug(f"Error scoring {symbol}: {e}")
                continue
        
        if len(ae_scores) < 5:
            logger.warning(f"Insufficient valid scores: {len(ae_scores)}")
            return None
        
        BASKET_STATS = BasketStatistics(
            ae_mean=float(np.mean(ae_scores)),
            ae_std=float(np.std(ae_scores)),
            if_mean=float(np.mean(if_scores)),
            if_std=float(np.std(if_scores)),
            n_securities=len(ae_scores),
            computed_at=datetime.utcnow().isoformat()
        )
        
        # Cache to file for future use
        try:
            BASKET_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(BASKET_STATS_FILE, 'w') as f:
                json.dump(BASKET_STATS.model_dump(), f, indent=2)
            logger.info(f"Saved basket statistics to {BASKET_STATS_FILE}")
        except Exception as e:
            logger.warning(f"Could not save basket stats: {e}")
        
        return BASKET_STATS
        
    except Exception as e:
        logger.error(f"Error computing basket stats: {e}")
        return None


def fetch_from_yahoo_and_save(symbol: str, db_path: Path) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance and save to database.
    
    Args:
        symbol: Ticker symbol
        db_path: Path to SQLite database
    
    Returns:
        DataFrame with price data
    """
    logger.info(f"Fetching {symbol} from Yahoo Finance...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3y")
        
        if df.empty:
            logger.warning(f"No data returned from Yahoo Finance for {symbol}")
            return pd.DataFrame()
        
        # Reset index and format
        df = df.reset_index()
        df['symbol'] = symbol
        df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Ensure date is string format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Save to database
        conn = sqlite3.connect(db_path)
        
        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        """)
        
        # Delete existing data for this symbol (replace)
        conn.execute("DELETE FROM market_prices WHERE symbol = ?", (symbol,))
        
        # Insert new data
        df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].to_sql(
            'market_prices', conn, if_exists='append', index=False
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Fetched and saved {len(df)} records for {symbol}")
        
        # Format for return (set date as index)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Yahoo Finance: {e}")
        return pd.DataFrame()


def ensure_symbol_data(symbol: str, db_path: Path, extractor: MarketFeatureExtractor) -> pd.DataFrame:
    """
    Ensure symbol data exists in database, fetching from Yahoo Finance if needed.
    
    Args:
        symbol: Ticker symbol
        db_path: Path to SQLite database
        extractor: MarketFeatureExtractor instance
    
    Returns:
        DataFrame with price data
    """
    # Try to load from database
    df = extractor.load_price_data(symbol)
    
    # If empty or insufficient data, fetch from Yahoo Finance
    if df.empty or len(df) < 10:  # Require at least 10 days of data
        logger.info(f"Symbol {symbol} not found in database or insufficient data, fetching from Yahoo Finance...")
        df = fetch_from_yahoo_and_save(symbol, db_path)
        
        # Reload from database after saving
        if not df.empty:
            df = extractor.load_price_data(symbol)
    
    return df


def get_current_prices(symbols: list) -> Dict[str, float]:
    """Get current prices for symbols, fetching from Yahoo Finance if not in database."""
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create database if it doesn't exist
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        """)
        conn.commit()
        conn.close()
    
    extractor = MarketFeatureExtractor(str(db_path))
    
    prices = {}
    for symbol in symbols:
        try:
            df = ensure_symbol_data(symbol, db_path, extractor)
            if not df.empty:
                prices[symbol] = float(df['close'].iloc[-1])
            else:
                logger.warning(f"No price data available for {symbol}")
                prices[symbol] = 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            prices[symbol] = 0.0
    
    return prices


def format_z_score_display(z_score: Optional[float]) -> Optional[str]:
    """Format z-score for human-readable display."""
    if z_score is None:
        return None
    
    abs_z = abs(z_score)
    sign = "+" if z_score > 0 else ""
    
    if abs_z > 3:
        return f"{sign}{z_score:.1f}σ (extreme)"
    elif abs_z > 2:
        return f"{sign}{z_score:.1f}σ (very unusual)"
    elif abs_z > 1:
        return f"{sign}{z_score:.1f}σ (somewhat unusual)"
    else:
        return f"{sign}{z_score:.1f}σ (normal range)"


def get_sector_for_symbol(symbol: str) -> Optional[str]:
    """
    Get sector information for a symbol using yfinance.
    Returns None if sector cannot be determined.
    """
    # Sector mapping cache to avoid repeated API calls
    SECTOR_CACHE = getattr(get_sector_for_symbol, '_cache', {})
    
    if symbol in SECTOR_CACHE:
        return SECTOR_CACHE[symbol]
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', None)
        SECTOR_CACHE[symbol] = sector
        get_sector_for_symbol._cache = SECTOR_CACHE
        return sector
    except Exception as e:
        logger.debug(f"Could not get sector for {symbol}: {e}")
        return None


def score_individual_securities(
    portfolio: Portfolio,
    ae_model: AutoencoderAnomalyDetector,
    if_model: IsolationForestAnomalyDetector,
    current_prices: Dict[str, float],
    weights: Dict[str, float],
    basket_stats: Optional[BasketStatistics] = None
) -> List[SecurityScore]:
    """
    Score each security individually for detailed analysis.
    
    Args:
        portfolio: Portfolio with positions
        ae_model: Autoencoder model
        if_model: Isolation Forest model
        current_prices: Current prices by symbol
        weights: Portfolio weights by symbol
        basket_stats: Optional basket statistics for z-score calculation
    
    Returns:
        List of SecurityScore objects with per-security detail
    """
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    extractor = MarketFeatureExtractor(str(db_path))
    
    # Materiality threshold - positions below this weight are considered immaterial
    MATERIALITY_THRESHOLD = 0.01  # 1%
    
    security_scores = []
    
    for symbol, position in portfolio.positions.items():
        try:
            # Load and calculate features for this security
            df = ensure_symbol_data(symbol, db_path, extractor)
            if df.empty or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            features = extractor.calculate_all_features(df)
            if features.empty:
                logger.warning(f"Could not calculate features for {symbol}")
                continue
            
            # Get most recent features
            latest = features.iloc[-1]
            feature_cols = extractor.get_feature_columns(features)
            X = np.array([[latest[col] for col in feature_cols]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Score with both models
            ae_score = float(ae_model.score(X)[0])
            ae_prediction = int(ae_model.predict(X)[0])
            ae_is_anomaly = (ae_prediction == -1)
            
            if_score = float(if_model.score(X)[0])
            if_prediction = int(if_model.predict(X)[0])
            if_is_anomaly = (if_prediction == -1)
            
            consensus_anomaly = ae_is_anomaly and if_is_anomaly
            
            # Get price and calculate position value
            price = current_prices.get(symbol, 0.0)
            position_value = position.shares * price
            weight = weights.get(symbol, 0.0)
            
            # Calculate cost basis and gain/loss if available
            cost_basis = position.cost_basis
            total_cost = None
            gain_loss_dollars = None
            gain_loss_percent = None
            
            if cost_basis is not None and cost_basis > 0:
                total_cost = position.shares * cost_basis
                gain_loss_dollars = position_value - total_cost
                gain_loss_percent = ((price / cost_basis) - 1) * 100
            
            # Get sector information
            sector = get_sector_for_symbol(symbol)
            
            # Calculate z-scores if basket stats available
            ae_z_score = None
            if_z_score = None
            if basket_stats:
                if basket_stats.ae_std > 0:
                    ae_z_score = (ae_score - basket_stats.ae_mean) / basket_stats.ae_std
                if basket_stats.if_std > 0:
                    if_z_score = (if_score - basket_stats.if_mean) / basket_stats.if_std
            
            # Format z-scores for display
            ae_z_display = format_z_score_display(ae_z_score)
            if_z_display = format_z_score_display(if_z_score)
            
            # Determine materiality
            is_material = weight >= MATERIALITY_THRESHOLD
            materiality_note = None
            if not is_material:
                materiality_note = f"Position is only {weight*100:.2f}% of portfolio - minimal impact"
            elif weight > 0.25:
                materiality_note = f"Large position ({weight*100:.1f}%) - significant portfolio impact"
            
            # Determine status and reason
            if consensus_anomaly:
                status = "anomaly"
                if is_material:
                    status_reason = "Both models flag this security as anomalous - HIGH PRIORITY due to position size"
                else:
                    status_reason = "Both models flag as anomalous, but position is too small to significantly impact portfolio"
            elif ae_is_anomaly or if_is_anomaly:
                status = "warning"
                if ae_is_anomaly:
                    status_reason = "Autoencoder detects unusual patterns"
                else:
                    status_reason = "Isolation Forest detects unusual patterns"
            else:
                status = "normal"
                status_reason = None
            
            # Add z-score context to status reason for extreme cases
            if ae_z_score is not None and abs(ae_z_score) > 3:
                z_context = f" - EXTREME deviation ({ae_z_display})"
                if status_reason:
                    status_reason += z_context
                else:
                    status_reason = f"Extreme z-score: {ae_z_display}"
                    if status == "normal":
                        status = "warning"
            elif ae_z_score is not None and abs(ae_z_score) > 2:
                z_context = f" - significant deviation ({ae_z_display})"
                if status_reason:
                    status_reason += z_context
            
            # Add gain/loss context to status reason for big winners/losers
            if gain_loss_percent is not None:
                if gain_loss_percent > 100:
                    if status_reason:
                        status_reason += f". Note: Position up {gain_loss_percent:.0f}% - consider taking some profits"
                elif gain_loss_percent < -50:
                    if status_reason:
                        status_reason += f". Note: Position down {abs(gain_loss_percent):.0f}% - evaluate if thesis still holds"
            
            security_scores.append(SecurityScore(
                symbol=symbol,
                shares=position.shares,
                current_price=price,
                position_value=position_value,
                weight=weight,
                cost_basis=cost_basis,
                total_cost=total_cost,
                gain_loss_dollars=gain_loss_dollars,
                gain_loss_percent=gain_loss_percent,
                sector=sector,
                ae_score=ae_score,
                ae_is_anomaly=ae_is_anomaly,
                ae_z_score=ae_z_score,
                ae_z_score_display=ae_z_display,
                if_score=if_score,
                if_is_anomaly=if_is_anomaly,
                if_z_score=if_z_score,
                if_z_score_display=if_z_display,
                consensus_anomaly=consensus_anomaly,
                status=status,
                status_reason=status_reason,
                is_material=is_material,
                materiality_note=materiality_note
            ))
            
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")
            continue
    
    # Sort by position value (largest first)
    security_scores.sort(key=lambda x: x.position_value, reverse=True)
    
    return security_scores


def compute_extended_metrics(
    security_scores: List[SecurityScore],
    weights: Dict[str, float],
    concentration_metrics: Dict,
    basket_stats: Optional[BasketStatistics] = None
) -> ExtendedPortfolioMetrics:
    """
    Compute extended portfolio metrics from per-security scores.
    
    Args:
        security_scores: List of individual security scores
        weights: Portfolio weights by symbol
        concentration_metrics: Concentration metrics dict
        basket_stats: Optional basket statistics
    
    Returns:
        ExtendedPortfolioMetrics with full detail
    """
    n_securities = len(security_scores)
    if n_securities == 0:
        raise ValueError("No security scores to compute metrics from")
    
    # Compute weighted aggregate scores
    ae_weighted_score = sum(s.ae_score * s.weight for s in security_scores)
    if_weighted_score = sum(s.if_score * s.weight for s in security_scores)
    
    # Compute weighted z-scores if available
    ae_weighted_z = None
    if_weighted_z = None
    if all(s.ae_z_score is not None for s in security_scores):
        ae_weighted_z = sum(s.ae_z_score * s.weight for s in security_scores)
    if all(s.if_z_score is not None for s in security_scores):
        if_weighted_z = sum(s.if_z_score * s.weight for s in security_scores)
    
    # Count anomalies
    ae_anomaly_count = sum(1 for s in security_scores if s.ae_is_anomaly)
    if_anomaly_count = sum(1 for s in security_scores if s.if_is_anomaly)
    consensus_anomaly_count = sum(1 for s in security_scores if s.consensus_anomaly)
    
    # Total value
    total_value = sum(s.position_value for s in security_scores)
    
    # Interpret portfolio vs market
    portfolio_vs_market = None
    if ae_weighted_z is not None:
        if ae_weighted_z > 2:
            portfolio_vs_market = "significantly_more_anomalous"
        elif ae_weighted_z > 1:
            portfolio_vs_market = "more_anomalous_than_typical"
        elif ae_weighted_z < -1:
            portfolio_vs_market = "calmer_than_typical"
        else:
            portfolio_vs_market = "similar_to_market"
    
    return ExtendedPortfolioMetrics(
        total_value=total_value,
        n_positions=n_securities,
        weights=weights,
        concentration_hhi=concentration_metrics['herfindahl_index'],
        max_position_weight=concentration_metrics['max_position'],
        top_3_concentration=concentration_metrics['top_3_concentration'],
        ae_weighted_score=ae_weighted_score,
        if_weighted_score=if_weighted_score,
        ae_weighted_z_score=ae_weighted_z,
        if_weighted_z_score=if_weighted_z,
        anomaly_rate_ae=ae_anomaly_count / n_securities,
        anomaly_rate_if=if_anomaly_count / n_securities,
        anomaly_rate_consensus=consensus_anomaly_count / n_securities,
        anomaly_count_ae=ae_anomaly_count,
        anomaly_count_if=if_anomaly_count,
        anomaly_count_consensus=consensus_anomaly_count,
        security_scores=security_scores,
        basket_stats=basket_stats,
        portfolio_vs_market=portfolio_vs_market
    )


def calculate_portfolio_features(portfolio: Portfolio) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Calculate features for portfolio, fetching from Yahoo Finance if needed."""
    symbols = list(portfolio.positions.keys())
    
    # Get current prices (this will also fetch missing symbols from Yahoo)
    current_prices = get_current_prices(symbols)
    
    # Calculate weights
    weights = calculate_portfolio_weights(portfolio.positions, current_prices)
    
    # Get feature extractor
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create database if it doesn't exist
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        """)
        conn.commit()
        conn.close()
    
    extractor = MarketFeatureExtractor(str(db_path))
    
    # Get latest features for each position
    position_features = []
    for symbol in symbols:
        try:
            # Ensure data exists (fetch from Yahoo if needed)
            df = ensure_symbol_data(symbol, db_path, extractor)
            if df.empty:
                logger.warning(f"No data available for {symbol} after fetch attempt")
                continue
            
            # Calculate features
            features = extractor.calculate_all_features(df)
            if not features.empty:
                # Use most recent row
                position_features.append(features.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            continue
    
    if not position_features:
        raise HTTPException(status_code=400, detail="Could not calculate features for portfolio")
    
    # Combine features (weighted average based on portfolio weights)
    feature_df = pd.DataFrame(position_features)
    feature_cols = [col for col in feature_df.columns if col != 'symbol']
    
    # Weight features by position size
    weighted_features = {}
    for col in feature_cols:
        weighted_features[col] = sum(
            feature_df[col].iloc[i] * weights.get(symbols[i], 0)
            for i in range(len(feature_df))
        )
    
    # Convert to array
    feature_values = [weighted_features[col] for col in feature_cols]
    X = np.array([feature_values])
    
    # Replace inf and NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Return both array and dictionary for interpretation
    return X, weights, current_prices, weighted_features


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Portfolio Anomaly Detection API",
        "version": "0.2.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/portfolio/analyze",
            "explain": "/portfolio/explain",
            "analyze_and_explain": "/portfolio/analyze-and-explain",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    # Check if models exist
    ae_path = MODEL_DIR / 'autoencoder' / 'model.pth'
    if_path = MODEL_DIR / 'isolation_forest' / 'model.joblib'
    
    # Check basket stats
    basket_stats = load_basket_stats()
    
    return {
        "status": "healthy",
        "models_loaded": {
            "autoencoder": "available" if ae_path.exists() else "missing",
            "isolation_forest": "available" if if_path.exists() else "missing"
        },
        "llm_available": is_llm_available(),
        "basket_stats_available": basket_stats is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/portfolio/analyze", response_model=PortfolioAnalysisResponse)
def analyze_portfolio(portfolio: Portfolio):
    """
    Analyze portfolio for anomalies and risk.
    
    This is the main endpoint - takes a portfolio and returns:
    - Anomaly detection results from both models
    - Risk assessment
    - Concentration metrics
    - Actionable recommendations
    """
    try:
        # Load models
        ae_model, if_model = load_models()
        
        # Calculate portfolio features
        X, weights, current_prices, weighted_features = calculate_portfolio_features(portfolio)
        
        # Run models
        ae_score = float(ae_model.score(X)[0])
        ae_prediction = int(ae_model.predict(X)[0])
        ae_is_anomaly = (ae_prediction == -1)
        ae_threshold = float(ae_model.threshold)
        
        if_score = float(if_model.score(X)[0])
        if_prediction = int(if_model.predict(X)[0])
        if_is_anomaly = (if_prediction == -1)
        
        # Consensus
        consensus = {
            "both_flagged": ae_is_anomaly and if_is_anomaly,
            "either_flagged": ae_is_anomaly or if_is_anomaly,
            "models_agree": ae_is_anomaly == if_is_anomaly,
            "confidence": "high" if ae_is_anomaly == if_is_anomaly else "medium"
        }
        
        # Calculate concentration metrics
        concentration_metrics = calculate_concentration_metrics(weights)
        concentration_risk = assess_concentration_risk(concentration_metrics)
        
        # Overall risk level
        risk_level = get_risk_level(ae_score, ae_threshold, concentration_risk)
        
        # Interpret features
        feature_conclusions = interpret_portfolio_features(weighted_features, ae_score, ae_threshold)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            risk_level,
            concentration_metrics,
            ae_is_anomaly
        )
        
        # Generate message
        message = generate_message(ae_is_anomaly, risk_level, ae_score, ae_threshold)
        
        # Portfolio metrics
        total_value = sum(
            pos.shares * current_prices.get(symbol, 0) 
            for symbol, pos in portfolio.positions.items()
        )
        
        portfolio_metrics = {
            "total_value": total_value,
            "n_positions": len(portfolio.positions),
            "weights": weights,
            "concentration_hhi": concentration_metrics['herfindahl_index'],
            "max_position_weight": concentration_metrics['max_position'],
            "top_3_concentration": concentration_metrics['top_3_concentration']
        }
        
        return PortfolioAnalysisResponse(
            timestamp=datetime.utcnow().isoformat(),
            model_results={
                "autoencoder": {
                    "score": ae_score,
                    "threshold": ae_threshold,
                    "is_anomaly": ae_is_anomaly
                },
                "isolation_forest": {
                    "score": if_score,
                    "is_anomaly": if_is_anomaly
                },
                "consensus": consensus
            },
            risk_level=risk_level,
            concentration_risk=concentration_risk,
            volatility_risk="medium",  
            portfolio_metrics=portfolio_metrics,
            recommendations=recommendations,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/explain", response_model=LLMExplanationResponse)
def explain_recommendations(request: LLMExplanationRequest):
    """
    Generate natural language explanation of portfolio recommendations using LLM.
    
    This endpoint uses a modular agentic flow to interpret recommendations.
    Can optionally include full portfolio context for more detailed explanations.
    """
    try:
        if not is_llm_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available. Set ANTHROPIC_API_KEY environment variable."
            )
        
        # Build context for agents
        context = {
            "recommendations": request.recommendations,
            "risk_level": request.risk_level,
            "portfolio_metrics": request.portfolio_metrics,
            "feature_conclusions": request.feature_conclusions or [],
            "security_scores": request.security_scores or []
        }
        
        # Add optional context if provided
        if request.include_portfolio and request.portfolio:
            context["portfolio"] = request.portfolio
        if request.analysis:
            context["analysis"] = request.analysis
        
        # Initialize agentic flow
        # Use portfolio context agent if full portfolio is provided, otherwise use recommendation interpreter
        if request.include_portfolio and request.portfolio:
            flow = AgenticFlow(agents=[PortfolioContextAgent()])
        else:
            flow = AgenticFlow(agents=[RecommendationInterpreterAgent()])
        
        # Execute flow
        results = flow.execute(context)
        
        # Get combined explanation
        explanation = None
        if results.get("explanations"):
            # Use the first successful explanation, or combine if multiple
            if len(results["explanations"]) == 1:
                explanation = results["explanations"][0]["explanation"]
            else:
                explanation = flow.get_combined_explanation(results)
        
        return LLMExplanationResponse(
            success=results.get("flow_success", False),
            explanation=explanation,
            agents_executed=results.get("agents_executed", []),
            errors=results.get("errors", []),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM explanation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/analyze-and-explain", response_model=OrchestratedAnalysisResponse)
def analyze_and_explain(request: OrchestratedAnalysisRequest):
    """
    Orchestrated endpoint that analyzes portfolio and generates LLM explanation.
    
    This endpoint:
    1. Analyzes the portfolio (same as /portfolio/analyze)
    2. Scores each security individually for detailed insights
    3. Computes z-scores relative to training basket for context
    4. Optionally generates LLM explanation of recommendations (if requested and LLM available)
    5. Returns both results in a single response
    
    This is a convenience endpoint that combines analysis and explanation in one call.
    """
    try:
        # Step 1: Load models and basket stats
        logger.info("Starting portfolio analysis...")
        ae_model, if_model = load_models()
        
        # Load or compute basket statistics
        basket_stats = load_basket_stats()
        if basket_stats is None and request.include_security_detail:
            logger.info("Computing basket statistics on demand...")
            basket_stats = compute_basket_stats_on_demand(ae_model, if_model)
        
        # Step 2: Calculate portfolio features (aggregate)
        X, weights, current_prices, weighted_features = calculate_portfolio_features(request.portfolio)
        
        # Step 3: Run models on aggregate portfolio
        ae_score = float(ae_model.score(X)[0])
        ae_prediction = int(ae_model.predict(X)[0])
        ae_is_anomaly = (ae_prediction == -1)
        ae_threshold = float(ae_model.threshold)
        
        if_score = float(if_model.score(X)[0])
        if_prediction = int(if_model.predict(X)[0])
        if_is_anomaly = (if_prediction == -1)
        
        # Consensus
        consensus = {
            "both_flagged": ae_is_anomaly and if_is_anomaly,
            "either_flagged": ae_is_anomaly or if_is_anomaly,
            "models_agree": ae_is_anomaly == if_is_anomaly,
            "confidence": "high" if ae_is_anomaly == if_is_anomaly else "medium"
        }
        
        # Calculate concentration metrics
        concentration_metrics = calculate_concentration_metrics(weights)
        concentration_risk = assess_concentration_risk(concentration_metrics)
        
        # Overall risk level
        risk_level = get_risk_level(ae_score, ae_threshold, concentration_risk)
        
        # Interpret features
        feature_conclusions = interpret_portfolio_features(weighted_features, ae_score, ae_threshold)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            risk_level,
            concentration_metrics,
            ae_is_anomaly
        )
        
        # Generate message
        message = generate_message(ae_is_anomaly, risk_level, ae_score, ae_threshold)
        
        # Step 4: Score individual securities if requested
        security_scores = []
        extended_metrics = None
        
        if request.include_security_detail:
            logger.info("Scoring individual securities...")
            security_scores = score_individual_securities(
                request.portfolio,
                ae_model,
                if_model,
                current_prices,
                weights,
                basket_stats
            )
            
            if security_scores:
                extended_metrics = compute_extended_metrics(
                    security_scores,
                    weights,
                    concentration_metrics,
                    basket_stats
                )
                logger.info(f"Scored {len(security_scores)} securities, "
                           f"anomaly rate: {extended_metrics.anomaly_rate_ae:.1%} (AE), "
                           f"{extended_metrics.anomaly_rate_if:.1%} (IF)")
        
        # Build portfolio metrics (extended if available)
        if extended_metrics:
            portfolio_metrics = extended_metrics.model_dump()
        else:
            total_value = sum(
                pos.shares * current_prices.get(symbol, 0) 
                for symbol, pos in request.portfolio.positions.items()
            )
            portfolio_metrics = {
                "total_value": total_value,
                "n_positions": len(request.portfolio.positions),
                "weights": weights,
                "concentration_hhi": concentration_metrics['herfindahl_index'],
                "max_position_weight": concentration_metrics['max_position'],
                "top_3_concentration": concentration_metrics['top_3_concentration']
            }
        
        # Create analysis response
        analysis_response = PortfolioAnalysisResponse(
            timestamp=datetime.utcnow().isoformat(),
            model_results={
                "autoencoder": {
                    "score": ae_score,
                    "threshold": ae_threshold,
                    "is_anomaly": ae_is_anomaly
                },
                "isolation_forest": {
                    "score": if_score,
                    "is_anomaly": if_is_anomaly
                },
                "consensus": consensus
            },
            risk_level=risk_level,
            concentration_risk=concentration_risk,
            volatility_risk="medium",  
            portfolio_metrics=portfolio_metrics,
            recommendations=recommendations,
            message=message
        )
        
        # Step 5: Generate LLM explanation if requested
        explanation_response = None
        explanation_available = False
        
        if request.include_explanation:
            logger.info("Generating LLM explanation...")
            
            if not is_llm_available():
                logger.warning("LLM not available, skipping explanation")
            else:
                try:
                    # Build context for agents with extended detail
                    context = {
                        "recommendations": recommendations,
                        "risk_level": risk_level,
                        "portfolio_metrics": portfolio_metrics,
                        "feature_conclusions": feature_conclusions,
                        "security_scores": [s.model_dump() for s in security_scores] if security_scores else [],
                        "basket_stats": basket_stats.model_dump() if basket_stats else None
                    }
                    
                    # Debug: Log feature conclusions being sent to LLM
                    logger.info("Feature conclusions being sent to LLM:")
                    for i, conclusion in enumerate(feature_conclusions, 1):
                        logger.info(f"  {i}. [{conclusion.get('category', 'Unknown')}] {conclusion.get('severity', 'unknown').upper()}: {conclusion.get('finding', '')}")
                    
                    if security_scores:
                        logger.info(f"Security scores being sent to LLM: {len(security_scores)} securities")
                        anomalous = [s for s in security_scores if s.consensus_anomaly]
                        if anomalous:
                            logger.info(f"  Anomalous securities: {[s.symbol for s in anomalous]}")
                    
                    # Add optional context if requested
                    if request.include_portfolio_context:
                        context["portfolio"] = request.portfolio
                        context["analysis"] = {
                            "risk_level": risk_level,
                            "concentration_risk": concentration_risk,
                            "volatility_risk": "medium",
                            "model_results": analysis_response.model_results,
                            "portfolio_metrics": portfolio_metrics
                        }
                        logger.info(f"Full portfolio context included: {len(request.portfolio.positions)} positions")
                    
                    # Initialize agentic flow
                    if request.include_portfolio_context:
                        flow = AgenticFlow(agents=[PortfolioContextAgent()])
                    else:
                        flow = AgenticFlow(agents=[RecommendationInterpreterAgent()])
                    
                    # Execute flow
                    results = flow.execute(context)
                    
                    # Get combined explanation
                    explanation = None
                    if results.get("explanations"):
                        if len(results["explanations"]) == 1:
                            explanation = results["explanations"][0]["explanation"]
                        else:
                            explanation = flow.get_combined_explanation(results)
                    
                    explanation_response = LLMExplanationResponse(
                        success=results.get("flow_success", False),
                        explanation=explanation,
                        agents_executed=results.get("agents_executed", []),
                        errors=results.get("errors", []),
                        timestamp=datetime.utcnow().isoformat()
                    )
                    
                    explanation_available = explanation_response.success and explanation is not None
                    
                except Exception as e:
                    logger.error(f"Error generating LLM explanation: {e}", exc_info=True)
                    # Don't fail the whole request if explanation fails
                    explanation_response = LLMExplanationResponse(
                        success=False,
                        explanation=None,
                        agents_executed=[],
                        errors=[{"error": str(e)}],
                        timestamp=datetime.utcnow().isoformat()
                    )
        
        return OrchestratedAnalysisResponse(
            analysis=analysis_response,
            explanation=explanation_response,
            timestamp=datetime.utcnow().isoformat(),
            explanation_requested=request.include_explanation,
            explanation_available=explanation_available
        )
        
    except Exception as e:
        logger.error(f"Orchestration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Check if models exist
    ae_path = MODEL_DIR / 'autoencoder' / 'model.pth'
    if_path = MODEL_DIR / 'isolation_forest' / 'model.joblib'
    
    if not ae_path.exists() or not if_path.exists():
        logger.warning("⚠️  Models not found!")
        logger.warning("Run 'make bootstrap' to fetch data and train models")
    else:
        logger.info("✓ Models found")
    
    # Load basket stats at startup
    basket_stats = load_basket_stats()
    if basket_stats:
        logger.info(f"✓ Basket statistics loaded ({basket_stats.n_securities} securities)")
    else:
        logger.warning("⚠️  Basket statistics not found, will compute on demand")
    
    logger.info("Starting Portfolio Anomaly Detection API...")
    logger.info("API documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")