#!/usr/bin/env python3
"""
Analysis and risk assessment of portfolio holdings.

This module provides various methods to calculate portfolio metrics, assess
concentration and anomaly risks, and generate actionable recommendations.

- Autoencoder is primary model
- Isolation Forest is secondary/confirmatory
- Weighted ensemble: AE 85%, IF 15%

Functions:
- calculate_portfolio_weights: Computes portfolio weights proportionate to asset values.
- calculate_concentration_metrics: Derives metrics related to portfolio asset concentration.
- assess_concentration_risk: Evaluates concentration risk as high, medium, or low.
- calculate_portfolio_returns: Calculates the portfolio's returns using asset weights.
- calculate_portfolio_volatility: Computes the annualized portfolio volatility.
- get_model_scores: Calculates normalized anomaly scores and flags for models.
- get_ensemble_score: Determines a weighted ensemble anomaly score.
- get_risk_level: Assesses overall portfolio risk level based on provided metrics.
- generate_recommendations: Generates portfolio insight based on risk and model scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from schemas import Portfolio, Position


def calculate_portfolio_weights(positions: Dict[str, Position], current_prices: Dict[str, float]) -> Dict[str, float]:
    """Calculate portfolio weights based on current prices."""
    total_value = sum(pos.shares * current_prices.get(symbol, 0) for symbol, pos in positions.items())
    
    if total_value == 0:
        return {symbol: 0.0 for symbol in positions}
    
    weights = {}
    for symbol, pos in positions.items():
        position_value = pos.shares * current_prices.get(symbol, 0)
        weights[symbol] = position_value / total_value
    
    return weights


def calculate_concentration_metrics(weights: Dict[str, float]) -> dict:
    """Calculate portfolio concentration metrics."""
    if not weights:
        return {
            'herfindahl_index': 0.0,
            'top_3_concentration': 0.0,
            'max_position': 0.0,
            'n_positions': 0
        }
    
    weight_values = list(weights.values())
    sorted_weights = sorted(weight_values, reverse=True)
    
    return {
        'herfindahl_index': sum(w**2 for w in weight_values),
        'top_3_concentration': sum(sorted_weights[:3]),
        'max_position': sorted_weights[0] if sorted_weights else 0.0,
        'n_positions': len(weights)
    }


def assess_concentration_risk(concentration_metrics: dict) -> str:
    """Assess concentration risk level."""
    hhi = concentration_metrics['herfindahl_index']
    max_pos = concentration_metrics['max_position']
    
    if hhi > 0.25 or max_pos > 0.30:
        return "high"
    elif hhi > 0.15 or max_pos > 0.20:
        return "medium"
    else:
        return "low"


def calculate_portfolio_returns(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame
) -> pd.Series:
    """Calculate portfolio returns given weights and asset returns."""
    if returns_matrix.empty:
        return pd.Series()
    
    available_assets = [symbol for symbol in weights if symbol in returns_matrix.columns]
    
    if not available_assets:
        return pd.Series()
    
    portfolio_returns = pd.Series(0.0, index=returns_matrix.index)
    for symbol in available_assets:
        portfolio_returns += returns_matrix[symbol] * weights[symbol]
    
    return portfolio_returns


def calculate_portfolio_volatility(portfolio_returns: pd.Series, window: int = 60) -> float:
    """Calculate annualized portfolio volatility."""
    if len(portfolio_returns) < window:
        return 0.0
    
    return portfolio_returns.rolling(window).std().iloc[-1] * np.sqrt(252)


def get_model_scores(
    ae_score: float, 
    ae_threshold: float,
    if_score: float = None,
    if_threshold: float = None
) -> Tuple[float, float, bool, bool]:
    """
    Calculate normalized scores and anomaly flags for both models.
    
    Returns:
        Tuple of (ae_ratio, if_ratio, ae_is_anomaly, if_is_anomaly)
    """
    # Autoencoder score ratio (higher = more anomalous)
    ae_ratio = ae_score / ae_threshold if ae_threshold > 0 else 0
    ae_is_anomaly = ae_ratio > 1.0
    
    # Isolation Forest score ratio
    if if_score is not None and if_threshold is not None:
        if_ratio = if_score / if_threshold if if_threshold > 0 else 0
        if_is_anomaly = if_ratio > 1.0
    else:
        if_ratio = 0
        if_is_anomaly = False
    
    return ae_ratio, if_ratio, ae_is_anomaly, if_is_anomaly


def get_ensemble_score(ae_ratio: float, if_ratio: float) -> float:
    """
    Calculate weighted ensemble score.
    
    Based on validation results:
    - Autoencoder: F1=0.63, reliable → weight 85%
    - Isolation Forest: F1=0.19, less reliable → weight 15%
    
    Scores are capped at 3.0 to prevent extreme values from dominating.
    """
    AE_WEIGHT = 0.85
    IF_WEIGHT = 0.15
    
    # Cap ratios to prevent extreme values
    ae_capped = min(ae_ratio, 3.0)
    if_capped = min(if_ratio, 3.0)
    
    return AE_WEIGHT * ae_capped + IF_WEIGHT * if_capped


def get_risk_level(
    ae_score: float, 
    ae_threshold: float, 
    concentration_risk: str,
    if_score: float = None,
    if_threshold: float = None
) -> str:
    """
    Determine overall portfolio risk level.
    
    - Primary decision based on Autoencoder (more reliable, F1=0.63)
    - Isolation Forest used as confirmatory signal only (F1=0.19)
    - Concentration risk can elevate but not lower risk level
    
    Risk Levels:
    - critical: Immediate action needed
    - high: Significant concern, review recommended
    - medium: Elevated risk, monitor closely
    - low: Within normal parameters
    """
    # Get normalized scores
    ae_ratio, if_ratio, ae_is_anomaly, if_is_anomaly = get_model_scores(
        ae_score, ae_threshold, if_score, if_threshold
    )
    
    # Calculate ensemble score
    ensemble_score = get_ensemble_score(ae_ratio, if_ratio)
    
    # === CRITICAL ===
    # Both models agree on anomaly (high confidence)
    if ae_is_anomaly and if_is_anomaly:
        return "critical"
    
    # AE flags anomaly AND high concentration (compounding risks)
    if ae_is_anomaly and concentration_risk == "high":
        return "critical"
    
    # Very high ensemble score (>2.0) with any elevated concentration
    if ensemble_score > 2.0 and concentration_risk in ("medium", "high"):
        return "critical"
    
    # === HIGH ===
    # Autoencoder flags anomaly (primary signal, trust it)
    if ae_is_anomaly:
        return "high"
    
    # High ensemble score without AE anomaly flag
    if ensemble_score > 1.5:
        return "high"
    
    # High concentration alone is significant risk
    if concentration_risk == "high":
        return "high"
    
    # === MEDIUM ===
    # IF flags anomaly alone (less reliable, so only medium)
    if if_is_anomaly:
        return "medium"
    
    # Elevated ensemble score
    if ensemble_score > 0.8:
        return "medium"
    
    # AE approaching threshold with medium concentration
    if ae_ratio > 0.7 and concentration_risk == "medium":
        return "medium"
    
    # Medium concentration with moderate ensemble score
    if concentration_risk == "medium" and ensemble_score > 0.5:
        return "medium"
    
    # === LOW ===
    return "low"


def generate_recommendations(
    risk_level: str,
    concentration_metrics: dict,
    is_anomaly: bool,
    ae_ratio: float = None,
    if_ratio: float = None,
    models_agree: bool = None
) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # === Anomaly-based recommendations ===
    if is_anomaly:
        if models_agree:
            recommendations.append(
                "HIGH CONFIDENCE: Both models detect unusual portfolio patterns - immediate review recommended"
            )
        elif ae_ratio and ae_ratio > 1.0:
            recommendations.append(
                "Portfolio exhibits unusual patterns compared to market norms (primary model flagged)"
            )
            if if_ratio and if_ratio < 1.0:
                recommendations.append(
                    "Note: Secondary model did not confirm anomaly - may be borderline case"
                )
        elif if_ratio and if_ratio > 1.0:
            recommendations.append(
                "Portfolio flagged by secondary model - recommend monitoring"
            )
    
    # === Concentration recommendations ===
    max_pos = concentration_metrics['max_position']
    if max_pos > 0.40:
        recommendations.append(
            f"CRITICAL: Largest position ({max_pos*100:.1f}%) exceeds 40% - severe concentration risk"
        )
    elif max_pos > 0.30:
        recommendations.append(
            f"WARNING: Largest position ({max_pos*100:.1f}%) exceeds 30% - high concentration risk"
        )
    elif max_pos > 0.20:
        recommendations.append(
            f"Consider reducing largest position ({max_pos*100:.1f}% of portfolio)"
        )
    
    top3 = concentration_metrics['top_3_concentration']
    if top3 > 0.80:
        recommendations.append(
            f"Top 3 positions are {top3*100:.1f}% of portfolio - highly concentrated"
        )
    elif top3 > 0.60:
        recommendations.append(
            f"Top 3 positions represent {top3*100:.1f}% of portfolio - consider diversification"
        )
    
    n_pos = concentration_metrics['n_positions']
    if n_pos < 5:
        recommendations.append(
            f"Portfolio has only {n_pos} positions - significant diversification risk"
        )
    elif n_pos < 10:
        recommendations.append(
            f"Portfolio has {n_pos} positions - consider adding more for diversification"
        )
    
    # === Risk level recommendations ===
    if risk_level == "critical":
        recommendations.append("CRITICAL RISK: Immediate portfolio review strongly recommended")
        recommendations.append("Consider significant rebalancing to reduce risk exposure")
    elif risk_level == "high":
        recommendations.append("Consider rebalancing to reduce risk exposure")
        recommendations.append("Review portfolio alignment with investment objectives")
    elif risk_level == "medium":
        recommendations.append("Monitor portfolio closely for further changes")
    elif risk_level == "low" and not recommendations:
        recommendations.append("✓ Portfolio within normal risk parameters")
    
    return recommendations


def generate_message(
    is_anomaly: bool,
    risk_level: str,
    ae_score: float,
    ae_threshold: float,
    if_score: float = None,
    if_threshold: float = None
) -> str:
    """Generate human-readable summary message."""
    
    ae_ratio, if_ratio, ae_is_anomaly, if_is_anomaly = get_model_scores(
        ae_score, ae_threshold, if_score, if_threshold
    )
    
    ensemble_score = get_ensemble_score(ae_ratio, if_ratio)
    
    if not is_anomaly:
        msg = (
            f"Portfolio patterns appear normal for current market conditions. "
            f"Risk Score: {ensemble_score:.2f} (threshold: 1.0). "
            f"Risk Level: {risk_level.upper()}."
        )
    else:
        if ae_is_anomaly and if_is_anomaly:
            confidence = "HIGH CONFIDENCE"
        elif ae_is_anomaly:
            confidence = "Primary model"
        else:
            confidence = "Secondary model"
        
        msg = (
            f"Portfolio flagged as anomalous ({confidence}). "
            f"Risk Score: {ensemble_score:.2f} (threshold: 1.0). "
            f"Risk Level: {risk_level.upper()}. "
            f"Review recommended."
        )
    
    return msg


def get_model_confidence(ae_is_anomaly: bool, if_is_anomaly: bool) -> str:
    """
    Determine confidence level based on model agreement.
    
    Returns: "high", "medium", or "low"
    """
    if ae_is_anomaly == if_is_anomaly:
        return "high"  # Models agree (both flag or both don't)
    elif ae_is_anomaly and not if_is_anomaly:
        return "medium"  # Primary model flags, secondary doesn't
    else:
        return "low"  # Only secondary model flags (less reliable signal)


def interpret_portfolio_features(features: Dict[str, float], anomaly_score: float, threshold: float) -> List[Dict[str, any]]:
    """
    Draw conclusions from feature patterns.
    
    Args:
        features: Dictionary of feature names to values
        anomaly_score: Anomaly score from model
        threshold: Anomaly threshold
    
    Returns:
        List of conclusion dictionaries with category, severity, finding, implication, recommendation
    """
    conclusions = []
    
    # Helper to safely get feature value with default
    def get_feature(name: str, default: float = 0.0) -> float:
        return features.get(name, default)
    
    # 1. Momentum Assessment
    rsi = get_feature('rsi_14', 50.0)
    returns_20d = get_feature('returns_20d', 0.0)
    
    if rsi > 70 and returns_20d > 0.05:
        conclusions.append({
            'category': 'Momentum Risk',
            'severity': 'high',
            'finding': 'Portfolio is overbought with strong recent gains',
            'implication': 'Elevated risk of near-term pullback',
            'recommendation': 'Consider taking profits or hedging'
        })
    elif rsi < 30 and returns_20d < -0.05:
        conclusions.append({
            'category': 'Momentum Risk',
            'severity': 'medium',
            'finding': 'Portfolio is oversold with recent losses',
            'implication': 'Potential bounce opportunity, but may indicate fundamental weakness',
            'recommendation': 'Investigate underlying causes before adding positions'
        })
    
    # 2. Volatility Assessment
    volatility_20d = get_feature('volatility_20d', 0.0)
    market_volatility = 0.16  # Approximate market average (~16% annualized)
    
    if volatility_20d > 0.30:
        vol_multiple = volatility_20d / market_volatility if market_volatility > 0 else 0
        conclusions.append({
            'category': 'Volatility Risk',
            'severity': 'high',
            'finding': f"Volatility ({volatility_20d:.2%}) is {vol_multiple:.1f}x market average",
            'implication': 'Large price swings expected',
            'recommendation': 'Ensure position sizing matches risk tolerance'
        })
    elif volatility_20d < 0.08:
        conclusions.append({
            'category': 'Volatility Risk',
            'severity': 'low',
            'finding': f"Volatility ({volatility_20d:.2%}) is below market average",
            'implication': 'Defensive positioning or low-risk assets',
            'recommendation': 'May indicate conservative allocation'
        })
    
    # 3. Price Position
    price_to_52d_high = get_feature('price_to_52d_high', 0.5)
    price_to_52d_low = get_feature('price_to_52d_low', 0.5)
    
    if price_to_52d_high > 0.95:
        conclusions.append({
            'category': 'Valuation Risk',
            'severity': 'medium',
            'finding': 'Portfolio trading at 52-week highs',
            'implication': 'Limited upside, elevated downside risk',
            'recommendation': 'Consider taking profits or tightening stops'
        })
    elif price_to_52d_low < 0.10:
        conclusions.append({
            'category': 'Valuation Risk',
            'severity': 'high',
            'finding': 'Portfolio trading near 52-week lows',
            'implication': 'Significant drawdown or fundamental issues',
            'recommendation': 'Review individual positions for fundamental problems'
        })
    
    # 4. Technical Trend
    above_ma_50 = get_feature('above_ma_50', 1.0)
    above_ma_200 = get_feature('above_ma_200', 1.0)
    macd_histogram = get_feature('macd_histogram', 0.0)
    
    if above_ma_50 == 0 and above_ma_200 == 0 and macd_histogram < 0:
        conclusions.append({
            'category': 'Trend Risk',
            'severity': 'high',
            'finding': 'Portfolio in technical downtrend',
            'implication': 'Multiple support levels broken, momentum negative',
            'recommendation': 'Consider defensive positioning or stop-losses'
        })
    elif above_ma_50 == 1 and above_ma_200 == 1 and macd_histogram > 0:
        conclusions.append({
            'category': 'Trend Strength',
            'severity': 'low',
            'finding': 'Portfolio in strong uptrend',
            'implication': 'Technical indicators bullish',
            'recommendation': 'Maintain positions, monitor for overextension'
        })
    
    # 5. Volume Analysis
    volume_ratio_20d = get_feature('volume_ratio_20d', 1.0)
    
    if volume_ratio_20d > 2.0:
        conclusions.append({
            'category': 'Activity Risk',
            'severity': 'medium',
            'finding': f"Volume {volume_ratio_20d:.1f}x normal levels",
            'implication': 'Unusual activity may indicate instability or news',
            'recommendation': 'Monitor for significant news or institutional activity'
        })
    elif volume_ratio_20d < 0.5:
        conclusions.append({
            'category': 'Liquidity Risk',
            'severity': 'low',
            'finding': f"Volume {volume_ratio_20d:.1f}x normal levels (low)",
            'implication': 'Low liquidity may impact execution',
            'recommendation': 'Verify liquidity before large trades'
        })
    
    # 6. Bollinger Band Position
    bb_position = get_feature('bb_position', 0.5)
    bb_width = get_feature('bb_width', 0.0)
    
    if bb_position > 0.95:
        conclusions.append({
            'category': 'Mean Reversion Risk',
            'severity': 'medium',
            'finding': 'Portfolio at upper Bollinger Band',
            'implication': 'Statistically overextended',
            'recommendation': 'Consider mean reversion risk in position sizing'
        })
    elif bb_position < 0.05:
        conclusions.append({
            'category': 'Mean Reversion Opportunity',
            'severity': 'low',
            'finding': 'Portfolio at lower Bollinger Band',
            'implication': 'Statistically oversold',
            'recommendation': 'Potential bounce, but verify fundamentals first'
        })
    
    # 7. Concentration inference from anomaly score
    if anomaly_score > threshold * 1.5:
        volatility = get_feature('volatility_20d', 0.0)
        returns_1d = get_feature('returns_1d', 0.0)
        
        if volatility > 0.35 and abs(returns_1d) > 0.02:
            conclusions.append({
                'category': 'Concentration Risk',
                'severity': 'critical',
                'finding': 'Portfolio behaving like concentrated position',
                'implication': 'High single-day moves suggest lack of diversification',
                'recommendation': 'Urgently review position sizes and correlations'
            })
    
    return conclusions