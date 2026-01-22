#!/usr/bin/env python3
"""Pydantic schemas for portfolio analysis API."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Position(BaseModel):
    """Individual position in portfolio."""
    shares: float = Field(..., description="Number of shares")
    cost_basis: Optional[float] = Field(None, description="Cost basis per share")


class Portfolio(BaseModel):
    """Portfolio request model."""
    positions: Dict[str, Position] = Field(
        ..., 
        description="Dictionary mapping ticker symbols to positions",
        json_schema_extra={
            "example": {
                "AAPL": {"shares": 100, "cost_basis": 150},
                "MSFT": {"shares": 50, "cost_basis": 300}
            }
        }
    )
    benchmark: Optional[str] = Field("SPY", description="Benchmark symbol for comparison")


class SecurityScore(BaseModel):
    """Individual security anomaly score details."""
    symbol: str = Field(..., description="Ticker symbol")
    shares: float = Field(..., description="Number of shares held")
    current_price: float = Field(..., description="Current price per share")
    position_value: float = Field(..., description="Total position value")
    weight: float = Field(..., description="Weight in portfolio (0-1)")
    
    # Cost basis and gain/loss
    cost_basis: Optional[float] = Field(None, description="Cost basis per share if provided")
    total_cost: Optional[float] = Field(None, description="Total cost basis (shares * cost_basis)")
    gain_loss_dollars: Optional[float] = Field(None, description="Unrealized gain/loss in dollars")
    gain_loss_percent: Optional[float] = Field(None, description="Unrealized gain/loss as percentage")
    
    # Sector information
    sector: Optional[str] = Field(None, description="Stock sector (e.g., Technology, Healthcare)")
    
    # Autoencoder scores
    ae_score: float = Field(..., description="Autoencoder reconstruction error score")
    ae_is_anomaly: bool = Field(..., description="Whether AE flags as anomaly")
    ae_z_score: Optional[float] = Field(None, description="AE score z-score vs training basket")
    ae_z_score_display: Optional[str] = Field(None, description="Human-readable z-score (e.g., '>3σ extreme')")
    
    # Isolation Forest scores (optional - may not be available for all model types)
    if_score: Optional[float] = Field(None, description="Isolation Forest anomaly score")
    if_is_anomaly: Optional[bool] = Field(None, description="Whether IF flags as anomaly")
    if_z_score: Optional[float] = Field(None, description="IF score z-score vs training basket")
    if_z_score_display: Optional[str] = Field(None, description="Human-readable z-score")
    
    # Consensus
    consensus_anomaly: bool = Field(..., description="Both models flag as anomaly (or AE only if IF unavailable)")
    
    # Interpretation
    status: str = Field("normal", description="Human-readable status: normal, warning, anomaly")
    status_reason: Optional[str] = Field(None, description="Reason for status")
    
    # Materiality assessment
    is_material: bool = Field(True, description="Whether position is large enough to matter (>1% weight)")
    materiality_note: Optional[str] = Field(None, description="Note about position significance")


class BasketStatistics(BaseModel):
    """Training basket statistics for z-score computation."""
    ae_mean: float = Field(..., description="Mean AE score across training basket")
    ae_std: float = Field(..., description="Std dev of AE scores")
    if_mean: Optional[float] = Field(None, description="Mean IF score across training basket (optional)")
    if_std: Optional[float] = Field(None, description="Std dev of IF scores (optional)")
    n_securities: int = Field(..., description="Number of securities in training basket")
    computed_at: Optional[str] = Field(None, description="When basket stats were computed")


class ExtendedPortfolioMetrics(BaseModel):
    """Extended portfolio metrics with per-security detail."""
    # Basic metrics
    total_value: float = Field(..., description="Total portfolio value")
    n_positions: int = Field(..., description="Number of positions")
    weights: Dict[str, float] = Field(..., description="Position weights by symbol")
    
    # Concentration metrics
    concentration_hhi: float = Field(..., description="Herfindahl-Hirschman Index")
    max_position_weight: float = Field(..., description="Largest position weight")
    top_3_concentration: float = Field(..., description="Top 3 positions combined weight")
    
    # Aggregate z-scores (portfolio-weighted)
    ae_weighted_score: float = Field(..., description="Portfolio-weighted AE score")
    if_weighted_score: Optional[float] = Field(None, description="Portfolio-weighted IF score (optional)")
    ae_weighted_z_score: Optional[float] = Field(None, description="Portfolio-weighted AE z-score")
    if_weighted_z_score: Optional[float] = Field(None, description="Portfolio-weighted IF z-score")
    
    # Anomaly rates
    anomaly_rate_ae: float = Field(..., description="Fraction of holdings flagged by AE")
    anomaly_rate_if: Optional[float] = Field(None, description="Fraction of holdings flagged by IF (optional)")
    anomaly_rate_consensus: float = Field(..., description="Fraction flagged by both models (or AE only if IF unavailable)")
    anomaly_count_ae: int = Field(..., description="Number of holdings flagged by AE")
    anomaly_count_if: Optional[int] = Field(None, description="Number of holdings flagged by IF (optional)")
    anomaly_count_consensus: int = Field(..., description="Number flagged by both models (or AE only if IF unavailable)")
    
    # Per-security detail
    security_scores: List[SecurityScore] = Field(
        default_factory=list, 
        description="Individual security scores and analysis"
    )
    
    # Basket comparison context
    basket_stats: Optional[BasketStatistics] = Field(
        None, 
        description="Training basket statistics for context"
    )
    
    # Z-score interpretation
    portfolio_vs_market: Optional[str] = Field(
        None,
        description="How portfolio compares to typical market behavior"
    )


class PortfolioAnalysisResponse(BaseModel):
    """Response from portfolio analysis."""
    portfolio_id: Optional[str] = None
    timestamp: str
    
    # Model results
    model_results: dict
    
    # Risk assessment
    risk_level: str
    concentration_risk: str
    volatility_risk: str
    
    # Metrics - now supports extended metrics
    portfolio_metrics: dict
    
    # Recommendations
    recommendations: List[str]
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict
    llm_available: bool
    basket_stats_available: bool = Field(False, description="Whether basket statistics are loaded")
    timestamp: str


class LLMExplanationRequest(BaseModel):
    """Request for LLM explanation of recommendations."""
    recommendations: List[str] = Field(..., description="List of recommendations to explain")
    risk_level: str = Field(..., description="Portfolio risk level")
    portfolio_metrics: Dict = Field(..., description="Portfolio metrics")
    include_portfolio: bool = Field(False, description="Whether to include full portfolio context")
    portfolio: Optional[Portfolio] = Field(None, description="Full portfolio data (optional)")
    analysis: Optional[Dict] = Field(None, description="Full analysis results (optional)")
    feature_conclusions: Optional[List[Dict]] = Field(None, description="Feature interpretation conclusions (optional)")
    security_scores: Optional[List[Dict]] = Field(None, description="Per-security scores (optional)")


class LLMExplanationResponse(BaseModel):
    """Response from LLM explanation."""
    success: bool
    explanation: Optional[str] = Field(None, description="LLM-generated explanation")
    agents_executed: List[str] = Field(default_factory=list, description="List of agents that executed")
    errors: List[Dict] = Field(default_factory=list, description="Any errors encountered")
    timestamp: str


class OrchestratedAnalysisRequest(BaseModel):
    """Request for orchestrated analyze + explain endpoint."""
    portfolio: Portfolio = Field(..., description="Portfolio to analyze")
    include_explanation: bool = Field(True, description="Whether to include LLM explanation")
    include_portfolio_context: bool = Field(True, description="Whether to include full portfolio context in explanation")
    include_security_detail: bool = Field(True, description="Whether to include per-security scoring detail")


class OrchestratedAnalysisResponse(BaseModel):
    """Combined response from orchestrated analyze + explain endpoint."""
    # Analysis results
    analysis: PortfolioAnalysisResponse

    # LLM explanation (optional - may be None if LLM unavailable or requested to skip)
    explanation: Optional[LLMExplanationResponse] = Field(None, description="LLM explanation if available and requested")

    # Metadata
    timestamp: str
    explanation_requested: bool = Field(True, description="Whether explanation was requested")
    explanation_available: bool = Field(False, description="Whether LLM explanation was successfully generated")


class Holding(BaseModel):
    """Individual holding with symbol and weight."""
    symbol: str = Field(..., description="Ticker symbol")
    weight: float = Field(..., description="Portfolio weight (0-1)")


class PortfolioHealthRequest(BaseModel):
    """Request for portfolio health analysis using experiment-based scoring."""
    holdings: List[Holding] = Field(
        ...,
        description="List of holdings with symbols and weights",
        json_schema_extra={
            "example": [
                {"symbol": "AAPL", "weight": 0.25},
                {"symbol": "MSFT", "weight": 0.20},
                {"symbol": "GOOGL", "weight": 0.15}
            ]
        }
    )
    analysis_date: str = Field(..., description="Analysis date in YYYY-MM-DD format")
    model_type: str = Field("cross_sectional", description="Model type name for scoring (default: cross_sectional)")
    market_proxy: str = Field("SPY", description="Market proxy symbol for comparison")
    contra_horizon: int = Field(5, description="Number of days for contra return calculation")


class Contributor(BaseModel):
    """Contribution of a single holding to portfolio score."""
    symbol: str = Field(..., description="Ticker symbol")
    weight: float = Field(..., description="Portfolio weight")
    ae_score: float = Field(..., description="Autoencoder anomaly score")
    contribution: float = Field(..., description="Contribution to portfolio score (weight * ae_score)")


class PortfolioHealthResponse(BaseModel):
    """Response from portfolio health analysis."""
    date: str = Field(..., description="Analysis date")
    portfolio_score: float = Field(..., description="Aggregate portfolio anomaly score")
    contra_return: float = Field(..., description="Portfolio return vs market over horizon")
    health_score: float = Field(..., description="Combined health score (structural + contra)")
    contributors: List[Contributor] = Field(..., description="Per-holding score contributions")