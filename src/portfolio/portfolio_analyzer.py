"""
Portfolio Analyzer - Unified interface for portfolio anomaly detection.

This module provides a high-level interface that orchestrates both:
1. Individual model approach: Detects unusual portfolio characteristics
2. Cross-sectional model approach: Identifies which stocks are driving divergence from market

Usage:
    from portfolio.portfolio_analyzer import PortfolioAnalyzer
    
    analyzer = PortfolioAnalyzer()
    result = analyzer.analyze(holdings, analysis_date='2024-01-15')
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
import sqlite3
import json

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector
from data.feature_extractor import MarketFeatureExtractor
from portfolio.analyze import analyze_portfolio_health

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """
    Unified portfolio analyzer that combines individual and cross-sectional approaches.
    
    The analyzer provides two complementary views:
    1. Risk Assessment (individual): Is this portfolio unusual?
    2. Attribution (cross_sectional): Which stocks are driving divergence?
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        individual_model_type: str = 'individual',
        cross_sectional_model_type: str = 'cross_sectional'
    ):
        """
        Initialize the portfolio analyzer.
        
        Args:
            model_dir: Root directory for models (default: project_root/models)
            db_path: Path to market data database (default: data/processed/market_data.sqlite)
            individual_model_type: Name of individual model type (default: 'individual')
            cross_sectional_model_type: Name of cross-sectional model type (default: 'cross_sectional')
        """
        self.project_root = project_root
        
        if model_dir is None:
            model_dir = self.project_root / 'models'
        self.model_dir = Path(model_dir)
        
        if db_path is None:
            db_path = self.project_root / 'data' / 'processed' / 'market_data.sqlite'
        self.db_path = Path(db_path)
        
        self.individual_model_type = individual_model_type
        self.cross_sectional_model_type = cross_sectional_model_type
        
        # Model caches
        self._individual_models = {}
        self._cross_sectional_models = {}
        self._feature_extractor = None
        
        logger.info(f"Initialized PortfolioAnalyzer")
        logger.info(f"  Model dir: {self.model_dir}")
        logger.info(f"  DB path: {self.db_path}")
        logger.info(f"  Individual model type: {self.individual_model_type}")
        logger.info(f"  Cross-sectional model type: {self.cross_sectional_model_type}")
    
    def _load_individual_models(self):
        """Load individual (risk assessment) models."""
        if self._individual_models:
            return
        
        individual_dir = self.model_dir / 'model_types' / self.individual_model_type
        
        if not individual_dir.exists():
            logger.warning(f"Individual models not found at {individual_dir}")
            logger.warning(f"Run 'python scripts/train_model.py {self.individual_model_type}' to train")
            return
        
        logger.info("Loading individual models...")
        
        # Load autoencoder
        ae_model = AutoencoderAnomalyDetector.load(individual_dir / 'autoencoder')
        
        # Try to load isolation forest (may not exist for all model types)
        if_model = None
        if_path = individual_dir / 'isolation_forest' / 'model.joblib'
        if if_path.exists():
            if_model = IsolationForestAnomalyDetector(
                sector='isolation_forest',
                model_dir=individual_dir
            )
            if_model.load()
        
        self._individual_models = {
            'autoencoder': ae_model,
            'isolation_forest': if_model
        }
        
        logger.info("✓ Individual models loaded")
    
    def _load_cross_sectional_models(self):
        """Load cross-sectional (attribution) models."""
        if self._cross_sectional_models:
            return
        
        cross_sectional_dir = self.model_dir / 'model_types' / self.cross_sectional_model_type
        
        if not cross_sectional_dir.exists():
            logger.warning(f"Cross-sectional models not found at {cross_sectional_dir}")
            logger.warning(f"Run 'python scripts/train_model.py {self.cross_sectional_model_type}' to train")
            return
        
        logger.info("Loading cross-sectional models...")
        
        # Load autoencoder only (cross-sectional approach)
        ae_model = AutoencoderAnomalyDetector.load(cross_sectional_dir / 'autoencoder')
        
        self._cross_sectional_models = {
            'autoencoder': ae_model
        }
        
        logger.info("✓ Cross-sectional models loaded")
    
    def _get_feature_extractor(self):
        """Get or create feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = MarketFeatureExtractor(str(self.db_path))
        return self._feature_extractor
    
    def analyze_risk(
        self,
        holdings: List[Dict],
        use_shares: bool = True
    ) -> Dict:
        """
        Assess portfolio risk using individual model approach.
        
        This answers: "Is this portfolio unusual?"
        
        Args:
            holdings: List of dicts with 'symbol' and either 'shares' or 'weight'
            use_shares: If True, use share counts; if False, use weights
        
        Returns:
            Dict with risk assessment:
                - risk_level: 'low', 'medium', 'high'
                - is_anomaly: bool
                - autoencoder_score: float
                - isolation_forest_score: float (if available)
                - consensus: bool (models agree)
                - recommendations: List[str]
        """
        self._load_individual_models()
        
        if not self._individual_models:
            raise RuntimeError("Individual models not available")
        
        # Extract symbols and weights
        symbols = [h['symbol'] for h in holdings]
        
        if use_shares:
            shares = {h['symbol']: h.get('shares', 0) for h in holdings}
            # Calculate market values and weights would go here
            # For now, convert to equal weights as placeholder
            weights = {s: 1.0/len(symbols) for s in symbols}
        else:
            weights = {h['symbol']: h.get('weight', 1.0/len(symbols)) for h in holdings}
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Get latest features for each security
        extractor = self._get_feature_extractor()
        
        portfolio_features = []
        feature_cols = None
        
        for symbol in symbols:
            # Load price data
            df = extractor.load_price_data(symbol)
            if df.empty or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Calculate features
            features = extractor.calculate_all_features(df)
            if features.empty:
                logger.warning(f"Could not calculate features for {symbol}")
                continue
            
            # Get most recent features
            latest_features = features.iloc[-1]
            
            # Get feature columns (exclude symbol)
            if feature_cols is None:
                feature_cols = extractor.get_feature_columns(features)
            
            # Extract feature values
            feature_values = np.array([latest_features[col] for col in feature_cols])
            
            # Replace inf and NaN
            feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Weight the features
            portfolio_features.append(feature_values * weights[symbol])
        
        if not portfolio_features:
            raise ValueError("No features available for any portfolio symbols")
        
        # Aggregate portfolio features
        portfolio_vector = np.sum(portfolio_features, axis=0)
        portfolio_vector = portfolio_vector.reshape(1, -1)
        
        # Replace any remaining inf/NaN
        portfolio_vector = np.nan_to_num(portfolio_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Score with autoencoder (always available)
        ae_model = self._individual_models['autoencoder']
        
        ae_score = float(ae_model.score(portfolio_vector)[0])
        ae_anomaly = ae_model.predict(portfolio_vector)[0] == -1
        
        # Score with isolation forest (if available)
        if_model = self._individual_models.get('isolation_forest')
        if_score = None
        if_anomaly = None
        consensus = None
        
        if if_model is not None:
            if_score = float(if_model.score(portfolio_vector)[0])
            if_anomaly = if_model.predict(portfolio_vector)[0] == -1
            consensus = ae_anomaly == if_anomaly
        
        # Overall assessment
        if if_model is not None:
            if ae_anomaly and if_anomaly:
                risk_level = 'high'
                is_anomaly = True
            elif ae_anomaly or if_anomaly:
                risk_level = 'medium'
                is_anomaly = True
            else:
                risk_level = 'low'
                is_anomaly = False
        else:
            # Only autoencoder available
            if ae_anomaly:
                risk_level = 'high' if ae_score > ae_model.threshold * 1.5 else 'medium'
                is_anomaly = True
            else:
                risk_level = 'low'
                is_anomaly = False
        
        # Generate recommendations
        recommendations = []
        if is_anomaly:
            recommendations.append("Portfolio exhibits unusual characteristics")
            recommendations.append("Consider reviewing position sizing and diversification")
            if if_model is not None and not consensus:
                recommendations.append("Mixed signals - proceed with caution")
        else:
            recommendations.append("Portfolio within normal risk parameters")
        
        result = {
            'risk_level': risk_level,
            'is_anomaly': is_anomaly,
            'model_results': {
                'autoencoder': {
                    'score': ae_score,
                    'is_anomaly': ae_anomaly,
                    'threshold': float(ae_model.threshold)
                },
                'consensus': {
                    'models_agree': consensus if consensus is not None else True,
                    'confidence': 'high' if consensus else 'medium' if consensus is not None else 'medium'
                }
            },
            'recommendations': recommendations
        }
        
        if if_model is not None:
            result['model_results']['isolation_forest'] = {
                'score': if_score,
                'is_anomaly': if_anomaly
            }
        
        return result
    
    def analyze_attribution(
        self,
        holdings: List[Dict],
        analysis_date: Optional[str] = None,
        market_proxy: str = 'SPY',
        contra_horizon: int = 5
    ) -> Dict:
        """
        Identify which stocks are driving portfolio divergence using cross-sectional model.
        
        This answers: "Which holdings are moving differently from the market?"
        
        Args:
            holdings: List of dicts with 'symbol' and 'weight'
            analysis_date: Date for analysis (YYYY-MM-DD), defaults to latest
            market_proxy: Market proxy symbol (default: SPY)
            contra_horizon: Days for return calculation (default: 5)
        
        Returns:
            Dict with attribution analysis:
                - portfolio_score: Overall divergence score
                - structural_z: Z-score vs market baseline
                - directional_score: Contra-directional return
                - health_score: Combined score
                - contributors: List of stocks sorted by contribution
        """
        if analysis_date is None:
            # Get latest date from database - try new table structure first
            conn = sqlite3.connect(self.db_path)
            
            result = conn.execute(
                "SELECT MAX(date) FROM market_daily_summary WHERE model_type = ?",
                (self.cross_sectional_model_type,)
            ).fetchone()

            conn.close()
            
            if result is None or result[0] is None:
                raise ValueError(
                    f"No scores found for model type '{self.cross_sectional_model_type}'. "
                    f"Run 'python scripts/score_universe.py {self.cross_sectional_model_type}' first"
                )
            
            analysis_date = result[0]
            logger.info(f"Using latest analysis date: {analysis_date}")
        
        # Use the existing analyze_portfolio_health function
        result = analyze_portfolio_health(
            holdings=holdings,
            analysis_date=analysis_date,
            model_type=self.cross_sectional_model_type,
            market_proxy=market_proxy,
            contra_horizon=contra_horizon,
            db_path=self.db_path
        )
        
        return result
    
    def analyze(
        self,
        holdings: List[Dict],
        analysis_date: Optional[str] = None,
        include_risk: bool = True,
        include_attribution: bool = True
    ) -> Dict:
        """
        Comprehensive portfolio analysis combining both approaches.
        
        Args:
            holdings: List of dicts with 'symbol' and either 'shares' or 'weight'
            analysis_date: Date for attribution analysis (defaults to latest)
            include_risk: Include individual model risk assessment
            include_attribution: Include cross-sectional model attribution
        
        Returns:
            Dict with complete analysis:
                - risk_assessment: Portfolio risk level and anomaly flags
                - attribution: Stock-level divergence contributors
                - summary: High-level interpretation
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'n_holdings': len(holdings),
                'symbols': [h['symbol'] for h in holdings]
            }
        }
        
        # Risk assessment (individual model approach)
        if include_risk:
            try:
                logger.info("Running risk assessment (individual model)...")
                risk_result = self.analyze_risk(holdings)
                result['risk_assessment'] = risk_result
            except Exception as e:
                logger.error(f"Risk assessment failed: {e}")
                result['risk_assessment'] = {'error': str(e)}
        
        # Attribution (cross-sectional model approach)
        if include_attribution:
            try:
                logger.info("Running attribution analysis (cross-sectional model)...")
                attribution_result = self.analyze_attribution(holdings, analysis_date)
                result['attribution'] = attribution_result
            except Exception as e:
                logger.error(f"Attribution analysis failed: {e}")
                result['attribution'] = {'error': str(e)}
        
        # Generate summary
        summary = []
        
        if include_risk and 'error' not in result.get('risk_assessment', {}):
            risk = result['risk_assessment']
            if risk['is_anomaly']:
                summary.append(f"⚠️  Portfolio risk level: {risk['risk_level'].upper()}")
            else:
                summary.append(f"✓ Portfolio risk level: {risk['risk_level']}")
        
        if include_attribution and 'error' not in result.get('attribution', {}):
            attr = result['attribution']
            
            # Find top contributors
            contributors = attr.get('contributors', [])
            if contributors:
                top_contributor = contributors[0]
                summary.append(
                    f"Top divergence contributor: {top_contributor['symbol']} "
                    f"(score: {top_contributor['ae_score']:.4f})"
                )
            
            # Health score interpretation
            health = attr.get('health_score', 0)
            if abs(health) > 2.0:
                summary.append(f"⚠️  Health score: {health:.2f} (significant divergence)")
            elif abs(health) > 1.0:
                summary.append(f"⚡ Health score: {health:.2f} (moderate divergence)")
            else:
                summary.append(f"✓ Health score: {health:.2f} (normal)")
        
        result['summary'] = summary
        
        return result


def format_analysis_report(result: Dict) -> str:
    """
    Format analysis result as a human-readable report.
    
    Args:
        result: Output from PortfolioAnalyzer.analyze()
    
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PORTFOLIO ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Holdings: {result['portfolio']['n_holdings']} securities")
    lines.append(f"Symbols: {', '.join(result['portfolio']['symbols'])}")
    lines.append("")
    
    # Summary
    if 'summary' in result:
        lines.append("SUMMARY")
        lines.append("-" * 80)
        for item in result['summary']:
            lines.append(f"  {item}")
        lines.append("")
    
    # Risk assessment
    if 'risk_assessment' in result and 'error' not in result['risk_assessment']:
        risk = result['risk_assessment']
        lines.append("RISK ASSESSMENT (Individual Model Approach)")
        lines.append("-" * 80)
        lines.append(f"Risk Level: {risk['risk_level'].upper()}")
        lines.append(f"Anomaly Detected: {'Yes' if risk['is_anomaly'] else 'No'}")
        lines.append("")
        
        lines.append("Model Results:")
        ae = risk['model_results']['autoencoder']
        lines.append(f"  Autoencoder: score={ae['score']:.4f}, "
                    f"anomaly={ae['is_anomaly']}, threshold={ae['threshold']:.4f}")
        
        if 'isolation_forest' in risk['model_results']:
            if_result = risk['model_results']['isolation_forest']
            lines.append(f"  Isolation Forest: score={if_result['score']:.4f}, "
                        f"anomaly={if_result['is_anomaly']}")
        
        consensus = risk['model_results']['consensus']
        lines.append(f"  Consensus: {consensus['models_agree']} "
                    f"(confidence: {consensus['confidence']})")
        lines.append("")
        
        lines.append("Recommendations:")
        for rec in risk['recommendations']:
            lines.append(f"  • {rec}")
        lines.append("")
    
    # Attribution
    if 'attribution' in result and 'error' not in result['attribution']:
        attr = result['attribution']
        lines.append("ATTRIBUTION ANALYSIS (Cross-Sectional Model Approach)")
        lines.append("-" * 80)
        lines.append(f"Analysis Date: {attr['date']}")
        lines.append(f"Model Type: {attr['model_type']}")
        lines.append("")
        
        lines.append("Portfolio Metrics:")
        lines.append(f"  Structural Score: {attr['structural_score']:.4f}")
        lines.append(f"  Structural Z-Score: {attr['structural_z']:+.3f}")
        lines.append(f"  Directional Score: {attr['directional_score']:+.4f}")
        lines.append(f"  Directional Z-Score: {attr['directional_z']:+.3f}")
        lines.append(f"  Health Score: {attr['health_score']:+.3f}")
        lines.append("")
        
        lines.append("Market Baseline:")
        baseline = attr['market_baseline']
        lines.append(f"  Mean: {baseline['ae_mean']:.4f}")
        lines.append(f"  Std: {baseline['ae_std']:.4f}")
        lines.append(f"  95th percentile: {baseline['ae_p95']:.4f}")
        lines.append(f"  99th percentile: {baseline['ae_p99']:.4f}")
        lines.append("")
        
        lines.append("Top Contributors (by divergence):")
        contributors = attr.get('contributors', [])[:10]
        for c in contributors:
            lines.append(
                f"  {c['symbol']:6s}: weight={c['weight']*100:5.2f}%, "
                f"score={c['ae_score']:7.4f}, contribution={c['contribution']:7.4f}"
            )
        lines.append("")
        
        lines.append("Coverage:")
        cov = attr['coverage']
        lines.append(f"  Requested symbols: {cov['n_requested']}")
        lines.append(f"  Scored symbols: {cov['n_scored']}")
        lines.append(f"  Used for returns: {cov['n_used_returns']}")
        if cov['missing_symbols']:
            lines.append(f"  Missing: {', '.join(cov['missing_symbols'])}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)
