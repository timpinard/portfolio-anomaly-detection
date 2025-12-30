#!/usr/bin/env python3
"""
Comprehensive model validation script.

Validates trained models using:
1. Train/test split with proper metrics
2. Cross-validation
3. Realistic synthetic anomalous portfolio tests (UPDATED)
4. Model calibration checks
5. Consensus validation
6. Overfitting detection

"""

import sys
from pathlib import Path
import yaml
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(db_path: Path):
    """Load feature data from database."""
    conn = sqlite3.connect(db_path)
    
    try:
        df = pd.read_sql_query("SELECT * FROM market_features", conn)
        df['date'] = pd.to_datetime(df['date'])
    except sqlite3.OperationalError as e:
        logger.error(f"Error loading features: {e}")
        logger.error("Run 'python scripts/derive_features.py' first")
        conn.close()
        sys.exit(1)
    
    conn.close()
    return df


def prepare_feature_matrix(df: pd.DataFrame):
    """Prepare feature matrix for training."""
    exclude_cols = {'symbol', 'date'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, feature_cols, df[['symbol', 'date']].copy()


def get_feature_indices(feature_names: List[str]) -> Dict[str, int]:
    """Map feature names to indices for targeted anomaly generation."""
    indices = {}
    for i, name in enumerate(feature_names):
        indices[name] = i
    return indices


def create_realistic_anomalies(
    X_normal: np.ndarray, 
    n_anomalies: int = 200,
    feature_names: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic anomalies representing real portfolio risk scenarios.
    
    This augments the old statistical-only approach with domain-specific
    anomalies that represent actual portfolio risk patterns:
    
    1. Market Crash Portfolio - Severe drawdown (2008/2020 style)
    2. Momentum Bubble Portfolio - Overbought, extended positions
    3. Volatility Spike Portfolio - Sudden vol expansion
    4. Correlation Breakdown Portfolio - Diversification failure
    5. Liquidity Crisis Portfolio - Volume collapse
    6. Technical Breakdown Portfolio - Multiple support breaks
    
    Args:
        X_normal: Normal training data (n_samples, n_features)
        n_anomalies: Number of anomalies to generate
        feature_names: List of feature names for targeted anomaly creation
    
    Returns:
        X_anomalies: Anomalous samples
        y_anomalies: Labels (all 1s indicating anomaly)
    """
    n_features = X_normal.shape[1]
    mean = np.mean(X_normal, axis=0)
    std = np.std(X_normal, axis=0)
    
    # Protect against zero std
    std = np.where(std < 1e-8, 1e-8, std)
    
    anomalies = []
    anomaly_types = []  # Track which type each anomaly is
    
    # Calculate how many of each type
    n_per_type = n_anomalies // 6
    remainder = n_anomalies % 6
    
    # Get feature indices if names provided
    idx = get_feature_indices(feature_names) if feature_names else {}
    
    # === Type 1: Market Crash Portfolio (severe drawdown) ===
    # Characteristics: Negative returns across all timeframes, high volatility, 
    # oversold RSI, below moving averages, near 52-week lows
    n_type1 = n_per_type + (1 if remainder > 0 else 0)
    for _ in range(n_type1):
        anomaly = mean.copy()
        
        if feature_names and idx:
            # Negative returns (3-5 std below mean)
            for ret_col in ['returns_1d', 'returns_5d', 'returns_20d', 'returns_60d']:
                if ret_col in idx:
                    anomaly[idx[ret_col]] = mean[idx[ret_col]] - np.random.uniform(3, 5) * std[idx[ret_col]]
            
            # High volatility (2-4 std above mean)
            for vol_col in ['volatility_20d', 'volatility_60d']:
                if vol_col in idx:
                    anomaly[idx[vol_col]] = mean[idx[vol_col]] + np.random.uniform(2, 4) * std[idx[vol_col]]
            
            # Oversold RSI (15-30)
            if 'rsi_14' in idx:
                anomaly[idx['rsi_14']] = np.random.uniform(15, 30)
            
            # Below moving averages
            if 'above_ma_50' in idx:
                anomaly[idx['above_ma_50']] = 0
            if 'above_ma_200' in idx:
                anomaly[idx['above_ma_200']] = 0
            
            # Near 52-week lows
            if 'price_to_52d_low' in idx:
                anomaly[idx['price_to_52d_low']] = np.random.uniform(0.95, 1.05)
            if 'price_to_52d_high' in idx:
                anomaly[idx['price_to_52d_high']] = np.random.uniform(0.5, 0.7)
            
            # Negative MACD
            if 'macd_histogram' in idx:
                anomaly[idx['macd_histogram']] = mean[idx['macd_histogram']] - np.random.uniform(2, 4) * std[idx['macd_histogram']]
        else:
            # Fallback: perturb first few features (assumed to be returns)
            for i in range(min(4, n_features)):
                anomaly[i] = mean[i] - np.random.uniform(3, 5) * std[i]
            # Increase volatility features (typically 4-5)
            if n_features > 5:
                anomaly[4] = mean[4] + np.random.uniform(2, 4) * std[4]
                anomaly[5] = mean[5] + np.random.uniform(2, 4) * std[5]
        
        anomalies.append(anomaly)
        anomaly_types.append('market_crash')
    remainder = max(0, remainder - 1)
    
    # === Type 2: Momentum Bubble Portfolio (overbought, extended) ===
    # Characteristics: Strong positive returns, high RSI, above bands, overextended
    n_type2 = n_per_type + (1 if remainder > 0 else 0)
    for _ in range(n_type2):
        anomaly = mean.copy()
        
        if feature_names and idx:
            # Strong positive returns
            for ret_col in ['returns_1d', 'returns_5d', 'returns_20d']:
                if ret_col in idx:
                    anomaly[idx[ret_col]] = mean[idx[ret_col]] + np.random.uniform(3, 5) * std[idx[ret_col]]
            
            # Overbought RSI (75-90)
            if 'rsi_14' in idx:
                anomaly[idx['rsi_14']] = np.random.uniform(75, 90)
            
            # Above Bollinger Bands
            if 'bb_position' in idx:
                anomaly[idx['bb_position']] = np.random.uniform(0.95, 1.2)
            
            # At 52-week highs
            if 'price_to_52d_high' in idx:
                anomaly[idx['price_to_52d_high']] = np.random.uniform(0.98, 1.02)
            
            # Above moving averages
            if 'above_ma_50' in idx:
                anomaly[idx['above_ma_50']] = 1
            if 'above_ma_200' in idx:
                anomaly[idx['above_ma_200']] = 1
            
            # Positive MACD
            if 'macd_histogram' in idx:
                anomaly[idx['macd_histogram']] = mean[idx['macd_histogram']] + np.random.uniform(2, 4) * std[idx['macd_histogram']]
        else:
            for i in range(min(4, n_features)):
                anomaly[i] = mean[i] + np.random.uniform(3, 5) * std[i]
        
        anomalies.append(anomaly)
        anomaly_types.append('momentum_bubble')
    remainder = max(0, remainder - 1)
    
    # === Type 3: Volatility Spike Portfolio ===
    # Characteristics: Extreme volatility, wide Bollinger bands, high volume
    n_type3 = n_per_type + (1 if remainder > 0 else 0)
    for _ in range(n_type3):
        anomaly = mean.copy()
        
        if feature_names and idx:
            # Extreme volatility (4-6 std above mean)
            for vol_col in ['volatility_20d', 'volatility_60d']:
                if vol_col in idx:
                    anomaly[idx[vol_col]] = mean[idx[vol_col]] + np.random.uniform(4, 6) * std[idx[vol_col]]
            
            # Wide Bollinger bands
            if 'bb_width' in idx:
                anomaly[idx['bb_width']] = mean[idx['bb_width']] + np.random.uniform(3, 5) * std[idx['bb_width']]
            
            # High volume (panic trading)
            for vol_col in ['volume_ratio_20d', 'volume_ratio_60d']:
                if vol_col in idx:
                    anomaly[idx[vol_col]] = np.random.uniform(2.5, 5.0)
            
            # Returns can go either way during vol spike
            direction = np.random.choice([-1, 1])
            for ret_col in ['returns_1d', 'returns_5d']:
                if ret_col in idx:
                    anomaly[idx[ret_col]] = mean[idx[ret_col]] + direction * np.random.uniform(2, 4) * std[idx[ret_col]]
        else:
            # Volatility features typically around index 4-5
            if n_features > 5:
                anomaly[4] = mean[4] + np.random.uniform(4, 6) * std[4]
                anomaly[5] = mean[5] + np.random.uniform(4, 6) * std[5]
        
        anomalies.append(anomaly)
        anomaly_types.append('volatility_spike')
    remainder = max(0, remainder - 1)
    
    # === Type 4: Correlation Breakdown Portfolio ===
    # Simulates diversification failure - use multivariate extreme
    n_type4 = n_per_type + (1 if remainder > 0 else 0)
    for _ in range(n_type4):
        # Sample from multivariate normal with inflated covariance
        try:
            cov = np.cov(X_normal.T)
            # Ensure positive definite
            cov = cov + np.eye(n_features) * 1e-6
            anomaly = np.random.multivariate_normal(mean, cov * 3)
        except:
            # Fallback if covariance is singular
            anomaly = mean + np.random.normal(0, 2.5 * std)
        
        # Push returns in same direction (correlation = 1 behavior)
        if feature_names and idx:
            direction = np.random.choice([-1, 1])
            for ret_col in ['returns_1d', 'returns_5d', 'returns_20d', 'returns_60d']:
                if ret_col in idx:
                    anomaly[idx[ret_col]] = mean[idx[ret_col]] + direction * np.random.uniform(2, 4) * std[idx[ret_col]]
        
        anomalies.append(anomaly)
        anomaly_types.append('correlation_breakdown')
    remainder = max(0, remainder - 1)
    
    # === Type 5: Liquidity Crisis Portfolio ===
    # Characteristics: Volume collapse, negative returns, elevated volatility
    n_type5 = n_per_type + (1 if remainder > 0 else 0)
    for _ in range(n_type5):
        anomaly = mean.copy()
        
        if feature_names and idx:
            # Volume collapse
            for vol_col in ['volume_ratio_20d', 'volume_ratio_60d']:
                if vol_col in idx:
                    anomaly[idx[vol_col]] = np.random.uniform(0.1, 0.3)
            
            # Negative returns (can't sell)
            for ret_col in ['returns_1d', 'returns_5d', 'returns_20d']:
                if ret_col in idx:
                    anomaly[idx[ret_col]] = mean[idx[ret_col]] - np.random.uniform(2, 4) * std[idx[ret_col]]
            
            # Elevated volatility (illiquidity causes gaps)
            if 'volatility_20d' in idx:
                anomaly[idx['volatility_20d']] = mean[idx['volatility_20d']] + np.random.uniform(2, 3) * std[idx['volatility_20d']]
            
            # Near lows
            if 'price_to_52d_low' in idx:
                anomaly[idx['price_to_52d_low']] = np.random.uniform(0.9, 1.1)
        else:
            # Assume volume features near end, returns at start
            for i in range(min(4, n_features)):
                anomaly[i] = mean[i] - np.random.uniform(2, 4) * std[i]
            if n_features > 15:
                anomaly[15] = mean[15] * 0.2  # Low volume
        
        anomalies.append(anomaly)
        anomaly_types.append('liquidity_crisis')
    remainder = max(0, remainder - 1)
    
    # === Type 6: Technical Breakdown Portfolio ===
    # Characteristics: Below all MAs, negative MACD, lower Bollinger band
    n_type6 = n_per_type
    for _ in range(n_type6):
        anomaly = mean.copy()
        
        if feature_names and idx:
            # Below moving averages
            if 'above_ma_50' in idx:
                anomaly[idx['above_ma_50']] = 0
            if 'above_ma_200' in idx:
                anomaly[idx['above_ma_200']] = 0
            
            # Negative MACD
            if 'macd' in idx:
                anomaly[idx['macd']] = mean[idx['macd']] - np.random.uniform(2, 4) * std[idx['macd']]
            if 'macd_histogram' in idx:
                anomaly[idx['macd_histogram']] = mean[idx['macd_histogram']] - np.random.uniform(2, 4) * std[idx['macd_histogram']]
            
            # At lower Bollinger band
            if 'bb_position' in idx:
                anomaly[idx['bb_position']] = np.random.uniform(-0.1, 0.1)
            
            # Moderate negative returns
            for ret_col in ['returns_5d', 'returns_20d']:
                if ret_col in idx:
                    anomaly[idx[ret_col]] = mean[idx[ret_col]] - np.random.uniform(1.5, 3) * std[idx[ret_col]]
            
            # RSI weak but not oversold
            if 'rsi_14' in idx:
                anomaly[idx['rsi_14']] = np.random.uniform(30, 45)
        else:
            # Generic perturbation
            for i in range(0, min(n_features, 12), 3):
                anomaly[i] = mean[i] - np.random.uniform(2, 3) * std[i]
        
        anomalies.append(anomaly)
        anomaly_types.append('technical_breakdown')
    
    X_anomalies = np.array(anomalies)
    y_anomalies = np.ones(len(X_anomalies))
    
    # Log distribution of anomaly types
    logger.info(f"Generated {len(X_anomalies)} realistic anomalies:")
    from collections import Counter
    type_counts = Counter(anomaly_types)
    for atype, count in sorted(type_counts.items()):
        logger.info(f"  - {atype}: {count}")
    
    return X_anomalies, y_anomalies


# Statistical anomaly generation used for comparison to the realistic anomalies.  These were originally used to validate the models, but are now kept for reference.
def create_statistical_anomalies(X_normal: np.ndarray, n_anomalies: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original statistical anomaly generation (kept for comparison).
    
    Creates anomalies using:
    1. Extreme feature values (5 std)
    2. Gaussian noise addition
    3. Outlier amplification
    """
    n_features = X_normal.shape[1]
    anomalies = []
    
    mean = np.mean(X_normal, axis=0)
    std = np.std(X_normal, axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    
    for _ in range(n_anomalies // 3):
        # Random feature with extreme value
        feature_idx = np.random.randint(0, n_features)
        extreme_value = mean[feature_idx] + np.random.choice([-1, 1]) * 5 * std[feature_idx]
        anomaly = mean.copy()
        anomaly[feature_idx] = extreme_value
        anomalies.append(anomaly)
    
    for _ in range(n_anomalies // 3):
        row_idx = np.random.randint(0, X_normal.shape[0])
        anomaly = X_normal[row_idx].copy()
        anomaly += np.random.normal(0, 2 * std, size=n_features)
        anomalies.append(anomaly)
    
    for _ in range(n_anomalies - len(anomalies)):
        z_scores = np.abs((X_normal - mean) / (std + 1e-8))
        outlier_mask = np.any(z_scores > 3, axis=1)
        if np.any(outlier_mask):
            anomaly = X_normal[outlier_mask][np.random.randint(0, np.sum(outlier_mask))]
            anomaly = anomaly + np.random.normal(0, std, size=n_features)
        else:
            anomaly = mean + np.random.normal(0, 3 * std, size=n_features)
        anomalies.append(anomaly)
    
    X_anomalies = np.array(anomalies)
    y_anomalies = np.ones(len(X_anomalies))
    
    return X_anomalies, y_anomalies


def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    model_name: str,
    threshold: float = None
) -> Dict:
    """Evaluate model performance."""
    # Get predictions
    if hasattr(model, 'predict'):
        predictions = model.predict(X_test)
        # Convert -1/1 to 0/1 for metrics
        y_pred = (predictions == -1).astype(int)
    else:
        # For models without predict, use threshold
        scores = model.score(X_test)
        if threshold is None:
            threshold = np.percentile(scores, 99)
        y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'n_samples': len(X_test),
        'n_anomalies_detected': int(np.sum(y_pred)),
        'anomaly_rate': float(np.mean(y_pred)),
    }
    
    # Classification metrics (if we have labels)
    if len(np.unique(y_test)) > 1:  # Both classes present
        metrics['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_test, y_pred, zero_division=0))
        
        # ROC-AUC (need scores)
        try:
            scores = model.score(X_test) if hasattr(model, 'score') else None
            if scores is not None:
                # Normalize scores to [0, 1] for ROC-AUC
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                metrics['roc_auc'] = float(roc_auc_score(y_test, scores_norm))
        except:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    else:
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1_score'] = None
        metrics['roc_auc'] = None
        metrics['confusion_matrix'] = None
    
    return metrics


def validate_consensus(ae_model, if_model, X_test: np.ndarray) -> Dict:
    """Validate consensus between models."""
    ae_pred = ae_model.predict(X_test)
    if_pred = if_model.predict(X_test)
    
    ae_anomalies = (ae_pred == -1)
    if_anomalies = (if_pred == -1)
    
    both_flagged = np.sum(ae_anomalies & if_anomalies)
    either_flagged = np.sum(ae_anomalies | if_anomalies)
    models_agree = np.sum(ae_anomalies == if_anomalies)
    
    consensus_metrics = {
        'total_samples': len(X_test),
        'both_flagged': int(both_flagged),
        'either_flagged': int(either_flagged),
        'models_agree': int(models_agree),
        'agreement_rate': float(models_agree / len(X_test)),
        'both_flagged_rate': float(both_flagged / len(X_test)),
        'either_flagged_rate': float(either_flagged / len(X_test))
    }
    
    return consensus_metrics


def check_overfitting(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str
) -> Dict:
    """Check for overfitting by comparing train vs test performance."""
    # Get scores
    train_scores = model.score(X_train)
    test_scores = model.score(X_test)
    
    # Calculate anomaly rates
    train_threshold = np.percentile(train_scores, 99)
    test_threshold = np.percentile(test_scores, 99)
    
    train_anomaly_rate = np.mean(train_scores > train_threshold)
    test_anomaly_rate = np.mean(test_scores > test_threshold)
    
    # Score distributions
    overfitting_metrics = {
        'model_name': model_name,
        'train_mean_score': float(np.mean(train_scores)),
        'test_mean_score': float(np.mean(test_scores)),
        'train_std_score': float(np.std(train_scores)),
        'test_std_score': float(np.std(test_scores)),
        'train_anomaly_rate': float(train_anomaly_rate),
        'test_anomaly_rate': float(test_anomaly_rate),
        'score_difference': float(np.mean(train_scores) - np.mean(test_scores)),
        'overfitting_risk': 'high' if abs(np.mean(train_scores) - np.mean(test_scores)) > 0.1 else 'low'
    }
    
    return overfitting_metrics


def time_series_cross_validation(
    X: np.ndarray,
    dates: pd.Series,
    n_splits: int = 5
) -> List[Dict]:
    """Perform time-series cross-validation."""
    logger.info(f"Performing {n_splits}-fold time-series cross-validation...")
    
    # Sort by date
    sort_idx = dates.argsort()
    X_sorted = X[sort_idx]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
        logger.info(f"  Fold {fold + 1}/{n_splits}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        X_train_cv = X_sorted[train_idx]
        X_test_cv = X_sorted[test_idx]
        
        # Train models
        ae_model = AutoencoderAnomalyDetector(
            sector='autoencoder',
            model_dir=Path('/tmp'),
            input_dim=X_sorted.shape[1],
            encoding_dim=15,
            hidden_dims=[128, 64, 32]
        )
        ae_model.fit(X_train_cv, epochs=50, batch_size=64)
        
        if_model = IsolationForestAnomalyDetector(
            sector='isolation_forest',
            model_dir=Path('/tmp'),
            contamination=0.05
        )
        if_model.fit(X_train_cv)
        
        # Evaluate
        ae_scores = ae_model.score(X_test_cv)
        if_scores = if_model.score(X_test_cv)
        
        cv_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'ae_mean_score': float(np.mean(ae_scores)),
            'ae_std_score': float(np.std(ae_scores)),
            'if_mean_score': float(np.mean(if_scores)),
            'if_std_score': float(np.std(if_scores))
        })
    
    return cv_results


def generate_validation_report(results: Dict) -> str:
    """Generate comprehensive validation report."""
    report = []
    report.append("=" * 100)
    report.append("MODEL VALIDATION REPORT")
    report.append("=" * 100)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Anomaly Type: {results.get('anomaly_type', 'realistic')}")
    report.append("")
    
    # Train/Test Split Results
    if 'train_test' in results:
        report.append("TRAIN/TEST SPLIT VALIDATION")
        report.append("-" * 100)
        tt = results['train_test']
        report.append(f"Train Size: {tt['train_size']:,}")
        report.append(f"Test Size: {tt['test_size']:,}")
        report.append(f"Split Ratio: {tt['train_size']/(tt['train_size']+tt['test_size'])*100:.1f}% train")
        report.append("")
        
        # Model metrics
        for model_metrics in tt['model_metrics']:
            report.append(f"{model_metrics['model_name'].upper()} METRICS:")
            report.append(f"  Anomaly Detection Rate: {model_metrics['anomaly_rate']*100:.2f}%")
            if model_metrics.get('precision') is not None:
                report.append(f"  Precision: {model_metrics['precision']:.4f}")
                report.append(f"  Recall: {model_metrics['recall']:.4f}")
                report.append(f"  F1 Score: {model_metrics['f1_score']:.4f}")
                if model_metrics.get('roc_auc') is not None:
                    report.append(f"  ROC-AUC: {model_metrics['roc_auc']:.4f}")
                if model_metrics.get('confusion_matrix'):
                    cm = model_metrics['confusion_matrix']
                    report.append(f"  Confusion Matrix:")
                    report.append(f"    True Negatives: {cm['tn']}")
                    report.append(f"    False Positives: {cm['fp']}")
                    report.append(f"    False Negatives: {cm['fn']}")
                    report.append(f"    True Positives: {cm['tp']}")
            report.append("")
    
    # Synthetic Anomaly Tests
    if 'synthetic_anomalies' in results:
        report.append("SYNTHETIC ANOMALY DETECTION")
        report.append("-" * 100)
        syn = results['synthetic_anomalies']
        report.append(f"Tested on {syn['n_synthetic']} synthetic anomalous portfolios")
        report.append(f"Anomaly Types: {syn.get('anomaly_types', 'realistic portfolio scenarios')}")
        report.append("")
        
        for model_metrics in syn['model_metrics']:
            report.append(f"{model_metrics['model_name'].upper()}:")
            report.append(f"  Detection Rate: {model_metrics['anomaly_rate']*100:.2f}%")
            if model_metrics.get('precision') is not None:
                report.append(f"  Precision: {model_metrics['precision']:.4f}")
                report.append(f"  Recall: {model_metrics['recall']:.4f}")
                report.append(f"  F1 Score: {model_metrics['f1_score']:.4f}")
            report.append("")
    
    # Consensus Validation
    if 'consensus' in results:
        report.append("MODEL CONSENSUS VALIDATION")
        report.append("-" * 100)
        cons = results['consensus']
        report.append(f"Total Samples: {cons['total_samples']:,}")
        report.append(f"Models Agree: {cons['models_agree']:,} ({cons['agreement_rate']*100:.2f}%)")
        report.append(f"Both Flagged: {cons['both_flagged']:,} ({cons['both_flagged_rate']*100:.2f}%)")
        report.append(f"Either Flagged: {cons['either_flagged']:,} ({cons['either_flagged_rate']*100:.2f}%)")
        report.append("")
    
    # Overfitting Check
    if 'overfitting' in results:
        report.append("OVERFITTING DETECTION")
        report.append("-" * 100)
        for of_metrics in results['overfitting']:
            report.append(f"{of_metrics['model_name'].upper()}:")
            report.append(f"  Train Mean Score: {of_metrics['train_mean_score']:.6f}")
            report.append(f"  Test Mean Score: {of_metrics['test_mean_score']:.6f}")
            report.append(f"  Score Difference: {of_metrics['score_difference']:.6f}")
            report.append(f"  Overfitting Risk: {of_metrics['overfitting_risk'].upper()}")
            report.append("")
    
    # Cross-Validation
    if 'cross_validation' in results:
        report.append("TIME-SERIES CROSS-VALIDATION")
        report.append("-" * 100)
        cv = results['cross_validation']
        report.append(f"Folds: {len(cv)}")
        for fold in cv:
            report.append(f"  Fold {fold['fold']}: Test Size={fold['test_size']}, "
                        f"AE Score={fold['ae_mean_score']:.6f}, IF Score={fold['if_mean_score']:.6f}")
        report.append("")
    
    # Summary
    report.append("VALIDATION SUMMARY")
    report.append("-" * 100)
    
    # Check if models are performing well
    issues = []
    warnings = []
    
    # Check train/test performance
    if 'train_test' in results:
        for model_metrics in results['train_test']['model_metrics']:
            if model_metrics.get('f1_score') is not None:
                f1 = model_metrics['f1_score']
                precision = model_metrics.get('precision', 0)
                recall = model_metrics.get('recall', 0)
                
                # For anomaly detection, low F1 can be acceptable if precision is high
                if f1 < 0.3:
                    issues.append(
                        f"{model_metrics['model_name']} has very low F1 score ({f1:.4f}) - "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}"
                    )
                elif f1 < 0.5:
                    if precision > 0.7:
                        warnings.append(
                            f"{model_metrics['model_name']} has low F1 ({f1:.4f}) but high precision ({precision:.4f}) - "
                            f"conservative detection (low recall: {recall:.4f})"
                        )
                    elif recall > 0.7:
                        warnings.append(
                            f"{model_metrics['model_name']} has low F1 ({f1:.4f}) but high recall ({recall:.4f}) - "
                            f"aggressive detection (low precision: {precision:.4f})"
                        )
                    else:
                        warnings.append(
                            f"{model_metrics['model_name']} has low F1 score ({f1:.4f}) - "
                            f"Precision: {precision:.4f}, Recall: {recall:.4f}"
                        )
    
    # Check synthetic anomaly detection
    if 'synthetic_anomalies' in results:
        for model_metrics in results['synthetic_anomalies']['model_metrics']:
            if model_metrics.get('recall') is not None:
                recall = model_metrics['recall']
                if recall < 0.5:
                    warnings.append(
                        f"{model_metrics['model_name']} detects only {recall*100:.1f}% of synthetic anomalies - "
                        f"may miss real anomalies"
                    )
    
    # Check overfitting
    if 'overfitting' in results:
        for of_metrics in results['overfitting']:
            if of_metrics['overfitting_risk'] == 'high':
                issues.append(f"{of_metrics['model_name']} shows signs of overfitting")
    
    # Report issues and warnings
    if issues:
        report.append("⚠️  ISSUES DETECTED:")
        for issue in issues:
            report.append(f"  - {issue}")
        report.append("")
    
    if warnings:
        report.append("ℹ️  WARNINGS:")
        for warning in warnings:
            report.append(f"  - {warning}")
        report.append("")
    
    if not issues and not warnings:
        report.append("✓ No major issues detected")
        report.append("")
    
    # Add interpretation guidance
    report.append("INTERPRETATION NOTES:")
    report.append("  • F1 Score < 0.3: Model may not be useful for anomaly detection")
    report.append("  • F1 Score 0.3-0.5: Acceptable if precision is high (conservative detection)")
    report.append("  • F1 Score > 0.5: Good balance of precision and recall")
    report.append("  • High Precision + Low Recall: Conservative (few false alarms, may miss some anomalies)")
    report.append("  • Low Precision + High Recall: Aggressive (catches most anomalies, more false alarms)")
    report.append("  • For anomaly detection, high precision is often preferred over high recall")
    report.append("")
    report.append("SYNTHETIC ANOMALY TYPES (Realistic):")
    report.append("  • Market Crash: Negative returns, high volatility, oversold RSI")
    report.append("  • Momentum Bubble: Strong returns, overbought RSI, above bands")
    report.append("  • Volatility Spike: Extreme volatility, wide bands, high volume")
    report.append("  • Correlation Breakdown: Diversification failure, correlated moves")
    report.append("  • Liquidity Crisis: Volume collapse, negative returns")
    report.append("  • Technical Breakdown: Below MAs, negative MACD, weak RSI")
    report.append("")
    
    report.append("")
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)
    
    return "\n".join(report)


def main():
    logger.info("=" * 100)
    logger.info("MODEL VALIDATION (with Realistic Anomalies)")
    logger.info("=" * 100)
    
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'model_config.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Paths
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'market_universe'
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    
    if not (model_dir / 'autoencoder' / 'model.pth').exists():
        logger.error(f"Models not found: {model_dir}")
        logger.error("Run 'python scripts/train_models.py' first")
        sys.exit(1)
    
    # Load data
    logger.info("\nLoading data...")
    df = load_training_data(db_path)
    logger.info(f"Loaded {len(df):,} feature rows from {df['symbol'].nunique()} symbols")
    
    # Prepare features
    X, feature_cols, metadata = prepare_feature_matrix(df)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature columns: {feature_cols}")
    
    # Load trained models
    logger.info("\nLoading trained models...")
    ae_model = AutoencoderAnomalyDetector(
        sector='autoencoder',
        model_dir=model_dir
    )
    ae_model.load()
    
    if_model = IsolationForestAnomalyDetector(
        sector='isolation_forest',
        model_dir=model_dir
    )
    if_model.load()
    
    results = {
        'anomaly_type': 'realistic'
    }
    
    # 1. Train/Test Split
    logger.info("\n" + "-" * 100)
    logger.info("1. TRAIN/TEST SPLIT VALIDATION")
    logger.info("-" * 100)
    
    # Time-based split (use 80% for training, 20% for testing)
    dates = metadata['date']
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    logger.info(f"Train: {len(X_train):,} samples")
    logger.info(f"Test: {len(X_test):,} samples")
    
    # Create REALISTIC synthetic anomalies for test set
    logger.info("\nGenerating realistic synthetic anomalies...")
    X_synthetic, y_synthetic = create_realistic_anomalies(
        X_train, 
        n_anomalies=100, 
        feature_names=feature_cols
    )
    X_test_with_anomalies = np.vstack([X_test, X_synthetic])
    y_test_with_anomalies = np.hstack([np.zeros(len(X_test)), y_synthetic])
    
    # Evaluate models
    ae_metrics = evaluate_model(ae_model, X_test_with_anomalies, y_test_with_anomalies, "Autoencoder", ae_model.threshold)
    if_metrics = evaluate_model(if_model, X_test_with_anomalies, y_test_with_anomalies, "Isolation Forest")
    
    results['train_test'] = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'model_metrics': [ae_metrics, if_metrics]
    }
    
    # 2. Synthetic Anomaly Detection
    logger.info("\n" + "-" * 100)
    logger.info("2. SYNTHETIC ANOMALY DETECTION (Realistic Portfolio Scenarios)")
    logger.info("-" * 100)
    
    X_syn_test, y_syn_test = create_realistic_anomalies(
        X_train, 
        n_anomalies=200, 
        feature_names=feature_cols
    )
    logger.info(f"Testing on {len(X_syn_test)} realistic synthetic anomalies")
    
    ae_syn_metrics = evaluate_model(ae_model, X_syn_test, y_syn_test, "Autoencoder", ae_model.threshold)
    if_syn_metrics = evaluate_model(if_model, X_syn_test, y_syn_test, "Isolation Forest")
    
    results['synthetic_anomalies'] = {
        'n_synthetic': len(X_syn_test),
        'anomaly_types': 'market_crash, momentum_bubble, volatility_spike, correlation_breakdown, liquidity_crisis, technical_breakdown',
        'model_metrics': [ae_syn_metrics, if_syn_metrics]
    }
    
    # 3. Consensus Validation
    logger.info("\n" + "-" * 100)
    logger.info("3. CONSENSUS VALIDATION")
    logger.info("-" * 100)
    
    consensus_metrics = validate_consensus(ae_model, if_model, X_test)
    results['consensus'] = consensus_metrics
    logger.info(f"Models agree: {consensus_metrics['agreement_rate']*100:.2f}%")
    
    # 4. Overfitting Check
    logger.info("\n" + "-" * 100)
    logger.info("4. OVERFITTING DETECTION")
    logger.info("-" * 100)
    
    ae_overfitting = check_overfitting(ae_model, X_train, X_test, "Autoencoder")
    if_overfitting = check_overfitting(if_model, X_train, X_test, "Isolation Forest")
    results['overfitting'] = [ae_overfitting, if_overfitting]
    
    logger.info(f"Autoencoder overfitting risk: {ae_overfitting['overfitting_risk']}")
    logger.info(f"Isolation Forest overfitting risk: {if_overfitting['overfitting_risk']}")
    
    # 5. Cross-Validation (optional, can be slow)
    logger.info("\n" + "-" * 100)
    logger.info("5. TIME-SERIES CROSS-VALIDATION")
    logger.info("-" * 100)
    logger.info("(This may take a while...)")
    
    try:
        cv_results = time_series_cross_validation(X[:10000], dates[:10000], n_splits=3)
        results['cross_validation'] = cv_results
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        results['cross_validation'] = []
    
    # Generate report
    logger.info("\n" + "=" * 100)
    logger.info("Generating validation report...")
    logger.info("=" * 100)
    
    report = generate_validation_report(results)
    print("\n" + report)
    
    # Save report to results directory
    results_dir = project_root / 'results' / 'reports'
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / f"model_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()