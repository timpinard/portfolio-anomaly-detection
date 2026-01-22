"""
Visualization functions for model training and validation.

This module provides plotting functions for documenting the complete
train/test/validation pipeline.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

def plot_data_split(features_df: pd.DataFrame, train_end_date: str, output_path: Optional[Path] = None):
    """Visualize temporal train/test split."""
    fig, ax = plt.subplots(figsize=(12, 3))
    
    dates = pd.to_datetime(features_df['date']).unique()
    dates = sorted(dates)
    train_end_dt = pd.to_datetime(train_end_date)
    
    train_dates = [d for d in dates if d <= train_end_dt]
    test_dates = [d for d in dates if d > train_end_dt]
    
    if train_dates:
        train_start = train_dates[0]
        train_duration = (train_dates[-1] - train_start).days
        ax.broken_barh([(mdates.date2num(train_start), train_duration)], 
                       (10, 9), facecolors='steelblue', label='Training Data')
    
    if test_dates:
        test_start = test_dates[0]
        test_duration = (test_dates[-1] - test_start).days
        ax.broken_barh([(mdates.date2num(test_start), test_duration)], 
                       (10, 9), facecolors='coral', label='Test/Validation Data')
    
    ax.set_ylim(5, 25)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Training & Validation Data Split', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_training_loss(history: Dict[str, List[float]], output_path: Optional[Path] = None):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'], label='Training Loss', linewidth=2, color='steelblue')
    
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, 
                color='coral', linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Autoencoder Training History', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_training_score_distribution(scores: np.ndarray, threshold: float, 
                                     percentile: float, output_path: Optional[Path] = None):
    """Plot score distribution on training data with threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full distribution
    ax1.hist(scores, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'{percentile}th percentile threshold')
    ax1.set_xlabel('Reconstruction Error (Score)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Training Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')  # Log scale to see tail
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tail only (above 50% of threshold)
    tail_scores = scores[scores > threshold * 0.5]
    if len(tail_scores) > 0:
        ax2.hist(tail_scores, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold: {threshold:.4f}')
        ax2.set_xlabel('Reconstruction Error (Score)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Training Score Distribution - Tail', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No tail data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Training Score Distribution - Tail', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_feature_correlation(features_df: pd.DataFrame, feature_cols: List[str], 
                            output_path: Optional[Path] = None, max_features: int = 50):
    """Plot correlation matrix of features."""
    # Limit features if too many
    if len(feature_cols) > max_features:
        feature_cols = feature_cols[:max_features]
        logger.warning(f"Limiting correlation matrix to {max_features} features")
    
    sample = features_df[feature_cols].sample(min(10000, len(features_df)))
    corr = sample.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax, xticklabels=False, yticklabels=False)
    
    ax.set_title(f'Feature Correlation Matrix ({len(feature_cols)} features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig

def plot_test_score_distribution(test_scores: np.ndarray, threshold: float,
                                 train_mean: Optional[float] = None, 
                                 train_std: Optional[float] = None,
                                 output_path: Optional[Path] = None):
    """Plot test score distribution with reference to training."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    ax.hist(test_scores, bins=100, color='steelblue', alpha=0.7, 
            edgecolor='black', label='Test Scores')
    
    # Threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Anomaly Threshold: {threshold:.4f}')
    
    # Training statistics reference (if provided)
    if train_mean is not None:
        ax.axvline(train_mean, color='green', linestyle=':', linewidth=2,
                   label=f'Training Mean: {train_mean:.4f}')
    
    ax.set_xlabel('Reconstruction Error (Score)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Test Score Distribution', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    stats_text = f'Test Statistics:\n'
    stats_text += f'Mean: {test_scores.mean():.4f}\n'
    stats_text += f'Std: {test_scores.std():.4f}\n'
    stats_text += f'Anomaly Rate: {(test_scores > threshold).mean()*100:.2f}%'
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_score_time_series(test_df: pd.DataFrame, threshold: float, 
                          output_path: Optional[Path] = None):
    """Plot daily mean and max scores over time."""
    daily_stats = test_df.groupby('date')['score'].agg(['mean', 'max', 'std']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Mean score over time
    ax1.plot(daily_stats['date'], daily_stats['mean'], linewidth=2,
             color='steelblue', label='Daily Mean Score')
    ax1.fill_between(daily_stats['date'], 
                     daily_stats['mean'] - daily_stats['std'],
                     daily_stats['mean'] + daily_stats['std'],
                     alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax1.axhline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    ax1.set_ylabel('Mean Score', fontsize=12)
    ax1.set_title('Daily Score Statistics Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Max score over time (most anomalous stock each day)
    ax2.plot(daily_stats['date'], daily_stats['max'], linewidth=2, 
             color='coral', label='Daily Max Score')
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, 
                label='Threshold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Max Score', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_anomaly_rate_by_period(test_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot anomaly detection rate by month."""
    test_df = test_df.copy()
    test_df['year_month'] = test_df['date'].dt.to_period('M')
    
    monthly = test_df.groupby('year_month').agg({
        'is_anomaly': ['sum', 'count', 'mean']
    }).reset_index()
    monthly.columns = ['year_month', 'anomaly_count', 'total_count', 'anomaly_rate']
    monthly['year_month'] = monthly['year_month'].astype(str)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Anomaly count
    ax1.bar(range(len(monthly)), monthly['anomaly_count'], 
            color='coral', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Anomaly Count', fontsize=12)
    ax1.set_title('Monthly Anomaly Detection', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Anomaly rate (percentage)
    ax2.plot(range(len(monthly)), monthly['anomaly_rate'] * 100, 
             marker='o', linewidth=2, markersize=8, color='steelblue')
    ax2.axhline(monthly['anomaly_rate'].mean() * 100, color='red', 
                linestyle='--', linewidth=2, label='Average Rate')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly['year_month'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_persistent_anomalies_heatmap(test_df: pd.DataFrame, top_n: int = 20,
                                     output_path: Optional[Path] = None):
    """Create heatmap showing which stocks are anomalies on which dates."""
    symbol_counts = test_df[test_df['is_anomaly'] == 1].groupby('symbol').size()
    
    if len(symbol_counts) == 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Top {top_n} Persistent Anomalies Over Time', 
                     fontsize=14, fontweight='bold')
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return
    
    top_symbols = symbol_counts.nlargest(top_n).index.tolist()
    
    pivot = test_df[test_df['symbol'].isin(top_symbols)].pivot_table(
        index='symbol', columns='date', values='is_anomaly', 
        aggfunc='max', fill_value=0
    )
    
    if pivot.shape[1] > 60:
        date_step = pivot.shape[1] // 60
        pivot = pivot.iloc[:, ::date_step]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(pivot, cmap=['lightblue', 'darkred'], cbar=False, 
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title(f'Top {top_n} Persistent Anomalies Over Time', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Symbol', fontsize=12)
    
    if pivot.shape[1] > 0:
        ax.set_xticklabels([pd.to_datetime(str(d)).strftime('%Y-%m-%d') 
                           for d in pivot.columns], rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_forward_return_comparison(analysis_df: pd.DataFrame, horizons: List[int] = [1, 5, 20],
                                  output_path: Optional[Path] = None):
    """Compare forward returns for anomaly vs normal groups."""
    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(6*n_horizons, 5))
    
    if n_horizons == 1:
        axes = [axes]
    
    for ax, horizon in zip(axes, horizons):
        col = f'fwd_return_{horizon}d'
        
        if col not in analysis_df.columns:
            ax.text(0.5, 0.5, f'No data for {horizon}d', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{horizon}-Day Forward Returns', fontsize=13, fontweight='bold')
            continue
        
        anomaly_returns = analysis_df[analysis_df['is_anomaly'] == 1][col].dropna()
        normal_returns = analysis_df[analysis_df['is_anomaly'] == 0][col].dropna()
        
        if len(anomaly_returns) == 0 or len(normal_returns) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{horizon}-Day Forward Returns', fontsize=13, fontweight='bold')
            continue
        
        data = [normal_returns * 100, anomaly_returns * 100]
        bp = ax.boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True,
                        showfliers=False, widths=0.6)
        
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Forward Return (%)', fontsize=12)
        ax.set_title(f'{horizon}-Day Forward Returns', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        stats_text = f'Normal: μ={normal_returns.mean()*100:.2f}%, σ={normal_returns.std()*100:.2f}%\n'
        stats_text += f'Anomaly: μ={anomaly_returns.mean()*100:.2f}%, σ={anomaly_returns.std()*100:.2f}%'
        
        ax.text(0.5, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_score_vs_forward_metrics(analysis_df: pd.DataFrame, horizon: int = 5,
                                  output_path: Optional[Path] = None):
    """Scatter plot of score vs forward return magnitude."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    col = f'fwd_return_{horizon}d'
    if col not in analysis_df.columns:
        ax1.text(0.5, 0.5, f'No data for {horizon}d', ha='center', va='center', 
                transform=ax1.transAxes)
        ax2.text(0.5, 0.5, f'No data for {horizon}d', ha='center', va='center', 
                transform=ax2.transAxes)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return
    
    valid_data = analysis_df[['score', col]].dropna()
    if len(valid_data) == 0:
        ax1.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                transform=ax2.transAxes)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return
    
    valid_data = valid_data.copy()
    valid_data['abs_return'] = valid_data[col].abs() * 100
    
    if len(valid_data) > 5000:
        valid_data = valid_data.sample(5000)
    
    scatter = ax1.scatter(valid_data['score'], valid_data['abs_return'], 
                         c=valid_data['score'], cmap='coolwarm', 
                         alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(valid_data['score'], valid_data['abs_return'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(valid_data['score'].min(), valid_data['score'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, 
             label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    corr = valid_data['score'].corr(valid_data['abs_return'])
    
    ax1.set_xlabel('Anomaly Score', fontsize=12)
    ax1.set_ylabel(f'|{horizon}d Forward Return| (%)', fontsize=12)
    ax1.set_title(f'Score vs Forward Return Magnitude (ρ={corr:.3f})', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Score')
    
    try:
        bins = pd.qcut(valid_data['score'], q=10, duplicates='drop')
        binned = valid_data.groupby(bins, observed=True)['abs_return'].mean().reset_index()
        bin_labels = [f'D{i+1}' for i in range(len(binned))]
        
        ax2.bar(range(len(binned)), binned['abs_return'], 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Score Decile (D1=lowest, D10=highest)', fontsize=12)
        ax2.set_ylabel(f'Mean |{horizon}d Forward Return| (%)', fontsize=12)
        ax2.set_title('Average Forward Return by Score Decile', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(binned)))
        ax2.set_xticklabels(bin_labels)
        ax2.grid(True, alpha=0.3, axis='y')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Binning error: {str(e)}', ha='center', va='center', 
                transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_anomalous_reconstructions(model, X_test: np.ndarray, feature_names: List[str],
                                   scores: np.ndarray, threshold: float,
                                   n_examples: int = 3,
                                   output_path: Optional[Path] = None):
    """
    Plot anomalous examples with original vs reconstructed in line plot style.
    
    Similar to the notebook style: shows original (blue), reconstructed (red),
    and filled error area (lightcoral).
    """
    try:
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        if scores.ndim == 0:
            scores = np.array([scores])
        
        anomalous_mask = scores > threshold
        anomalous_indices = np.where(anomalous_mask)[0]
        
        if len(anomalous_indices) == 0:
            anomalous_indices = np.argsort(scores)[-n_examples:]
        else:
            anomalous_scores = scores[anomalous_indices]
            top_anomalous = np.argsort(anomalous_scores)[-n_examples:]
            anomalous_indices = anomalous_indices[top_anomalous]
        
        n_examples = min(n_examples, len(anomalous_indices), len(X_test))
        if n_examples == 0:
            raise ValueError("No examples to plot")
        
        anomalous_indices = anomalous_indices[:n_examples]
        
        import torch
        model.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(model.device)
            reconstructions = model.model(X_tensor).cpu().numpy()
        
        if reconstructions.ndim == 1:
            reconstructions = reconstructions.reshape(1, -1)
        
        fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4*n_examples))
        if n_examples == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, anomalous_indices):
            idx = int(idx)
            
            original = X_test[idx].flatten()
            reconstructed = reconstructions[idx].flatten()
            error = np.abs(original - reconstructed)
            
            max_features = min(len(original), 100)
            if len(original) > max_features:
                # Sample evenly
                indices = np.linspace(0, len(original)-1, max_features, dtype=int)
                original = original[indices]
                reconstructed = reconstructed[indices]
                error = error[indices]
                x = np.arange(max_features)
                labels = [feature_names[i] if i < len(feature_names) else f'F{i}' 
                         for i in indices]
            else:
                x = np.arange(len(original))
                labels = feature_names[:len(original)] if len(feature_names) >= len(original) else [f'F{i}' for i in range(len(original))]
            
            ax.plot(x, original, 'b-', linewidth=2, label='Original', alpha=0.8)
            
            ax.plot(x, reconstructed, 'r-', linewidth=2, label='Reconstruction', alpha=0.8)
            
            ax.fill_between(x, reconstructed, original,
                           where=(original >= reconstructed),
                           color='lightcoral', alpha=0.3, label='Error')
            ax.fill_between(x, reconstructed, original, 
                           where=(original < reconstructed),
                           color='lightcoral', alpha=0.3)
            
            score_val = float(scores[idx])
            
            ax.set_xlabel('Feature Index', fontsize=12)
            ax.set_ylabel('Normalized Value', fontsize=12)
            ax.set_title(f'Anomalous Example (Score: {score_val:.4f}, Threshold: {threshold:.4f})', 
                         fontsize=13, fontweight='bold')
            
            if len(x) <= 30:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            else:
                step = max(1, len(x) // 20)
                ax.set_xticks(x[::step])
                ax.set_xticklabels([labels[i] for i in range(0, len(x), step)], 
                                  rotation=45, ha='right', fontsize=8)
            
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    except Exception as e:
        import traceback
        error_msg = f'Error generating anomalous reconstructions plot: {str(e)}\n{traceback.format_exc()}'
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', 
               transform=ax.transAxes, fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Anomalous Reconstructions Plot Error', fontsize=14, fontweight='bold')
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


def plot_reconstruction_examples(model, X_test: np.ndarray, feature_names: List[str], 
                                indices: Optional[List[int]] = None,
                                output_path: Optional[Path] = None):
    """Plot original vs reconstructed feature vectors for specific examples."""
    try:
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        reconstructions = None
        if hasattr(model, 'predict'):
            try:
                pred = model.predict(X_test)
                if pred is not None and len(pred.shape) == 1:
                    reconstructions = None  # Will use fallback
            except:
                reconstructions = None
        
        if reconstructions is None:
            # Fallback: use model directly to get reconstructions
            import torch
            model.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(model.device)
                reconstructions = model.model(X_tensor).cpu().numpy()
        
        if reconstructions.ndim == 1:
            reconstructions = reconstructions.reshape(1, -1)
        
        scores = model.score(X_test)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        if scores.ndim == 0:
            scores = np.array([scores])
        
        if indices is None:
            if len(scores) > 0:
                high_score_indices = np.argsort(scores)[-3:].tolist()
            else:
                high_score_indices = [0] if len(X_test) > 0 else []
        else:
            high_score_indices = list(indices[:3])
        
        high_score_indices = [idx for idx in high_score_indices if 0 <= idx < len(X_test)]
        if not high_score_indices:
            high_score_indices = [0] if len(X_test) > 0 else []
        
        if not high_score_indices:
            raise ValueError("No valid examples to plot")
        
        n_examples = len(high_score_indices)
        fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4*n_examples))
        
        if n_examples == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, high_score_indices):
            idx = int(idx)
            
            original = X_test[idx]
            reconstructed = reconstructions[idx]
            error = np.abs(original - reconstructed)
            
            if original.ndim > 1:
                original = original.flatten()
            if reconstructed.ndim > 1:
                reconstructed = reconstructed.flatten()
            if error.ndim > 1:
                error = error.flatten()
            
            max_features_to_show = min(len(feature_names), 50)
            if len(feature_names) > max_features_to_show:
                feature_indices = np.linspace(0, len(feature_names)-1, max_features_to_show, dtype=int)
                original = original[feature_indices]
                reconstructed = reconstructed[feature_indices]
                error = error[feature_indices]
                shown_feature_names = [feature_names[i] for i in feature_indices]
            else:
                feature_indices = np.arange(len(feature_names))
                shown_feature_names = feature_names
            
            x = np.arange(len(original))
            width = 0.35
            
            ax.bar(x - width/2, original, width, label='Original', 
                   color='steelblue', alpha=0.7, edgecolor='black')
            ax.bar(x + width/2, reconstructed, width, label='Reconstructed', 
                   color='coral', alpha=0.7, edgecolor='black')
            
            for i, e in enumerate(error):
                if e > 0.5:  # Threshold for significant error
                    ax.plot([i-width/2, i+width/2], [original[i], reconstructed[i]], 
                           'r-', linewidth=2, alpha=0.7)
            
            score_val = float(scores[idx]) if idx < len(scores) else 0.0
            
            ax.set_xlabel('Feature Index', fontsize=12)
            ax.set_ylabel('Normalized Value', fontsize=12)
            ax.set_title(f'Reconstruction Example (Score: {score_val:.4f})', 
                         fontsize=13, fontweight='bold')
            
            if len(x) <= 20:
                ax.set_xticks(x)
                ax.set_xticklabels(shown_feature_names, rotation=45, ha='right', fontsize=8)
            else:
                step = max(1, len(x) // 20)
                ax.set_xticks(x[::step])
                ax.set_xticklabels([shown_feature_names[i] for i in range(0, len(x), step)], 
                                  rotation=45, ha='right', fontsize=8)
            
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    except Exception as e:
        import traceback
        error_msg = f'Error generating reconstruction plot: {str(e)}\n{traceback.format_exc()}'
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', 
               transform=ax.transAxes, fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Reconstruction Plot Error', fontsize=14, fontweight='bold')
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


def create_validation_dashboard(
    test_df: pd.DataFrame, 
    analysis_df: Optional[pd.DataFrame],
    ae_model,
    training_meta: Dict[str, Any],
    forward_horizons: List[int] = [1, 5, 20],
    output_path: Optional[Path] = None
):
    """Create comprehensive validation dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Score distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(test_df['score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(ae_model.threshold, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Score over time (top middle + right)
    ax2 = fig.add_subplot(gs[0, 1:])
    daily_mean = test_df.groupby('date')['score'].mean()
    ax2.plot(daily_mean.index, daily_mean.values, linewidth=2, color='steelblue')
    ax2.axhline(ae_model.threshold, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Daily Mean Score Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Anomaly rate by month (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    monthly = test_df.copy()
    monthly['year_month'] = monthly['date'].dt.to_period('M').astype(str)
    monthly_rate = monthly.groupby('year_month')['is_anomaly'].mean() * 100
    ax3.bar(range(len(monthly_rate)), monthly_rate.values, 
            color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Anomaly Rate (%)')
    ax3.set_title('Monthly Anomaly Rate')
    ax3.set_xticks(range(len(monthly_rate)))
    ax3.set_xticklabels(monthly_rate.index, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Forward returns comparison (middle middle + right)
    ax4 = fig.add_subplot(gs[1, 1:])
    has_data = False
    if analysis_df is not None:
        for horizon in forward_horizons:
            col = f'fwd_return_{horizon}d'
            if col in analysis_df.columns:
                anomaly_returns = analysis_df[analysis_df['is_anomaly'] == 1][col].dropna()
                normal_returns = analysis_df[analysis_df['is_anomaly'] == 0][col].dropna()
                
                if len(anomaly_returns) > 0 and len(normal_returns) > 0:
                    has_data = True
                    positions = [horizon - 0.2, horizon + 0.2]
                    bp = ax4.boxplot([normal_returns * 100, anomaly_returns * 100], 
                                     positions=positions, widths=0.3, 
                                     patch_artist=True, showfliers=False)
                    bp['boxes'][0].set_facecolor('steelblue')
                    bp['boxes'][1].set_facecolor('coral')
    
    if not has_data:
        ax4.text(0.5, 0.5, 'Forward return data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Forward Horizon (days)')
    ax4.set_ylabel('Forward Return (%)')
    ax4.set_title('Forward Returns: Normal vs Anomaly')
    ax4.set_xticks(forward_horizons)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Top persistent anomalies (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    symbol_counts = test_df[test_df['is_anomaly'] == 1].groupby('symbol').size()
    if len(symbol_counts) > 0:
        top_symbols = symbol_counts.nlargest(10)
        ax5.barh(range(len(top_symbols)), top_symbols.values, 
                 color='coral', alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(top_symbols)))
        ax5.set_yticklabels(top_symbols.index)
        ax5.set_xlabel('Anomaly Count')
        ax5.set_title('Top 10 Persistent Anomalies')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
    else:
        ax5.text(0.5, 0.5, 'No anomalies', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Top 10 Persistent Anomalies')
    
    # 6. Score vs forward return correlation (bottom middle + right)
    ax6 = fig.add_subplot(gs[2, 1:])
    has_data_ax6 = False
    if analysis_df is not None:
        for horizon in forward_horizons:
            col = f'fwd_return_{horizon}d'
            if col in analysis_df.columns:
                valid = analysis_df[['score', col]].dropna()
                if len(valid) > 0:
                    valid = valid.copy()
                    valid['abs_return'] = valid[col].abs() * 100
                    
                    try:
                        bins = pd.qcut(valid['score'], q=10, duplicates='drop')
                        binned = valid.groupby(bins, observed=True)['abs_return'].mean()
                        if len(binned) > 0:
                            has_data_ax6 = True
                            ax6.plot(range(len(binned)), binned.values, 
                                    marker='o', linewidth=2, markersize=8, label=f'{horizon}d')
                    except:
                        pass
    
    if not has_data_ax6:
        ax6.text(0.5, 0.5, 'Forward return data not available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    
    ax6.set_xlabel('Score Decile (1=lowest, 10=highest)')
    ax6.set_ylabel('Mean |Forward Return| (%)')
    ax6.set_title('Score vs Forward Return Magnitude by Decile')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    metadata_text = f"Experiment: {training_meta.get('model_type', 'Unknown')}\n"
    metadata_text += f"Training: {training_meta.get('train_start_date', 'N/A')} to {training_meta.get('train_end_date', 'N/A')}\n"
    metadata_text += f"Test: {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}\n"
    metadata_text += f"Threshold: {ae_model.threshold:.4f}\n"
    metadata_text += f"Anomaly Rate: {(test_df['is_anomaly'].mean()*100):.2f}%"
    
    fig.text(0.99, 0.01, metadata_text, fontsize=9, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Validation Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig
