#!/usr/bin/env python3
"""
Visual Diagnostic: How are the models scoring data points?

This creates visualizations to understand:
1. Score distributions - is there natural separation?
2. Feature space - what do anomalies look like?
3. Time series - when do anomalies occur?
4. Model agreement - where do models agree/disagree?

Run this and examine the generated plots to validate model behavior.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector


def load_data_and_score(project_root: Path):
    """Load data and get model scores."""
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'market_universe'
    
    # Load features
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM market_features", conn)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    conn.close()
    
    # Prepare features
    exclude_cols = {'symbol', 'date'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Load and score with both models
    ae_model = AutoencoderAnomalyDetector(sector='autoencoder', model_dir=model_dir)
    ae_model.load()
    
    if_model = IsolationForestAnomalyDetector(sector='isolation_forest', model_dir=model_dir)
    if_model.load()
    
    # Get scores and predictions
    df['ae_score'] = ae_model.score(X)
    df['ae_anomaly'] = ae_model.predict(X) == -1
    df['ae_threshold'] = ae_model.threshold
    
    df['if_score'] = if_model.score(X)
    df['if_anomaly'] = if_model.predict(X) == -1
    
    return df, feature_cols, ae_model, if_model


def plot_score_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Score distributions with thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AE score histogram
    ax = axes[0, 0]
    ax.hist(df['ae_score'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(df['ae_threshold'].iloc[0], color='red', linestyle='--', linewidth=2, label=f'Threshold: {df["ae_threshold"].iloc[0]:.4f}')
    ax.set_xlabel('Autoencoder Score (Reconstruction Error)')
    ax.set_ylabel('Count')
    ax.set_title('Autoencoder Score Distribution')
    ax.legend()
    ax.set_yscale('log')
    
    # AE score - zoomed on tail
    ax = axes[0, 1]
    threshold = df['ae_threshold'].iloc[0]
    tail_data = df[df['ae_score'] > threshold * 0.5]['ae_score']
    ax.hist(tail_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Autoencoder Score')
    ax.set_ylabel('Count')
    ax.set_title('Autoencoder Score - Tail (>50% of threshold)')
    ax.legend()
    
    # IF score histogram
    ax = axes[1, 0]
    ax.hist(df['if_score'], bins=100, alpha=0.7, color='forestgreen', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Threshold (0)')
    ax.set_xlabel('Isolation Forest Score (higher = more anomalous)')
    ax.set_ylabel('Count')
    ax.set_title('Isolation Forest Score Distribution')
    ax.legend()
    ax.set_yscale('log')
    
    # Score correlation
    ax = axes[1, 1]
    # Sample for performance
    sample_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
    sample = df.iloc[sample_idx]
    
    colors = ['red' if (a or b) else 'steelblue' 
              for a, b in zip(sample['ae_anomaly'], sample['if_anomaly'])]
    ax.scatter(sample['ae_score'], sample['if_score'], c=colors, alpha=0.3, s=10)
    ax.axvline(df['ae_threshold'].iloc[0], color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Autoencoder Score')
    ax.set_ylabel('Isolation Forest Score')
    ax.set_title('Model Score Correlation (red = flagged by either)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'score_distributions.png'}")


def plot_anomalies_over_time(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Anomaly scores over time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Aggregate by date (mean score per day)
    daily = df.groupby('date').agg({
        'ae_score': 'mean',
        'if_score': 'mean',
        'ae_anomaly': 'sum',
        'if_anomaly': 'sum',
        'symbol': 'count'
    }).rename(columns={'symbol': 'n_samples'})
    
    daily['ae_anomaly_rate'] = daily['ae_anomaly'] / daily['n_samples']
    daily['if_anomaly_rate'] = daily['if_anomaly'] / daily['n_samples']
    
    # AE score over time
    ax = axes[0]
    ax.plot(daily.index, daily['ae_score'], color='steelblue', alpha=0.7, linewidth=0.5)
    ax.axhline(df['ae_threshold'].iloc[0], color='red', linestyle='--', label='Threshold')
    
    # Highlight high-anomaly periods
    high_anomaly_dates = daily[daily['ae_anomaly_rate'] > 0.1].index
    for date in high_anomaly_dates:
        ax.axvline(date, color='red', alpha=0.1)
    
    ax.set_ylabel('Mean AE Score')
    ax.set_title('Autoencoder: Daily Mean Score Over Time')
    ax.legend()
    
    # IF score over time
    ax = axes[1]
    ax.plot(daily.index, daily['if_score'], color='forestgreen', alpha=0.7, linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', label='Threshold')
    ax.set_ylabel('Mean IF Score')
    ax.set_title('Isolation Forest: Daily Mean Score Over Time')
    ax.legend()
    
    # Anomaly rate over time
    ax = axes[2]
    ax.bar(daily.index, daily['ae_anomaly_rate'] * 100, alpha=0.5, color='steelblue', label='AE')
    ax.bar(daily.index, daily['if_anomaly_rate'] * 100, alpha=0.5, color='forestgreen', label='IF')
    ax.set_ylabel('Anomaly Rate (%)')
    ax.set_xlabel('Date')
    ax.set_title('Daily Anomaly Rate by Model')
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomalies_over_time.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'anomalies_over_time.png'}")


def plot_feature_comparison(df: pd.DataFrame, feature_cols: list, output_dir: Path):
    """Plot 3: Feature distributions for anomalies vs normal."""
    # Select key features
    key_features = [
        'returns_1d', 'returns_5d', 'returns_20d',
        'volatility_20d', 'rsi_14', 'bb_position',
        'volume_ratio_20d', 'macd_histogram'
    ]
    key_features = [f for f in key_features if f in feature_cols][:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    normal = df[~df['ae_anomaly']]
    anomaly = df[df['ae_anomaly']]
    
    for i, feat in enumerate(key_features):
        ax = axes[i]
        
        # Plot distributions
        ax.hist(normal[feat], bins=50, alpha=0.5, color='steelblue', 
                label=f'Normal (n={len(normal)})', density=True)
        ax.hist(anomaly[feat], bins=50, alpha=0.5, color='red', 
                label=f'Anomaly (n={len(anomaly)})', density=True)
        
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_title(f'{feat}')
    
    # Hide unused subplots
    for i in range(len(key_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions: Normal vs AE-Flagged Anomalies', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'feature_comparison.png'}")


def plot_top_anomalies(df: pd.DataFrame, feature_cols: list, output_dir: Path):
    """Plot 4: Examine the top anomalies in detail."""
    # Get top 20 anomalies by AE score
    top_anomalies = df.nlargest(20, 'ae_score')[['date', 'symbol', 'ae_score', 'if_score', 'ae_anomaly', 'if_anomaly'] + feature_cols[:10]]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top anomalies table-like plot
    ax = axes[0, 0]
    ax.axis('off')
    table_data = top_anomalies[['date', 'symbol', 'ae_score', 'if_score']].head(10)
    table_data['date'] = table_data['date'].astype(str).str[:10]
    table_data['ae_score'] = table_data['ae_score'].apply(lambda x: f'{x:.4f}')
    table_data['if_score'] = table_data['if_score'].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(
        cellText=table_data.values,
        colLabels=['Date', 'Symbol', 'AE Score', 'IF Score'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Top 10 Anomalies by AE Score', fontsize=12, fontweight='bold')
    
    # Score comparison for top anomalies
    ax = axes[0, 1]
    top_20 = df.nlargest(50, 'ae_score')
    ax.scatter(top_20['ae_score'], top_20['if_score'], c='red', alpha=0.6, s=50)
    ax.set_xlabel('AE Score')
    ax.set_ylabel('IF Score')
    ax.set_title('Top 50 AE Anomalies: Score Comparison')
    
    # Anomalies by symbol
    ax = axes[1, 0]
    symbol_counts = df[df['ae_anomaly']].groupby('symbol').size().sort_values(ascending=True).tail(15)
    symbol_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Anomalies')
    ax.set_title('Anomalies by Symbol (Top 15)')
    
    # Anomalies by month
    ax = axes[1, 1]
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg({
        'ae_anomaly': 'sum',
        'symbol': 'count'
    })
    monthly['rate'] = monthly['ae_anomaly'] / monthly['symbol'] * 100
    monthly['rate'].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Anomaly Rate (%)')
    ax.set_title('Anomaly Rate by Month')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_anomalies.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'top_anomalies.png'}")


def plot_model_agreement(df: pd.DataFrame, output_dir: Path):
    """Plot 5: Where do models agree/disagree?"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Categorize samples
    both_normal = (~df['ae_anomaly']) & (~df['if_anomaly'])
    ae_only = df['ae_anomaly'] & (~df['if_anomaly'])
    if_only = (~df['ae_anomaly']) & df['if_anomaly']
    both_anomaly = df['ae_anomaly'] & df['if_anomaly']
    
    # Agreement pie chart
    ax = axes[0, 0]
    sizes = [both_normal.sum(), ae_only.sum(), if_only.sum(), both_anomaly.sum()]
    labels = [
        f'Both Normal\n({both_normal.sum():,})',
        f'AE Only\n({ae_only.sum():,})',
        f'IF Only\n({if_only.sum():,})',
        f'Both Anomaly\n({both_anomaly.sum():,})'
    ]
    colors = ['lightgreen', 'steelblue', 'forestgreen', 'red']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Model Agreement Breakdown')
    
    # Score distributions by category
    ax = axes[0, 1]
    categories = {
        'Both Normal': df[both_normal]['ae_score'],
        'AE Only': df[ae_only]['ae_score'],
        'IF Only': df[if_only]['ae_score'],
        'Both Anomaly': df[both_anomaly]['ae_score']
    }
    ax.boxplot([v for v in categories.values() if len(v) > 0], 
               labels=[k for k, v in categories.items() if len(v) > 0])
    ax.set_ylabel('AE Score')
    ax.set_title('AE Score by Agreement Category')
    
    # Disagreement analysis - AE only cases
    ax = axes[1, 0]
    if ae_only.sum() > 0:
        ae_only_df = df[ae_only]
        # What makes AE flag but not IF?
        feature_diffs = []
        for col in ['volatility_20d', 'returns_20d', 'rsi_14', 'bb_position']:
            if col in df.columns:
                normal_mean = df[both_normal][col].mean()
                ae_only_mean = ae_only_df[col].mean()
                feature_diffs.append((col, ae_only_mean - normal_mean))
        
        if feature_diffs:
            features, diffs = zip(*feature_diffs)
            colors = ['red' if d > 0 else 'blue' for d in diffs]
            ax.barh(features, diffs, color=colors, alpha=0.7)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Difference from Normal Mean')
            ax.set_title('AE-Only Anomalies: Feature Differences')
    else:
        ax.text(0.5, 0.5, 'No AE-only anomalies', ha='center', va='center')
        ax.set_title('AE-Only Anomalies: Feature Differences')
    
    # Timeline of disagreements
    ax = axes[1, 1]
    df['agreement'] = 'Both Normal'
    df.loc[ae_only, 'agreement'] = 'AE Only'
    df.loc[if_only, 'agreement'] = 'IF Only'
    df.loc[both_anomaly, 'agreement'] = 'Both Anomaly'
    
    daily_agreement = df.groupby(['date', 'agreement']).size().unstack(fill_value=0)
    if 'Both Anomaly' in daily_agreement.columns:
        daily_agreement['Both Anomaly'].plot(ax=ax, label='Both Anomaly', color='red', alpha=0.7)
    if 'AE Only' in daily_agreement.columns:
        daily_agreement['AE Only'].plot(ax=ax, label='AE Only', color='steelblue', alpha=0.7)
    if 'IF Only' in daily_agreement.columns:
        daily_agreement['IF Only'].plot(ax=ax, label='IF Only', color='forestgreen', alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Daily Anomaly Counts by Category')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_agreement.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'model_agreement.png'}")


def archive_existing_diagnostics(diagnostics_dir: Path, archive_base_dir: Path):
    """Archive existing diagnostics directory if it contains files."""
    if not diagnostics_dir.exists():
        return False
    
    # Check if directory has any files (excluding archive subdirectory)
    existing_files = [f for f in diagnostics_dir.glob('*') 
                      if f.is_file() and f.name != 'archive']
    if not existing_files:
        return False
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = archive_base_dir / f"diagnostics_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Move all files to archive
    for file_path in existing_files:
        shutil.move(str(file_path), str(archive_dir / file_path.name))
    
    print(f"✓ Archived existing diagnostics to {archive_dir}")
    return True


def generate_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Generate text summary of findings."""
    summary = []
    summary.append("=" * 80)
    summary.append("MODEL BEHAVIOR DIAGNOSTIC SUMMARY")
    summary.append("=" * 80)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Data overview
    summary.append("DATA OVERVIEW")
    summary.append("-" * 40)
    summary.append(f"Total samples: {len(df):,}")
    summary.append(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    summary.append(f"Symbols: {df['symbol'].nunique()}")
    summary.append("")
    
    # AE stats
    summary.append("AUTOENCODER")
    summary.append("-" * 40)
    ae_anomalies = df['ae_anomaly'].sum()
    summary.append(f"Anomalies flagged: {ae_anomalies:,} ({ae_anomalies/len(df)*100:.2f}%)")
    summary.append(f"Threshold: {df['ae_threshold'].iloc[0]:.6f}")
    summary.append(f"Score range: {df['ae_score'].min():.6f} to {df['ae_score'].max():.6f}")
    summary.append(f"Score mean: {df['ae_score'].mean():.6f}")
    summary.append(f"Score median: {df['ae_score'].median():.6f}")
    summary.append("")
    
    # IF stats
    summary.append("ISOLATION FOREST")
    summary.append("-" * 40)
    if_anomalies = df['if_anomaly'].sum()
    summary.append(f"Anomalies flagged: {if_anomalies:,} ({if_anomalies/len(df)*100:.2f}%)")
    summary.append(f"Score range: {df['if_score'].min():.6f} to {df['if_score'].max():.6f}")
    summary.append(f"Score mean: {df['if_score'].mean():.6f}")
    summary.append("")
    
    # Agreement
    summary.append("MODEL AGREEMENT")
    summary.append("-" * 40)
    both_flag = (df['ae_anomaly'] & df['if_anomaly']).sum()
    either_flag = (df['ae_anomaly'] | df['if_anomaly']).sum()
    agree = (df['ae_anomaly'] == df['if_anomaly']).sum()
    summary.append(f"Both flag: {both_flag:,} ({both_flag/len(df)*100:.2f}%)")
    summary.append(f"Either flag: {either_flag:,} ({either_flag/len(df)*100:.2f}%)")
    summary.append(f"Agreement rate: {agree/len(df)*100:.2f}%")
    summary.append("")
    
    # Top anomaly periods
    summary.append("HIGH ANOMALY PERIODS")
    summary.append("-" * 40)
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg({
        'ae_anomaly': ['sum', 'mean'],
        'if_anomaly': ['sum', 'mean']
    })
    monthly.columns = ['ae_count', 'ae_rate', 'if_count', 'if_rate']
    top_months = monthly.nlargest(5, 'ae_rate')
    for month, row in top_months.iterrows():
        summary.append(f"  {month}: AE={row['ae_rate']*100:.1f}%, IF={row['if_rate']*100:.1f}%")
    summary.append("")
    
    # Top anomaly symbols
    summary.append("SYMBOLS WITH MOST ANOMALIES")
    summary.append("-" * 40)
    symbol_anomalies = df[df['ae_anomaly']].groupby('symbol').size().sort_values(ascending=False).head(10)
    symbol_totals = df.groupby('symbol').size()
    for symbol, count in symbol_anomalies.items():
        total = symbol_totals[symbol]
        summary.append(f"  {symbol}: {count}/{total} ({count/total*100:.1f}%)")
    
    summary.append("")
    summary.append("=" * 80)
    
    summary_text = "\n".join(summary)
    
    # Save and print
    with open(output_dir / 'diagnostic_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSaved: {output_dir / 'diagnostic_summary.txt'}")


def main():
    print("=" * 80)
    print("MODEL BEHAVIOR VISUAL DIAGNOSTIC")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / 'diagnostics'
    archive_base_dir = project_root / 'results' / 'diagnostics' / 'archive'
    
    # Archive existing diagnostics if they exist
    archive_existing_diagnostics(output_dir, archive_base_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print("\nLoading data and scoring...")
    
    df, feature_cols, ae_model, if_model = load_data_and_score(project_root)
    
    print(f"Loaded {len(df):,} samples")
    print(f"AE anomalies: {df['ae_anomaly'].sum():,}")
    print(f"IF anomalies: {df['if_anomaly'].sum():,}")
    
    print("\nGenerating visualizations...")
    
    plot_score_distributions(df, output_dir)
    plot_anomalies_over_time(df, output_dir)
    plot_feature_comparison(df, feature_cols, output_dir)
    plot_top_anomalies(df, feature_cols, output_dir)
    plot_model_agreement(df, output_dir)
    
    print("\nGenerating summary...")
    generate_summary_stats(df, output_dir)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - score_distributions.png")
    print("  - anomalies_over_time.png")
    print("  - feature_comparison.png")
    print("  - top_anomalies.png")
    print("  - model_agreement.png")
    print("  - diagnostic_summary.txt")


if __name__ == '__main__':
    main()