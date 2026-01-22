# Cross-Sectional Model Microcosm - Interactive Tutorial

## Overview

The `portfolio_cross_sectional_model_microcosm.ipynb` notebook is a **self-contained, educational demonstration** of the cross-sectional attribution model. It uses synthetic data to illustrate the complete workflow in a simplified, transparent environment where you can see exactly how each component works.

**Purpose:** Understand the cross-sectional model's inner workings without the complexity of real market data, production code, or large-scale training.

## What This Notebook Demonstrates

### The Complete Pipeline

1. **Synthetic Market Universe Creation**
   - Generates market proxy returns (SPY-like)
   - Creates 30 synthetic stocks with realistic sector exposures
   - Includes familiar ticker symbols (AAPL, MSFT, GOOGL, etc.)
   - Simulates returns driven by market factors, sector factors, and idiosyncratic noise

2. **Cross-Sectional Feature Engineering**
   - Relative returns (stock - market)
   - Rolling correlations with market
   - Beta calculation (market sensitivity)
   - Rank extremity (percentile position)
   - Universe z-scores (how different from average)

3. **Autoencoder Training**
   - Small neural network (6 features → 16 → 8 → 4 → 8 → 16 → 6)
   - Trained on the synthetic universe
   - Learns "normal" market-relative patterns

4. **Universe Scoring**
   - Scores all stocks daily
   - Computes reconstruction errors
   - Generates market baseline statistics (mean, std, percentiles)

5. **Portfolio Analysis**
   - **Structural divergence:** Portfolio-weighted reconstruction error
   - **Directional divergence:** Portfolio return vs market return
   - **Attribution:** Which holdings contribute most to divergence
   - **Health score:** Combined structural + directional metric

6. **Demonstration Portfolios**
   - **TECH:** Concentrated in large-cap tech (AAPL, MSFT, GOOGL, AMZN, META)
   - **DEFENSIVE:** Consumer staples (JNJ, PG, KO, PEP)
   - **ERRATIC:** Mix with intentionally volatile components (includes TSLA, NVDA)

## Why Use Synthetic Data?

**Advantages of the microcosm approach:**

1. **Transparency:** You control the data-generating process, so you know ground truth
2. **Speed:** Runs in seconds, not hours
3. **Reproducibility:** Seeded random number generation ensures consistent results
4. **Educational:** Simple enough to understand every step
5. **Debugging:** Easy to experiment with different parameters
6. **No Dependencies:** No API keys, no data downloads, no external services

**What you sacrifice:**
- Real market complexity (regimes, events, structural breaks)
- Scale (30 stocks vs 500)
- Historical validation against actual market events

## Notebook Structure

### Section 1: Synthetic Market Universe
```python
# Creates 252 days of synthetic returns for 30 stocks
# Driven by: market_factor + sector_factor + idiosyncratic_noise
# Includes sector labels: TECH, HEALTHCARE, CONSUMER, etc.
```

**Key Output:** `returns` DataFrame with daily returns for all stocks + market proxy

### Section 2: Cross-Sectional Features
```python
# Computes 6 simplified cross-sectional features:
# - relative_return_5d
# - corr_20d
# - corr_change_20d
# - beta_60d
# - rank_extremity_5d
# - zscore_vs_universe_20d
```

**Key Output:** `feat` DataFrame with features for all stocks on all dates

### Section 3: Autoencoder Training
```python
# Small network: 6 → 16 → 8 → 4 (latent) → 8 → 16 → 6
# Trains for 50 epochs
# Uses StandardScaler for normalization
```

**Key Output:** Trained autoencoder model + scaler

### Section 4: Universe Scoring
```python
# Scores all stocks using the trained autoencoder
# Computes reconstruction error (MSE)
# Generates daily market summaries
```

**Key Output:** 
- `scored`: Daily scores for all stocks
- `daily`: Market baseline statistics (mean, std, p95, p99)

### Section 5: Portfolio Analysis Function
```python
def analyze_portfolio_health(holdings, analysis_date, contra_horizon=5):
    """
    Returns:
    - portfolio_score: Weighted average of reconstruction errors
    - structural_z: Z-score vs market baseline
    - contra_return: Portfolio return - market return
    - directional_z: Z-score of contra return
    - health_score: Combined metric
    - contributors: Per-holding attribution
    """
```

**This function is the conceptual equivalent of your production `analyze_portfolio_health()`**

### Section 6: Demonstration Portfolios

Three example portfolios show different behaviors:

**TECH Portfolio:**
```python
[AAPL: 25%, MSFT: 25%, GOOGL: 20%, AMZN: 15%, META: 15%]
```
- Concentrated in large-cap tech
- High correlation with market
- Low structural divergence (these stocks "behave normally")

**DEFENSIVE Portfolio:**
```python
[JNJ: 30%, PG: 30%, KO: 20%, PEP: 20%]
```
- Consumer staples / healthcare
- Lower beta (defensive)
- May show directional divergence when market rallies

**ERRATIC Portfolio:**
```python
[AAPL: 40%, MSFT: 40%, TSLA: 5%, NVDA: 5%, META: 10%]
```
- Includes intentionally volatile components
- Higher structural divergence
- Attribution clearly shows which holdings drive anomaly

### Section 7: Mapping to Production System

Shows how notebook concepts map to your production code:

| Notebook Component | Production System |
|--------------------|-------------------|
| `returns` DataFrame | `market_prices` table |
| `feat` cross-sectional features | `src/data/cross_sectional.py` |
| Autoencoder training | `train_model.py cross_sectional` |
| Universe scoring | `score_universe.py cross_sectional` |
| `analyze_portfolio_health()` | `portfolio/analyze.py` |
| `daily` summaries | `market_daily_summary` table |

## How to Use the Notebook

### Basic Usage

1. **Open in Jupyter:**
   ```bash
   jupyter notebook portfolio_cross_sectional_model_microcosm.ipynb
   ```

2. **Run all cells:**
   - Menu: Cell → Run All
   - Or: Shift + Enter through each cell

3. **Examine outputs:**
   - DataFrame previews
   - Training loss curves
   - Portfolio analysis results
   - Attribution bar charts

### Experiments to Try

**Modify the synthetic data:**
```python
# Make market more volatile
market = np.random.randn(n_days) * 0.03  # was 0.02

# Add more erratic stocks
erratic_assets.extend(['TSLA', 'NVDA', 'GME', 'AMC'])
```

**Change feature windows:**
```python
# Use longer correlation window
corr_40d = returns.rolling(40).corr(market)

# Add new feature
momentum_diff = r5 - r20  # Short-term vs long-term momentum
```

**Adjust autoencoder architecture:**
```python
# Deeper network
hidden_dims = [32, 16, 8]  # was [16, 8]
latent_dim = 3  # was 4 (more compression)
```

**Create custom portfolios:**
```python
my_portfolio = [
    {"symbol": "AAPL", "weight": 0.30},
    {"symbol": "JNJ", "weight": 0.30},
    {"symbol": "TSLA", "weight": 0.40}  # Mix defensive + erratic
]

result = analyze_portfolio_health(my_portfolio, analysis_date)
```

## Key Insights from the Notebook

### 1. Structural Divergence is Different from Performance

A portfolio can have:
- **High return, low divergence:** Moving with market but amplified (high beta tech)
- **Low return, low divergence:** Moving with market but muted (defensive stocks)
- **High return, high divergence:** Moving against market successfully (contrarian)
- **Low return, high divergence:** Moving against market unsuccessfully

Divergence measures **how unusual the pattern is**, not whether returns are good or bad.

### 2. Attribution is Portfolio-Weight Dependent

From the notebook's contributor analysis:
```python
# Holding with:
# - High divergence score (0.020)
# - Low weight (5%)
# → Low contribution (0.001)

# Holding with:
# - Medium divergence score (0.012)
# - High weight (40%)
# → High contribution (0.0048)
```

**The biggest contributors are often moderate divergence × large weights**

### 3. Market Baseline Provides Context

Without baseline normalization:
```
Portfolio score: 0.015
```
Hard to interpret - is this high or low?

With z-score normalization:
```
Portfolio score: 0.015
Market mean: 0.010
Market std: 0.003
Structural z-score: +1.67
```
Now clear: Portfolio is 1.67 standard deviations above typical market divergence.

### 4. Health Score Combines Structural + Directional

```python
health_score = structural_z + directional_z
```

This captures both:
- How unusual the portfolio's features are (structural)
- How much it's moving against the market (directional)

High health scores (> +2) warrant investigation.

## Comparing Microcosm to Production

### Similarities ✅

- Same conceptual flow (universe → features → train → score → analyze)
- Same cross-sectional features (relative returns, correlations, beta)
- Same autoencoder architecture pattern (encoder → latent → decoder)
- Same attribution logic (weighted scores → z-score normalization)
- Same portfolio analysis outputs (structural_z, health_score, contributors)

### Differences ⚠️

| Aspect | Microcosm | Production |
|--------|-----------|------------|
| **Data** | Synthetic, 252 days | Real S&P 500, 10 years |
| **Universe** | 30 stocks | 500+ stocks |
| **Features** | 6 simplified | 15 comprehensive |
| **Architecture** | 6→16→8→4 | 15→128→64→32→15 |
| **Training** | 50 epochs, seconds | 100 epochs, minutes |
| **Storage** | In-memory DataFrames | SQLite database |
| **Scale** | Single notebook | Multi-module system |

### When to Use Each

**Use the Microcosm when:**
- Learning how the cross-sectional model works
- Testing new feature ideas quickly
- Debugging attribution logic
- Teaching others the concepts
- Validating mathematical formulas

**Use the Production System when:**
- Analyzing real portfolios
- Making actual investment decisions
- Need full S&P 500 universe coverage
- Require historical data and backtesting
- Building client-facing applications

## Educational Value

### For Understanding the Model

The notebook makes it easy to see:

1. **How features capture divergence:**
   - Print `feat.head()` to see actual values
   - Observe how relative_return_5d differs from absolute returns

2. **How the autoencoder learns:**
   - Watch training loss decrease
   - See reconstruction error patterns
   - Understand what "normal" means to the model

3. **How attribution works:**
   - Step through `analyze_portfolio_health()` line by line
   - See weight × score = contribution calculation
   - Visualize contributions with bar charts

4. **How portfolios differ:**
   - Compare TECH vs DEFENSIVE vs ERRATIC
   - See which holdings drive each portfolio's divergence
   - Understand structural vs directional components

### For Experimentation

The synthetic data lets you:

1. **Create extreme scenarios:**
   ```python
   # Make TSLA extremely erratic
   returns.loc[returns.symbol == 'TSLA', 'return'] *= 3
   ```

2. **Test edge cases:**
   ```python
   # What if portfolio is 100% market proxy?
   spy_only = [{"symbol": "SPY", "weight": 1.0}]
   # Expect: structural_z ≈ 0, low divergence
   ```

3. **Validate intuitions:**
   ```python
   # Should defensive stocks have negative beta?
   feat[feat.symbol.isin(['JNJ', 'PG'])]['beta_60d'].describe()
   ```

## Next Steps After the Notebook

Once you understand the microcosm:

1. **Explore the production code:**
   - `src/data/cross_sectional.py` - See all 15 features
   - `train_model.py cross_sectional` - Full training pipeline
   - `portfolio/analyze.py` - Production analysis function

2. **Run the production system:**
   ```bash
   python train_model.py cross_sectional --train-end 2024-06-30
   python score_universe.py cross_sectional
   python analyze_portfolio.py AAPL MSFT GOOGL
   ```

3. **Compare outputs:**
   - Notebook portfolio analysis results
   - Production system results for same holdings
   - Understand differences (scale, features, training data)

4. **Extend both:**
   - Add new features in notebook first (fast iteration)
   - Validate with synthetic data
   - Then implement in production code

## Common Questions

**Q: Why does the notebook use only 6 features instead of 15?**  
A: Simplicity for education. The full production system uses 15 features for more comprehensive market-relative analysis, but 6 features are enough to demonstrate the concepts.

**Q: Can I use this notebook with real data?**  
A: Yes, but you'd need to modify the data loading section. The synthetic data structure matches what you'd get from `yfinance`, so the rest of the notebook would work with minimal changes.

**Q: Why is the architecture different (smaller)?**  
A: The microcosm uses a smaller network (6→16→8→4) because it has fewer features and less data. The production system needs more capacity (15→128→64→32→15) for the larger, more complex real-world dataset.

**Q: How accurate is the synthetic data?**  
A: It captures key statistical properties (correlations, volatility patterns, factor exposures) but lacks real market complexity (regimes, events, microstructure). It's good for understanding concepts, not for making investment decisions.

**Q: Should I train production models with this notebook?**  
A: No. This is for education and experimentation only. Use the production training pipeline (`train_model.py`) for models that analyze real portfolios.

**Q: Can I modify the notebook?**  
A: Absolutely! That's the point. Experiment freely - change parameters, add features, try different architectures. It's a sandbox for learning.

## Conclusion

The microcosm notebook is your gateway to understanding the cross-sectional attribution model. It strips away the complexity of production code and real market data to reveal the core concepts in a transparent, interactive environment.

**Use it to:**
- ✅ Learn how cross-sectional features work
- ✅ Understand autoencoder-based anomaly detection
- ✅ See portfolio attribution in action
- ✅ Experiment with different approaches
- ✅ Build intuition before diving into production code

**Then move to the production system to:**
- 📊 Analyze real portfolios
- 🎯 Make actual investment decisions
- 📈 Leverage full S&P 500 universe
- 🔍 Access historical data and backtesting
- 🚀 Build production applications

---

## Quick Reference

**Run the notebook:**
```bash
jupyter notebook portfolio_cross_sectional_model_microcosm.ipynb
```

**Key functions:**
- `create_synthetic_universe()` - Generate market data
- `compute_features()` - Calculate cross-sectional metrics
- `train_autoencoder()` - Train the model
- `score_universe()` - Generate anomaly scores
- `analyze_portfolio_health()` - Analyze portfolio divergence

**Three demo portfolios:**
- `p_tech` - Large-cap tech concentration
- `p_def` - Defensive consumer staples
- `p_err` - Mix with erratic components

**Key outputs:**
- `returns` - Daily returns for all stocks
- `feat` - Cross-sectional features
- `scored` - Daily anomaly scores
- `daily` - Market baseline summaries
- Portfolio analysis with attribution

**Runtime:** < 1 minute on modern laptop
