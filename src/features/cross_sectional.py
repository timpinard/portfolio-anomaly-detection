"""Cross-sectional feature extractor for detecting divergence from market."""

import logging
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class CrossSectionalFeatureExtractor(BaseFeatureExtractor):
    """
    Extract cross-sectional features that compare stocks to the market/universe.

    This approach detects stocks that are DIVERGING from the broader market -
    i.e., moving in opposite directions or with different magnitude than peers.

    Best for: Detecting correlation breakdown, relative strength shifts
    Features: Relative returns, rolling beta/correlation, cross-sectional rank
    """

    name = "cross_sectional"
    description = "Cross-sectional features"

    def __init__(self, db_path: str, config: Dict[str, Any] = None):
        super().__init__(db_path, config)

        self.market_proxy = self.config.get('market_proxy', 'SPY')
        self.correlation_windows = self.config.get('correlation_windows', [20, 60])
        self.beta_window = self.config.get('beta_window', 60)
        self.rank_window = self.config.get('rank_window', 20)

        self._market_returns = None
        self._universe_returns = None

    def _load_market_returns(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """Load market proxy returns (e.g., SPY)."""
        if self._market_returns is not None:
            return self._market_returns

        df = self.load_price_data(self.market_proxy, start_date, end_date)

        if df.empty:
            logger.warning(f"No data for market proxy {self.market_proxy}")
            return pd.Series(dtype=float)

        self._market_returns = df['close'].pct_change()
        self._market_returns.name = 'market_return'

        return self._market_returns

    def _load_universe_returns(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load returns for entire universe to calculate cross-sectional stats."""
        if self._universe_returns is not None:
            return self._universe_returns

        logger.info(f"Loading universe returns for {len(symbols)} symbols...")

        returns_dict = {}
        for symbol in symbols:
            try:
                df = self.load_price_data(symbol, start_date, end_date)
                if not df.empty:
                    returns_dict[symbol] = df['close'].pct_change()
            except Exception as e:
                logger.debug(f"Could not load {symbol}: {e}")

        self._universe_returns = pd.DataFrame(returns_dict)
        logger.info(f"Loaded returns for {len(returns_dict)} symbols")

        return self._universe_returns

    def calculate_relative_returns(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate returns relative to market.

        Args:
            stock_returns: Stock's daily returns
            market_returns: Market's daily returns

        Returns:
            DataFrame with relative return features
        """
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if aligned.empty:
            return pd.DataFrame(index=stock_returns.index)

        features = pd.DataFrame(index=aligned.index)

        features['relative_return_1d'] = aligned['stock'] - aligned['market']

        for window in [5, 20]:
            stock_cum = (1 + aligned['stock']).rolling(window).apply(lambda x: x.prod()) - 1
            market_cum = (1 + aligned['market']).rolling(window).apply(lambda x: x.prod()) - 1
            features[f'relative_return_{window}d'] = stock_cum - market_cum

        features['same_direction'] = (
            np.sign(aligned['stock']) == np.sign(aligned['market'])
        ).astype(int)

        features['direction_agreement_20d'] = features['same_direction'].rolling(20).mean()

        return features

    def calculate_correlation_features(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        windows: List[int] = [20, 60]
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation with market.

        Args:
            stock_returns: Stock's daily returns
            market_returns: Market's daily returns
            windows: Rolling window sizes

        Returns:
            DataFrame with correlation features
        """
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if aligned.empty:
            return pd.DataFrame(index=stock_returns.index)

        features = pd.DataFrame(index=aligned.index)

        for window in windows:
            features[f'correlation_{window}d'] = aligned['stock'].rolling(window).corr(aligned['market'])

            corr = features[f'correlation_{window}d']
            features[f'correlation_change_{window}d'] = corr - corr.shift(window)

        return features

    def calculate_beta_features(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling beta (sensitivity to market).

        Args:
            stock_returns: Stock's daily returns
            market_returns: Market's daily returns
            window: Rolling window size

        Returns:
            DataFrame with beta features
        """
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if aligned.empty:
            return pd.DataFrame(index=stock_returns.index)

        features = pd.DataFrame(index=aligned.index)

        # Calculate rolling beta (covariance/variance ratio)
        def calc_beta(stock, market):
            if len(stock) < 2:
                return np.nan
            cov = np.cov(stock, market)[0, 1]
            var = np.var(market)
            return cov / var if var > 0 else np.nan

        beta_values = []
        for i in range(len(aligned)):
            if i < window:
                beta_values.append(np.nan)
            else:
                stock_window = aligned['stock'].iloc[i-window:i]
                market_window = aligned['market'].iloc[i-window:i]
                beta_values.append(calc_beta(stock_window, market_window))

        features[f'beta_{window}d'] = beta_values

        features[f'beta_deviation_{window}d'] = abs(features[f'beta_{window}d'] - 1)

        features[f'beta_change_{window}d'] = features[f'beta_{window}d'].diff(window // 2)

        return features

    def calculate_cross_sectional_rank(
        self,
        stock_returns: pd.Series,
        universe_returns: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate where this stock ranks vs the universe.

        Args:
            stock_returns: Stock's daily returns
            universe_returns: DataFrame with returns for all stocks
            window: Rolling window for cumulative return ranking

        Returns:
            DataFrame with rank features
        """
        if universe_returns.empty:
            return pd.DataFrame()

        common_dates = stock_returns.dropna().index.intersection(universe_returns.index)
        if len(common_dates) == 0:
            return pd.DataFrame()

        features = pd.DataFrame(index=common_dates)

        def daily_rank(date):
            if date not in universe_returns.index:
                return np.nan
            day_returns = universe_returns.loc[date].dropna()
            if len(day_returns) < 5:
                return np.nan
            stock_ret = stock_returns.get(date, np.nan)
            if pd.isna(stock_ret):
                return np.nan
            return (day_returns < stock_ret).mean()

        features['return_rank_daily'] = [daily_rank(d) for d in common_dates]

        sample_dates = common_dates[::5]
        cum_returns = (1 + universe_returns).rolling(window).apply(lambda x: x.prod()) - 1

        def rolling_rank(date):
            if date not in cum_returns.index:
                return np.nan
            period_returns = cum_returns.loc[date].dropna()
            if len(period_returns) < 5:
                return np.nan
            try:
                stock_cum = (1 + stock_returns.loc[:date].tail(window)).prod() - 1
                return (period_returns < stock_cum).mean()
            except Exception:
                return np.nan
        rank_values = {d: rolling_rank(d) for d in sample_dates}
        features[f'return_rank_{window}d'] = pd.Series(rank_values).reindex(common_dates).ffill()

        features['rank_extremity'] = abs(features['return_rank_daily'] - 0.5) * 2

        return features

    def calculate_dispersion_features(
        self,
        stock_returns: pd.Series,
        universe_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate how dispersed this stock is from universe average.

        Args:
            stock_returns: Stock's daily returns
            universe_returns: DataFrame with returns for all stocks

        Returns:
            DataFrame with dispersion features
        """
        if universe_returns.empty:
            return pd.DataFrame()

        universe_mean = universe_returns.mean(axis=1)
        universe_std = universe_returns.std(axis=1)

        aligned = pd.DataFrame({
            'stock': stock_returns,
            'universe_mean': universe_mean,
            'universe_std': universe_std
        }).dropna()

        if aligned.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=aligned.index)

        features['zscore_vs_universe'] = (
            (aligned['stock'] - aligned['universe_mean']) /
            (aligned['universe_std'] + 1e-8)
        )

        features['zscore_vs_universe_20d'] = features['zscore_vs_universe'].rolling(20).mean()

        features['zscore_abs_vs_universe'] = features['zscore_vs_universe'].abs()

        return features

    def calculate_features(
        self,
        df: pd.DataFrame,
        market_returns: pd.Series = None,
        universe_returns: pd.DataFrame = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate all cross-sectional features for a stock.

        Args:
            df: DataFrame with OHLCV data for one symbol
            market_returns: Market proxy returns (required)
            universe_returns: Returns for all symbols (optional, for ranking)

        Returns:
            DataFrame with cross-sectional features
        """
        if df.empty:
            return pd.DataFrame()

        if market_returns is None:
            raise ValueError("market_returns is required for cross-sectional features")

        stock_returns = df['close'].pct_change()
        stock_returns.name = df['symbol'].iloc[0] if 'symbol' in df.columns else 'stock'

        # Find common dates between stock and market
        common_dates = stock_returns.dropna().index.intersection(market_returns.dropna().index)
        if len(common_dates) == 0:
            return pd.DataFrame()

        relative = self.calculate_relative_returns(stock_returns, market_returns)

        correlation = self.calculate_correlation_features(
            stock_returns, market_returns, self.correlation_windows
        )

        beta = self.calculate_beta_features(
            stock_returns, market_returns, self.beta_window
        )

        if universe_returns is not None and not universe_returns.empty:
            rank = self.calculate_cross_sectional_rank(
                stock_returns, universe_returns, self.rank_window
            )
            dispersion = self.calculate_dispersion_features(
                stock_returns, universe_returns
            )
        else:
            rank = pd.DataFrame()
            dispersion = pd.DataFrame()

        # Combine all features - use outer join and then filter to common dates
        all_dfs = [relative, correlation, beta, rank, dispersion]
        all_dfs = [df for df in all_dfs if not df.empty]

        if not all_dfs:
            return pd.DataFrame()

        features = pd.concat(all_dfs, axis=1, join='outer')

        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else None
        features['symbol'] = symbol

        return features

    def prepare_training_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare training data with cross-sectional features.

        Args:
            symbols: List of symbols (None = all symbols in database)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Combined DataFrame with features for all symbols
        """
        if symbols is None:
            symbols = self.get_all_symbols()

        symbols = [s for s in symbols if s != self.market_proxy]

        logger.info(f"Calculating cross-sectional features for {len(symbols)} symbols...")

        market_returns = self._load_market_returns(start_date, end_date)
        if market_returns.empty:
            raise ValueError(f"Could not load market proxy {self.market_proxy}")

        universe_returns = self._load_universe_returns(symbols, start_date, end_date)

        all_features = []

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"  [{i}/{len(symbols)}] Processing {symbol}...")

                df = self.load_price_data(symbol, start_date, end_date)

                if df.empty:
                    logger.warning(f"    No data for {symbol}")
                    continue

                features = self.calculate_features(
                    df,
                    market_returns=market_returns,
                    universe_returns=universe_returns
                )
                features = features.dropna()

                if not features.empty:
                    all_features.append(features)
                    logger.info(f"    ✓ {len(features)} feature rows")

            except Exception as e:
                logger.error(f"    ✗ Error processing {symbol}: {e}")
                continue

        if not all_features:
            logger.error("No features calculated for any symbol")
            return pd.DataFrame()

        combined = pd.concat(all_features, axis=0)
        logger.info(f"Total feature rows: {len(combined)}")

        return combined
