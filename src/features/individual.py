"""Individual stock feature extractor - the original/baseline approach."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class IndividualFeatureExtractor(BaseFeatureExtractor):
    """
    Extract individual stock features for anomaly detection.

    This is the original/baseline approach that calculates features
    based on each stock's own price history (returns, volatility,
    technical indicators, etc.)

    Best for: Detecting unusual patterns in individual stocks
    Limitation: Doesn't capture cross-sectional relationships
    """

    name = "individual"
    description = "Individual stock features (returns, volatility, technicals)"

    def calculate_returns(self, prices: pd.Series, windows: List[int] = [1, 5, 20, 60]) -> pd.DataFrame:
        """Calculate returns for multiple windows."""
        returns = pd.DataFrame(index=prices.index)
        for window in windows:
            returns[f'returns_{window}d'] = prices.pct_change(window)
        return returns

    def calculate_volatility(self, returns: pd.Series, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """Calculate rolling volatility."""
        volatility = pd.DataFrame(index=returns.index)
        for window in windows:
            volatility[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        return volatility

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        band_range = upper - lower
        position = (prices - lower) / band_range

        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': ma,
            'bb_lower': lower,
            'bb_position': position,
            'bb_width': band_range / ma
        })

    def calculate_volume_features(self, volume: pd.Series, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=volume.index)
        for window in windows:
            ma = volume.rolling(window).mean()
            features[f'volume_ratio_{window}d'] = volume / ma
            features[f'volume_std_{window}d'] = volume.rolling(window).std()
        return features

    def calculate_price_position(self, prices: pd.Series, windows: List[int] = [52, 200]) -> pd.DataFrame:
        """Calculate price position relative to highs/lows."""
        features = pd.DataFrame(index=prices.index)
        for window in windows:
            high = prices.rolling(window).max()
            low = prices.rolling(window).min()
            features[f'price_to_{window}d_high'] = prices / high
            features[f'price_to_{window}d_low'] = prices / low

        features['above_ma_50'] = (prices > prices.rolling(50).mean()).astype(int)
        features['above_ma_200'] = (prices > prices.rolling(200).mean()).astype(int)

        return features

    def calculate_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate all individual features for a price dataset.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all features
        """
        if df.empty:
            return pd.DataFrame()

        prices = df['close']

        # Returns
        returns = self.calculate_returns(prices)

        # Volatility
        volatility = self.calculate_volatility(returns['returns_1d'])

        # Technical indicators
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        bb = self.calculate_bollinger_bands(prices)

        # Volume features
        volume_features = self.calculate_volume_features(df['volume'])

        # Price position
        price_position = self.calculate_price_position(prices)

        # Combine all features
        features = pd.concat([
            returns,
            volatility,
            pd.DataFrame({'rsi_14': rsi}),
            macd,
            bb,
            volume_features,
            price_position,
        ], axis=1)

        # Add symbol
        features['symbol'] = df['symbol'].iloc[0] if 'symbol' in df.columns else None

        return features

    def prepare_training_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare training data for all symbols.

        Args:
            symbols: List of symbols (None = all symbols in database)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Combined DataFrame with features for all symbols
        """
        if symbols is None:
            symbols = self.get_all_symbols()

        logger.info(f"Calculating individual features for {len(symbols)} symbols...")

        all_features = []

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"  [{i}/{len(symbols)}] Processing {symbol}...")

                df = self.load_price_data(symbol, start_date, end_date)

                if df.empty:
                    logger.warning(f"    No data for {symbol}")
                    continue

                features = self.calculate_features(df)
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
