#!/usr/bin/env python3
"""Feature extraction for market data."""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MarketFeatureExtractor:
    """Extract features from market price data for anomaly detection."""
    
    def __init__(self, db_path: str = "data/processed/market_data.sqlite"):
        """
        Initialize extractor.
        
        Args:
            db_path: Path to SQLite database with market_prices table
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def load_price_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load price data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD) or None for all
            end_date: End date (YYYY-MM-DD) or None for all
        
        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM market_prices WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df.set_index('date', inplace=True)
        
        return df
    
    def calculate_returns(self, prices: pd.Series, windows: List[int] = [1, 5, 20, 60]) -> pd.DataFrame:
        """
        Calculate returns for multiple windows.
        
        Args:
            prices: Series of closing prices
            windows: List of window sizes in days
        
        Returns:
            DataFrame with return columns
        """
        returns = pd.DataFrame(index=prices.index)
        
        for window in windows:
            returns[f'returns_{window}d'] = prices.pct_change(window)
        
        return returns
    
    def calculate_volatility(self, returns: pd.Series, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Series of daily returns
            windows: List of window sizes in days
        
        Returns:
            DataFrame with volatility columns
        """
        volatility = pd.DataFrame(index=returns.index)
        
        for window in windows:
            volatility[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        return volatility
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of closing prices
            period: RSI period (typically 14)
        
        Returns:
            Series of RSI values
        """
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
        """
        Calculate MACD indicator.
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            DataFrame with macd, signal, and histogram
        """
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
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of closing prices
            window: Moving average period
            num_std: Number of standard deviations
        
        Returns:
            DataFrame with upper, middle, lower bands and position
        """
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        
        # Calculate position within bands (0 = lower, 0.5 = middle, 1 = upper)
        band_range = upper - lower
        position = (prices - lower) / band_range
        
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': ma,
            'bb_lower': lower,
            'bb_position': position,
            'bb_width': band_range / ma  # Normalized width
        })
    
    def calculate_volume_features(self, volume: pd.Series, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            volume: Series of trading volume
            windows: List of window sizes
        
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=volume.index)
        
        for window in windows:
            ma = volume.rolling(window).mean()
            features[f'volume_ratio_{window}d'] = volume / ma
            features[f'volume_std_{window}d'] = volume.rolling(window).std()
        
        return features
    
    def calculate_price_position(self, prices: pd.Series, windows: List[int] = [52, 200]) -> pd.DataFrame:
        """
        Calculate price position relative to highs/lows.
        
        Args:
            prices: Series of closing prices
            windows: List of window sizes in days
        
        Returns:
            DataFrame with price position features
        """
        features = pd.DataFrame(index=prices.index)
        
        for window in windows:
            high = prices.rolling(window).max()
            low = prices.rolling(window).min()
            
            features[f'price_to_{window}d_high'] = prices / high
            features[f'price_to_{window}d_low'] = prices / low
        
        # Moving averages
        features['above_ma_50'] = (prices > prices.rolling(50).mean()).astype(int)
        features['above_ma_200'] = (prices > prices.rolling(200).mean()).astype(int)
        
        return features
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for a price dataset.
        
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
            price_position
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
        # Get symbols from database if not provided
        if symbols is None:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_prices")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
        
        logger.info(f"Calculating features for {len(symbols)} symbols...")
        
        all_features = []
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"  [{i}/{len(symbols)}] Processing {symbol}...")
                
                # Load price data
                df = self.load_price_data(symbol, start_date, end_date)
                
                if df.empty:
                    logger.warning(f"    No data for {symbol}")
                    continue
                
                # Calculate features
                features = self.calculate_all_features(df)
                
                # Drop rows with NaN (from window calculations)
                features = features.dropna()
                
                if not features.empty:
                    all_features.append(features)
                    logger.info(f"    ✓ {len(features)} feature rows")
                
            except Exception as e:
                logger.error(f"    ✗ Error processing {symbol}: {e}")
                continue
        
        # Combine all features
        if not all_features:
            logger.error("No features calculated for any symbol")
            return pd.DataFrame()
        
        combined = pd.concat(all_features, axis=0)
        logger.info(f"Total feature rows: {len(combined)}")
        
        return combined
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata).
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of feature column names
        """
        exclude_cols = {'symbol', 'date'}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for model input.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].values
        
        # Replace inf and NaN with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
