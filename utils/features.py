import pandas as pd
import numpy as np
from typing import Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical features to a DataFrame for ML model training.
    
    Args:
        df (pd.DataFrame): DataFrame containing market data with columns: 
                           ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        pd.DataFrame: DataFrame with added technical features
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to add_technical_features")
        return df

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Ensure we have the necessary columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return df

    # 1. RSI: 14-period Relative Strength Index
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # 2. SMA_Ratio: Close price divided by SMA_20 (normalization)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_ratio'] = df['close'] / df['sma_20']
    
    # 3. Log Returns: np.log(df.close / df.close.shift(1))
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 4. Volatility: Rolling standard deviation of log returns (20-period)
    df['volatility'] = df['log_return'].rolling(window=20).std()
    
    # 5. ATR (Average True Range)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # 6. Lagged Values: Shifted returns (t-1, t-2) for context
    df['log_return_lag1'] = df['log_return'].shift(1)
    df['log_return_lag2'] = df['log_return'].shift(2)
    
    # 7. Price position relative to high-low range
    df['hl_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 8. Volume moving average ratio
    if df['volume'].sum() == 0:
        df['volume_sma_ratio'] = 1.0
    else:
        volume_sma = df['volume'].rolling(window=20).mean()
        # Handle division by zero if some volumes are 0
        df['volume_sma_ratio'] = df['volume'] / volume_sma
        df['volume_sma_ratio'] = df['volume_sma_ratio'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    # 9. MACD
    macd, signal, hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    # 10. Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], period=20)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_upper - bb_lower
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # 11. Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # 12. Williams %R
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])

    # 13. On-Balance Volume (OBV)
    df['obv'] = calculate_obv(df['close'], df['volume'])

    # 14. Slope (Linear Regression Slope of Close)
    df['slope'] = calculate_slope(df['close'], period=10)

    # 15. Lagged Indicators (Context)
    # Lag RSI, MACD Hist, and Volatility to give the model trend context
    for lag in [1, 2, 3]:
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
        df[f'macd_hist_lag{lag}'] = df['macd_hist'].shift(lag)
        df[f'volatility_lag{lag}'] = df['volatility'].shift(lag)
        df[f'obv_lag{lag}'] = df['obv'].shift(lag)
    
    # Clean: Drop NaN values resulting from rolling windows
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    
    if initial_rows != final_rows:
        logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values after feature engineering")
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        prices (pd.Series): Series of closing prices
        period (int): Period for RSI calculation
    
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices (pd.Series): Series of closing prices
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line EMA period
    
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, and histogram
    """
    exp1 = prices.ewm(span=fast_period).mean()
    exp2 = prices.ewm(span=slow_period).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices (pd.Series): Series of closing prices
        period (int): Period for moving average
        std_dev (int): Number of standard deviations
    
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: Upper band, middle band (SMA), and lower band
    """
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return upper_band, sma, lower_band


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of closing prices
        period (int): Period for ATR calculation
    
    Returns:
        pd.Series: ATR values
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D)
    """
    # Calculate %K
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Smooth %K
    k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
    
    # Calculate %D (SMA of %K)
    d_percent = k_percent_smooth.rolling(window=smooth_d).mean()
    
    return k_percent_smooth, d_percent


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    """
    obv = pd.Series(index=close.index, dtype='float64')
    obv.iloc[0] = volume.iloc[0]
    
    # Vectorized calculation
    change = close.diff()
    direction = np.where(change > 0, 1, np.where(change < 0, -1, 0))
    
    # We need to handle the first element being NaN from diff()
    direction[0] = 0 
    
    # Calculate cumulative sum of signed volume
    obv = (direction * volume).cumsum()
    return obv


def calculate_slope(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate the slope of the linear regression line for a rolling window.
    """
    def linear_slope(y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y))
        # Simple linear regression slope: cov(x, y) / var(x)
        # Using numpy polyfit is cleaner but slower in rolling apply
        # Let's use a simplified formula for slope
        x_mean = (len(y) - 1) / 2
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        return numerator / denominator if denominator != 0 else 0

    return series.rolling(window=period).apply(linear_slope, raw=True)


def prepare_training_data(df: pd.DataFrame, target_period: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training by creating features and target variables.
    
    Args:
        df (pd.DataFrame): DataFrame with market data and features
        target_period (int): Number of periods ahead to predict (default: 1)
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to prepare_training_data")
        return pd.DataFrame(), pd.Series()
    
    df = df.copy()
    
    # Calculate the target: 1 if price goes up in target_period, 0 if it goes down
    df['future_close'] = df['close'].shift(-target_period)
    df['target'] = np.where(df['future_close'] > df['close'], 1, 0)
    
    # Select feature columns for the model (exclude target-related columns and non-numeric columns)
    feature_columns = [col for col in df.columns if col not in ['target', 'future_close'] and pd.api.types.is_numeric_dtype(df[col])]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    # Remove rows where target is NaN (happens when we can't calculate future price)
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    if X.empty:
        logger.warning("No valid data for training after target calculation")
        return pd.DataFrame(), pd.Series()
    
    logger.info(f"Prepared training data: {len(X)} samples with {len(X.columns)} features")
    
    return X, y


if __name__ == "__main__":
    # Example usage
    # Create sample data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.rand(100) * 10 + 100,
        'high': np.random.rand(100) * 10 + 105,
        'low': np.random.rand(100) * 10 + 95,
        'close': np.random.rand(100) * 10 + 100,
        'volume': np.random.randint(1000, 5000, size=100)
    }, index=dates)

    # Add technical features
    sample_data_with_features = add_technical_features(sample_data)

    print("Sample Data with Features:")
    print(sample_data_with_features.head(10))
    print("\nColumns with Features:")
    print(sample_data_with_features.columns.tolist())
    
    # Prepare training data
    X, y = prepare_training_data(sample_data_with_features)
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution: {y.value_counts()}")