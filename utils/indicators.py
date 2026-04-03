import pandas as pd
import numpy as np
import talib
from typing import Union


def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average
    """
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average
    """
    return data.ewm(span=period).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns MACD line, signal line, and histogram
    """
    exp1 = calculate_ema(data, fast_period)
    exp2 = calculate_ema(data, slow_period)
    macd = exp1 - exp2
    signal = calculate_ema(macd, signal_period)
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2):
    """
    Calculate Bollinger Bands
    Returns upper band, middle band (SMA), and lower band
    """
    sma = calculate_sma(data, period)
    rolling_std = data.rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return upper_band, sma, lower_band


def calculate_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    """
    Calculate Stochastic Oscillator
    Returns %K and %D lines
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Calculate Average True Range
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Calculate Williams %R
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    return williams_r


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20):
    """
    Calculate Commodity Channel Index
    """
    tp = (high + low + close) / 3  # Typical price
    ma = tp.rolling(window=period).mean()
    mean_deviation = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
    
    cci = (tp - ma) / (0.015 * mean_deviation)
    
    return cci


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Calculate Average Directional Index
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    # Calculate Positive and Negative Directional Movement
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate Smoothed True Range and Directional Movements
    smooth_tr = true_range.rolling(window=period).sum()
    smooth_plus_dm = pd.Series(plus_dm).rolling(window=period).sum()
    smooth_minus_dm = pd.Series(minus_dm).rolling(window=period).sum()
    
    # Calculate Directional Indicators
    plus_di = (smooth_plus_dm / smooth_tr) * 100
    minus_di = (smooth_minus_dm / smooth_tr) * 100
    
    # Calculate DX and ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_obv(close: pd.Series, volume: pd.Series):
    """
    Calculate On-Balance Volume
    """
    obv = np.where(close > close.shift(), volume, np.where(close < close.shift(), -volume, 0))
    obv = pd.Series(obv).cumsum()
    return pd.Series(obv, index=close.index)


# Using TA-Lib for additional indicators
def calculate_talib_indicators(df: pd.DataFrame):
    """
    Calculate various indicators using TA-Lib
    """
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    volumes = df['volume'].values
    
    # Calculate various indicators using TA-Lib
    indicators = {}
    
    # Moving Averages
    indicators['sma_10'] = talib.SMA(close_prices, timeperiod=10)
    indicators['sma_30'] = talib.SMA(close_prices, timeperiod=30)
    indicators['ema_10'] = talib.EMA(close_prices, timeperiod=10)
    indicators['ema_30'] = talib.EMA(close_prices, timeperiod=30)
    
    # RSI
    indicators['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['macd'] = macd
    indicators['macd_signal'] = macdsignal
    indicators['macd_hist'] = macdhist
    
    # Bollinger Bands
    indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close_prices)
    
    # Stochastic
    slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
    indicators['stoch_k'] = slowk
    indicators['stoch_d'] = slowd
    
    # ATR
    indicators['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
    
    # ADX
    indicators['adx_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
    
    # CCI
    indicators['cci_14'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
    
    # Williams %R
    indicators['willr'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
    
    # Return indicators as a DataFrame
    result_df = pd.DataFrame(index=df.index)
    for name, values in indicators.items():
        result_df[name] = values
        
    return result_df


def add_technical_indicators(df: pd.DataFrame, include_talib=True):
    """
    Add technical indicators to a DataFrame
    """
    if df.empty:
        return df
    
    # Calculate our custom indicators
    df = df.copy()
    
    # Moving averages
    df['sma_20'] = calculate_sma(df['close'], 20)
    df['sma_50'] = calculate_sma(df['close'], 50)
    df['ema_12'] = calculate_ema(df['close'], 12)
    df['ema_26'] = calculate_ema(df['close'], 26)
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    # ATR
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic_oscillator(df['high'], df['low'], df['close'], 14)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    # Williams %R
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'], 14)
    
    # CCI
    df['cci_20'] = calculate_cci(df['high'], df['low'], df['close'], 20)
    
    # ADX
    df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    
    # OBV
    df['obv'] = calculate_obv(df['close'], df['volume'])
    
    # If TA-Lib is available, add those indicators too
    if include_talib:
        try:
            talib_indicators = calculate_talib_indicators(df)
            df = pd.concat([df, talib_indicators], axis=1)
        except:
            print("TA-Lib not available, skipping TA-Lib indicators")
    
    return df


if __name__ == "__main__":
    # Example usage
    # Create sample data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.rand(100) * 100 + 100,
        'high': np.random.rand(100) * 10 + 105,
        'low': np.random.rand(100) * 10 + 95,
        'close': np.random.rand(100) * 10 + 100,
        'volume': np.random.randint(1000, 5000, size=100)
    }, index=dates)
    
    # Add technical indicators
    sample_data_with_indicators = add_technical_indicators(sample_data)
    
    print("Sample Data with Indicators:")
    print(sample_data_with_indicators.head(10))
    print("\nColumns with Indicators:")
    print(sample_data_with_indicators.columns.tolist())