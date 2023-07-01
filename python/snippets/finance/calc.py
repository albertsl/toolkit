import numpy as np

def calc_rsi(prices, period: int = 14) -> float:
    """Calculate RSI for a price data time series

    Args:
        prices (pd.Series): Adjusted close price data. Index is the date
        period (int, optional): Period. Defaults to 14.

    Returns:
        float: RSI
    """
    #In case there is not enough data to calc RSI return None
    if len(prices) <= period:
        return None
    
    # Calculate daily price changes
    deltas = np.diff(prices)
    
    # Separate changes into gains and losses
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    
    # Calculate average gains and losses
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.abs(np.mean(losses[-period:]))
    if avg_loss == 0:
        # Avoid division by zero
        return 100
    else:
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))