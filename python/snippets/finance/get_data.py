import os
import pickle
import yfinance as yf
from datetime import datetime, timedelta

def get_price_data(ticker: str):
    """Download stock market price data for the given ticker

    Args:
        ticker (str): Ticker Symbol

    Returns:
        pd.DataFrame: DataFrame containing Open, High, Low, Close, Adj Close and Volume data for the last month
    """
    start_date = datetime.now() - timedelta(days=30)
    today = datetime.now().strftime("%Y-%m-%d")

    file_path = f"data/{ticker}_{today}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = yf.download(ticker, start=start_date)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    return data