"""
Yahoo Finance market data provider.
"""
import pandas as pd
import yfinance as yf


class YFinanceMarketData:
    def get_history(self, ticker: str, period: str) -> pd.DataFrame:
        return yf.Ticker(ticker).history(period=period)
