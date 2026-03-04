"""
Yahoo Finance market data provider.
"""
import pandas as pd
import yfinance as yf


class YFinanceMarketData:
    def get_history(self, ticker: str, period: str) -> pd.DataFrame:
        return yf.Ticker(ticker).history(period=period)

    def get_company_name(self, ticker: str) -> str:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ""
