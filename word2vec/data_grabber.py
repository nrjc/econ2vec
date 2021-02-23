import argparse

import yfinance as yf
import pandas as pd


def download(ticker, start_date='2020-01-01', end_date='2020-03-31'):
    ticker_df = yf.download(ticker,
                            start=start_date,
                            end=end_date,
                            auto_adjust=True,
                            progress=False)
    return ticker_df


def get_ts(ticker_df):
    high = ticker_df['High']
    low = ticker_df['Low']
    close = ticker_df['Close']
    return high, low, close


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Enter a list of Yahoo Finance security tickers", nargs="+", type=str,
                        default=['^GSPC', '^STI', '^HSI', '^FTSE', '^IXIC', '^TNX', 'GC=F', 'KODK', 'TSLA', 'MSFT',
                                'FB', 'AMZN', 'AAPL', 'GOOG', 'NFLX', 'JPM', 'BAC', 'BA', 'MA', 'GBPUSD=X'])
    parser.add_argument("--start", help="Start date of format YYY-MM-DD", type=str, default='2020-01-01')
    parser.add_argument("--end", help="End date of format YYY-MM-DD", type=str, default='2020-03-31')
    args = parser.parse_args()
    tickers = args.tickers
    start_date = args.start
    end_date = args.end

    for ticker in tickers:
        df = download(ticker, start_date=start_date, end_date=end_date)
        high, low, close = get_ts(df)