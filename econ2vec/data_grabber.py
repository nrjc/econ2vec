import argparse
import math
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Dict

import gin
import torch
import yfinance as yf
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


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


@gin.configurable
@dataclass
class YahooFinanceETL(Dataset):
    tickers: List[str] = ('^GSPC', '^STI', '^HSI', '^FTSE', '^IXIC', '^TNX', 'GC=F', 'KODK', 'TSLA', 'MSFT', 'FB',
                          'AMZN', 'AAPL', 'GOOG', 'NFLX', 'JPM', 'BAC', 'BA', 'MA', 'GBPUSD=X')
    start: str = '2020-01-01'
    end: str = '2020-03-31'
    neighborhood_size: int = 2
    inital_embedding_size: int = field(init=False)
    dataset: pd.DataFrame = field(init=False)
    data_length: int = field(init=False)

    def __post_init__(self):
        stock_dfs = [download(ticker, start_date=self.start, end_date=self.end) for ticker in self.tickers]
        close_prices = [get_ts(df)[2] for df in stock_dfs]
        # Changing names of columns
        close_prices = [df.rename(self.tickers[i]) for i, df in enumerate(close_prices)]
        df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), close_prices)
        self.dataset = df_final
        self.inital_embedding_size = len(self.tickers)
        self.data_length = len(df_final)

    def __getitem__(self, index) -> T_co:
        """
        Returns two tensors [u_embedding, v_embedding]. U is a numpy array of [1, initial_embedding_size],
        V is a numpy array of [N, initial_embedding size], where N is usually neighborhood_size * 2
        :param index:
        :return:
        """
        u_embedding = self.dataset.iloc[index, :].to_numpy()
        v_indexes = [abs(i) for i in range(index - self.neighborhood_size, index + self.neighborhood_size) if
                     i != index]
        v_embedding = self.dataset.iloc[v_indexes, :].to_numpy()
        return u_embedding, v_embedding

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v in batch if len(batch) > 0]
        return torch.FloatTensor(all_u), torch.FloatTensor(all_v)

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
    etl_class = YahooFinanceETL(start=start_date, end=end_date)
