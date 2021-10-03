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

DEFAULT_START = '2016-01-01'
DEFAULT_END = '2021-04-30'

def download(ticker, start_date=DEFAULT_START, end_date=DEFAULT_END):
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
                          'AMZN', 'AAPL', 'GOOG', 'NFLX', 'JPM', 'BAC', 'BA', 'MA', 'GBPUSD=X', 'GOOGL')
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    neighborhood_size: int = 2
    initial_embedding_size: int = field(init=False)
    dataset: pd.DataFrame = field(init=False)
    data_length: int = field(init=False)

    def __post_init__(self):
        stock_dfs = [download(ticker, start_date=self.start, end_date=self.end) for ticker in self.tickers]
        close_prices = [get_ts(df)[2] for df in stock_dfs]
        # Changing names of columns
        close_prices = [df.rename(self.tickers[i]) for i, df in enumerate(close_prices)]
        df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), close_prices)
        self.dataset = df_final
        self.initial_embedding_size = len(self.tickers)
        self.data_length = len(df_final)
        self.id2ts = dict()
        self.ts2id = dict()
        for i, ticker in enumerate(self.tickers):
            self.id2ts[i] = ticker
            self.ts2id[ticker] = i

    def __len__(self):
        return self.data_length

    def get_emb_size(self):
        return self.initial_embedding_size

    def derive_relevant_indices(self, idx) -> List[int]:
        unfiltered_range = range(idx - self.neighborhood_size, idx + self.neighborhood_size + 1)

        def mirror(i):
            if i > self.data_length - 1:
                return 2 * self.data_length - i - 1
            if i < 0:
                return -i
            return i

        return list(map(mirror, filter(lambda x: x != idx, unfiltered_range)))

    def __getitem__(self, index):
        """
        Returns two tensors [u_embedding, v_embedding].
        U is a numpy array of [1, initial_embedding_size],
        V is a numpy array of [N, initial_embedding size], where N is usually neighborhood_size * 2
        :param index:
        :return:
        """
        u_embedding = self.dataset.iloc[index, :].to_numpy().reshape(1, -1)
        v_embedding = self.dataset.iloc[self.derive_relevant_indices(index), :].to_numpy()
        return u_embedding, v_embedding

    @staticmethod
    def collate(batches):
        all_u = [u for u, _ in batches if len(batches) > 0]
        all_v = [v for _, v in batches if len(batches) > 0]
        return torch.FloatTensor(all_u), torch.FloatTensor(all_v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Enter a list of Yahoo Finance security tickers", nargs="+", type=str,
                        default=['^GSPC', '^STI', '^HSI', '^FTSE', '^IXIC', '^TNX', 'GC=F', 'KODK', 'TSLA', 'MSFT',
                                 'FB', 'AMZN', 'AAPL', 'GOOG', 'NFLX', 'JPM', 'BAC', 'BA', 'MA', 'GBPUSD=X', 'GOOGL'])
    parser.add_argument("--start", help="Start date of format YYY-MM-DD", type=str, default=DEFAULT_START)
    parser.add_argument("--end", help="End date of format YYY-MM-DD", type=str, default=DEFAULT_END)
    args = parser.parse_args()
    tickers = args.tickers
    start_date = args.start
    end_date = args.end
    etl_class = YahooFinanceETL(start=start_date, end=end_date)
