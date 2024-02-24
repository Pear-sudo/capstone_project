import re
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True


class LoadingStrategy:
    def __init__(self):
        self.include: list[str] = []
        self.exclude: list[str] = []
        self.conditions: Mapping[str, Any] = {}

        # raw (unprocessed) columns count
        self.raw_columns: Optional[int] = None
        # how many columns you would expect after all the processing
        self.expected_columns: Optional[int] = None

        self.beginning_year = None


class StockLoadingStrategy(LoadingStrategy):
    def __init__(self):
        super().__init__()
        self.conditions = {
            "trdsta": 1,
        }
        self.exclude = ['markettype', 'capchgdt', 'trdsta']
        self.exclude.append('stkcd')  # use embedding to handle 300 hundred categorical data; needs a lot of work;
        # forget about it for now
        self.exclude += ['ahshrtrd_d', 'ahvaltrd_d']  # these columns often produce NaN after normalization

        self.raw_columns = 21
        self.expected_columns = 21 - len(self.exclude) + 6 * 1  # one date column would need additional 6 columns

        self.beginning_year = 1991


class Normalizer:
    def __init__(self, table: str, ordering_column: str, columns: list[str], std_columns: list[str]):
        self.moving_average_window = 7
        self.moving_average_template = f"""
with stats as (select opnprc,
                      avg(opnprc) over win    as avg,
                      stddev(opnprc) over win as std
               from {table}
               where stkcd = '000001'
               window win as (order by {ordering_column} rows between {self.moving_average_window - 1} preceding and current row))
select case
           when std = 0 then null
           else (opnprc - avg) / std
           end
from stats
"""

    def sql(self):
        pass


class Preprocessor:
    def __init__(self, strategy: LoadingStrategy):
        self.strategy = strategy

    def load_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def load_normalized_dataset(self, path: str) -> Optional[pd.DataFrame]:
        df = self.load_dataset(path)
        df = self.normalize_dataset(df)
        return df

    def split_to_dataframes(self, df: pd.DataFrame, ratio: tuple[float, float, float] = (0.7, 0.2, 0.1)) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tolerance = 1e-6
        if not abs(sum(ratio) - 1) < tolerance:
            raise ValueError(f"The sum of ratio must be equal to 1, current sum: {sum(ratio)}")

        n = len(df)
        cut_train_val = int(n * ratio[0])
        cut_val_test = int(n * (ratio[0] + ratio[1]))

        train_df: pd.DataFrame = df[:cut_train_val]
        val_df: pd.DataFrame = df[cut_train_val:cut_val_test]
        test_df: pd.DataFrame = df[cut_val_test:]

        if train_df is None or val_df is None or test_df is None:
            raise RuntimeError('Dataset is empty for unknown reasons.')

        return train_df, val_df, test_df

    def normalize_date(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        sample = df[date_column].iloc[0]
        if not self.is_valid_date(sample):
            raise ValueError('It seems like the specified column is not of the form YYYY-MM-DD.')

        df[date_column] = pd.to_datetime(df[date_column])

        base_year = self.strategy.beginning_year
        if base_year is None:
            base_year = df[date_column].dt.year.min()
        df[f'{date_column}_year_reduced'] = df[date_column].dt.year - base_year

        df[f'{date_column}_sin_month'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
        df[f'{date_column}_cos_month'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)

        df[f'{date_column}_sin_day'] = np.sin(2 * np.pi * df[date_column].dt.day / 31)
        df[f'{date_column}_cos_day'] = np.cos(2 * np.pi * df[date_column].dt.day / 31)

        # Monday=0, Sunday=6
        df[f'{date_column}_sin_dayofweek'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
        df[f'{date_column}_cos_dayofweek'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)

        df.drop(date_column, axis=1, inplace=True)
        return df

    def normalize_nan(self, df: pd.DataFrame):
        """
        Delete the columns whose values are all nan (not a number).
        :param df:
        :return:
        """
        df.dropna(axis=1, how="all", inplace=True)  # first deal columns
        df.dropna(axis=0, how="any", inplace=True)  # then rows

    def normalize_values(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # todo this is not correct: you should use moving averages
        """
        (v - mean) / std
        :param
        :return:
        """
        train_mean = train.mean()
        train_std = train.std()

        train = (train - train_mean) / train_std
        val = (val - train_mean) / train_std  # again, this is to prevent data leakage
        test = (test - train_mean) / train_std

        return train, val, test

    def is_valid_date(self, s: str) -> bool:
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(s)))  # that's why I hate python

    def normalize_dataset(self, df: pd.DataFrame) -> Optional[
        pd.DataFrame]:
        if len(df.columns) != self.strategy.raw_columns and self.strategy.raw_columns is not None:
            raise ValueError(f"Expected {self.strategy.raw_columns} raw columns, found {len(df.columns)}")

        for column in df.columns:
            if column in self.strategy.conditions:
                df = df[df[column] == self.strategy.conditions[column]]
            if column in self.strategy.exclude:
                df.drop(column, axis=1, inplace=True)

        try:
            date_cols: list[str] = [col for col in df.columns if self.is_valid_date(df[col].iloc[0])]
        except IndexError:
            # this df has no rows after filtering
            return None
        for date_col in date_cols:
            df = self.normalize_date(df, date_col)

        return df

    def post_normalize(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        for normalization that needs to be done differently in each dataset
        :param
        :return:
        """
        train, val, test = self.normalize_values(train, val, test)

        # nan guard
        self.normalize_nan(train)
        self.normalize_nan(val)
        self.normalize_nan(test)

        if len(train.columns) != self.strategy.expected_columns and self.strategy.expected_columns is not None:
            raise ValueError(
                f"Expected {self.strategy.expected_columns}, got {len(train.columns)} during post normalization")
        return train, val, test
