import re

import numpy as np
import pandas as pd


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


def normalize_date(df: pd.DataFrame, date_column: str):
    sample = df[date_column].iloc[0]
    if not is_valid_date(sample):
        raise ValueError('It seems like the specified column is not of the form YYYY-MM-DD.')

    df[date_column] = pd.to_datetime(df[date_column])

    df[f'{date_column}_year_reduced'] = df[date_column].dt.year - df[date_column].dt.year.min()

    df[f'{date_column}_sin_month'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
    df[f'{date_column}_cos_month'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)

    df[f'{date_column}_sin_day'] = np.sin(2 * np.pi * df[date_column].dt.day / 31)
    df[f'{date_column}_cos_day'] = np.cos(2 * np.pi * df[date_column].dt.day / 31)

    # Monday=0, Sunday=6
    df[f'{date_column}_sin_dayofweek'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
    df[f'{date_column}_cos_dayofweek'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)

    df.drop(date_column, axis=1, inplace=True)


def normalize_nan(df: pd.DataFrame):
    """
    Delete the columns whose values are all nan (not a number).
    :param df:
    :return:
    """
    df.dropna(axis=1, how="all", inplace=True)  # first deal columns
    df.dropna(axis=0, how="any", inplace=True)  # then rows


def is_valid_date(s: str) -> bool:
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(s)))  # that's why I hate python


def normalize_dataset(df: pd.DataFrame) -> None:
    normalize_nan(df)

    date_cols: list[str] = [col for col in df.columns if is_valid_date(df[col].iloc[0])]
    for date_col in date_cols:
        normalize_date(df, date_col)
