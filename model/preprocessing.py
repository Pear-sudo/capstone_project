import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from model.config import DataConfig, DataConfigLayout
from model.loader import CsmarData, CsmarColumnInfo

pd.options.mode.copy_on_write = True

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


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


class Translation(Enum):
    DELTA = 'delta'


class Preprocessor:
    def __init__(self, strategy: LoadingStrategy):
        self.strategy = strategy
        self.granularity_dic = {
            'year': ['Sgnyea'],
            'day': ['Trddt']
        }

    @staticmethod
    def expect_file(path: str):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"\"{path}\" is not a valid file")

    @staticmethod
    def load_dataset(path: str) -> pd.DataFrame:
        """
        Transform a csv file into a pandas dataframe
        :param path: A csv file path
        :return: pandas dataframe
        """
        df = pd.read_csv(path)
        return df

    def load_normalized_dataset(self, path: str) -> Optional[pd.DataFrame]:
        """
        Transform a csv file into a pandas dataframe and normalize it according to the strategy
        :param path: A csv file path
        :return: A normalized pandas dataframe; None if no data is left after filtering/normalization
        """
        df = self.load_dataset(path)
        df = self.normalize_dataset(df)
        return df

    def load_normalized_csmar_data(self, datas: list[CsmarData]) -> pd.DataFrame:
        combined = pd.DataFrame()
        for data in datas:
            csmar_directory = data.csmar_directory
            data_path = csmar_directory.data
            data_sheet = data.csmar_datasheet

            # skip disabled data
            if data_sheet.disabled:
                continue

            enabled_columns = [c for c in data_sheet.column_infos if c.enabled]
            if len(enabled_columns) == 0:
                continue

            # load the dataframe
            enabled_column_names: list[str] = [info.column_name for info in enabled_columns]
            all_column_names = [info.column_name for info in data_sheet.column_infos]
            granularity_col_name: str = self.detect_granularity(all_column_names)[1]
            if granularity_col_name not in enabled_column_names:
                enabled_column_names.append(granularity_col_name)
            df: DataFrame | None = None
            with open(data_path, 'r') as f:
                # ensure column len match
                df = pd.read_csv(f, usecols=enabled_column_names)
                if len(df.columns) != len(enabled_column_names):
                    raise RuntimeError(f"Number of columns do not match: "
                                       f"expected {len(enabled_column_names)}, "
                                       f"loaded {len(df.columns)}")
                logger.debug(f"Read {len(df.columns)} columns from {data_path}:\n{df.columns}")

            # null safety check
            if df is None:
                logger.warning(f'Skipping {data_path} because pd loaded without data')
                continue

            # filter the data:
            columns_to_filter = [info for info in enabled_columns if info.filter.strip() != '']
            for info in columns_to_filter:
                # split the filter's instructions
                fs: list[str] = [f.strip() for f in info.filter.strip().split(',')]
                # todo this is not safe, add checking
                fs_series = pd.Series(fs).astype(int)
                df = df[df[info.column_name].isin(fs_series)]

            # transform the data
            special_columns = []
            ordinary_columns = []
            for c in enabled_columns:
                if 's' in self.split_instructions(c.instruction):
                    special_columns.append(c)
                else:
                    ordinary_columns.append(c)

            for info in special_columns:
                df = self.execute_instruction(df, info)
            for info in ordinary_columns:
                df = self.execute_instruction(df, info)

            combined = self.combine_dataframes(combined, df)
            logger.info(f"Successfully loaded normalized {data_sheet.data_name}")
        logger.info("Loading finished.")

    @staticmethod
    def split_instructions(ins: str) -> list[str]:
        return [ins.strip() for ins in ins.strip().split(',')]

    def execute_instruction(self, df: pd.DataFrame, info: CsmarColumnInfo) -> pd.DataFrame:
        if info.instruction.strip() == '':
            # perform default transform
            self.auto_transform_column(df, info.column_name)
            return df

        instructions: list[str] = self.split_instructions(info.instruction)
        for i in instructions:
            match i:
                case 's':
                    # split into columns
                    unique = df[info.column_name].unique()
                    new_df = pd.DataFrame()
                    for col in unique:
                        partition = df[df[info.column_name] == col].copy()
                        partition = partition.drop(columns=[info.column_name])
                        partition.columns = [f"{name}_{col}" for name in partition.columns]
                        new_df = self.combine_dataframes(new_df, partition)
                    df = new_df
                case _:
                    raise RuntimeError(f"Invalid instruction: {i}")
        return df

    def detect_granularity(self, column_names: list[str], strict: bool = True, granularity_dic=None) -> tuple[str, str]:
        if granularity_dic is None:
            granularity_dic = self.granularity_dic
        detected_granularity = {}

        for name in column_names:
            for granularity, keywords in granularity_dic.items():
                if strict:
                    if name in keywords:
                        detected_granularity.setdefault(granularity, []).append(name)
                else:
                    for k in keywords:
                        if name.find(k) != -1:
                            detected_granularity.setdefault(granularity, []).append(name)

        granularity_len = len(detected_granularity)

        # error checking
        if granularity_len == 0:
            raise RuntimeError('Cannot detect any time granularity')
        elif granularity_len == 1:
            granularity_cols = list(detected_granularity.values())[0]
            if len(detected_granularity) == 1:
                pass
            elif len(detected_granularity) > 1:
                raise RuntimeError(f'Detected more than 1 column in current granularity {detected_granularity}')
            else:
                raise RuntimeError('Unknown condition')
        elif granularity_len > 1:
            raise RuntimeError(f'Detected more than one time granularity: {detected_granularity}')
        else:
            raise RuntimeError('Unknown condition')

        col_name = list(detected_granularity.values())[0][0]
        granularity = list(detected_granularity.keys())[0]

        return granularity, col_name

    def combine_dataframes(self,
                           origin: pd.DataFrame,
                           new: pd.DataFrame,
                           origin_dc: str | None = None,
                           new_dc: str | None = None) -> pd.DataFrame:
        """
        Combines two dataframes, respecting the time.
        :param new_dc: date column
        :param origin_dc: date column
        :param origin:
        :param new:
        :return:
        """
        if len(origin.columns) == 0:
            return new

        if origin_dc is not None and origin_dc not in origin.columns:
            raise ValueError(f"'{origin_dc}' not found in '{origin.columns}'")
        if new_dc is not None and new_dc not in new.columns:
            raise ValueError(f"'{new_dc}' not found in '{new.columns}'")

        if origin_dc is None:
            origin_dc = self.detect_granularity(origin.columns, strict=False)[1]
        if new_dc is None:
            new_dc = self.detect_granularity(new.columns, strict=False)[1]

        combined = pd.merge(origin, new, left_on=origin_dc, right_on=new_dc, how='outer')

        combined.drop(columns=[new_dc], inplace=True)

        return combined

    @staticmethod
    def summarize_csmar_data(datas: list[CsmarData]):
        headers = ['Series Number', 'Acronym', 'Fullname', 'Tran', 'Descriptions']
        count = 0
        table_data = []
        for data in datas:
            data_sheet = data.csmar_datasheet
            if data_sheet.disabled:
                continue
            for column_info in data_sheet.column_infos:
                if not column_info.enabled:
                    continue
                number = count + 1  # begin with 1
                count += 1

                acronym = column_info.column_name
                fullname = column_info.full_name
                tran = Translation.DELTA.value
                description = column_info.full_name + column_info.explanation

                table_data.append([number, acronym, fullname, tran, description])
        tab = tabulate(table_data, headers=headers)
        print(tab)

    @staticmethod
    def split_to_dataframes(df: pd.DataFrame, ratio: tuple[float, float, float] = (0.7, 0.2, 0.1)) \
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

    def auto_transform_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        pass

    def delta(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def ln(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

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


if __name__ == "__main__":
    config = DataConfig(DataConfigLayout(Path('./config/data')))
    config.auto_config(r'/Users/a/playground/freestyle/')
    preprocessor = Preprocessor(StockLoadingStrategy())
    preprocessor.load_normalized_csmar_data(config.derived_csmar_datas)
