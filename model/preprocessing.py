import logging
import random
import re
import string
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from model.config import DataConfig, DataConfigLayout
from model.loader import CsmarData, CsmarColumnInfo

"""
This module preprocesses the data loaded by the 'loader' before passing it to 'window' module
"""

pd.options.mode.copy_on_write = True

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def count_words(s: str):
    return len(s.strip().split())


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
        self.ending_year = None


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

        self.beginning_year = 2020
        self.ending_year = 2023


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


class Granularity(Enum):
    DAILY = 'daily'
    YEARLY = 'yearly'
    MONTHLY = 'monthly'


class Preprocessor:
    def __init__(self, strategy: LoadingStrategy):
        self.strategy = strategy
        self.granularity_dic = {
            Granularity.YEARLY: ['Sgnyea'],
            Granularity.MONTHLY: ['Staper'],
            Granularity.DAILY: ['Trddt', 'Exchdt', 'Clsdt', 'Date']
        }
        self.column_map: Dict[str: list[CsmarColumnInfo]] = {}

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

    def load_partial_df(self, path: Path, enabled_column_names: list[str], all_column_names: list[str]) -> pd.DataFrame:
        granularity_col_name: str = self.detect_granularity(all_column_names)[1]
        if granularity_col_name not in enabled_column_names:
            enabled_column_names.append(granularity_col_name)
        df: DataFrame | None = None
        with open(path, 'r') as f:
            # ensure column len match
            df = pd.read_csv(f, usecols=enabled_column_names)
            if len(df.columns) != len(enabled_column_names):
                raise RuntimeError(f"Number of columns do not match: "
                                   f"expected {len(enabled_column_names)}, "
                                   f"loaded {len(df.columns)}")
            logger.debug(f"Read {len(df.columns)} columns from {path}:\n{df.columns}")

        # only select data within the date range
        temporary_date_column_name = 'date_column_DF2ABBF3'
        df[temporary_date_column_name] = pd.to_datetime(df[granularity_col_name])
        df = df[(df[temporary_date_column_name].dt.year >= self.strategy.beginning_year)
                & (df[temporary_date_column_name].dt.year <= self.strategy.ending_year)]
        df.drop(temporary_date_column_name, axis=1, inplace=True)
        return df

    def load_normalized_csmar_data(self, datas: list[CsmarData], no_transform: bool = False,
                                   anchor: CsmarData = None) -> pd.DataFrame:
        combined = pd.DataFrame()
        if anchor:
            datas.remove(anchor)
            combined = self.load_normalized_csmar_data([anchor], no_transform=no_transform)
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
            all_column_names: list[str] = [info.column_name for info in data_sheet.column_infos]
            df = self.load_partial_df(path=data_path,
                                      enabled_column_names=enabled_column_names,
                                      all_column_names=all_column_names)

            # null safety check
            if df is None:
                logger.warning(f'Skipping {data_path} because pd loaded without data')
                continue

            # filter the data:
            columns_to_filter = [info for info in enabled_columns if info.filter.strip() != '']
            for info in columns_to_filter:
                # split the filter's instructions
                fs: list[str] = [f.strip() for f in info.filter.strip().split(',')]
                if not fs[0] in string.ascii_letters:
                    fs_series = pd.Series(fs).astype(int)
                else:
                    fs_series = pd.Series(fs)
                df = df[df[info.column_name].isin(fs_series)]
                if fs[0] in string.ascii_letters:
                    df.drop(info.column_name, axis=1, inplace=True)
                    enabled_columns.remove(info)

            # transform the data
            df = self.execute_instruction(df, enabled_columns, no_transform=no_transform)

            left_join = False if anchor is None else True
            combined = self.combine_dataframes(combined, df, left_join=left_join)
            logger.info(f"Successfully loaded normalized {data_sheet.data_name}")
        logger.info("Loading finished.")
        return combined

    @staticmethod
    def split_instructions(ins: str) -> list[str]:
        return [ins.strip() for ins in ins.strip().split(',')]

    def execute_instruction(self, df: pd.DataFrame, column_infos: list[CsmarColumnInfo],
                            no_transform: bool = False) -> pd.DataFrame:
        special_columns = []
        ordinary_columns = []
        split_column_infos: list[CsmarColumnInfo] = []
        _, granularity_column = self.detect_granularity(df.columns)
        for c in column_infos:
            if 's' in self.split_instructions(c.instruction):
                special_columns.append(c)
            else:
                ordinary_columns.append(c)

        suffixes: list[str] = []
        for info in special_columns:
            instructions: list[str] = self.split_instructions(info.instruction)
            for i in instructions:
                match i:
                    case 's':
                        # split into columns
                        split_column_infos.append(info)
                        suffixes = df[info.column_name].unique()
                        new_df = pd.DataFrame()
                        for suffix in suffixes:
                            partition = df[df[info.column_name] == suffix].copy()
                            partition = partition.drop(columns=[info.column_name])
                            partition.columns = [f"{name}_{suffix}" if name != granularity_column else name
                                                 for name in partition.columns]
                            new_df = self.combine_dataframes(new_df, partition, fill=False)
                        df = new_df
                    case _:
                        raise RuntimeError(f"Invalid instruction: {i}")

        for info in ordinary_columns:
            # first check if the name is changed due to split (s) operation, if so, correct it
            original_column_name = info.column_name
            column_names = []
            if len(suffixes) > 0:
                for suffix in suffixes:
                    column_names.append(f"{original_column_name}_{suffix}")
            else:
                column_names.append(original_column_name)

            for column_name in column_names:
                # note the column_name may not be the original one if there is a combination
                discarded = False
                if info.instruction.strip() == '':
                    # perform default transform
                    if not no_transform:
                        df = self.auto_transform_column(df, column_name)
                else:
                    instructions: list[str] = self.split_instructions(info.instruction)
                    for i in instructions:
                        match i:
                            case 'max':
                                idx = df.groupby(granularity_column)[column_name].idxmax()
                                df = df.loc[idx]
                            case 'min':
                                idx = df.groupby(granularity_column)[column_name].idxmin()
                                df = df.loc[idx]
                            case 'd':
                                # drop/delete the row
                                df = df.drop(column_name, axis=1)
                                discarded = True  # do not perform transform on this column
                            case _:
                                raise RuntimeError(f"Invalid instruction: {i}")
                    if not no_transform and not discarded:
                        df = self.auto_transform_column(df, column_name)

                if not discarded:
                    if column_name in self.column_map:
                        raise RuntimeError(f"Column {column_name} already exists")
                    self.column_map[column_name] = [info] + split_column_infos

        return df

    def detect_granularity(self,
                           column_names: list[str],
                           strict: bool = True,
                           granularity_dic=None) -> tuple[Granularity, str]:
        """

        :param column_names:
        :param strict:
        :param granularity_dic:
        :return: (granularity, col_name)
        """
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
            raise RuntimeError(f'Cannot detect any time granularity in the following columns: {column_names}')
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
                           new_dc: str | None = None,
                           date_column_name: str | None = 'Date',
                           fill: bool = True,
                           left_join: bool = False
                           ) -> pd.DataFrame:
        """
        Combines two dataframes, respecting the time.
        :param left_join:
        :param fill: if fill to date
        :param date_column_name:
        :param new_dc: date column
        :param origin_dc: date column
        :param origin:
        :param new:
        :return:
        """

        def detect_date_column_name(date_column_to_detect: str | None, df: pd.DataFrame) -> None | str:
            if date_column_name is not None and date_column_to_detect is None:
                if date_column_name in df.columns:
                    return date_column_name
            return date_column_to_detect

        def check_df_key(df: pd.DataFrame, key_col: str):
            if df[key_col].isna().any():
                raise RuntimeError(f"Key {key_col} has NaN values")
            d = df[key_col].duplicated()
            if d.any():
                criminal = df[df.duplicated(key_col, keep=False)]
                raise RuntimeError(f"Key {key_col} has duplicated values")

        if len(origin.columns) == 0:
            return new

        if origin_dc is not None and origin_dc not in origin.columns:
            raise ValueError(f"'{origin_dc}' not found in '{origin.columns}'")
        if new_dc is not None and new_dc not in new.columns:
            raise ValueError(f"'{new_dc}' not found in '{new.columns}'")

        origin_dc = detect_date_column_name(origin_dc, origin)
        new_dc = detect_date_column_name(new_dc, new)

        origin_g = None

        if origin_dc is None:
            if fill:
                _, origin_dc, origin = self.detect_fill_granularity(origin, strict=False)
            else:
                origin_g, origin_dc = self.detect_granularity(origin.columns, strict=False)
        if new_dc is None:
            if fill:
                _, new_dc, new = self.detect_fill_granularity(new, strict=False)
            else:
                _, new_dc = self.detect_granularity(new.columns, strict=False)

        check_df_key(origin, origin_dc)
        check_df_key(new, new_dc)
        combined = None
        if not left_join:
            if origin_dc == new_dc:
                combined = pd.merge(origin, new, on=origin_dc, how='outer')
            else:
                # we need to mimic coalesce in sql, otherwise there will be nan in the key, which is fatal
                combined = pd.merge(origin, new, left_on=origin_dc, right_on=new_dc, how='outer')
                intermediate_col = ''.join(random.choices(string.ascii_letters, k=10))
                combined[intermediate_col] = combined[origin_dc].combine_first(combined[new_dc])
                combined = combined.drop([origin_dc, new_dc], axis=1)
                combined.rename(columns={intermediate_col: origin_dc}, inplace=True)
        else:
            if origin_dc == new_dc:
                combined = pd.merge(origin, new, on=origin_dc, how='left')
            else:
                combined = pd.merge(origin, new, left_on=origin_dc, right_on=new_dc, how='left')
                combined.drop(new_dc, axis=1, inplace=True)

        # let's do some security check
        check_df_key(combined, origin_dc)

        if date_column_name and date_column_name not in combined.columns:
            if fill or origin_g is Granularity.DAILY:
                combined.rename(columns={origin_dc: date_column_name}, inplace=True)

        return combined

    def detect_fill_granularity(self,
                                df: pd.DataFrame,
                                strict: bool = True,
                                granularity_dic=None) -> tuple[Granularity, str, pd.DataFrame]:
        granularity, col_name = self.detect_granularity(df.columns.tolist(),
                                                        strict=strict,
                                                        granularity_dic=granularity_dic)
        if granularity is Granularity.DAILY:
            pass
        elif granularity is Granularity.YEARLY:
            df = self.fill_yearly_data(df, col_name)
        elif granularity is Granularity.MONTHLY:
            df = self.fill_monthly_data(df, col_name)
        else:
            raise RuntimeError(f"Unrecognized granularity")

        return granularity, col_name, df

    @staticmethod
    def fill_monthly_data(df: pd.DataFrame, year_column: str) -> pd.DataFrame:
        expanded_df = pd.DataFrame()

        for i, row in df.iterrows():
            s: str = row[year_column]
            tokens = s.split('-')
            year = int(tokens[0])
            month = int(tokens[1])
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            row_data = row.drop(year_column).to_dict()
            year_df = pd.DataFrame({
                year_column: dates,
                **row_data
            })

            expanded_df = pd.concat([expanded_df, year_df], ignore_index=True)

        # correct the type
        expanded_df[year_column] = expanded_df[year_column].dt.strftime('%Y-%m-%d')

        return expanded_df

    @staticmethod
    def fill_yearly_data(df: pd.DataFrame, year_column: str) -> pd.DataFrame:
        expanded_df = pd.DataFrame()

        for i, row in df.iterrows():
            year = row[year_column].astype(int)
            start_date = pd.Timestamp(year=year, month=1, day=1)
            end_date = pd.Timestamp(year=year, month=12, day=31)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            row_data = row.drop(year_column).to_dict()
            year_df = pd.DataFrame({
                year_column: dates,
                **row_data
            })

            expanded_df = pd.concat([expanded_df, year_df], ignore_index=True)

        # correct the type
        expanded_df[year_column] = expanded_df[year_column].dt.strftime('%Y-%m-%d')

        return expanded_df

    def summarize_csmar_data(self, datas: list[CsmarData]):
        summary = {'Series Number': [],
                   'Frequency': [],
                   'Acronym': [],
                   'Fullname': [],
                   'Description': []}
        stat = {
            'Acronym': [],
            'Obs': [],
            'Mean': [],
            'Max': [],
            'Min': [],
            'Std': [],
            'Skew': [],
            'Kurt': []
        }

        anchor = None
        for data in datas:
            name = data.csmar_datasheet.data_name
            if name == 'Aggregated Daily Market Returns':
                anchor = data
                break
        df = self.load_normalized_csmar_data(datas, no_transform=True, anchor=anchor)

        count = 0
        granularity, granularity_col_name = self.detect_granularity(df.columns)
        # we do not need to summarize date
        column_names: list[str] = [c for c in df.columns if c != granularity_col_name]
        # do not use the infos in the data sheet since some column may be discarded according to loading strategy
        for column_name in column_names:
            column_infos = self.column_map[column_name]
            if len(column_infos) > 2:
                raise RuntimeError('Do not currently support more than 2 foreign columns')
            self_column_info: CsmarColumnInfo = column_infos[0]
            foreign_column_info: CsmarColumnInfo | None = None
            if len(column_infos) == 2:
                foreign_column_info = column_infos[1]
            if not self_column_info.enabled:
                continue

            number = count + 1  # begin with 1
            summary['Series Number'].append(number)
            count += 1

            frequency = granularity.value
            summary['Frequency'].append(frequency)

            acronym = column_name
            summary['Acronym'].append(acronym)

            fullname = self_column_info.full_name
            if count_words(fullname) <= 1:
                fullname = self_column_info.explanation
            if foreign_column_info:
                foreign_explanation = foreign_column_info.explanation
                token_dic: Dict[str, str] = {}
                if ';' in foreign_explanation:
                    tokens = [t.strip() for t in foreign_explanation.split(';')]
                    for token in tokens:
                        kv = token.split('=')
                        k = kv[0].strip()
                        if k == 'JYP':
                            k = 'JPY'
                        v = kv[1].strip()
                        token_dic[k] = v
                elif ':' in foreign_explanation:
                    split_pattern = r'(?<=\D)(?=\d{6})'
                    tokens = re.split(split_pattern, foreign_explanation)
                    for token in tokens:
                        kv = token.split(':')
                        k = kv[0].strip()
                        if all(c in string.digits for c in k):
                            k = str(int(k))
                        v = kv[1].strip()
                        token_dic[k] = v
                if len(token_dic) > 0:
                    k = acronym.split('_')[-1]
                    v = token_dic[k]
                    fullname = fullname + f' ({v})'
            summary['Fullname'].append(fullname)

            description = self_column_info.explanation  # let's treat full name as description
            summary['Description'].append(description)

            stat['Acronym'].append(acronym)

            obs = len(df)
            stat['Obs'].append(obs)

            mean = df[column_name].mean()
            stat['Mean'].append(mean)

            max_val = df[column_name].max()
            stat['Max'].append(max_val)

            min_val = df[column_name].min()
            stat['Min'].append(min_val)

            std = df[column_name].std()
            stat['Std'].append(std)

            skew = df[column_name].skew()
            stat['Skew'].append(skew)

            kurt = df[column_name].kurt()
            stat['Kurt'].append(kurt)

        summary_df = pd.DataFrame(summary)
        stat_df = pd.DataFrame(stat)

        summary_s = summary_df.to_string(index=False)
        stat_s = stat_df.to_string(index=False)
        print(summary_s)
        print()
        print(stat_s)

        with pd.ExcelWriter('../out/data_stat.xlsx') as writer:
            summary_df.to_excel(writer, sheet_name='description', index=False)
            stat_df.to_excel(writer, sheet_name='statistics', index=False)

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
        df[column_name] = self.delta(df[column_name])
        # I am wondering if it is better to handle NaN in the final combined df rather than at here
        return df

    @staticmethod
    def delta(df: pd.DataFrame) -> pd.DataFrame:
        return df.diff()

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

    @staticmethod
    def drop_nan(df: pd.DataFrame):
        """
        Delete:
        1. The columns whose values are all NaN
        2. The rows that contain any NaN
        :param df:
        :return:
        """
        df.dropna(axis=1, how="all", inplace=True)  # first deal columns
        df.dropna(axis=0, how="any", inplace=True)  # then rows

    @staticmethod
    def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def normalize_values(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) \
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

    @staticmethod
    def is_valid_date(s: str) -> bool:
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(s)))  # that's why I hate python

    def normalize_dataset(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
        self.drop_nan(train)
        self.drop_nan(val)
        self.drop_nan(test)

        if len(train.columns) != self.strategy.expected_columns and self.strategy.expected_columns is not None:
            raise ValueError(
                f"Expected {self.strategy.expected_columns}, got {len(train.columns)} during post normalization")
        return train, val, test


if __name__ == "__main__":
    config = DataConfig(DataConfigLayout(Path('./config/data')))
    config.auto_config(r'/Users/a/playground/freestyle/')
    preprocessor = Preprocessor(StockLoadingStrategy())
    # preprocessor.load_normalized_csmar_data(config.derived_csmar_datas, no_transform=True)
    preprocessor.summarize_csmar_data(config.derived_csmar_datas)
