from __future__ import annotations

import os.path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from pandas import DataFrame

from model.preprocessing import Preprocessor, StockLoadingStrategy
from model.stocks import StockColumn


class WindowGenerator:
    """
    """

    @staticmethod
    def get_single_step_window(
            label_columns: list[str],
            data: Optional[str | pd.DataFrame] = None,
            train_df: Optional[DataFrame] = None,
            val_df: Optional[DataFrame] = None,
            test_df: Optional[DataFrame] = None) -> WindowGenerator:
        return WindowGenerator(1, 1, 1, label_columns, data, train_df, val_df, test_df)

    def __init__(self,
                 input_width: int, label_width: int, shift: int,
                 label_columns: list[str],
                 data: Optional[str | pd.DataFrame] = None,
                 train_df: Optional[DataFrame] = None,
                 val_df: Optional[DataFrame] = None,
                 test_df: Optional[DataFrame] = None):

        self.preprocessor = Preprocessor(StockLoadingStrategy())

        self.is_mixed_dataset = False

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # for mixed dataset only
        self.train_dfs: tuple[DataFrame, ...] | None = None
        self.val_dfs: tuple[DataFrame, ...] | None = None
        self.test_dfs: tuple[DataFrame, ...] | None = None

        self.data = data
        self.path = None

        self.import_data()

        # Work out the label column indices (as a map).
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        if not self.is_mixed_dataset:
            self.column_indices: dict[str:int] = {name: i for i, name in  # name is of type str
                                                  enumerate(self.train_df.columns)}
        else:
            self.column_indices: dict[str:int] = {name: i for i, name in
                                                  enumerate(self.train_dfs[0].columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width  # prediction
        self.shift = shift  # shift includes label width

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]  # a np array to identify inputs

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def import_data(self):
        if self.train_df is not None and self.val_df is not None and self.test_df is not None:
            # when the data has been provided in the initializer
            pass
        else:
            if type(self.data) is str:
                self.path = self.data
                if os.path.isfile(self.data):
                    self.import_from_file()
                elif os.path.isdir(self.data):
                    self.data = os.path.abspath(self.data)
                    self.is_mixed_dataset = True
                    self.import_from_directory()
                else:
                    raise ValueError(f"{self.data} is not a valid file or directory")
            elif type(self.data) is pd.DataFrame:
                self.import_from_dataframe()
            else:
                raise ValueError('Neither dfs nor data is valid.')

    def import_from_dataframe(self):
        self.train_df, self.val_df, self.test_df = self.preprocessor.split_to_dataframes(self.data)

        # security check
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise RuntimeError('Some df is None for unknown reasons.')
        else:
            self.train_df, self.val_df, self.test_df = self.preprocessor.post_normalize(self.train_df,
                                                                                        self.val_df, self.test_df, )

    def import_from_file(self):
        self.data = self.preprocessor.load_normalized_dataset(self.data)
        if self.data is None:
            raise ValueError(f'No data is left after filtering {self.path}')
        self.import_from_dataframe()

    def import_from_directory(self):
        files = os.listdir(self.data)
        datasets = [self.preprocessor.load_normalized_dataset(os.path.join(self.data, file)) for file in files]
        datasets = [dataset for dataset in datasets if dataset is not None]
        features = [dataset.shape[1] for dataset in datasets]
        self.check_features(features)
        del files
        spilt_datasets = [self.preprocessor.split_to_dataframes(dataset) for dataset in datasets]
        del datasets
        post_normalized_spilt_datasets = [self.preprocessor.post_normalize(*s_dataset) for s_dataset in
                                          spilt_datasets]
        del spilt_datasets
        features = [t[0].shape[1] for t in post_normalized_spilt_datasets]
        self.check_features(features)  # this often goes wrong
        # the memory consumption at this point is about 559M
        self.train_dfs, self.val_dfs, self.test_dfs = zip(*post_normalized_spilt_datasets)

    def check_features(self, features: list[int]):
        if len(set(features)) != 1:
            print(len(set(features)))
            print(features)
            print(set(features))
            exit(1)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    @property
    def train(self):
        if self.is_mixed_dataset:
            return self.make_mixed_dataset(self.train_dfs)
        else:
            return self.make_dataset(self.train_df)

    @property
    def val(self):
        if self.is_mixed_dataset:
            return self.make_mixed_dataset(self.val_dfs)
        else:
            return self.make_dataset(self.val_df)

    @property
    def test(self):
        if self.is_mixed_dataset:
            return self.make_mixed_dataset(self.test_dfs)
        else:
            return self.make_dataset(self.test_df)

    @property
    def example(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Get and cache a training example batch of `inputs, labels` for plotting.
        :return: [inputs tensor, labels tensor]
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def split_window(self, features: tf.Tensor):
        # (batch, time, features)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        # Also note that you only sliced the second axis.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, plot_col: str, model: Model = None, max_subplots: int = 3):
        """
        The Plot is based on the example dataset.
        :param plot_col: What you want to predict
        :param model: If provided, the prediction made by the model will be plotted;
        otherwise, only inputs (of one feature, usually the target itself) and labels are plotted.
        :param max_subplots:
        :return:
        """
        inputs, labels = self.example  # they are just tensors
        if type(plot_col) is not str:
            raise ValueError('plot_col must be a string')
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))  # len(inputs) equals to inputs.shape[0], which is the batch axis

        plt.figure(figsize=(12, 8))
        # plot one window at a time (choose one from batch axis)
        for n in range(max_n):
            # region inputs
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col}')
            # x: input width start from 0; y: (batch, time width, feature), the label
            input_to_plot = inputs[n, :, plot_col_index]
            plt.plot(self.input_indices, input_to_plot,
                     label='Inputs', marker='.', zorder=-10)
            # endregion

            # region labels
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            # endregion

            # region predictions
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
            # endregion

            if n == 0:
                plt.legend()

        plt.xlabel('Time [d]')

    def make_dataset(self, data: DataFrame):  # there should be 29 features
        """
        This function assumes that the input data only contains one stock's data.
        To train multi-stock models, you need to first divide each stock into sequences and then mixed them.
        Use make_mixed_dataset() to achieve this.
        :param data:
        :return:
        """
        # note that this function's input is divided to train, validation and test,
        # so shuffling should not bring leakage issues
        data = np.array(data, dtype=np.float32)
        ds: tf.data.Dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,  # the origin of time axis
            sequence_stride=1,
            # There are actually three levels of order:
            # Whole array --- batches
            # One batch --- windows in each batch
            # One window --- timepoints in each window
            # It's VERY important to know that this shuffle only impacts the first level order;
            # Time order in other levels are preserved.
            shuffle=True,
            batch_size=32, )  # that's how you get the batch axis in split_window's input

        ds = ds.map(self.split_window)  # (batch, time, features)

        return ds

    def make_mixed_dataset(self, dataframes: tuple[DataFrame, ...]):
        # dataframes = [d for d in dataframes if d.shape[1] == 28]
        # print(len(dataframes))
        datasets = []

        for dataframe in dataframes:
            data = np.array(dataframe, dtype=np.float32)
            del dataframe

            ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,  # Shuffling happens after combining the datasets
                batch_size=32,
            )

            ds = ds.map(self.split_window)
            datasets.append(ds)

        combined_ds = tf.data.Dataset.sample_from_datasets(datasets)

        combined_ds = combined_ds.shuffle(buffer_size=1000)

        return combined_ds


class WindowGeneratorStock(WindowGenerator):
    """
    A window generator tailored for stock data.
    """

    label_column = StockColumn.clsprc.name
    label_columns = [label_column]

    @classmethod
    def get_single_step_window(
            cls,
            label_columns: Optional[list[str]] = None,
            data: Optional[str | pd.DataFrame] = None,
            train_df: Optional[DataFrame] = None,
            val_df: Optional[DataFrame] = None,
            test_df: Optional[DataFrame] = None) -> WindowGeneratorStock:
        if label_columns is None:
            label_columns = cls.label_columns
        return WindowGeneratorStock(1, 1, 1, label_columns, data, train_df, val_df, test_df)

    @classmethod
    def get_wide_step_window(
            cls,
            label_columns: Optional[list[str]] = None,
            data: Optional[str | pd.DataFrame] = None,
            train_df: Optional[DataFrame] = None,
            val_df: Optional[DataFrame] = None,
            test_df: Optional[DataFrame] = None) -> WindowGeneratorStock:
        if label_columns is None:
            label_columns = cls.label_columns
        return WindowGeneratorStock(24, 24, 1, label_columns, data, train_df, val_df, test_df)

    def __init__(self,
                 input_width: int, label_width: int, shift: int,
                 label_columns: Optional[list[str]] = None,
                 data: Optional[str | pd.DataFrame] = None,
                 train_df: Optional[DataFrame] = None,
                 val_df: Optional[DataFrame] = None,
                 test_df: Optional[DataFrame] = None):
        # because the default value shall not be mutable, init the default value (which is a list) here.
        if label_columns is None:
            label_columns = WindowGeneratorStock.label_columns

        super().__init__(input_width, label_width, shift, label_columns, data, train_df, val_df, test_df)

    def plot(self, plot_col: Optional[str] = None, model: Model = None, max_subplots: int = 3):
        if plot_col is None:
            plot_col = WindowGeneratorStock.label_column
        super().plot(plot_col, model, max_subplots)
