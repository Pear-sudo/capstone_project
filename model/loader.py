import pandas as pd
from model.preprocessing import *


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_normalized_dataset(path: str) -> pd.DataFrame:
    df = load_dataset(path)
    normalize_dataset(df)
    return df


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
