import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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

    return train_df, val_df, test_df
