import pandas as pd
from model.loader import split_to_dataframes


def test_split_to_dataframes():
    # Create a stub DataFrame
    df = pd.DataFrame({
        'col1': range(100),
        'col2': range(100, 200)
    })

    train_df, val_df, test_df = split_to_dataframes(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
