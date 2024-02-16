from model.loader import *
from model.preprocessing import *
from model.stocks import StockColumn


def test_normalize_date():
    df = load_dataset('./data/sample.csv')
    date_col = StockColumn.trddt.name
    normalize_date(df, date_col)
    assert all(column in df.columns for column in ['sin_dayofweek', 'sin_day', 'sin_month', 'year_reduced'])
    assert date_col not in df.columns


def test_is_valid_date():
    assert is_valid_date('2024-01-01') is True
    assert is_valid_date('000001') is False
