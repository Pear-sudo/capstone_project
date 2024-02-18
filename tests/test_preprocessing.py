import pytest

from model.loader import *
from model.preprocessing import *
from model.stocks import StockColumn


@pytest.fixture(scope="module")
def data():
    return load_dataset('./data/sample.csv')


def test_normalize_date(data):
    df = data

    date_col = StockColumn.trddt.name
    normalize_date(df, date_col)

    expected_cols = ['sin_dayofweek', 'sin_day', 'sin_month', 'year_reduced']
    expected_cols = [date_col + '_' + col for col in expected_cols]
    assert all(column in df.columns for column in expected_cols)

    assert date_col not in df.columns


def test_is_valid_date():
    assert is_valid_date('2024-01-01') is True
    assert is_valid_date('000001') is False
