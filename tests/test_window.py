import pytest

from model.loader import *
from model.stocks import StockColumn
from model.window import WindowGenerator


@pytest.fixture(scope="module")
def window():
    return WindowGenerator(24, 24, 1,
                           [StockColumn.clsprc.name],
                           *split_to_dataframes(load_normalized_dataset('./data/sample.csv')))


def test_window_generator_column_indices():
    """
    0. stkcd
    1. trddt
    2. trdsta
    3. opnprc
    4. hiprc
    5. loprc
    6. clsprc
    7. dnshrtrd
    8. dnvaltrd
    9. dsmvosd
    10. dsmvtll
    11. dretwd
    12. dretnd
    13. adjprcwd
    14. adjprcnd
    15. markettype
    16. capchgdt
    17. ahshrtrd_d
    18. ahvaltrd_d
    19. precloseprice
    20. changeratio
    """
    window = WindowGenerator(24, 24, 1,
                             [StockColumn.clsprc.name],
                             *split_to_dataframes(
                                 load_dataset('./data/sample.csv')))  # here we do not normalize it to preserve order
    assert window.column_indices[StockColumn.stkcd.name] == 0
    assert window.column_indices[StockColumn.precloseprice.name] == 19


def test_example(window):
    example = window.example
    # 33 is the total number of features (including the label) (21 + 6 * 2, for two date columns)
    assert example[0].shape == (32, 24, 33)
    assert example[0].shape[0] == len(example[0])
    assert example[1].shape == (32, 24, 1)
