import pytest

from model.loader import *
from model.stocks import StockColumn
from model.window import WindowGenerator, WindowGeneratorStock


@pytest.fixture(scope="module")
def window():
    return WindowGenerator(24, 24, 1,
                           [StockColumn.clsprc.name],
                           *split_to_dataframes(load_normalized_dataset('./data/sample.csv')))


def test_window_generator_column_indices(window):
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
    17. ahshrtrd_d nan
    18. ahvaltrd_d nan
    19. precloseprice
    20. changeratio
    """
    indices_test(window)


def test_example(window):
    example = window.example
    # 33 is the total number of features (including the label) (21 + 6 * 2, for two date columns)
    assert example[0].shape == (32, 24, 33)
    assert example[0].shape[0] == len(example[0])
    assert example[1].shape == (32, 24, 1)


def test_window_generator():
    w = WindowGenerator(1, 1, 1, [StockColumn.clsprc.name], './data/sample.csv')
    indices_test(w)


def test_window_generator_stock():
    window = WindowGeneratorStock(24, 24, 1, data='./data/sample.csv')
    indices_test(window)


def test_get_single_step_window():
    window = WindowGenerator.get_single_step_window([StockColumn.clsprc.name], data='./data/sample.csv')
    indices_test(window)


def indices_test(window: WindowGenerator):
    assert window.column_indices[StockColumn.stkcd.name] == 0
    assert window.column_indices[StockColumn.precloseprice.name] == 17
