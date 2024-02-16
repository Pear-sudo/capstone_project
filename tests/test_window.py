from model.loader import split_to_dataframes, load_dataset
from model.stocks import StockColumn
from model.window import WindowGenerator


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
    w = WindowGenerator(1, 1, 1, [StockColumn.clsprc.name], *split_to_dataframes(load_dataset('./data/sample.csv')))

    assert w.column_indices[StockColumn.stkcd.name] == 0
    assert w.column_indices[StockColumn.precloseprice.name] == 19
