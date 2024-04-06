import csv
import os
from enum import Enum, auto
from pathlib import Path
from string import Template
from typing import Optional

import pandas as pd
from psycopg import Cursor
from psycopg.rows import Row

from model.db import Database
from model.loader import head, tail

"""
This module contains the specialized functions for stock data
"""


class DataFragment(Enum):
    TRAINING = "training"
    TESTING = "testing"
    VALIDATION = "validation"
    ALL = "all"


class StockColumn(Enum):
    stkcd = auto()
    trddt = auto()
    trdsta = auto()
    opnprc = auto()
    hiprc = auto()
    loprc = auto()
    clsprc = auto()
    dnshrtrd = auto()
    dnvaltrd = auto()
    dsmvosd = auto()
    dsmvtll = auto()
    dretwd = auto()
    dretnd = auto()
    adjprcwd = auto()
    adjprcnd = auto()
    markettype = auto()
    capchgdt = auto()
    ahshrtrd_d = auto()
    ahvaltrd_d = auto()
    precloseprice = auto()
    changeratio = auto()


class Stocks(Database):
    def __init__(self):
        super().__init__()
        self.table_name = "stock"
        self.stocks = None

    def list_stocks(self) -> list[str]:
        if self.stocks is None:
            stocks = self.get_connection().execute(f'select stkcd from {self.table_name} group by stkcd').fetchall()
            stocks = [row[0] for row in stocks]
            stocks.sort()
            self.stocks = stocks
        return self.stocks

    def export_to_csv(self, filename: str = "../data/stocks.csv",
                      dir_name: str = "../data/raw/stocks",
                      stocks: list[str] = None) -> None:
        stocks_to_export = stocks
        if stocks is None:
            stocks_to_export = self.list_stocks()
        else:
            raise ValueError("Stock selection is not supported")  # TODO implement stock selection

        conn = self.get_connection()
        cur = conn.cursor(name="csv_exporter")
        cur.itersize = 10_000
        try:
            for stock in stocks_to_export:
                filename = os.path.join(dir_name, f"{stock}.csv")

                cur.execute(f"SELECT * FROM {self.table_name} where stkcd = '{stock}' order by trddt")

                # Open the CSV file for writing
                with open(filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)

                    # Write the column headers
                    col_names = [desc[0] for desc in cur.description]
                    csvwriter.writerow(col_names)

                    # Write the rows of data
                    while True:
                        rows = cur.fetchmany(cur.itersize)
                        if not rows:
                            break
                        csvwriter.writerows(rows)

                print(f"Stock {stock} exported to {filename}")

        except Exception as e:
            print("Error: ", e)

        finally:
            cur.close()
            conn.close()
        print('Export completed without any error.')


class Stock(Database):
    def __init__(self, stk_id: str):
        super().__init__()
        self.stk_id: str = stk_id
        self.columns = "stkcd, trddt, trdsta, opnprc, hiprc, loprc, clsprc, dnshrtrd"

        self._count: Optional[int] = None
        self.boundary_training_validating = None
        self.boundary_validating_testing = None

        self.query_template = Template(
            f"select {self.columns} from stock where stkcd = '000001' order by trddt offset $offset limit $limit")

    @property
    def count(self) -> int:
        if self._count is None:
            query = (
                f"select count(*) from stock where stkcd = '{self.stk_id}'"
            )
            self._count = self.get_connection().execute(query).fetchone()[0]
        if self.boundary_validating_testing is None and self.boundary_training_validating is None:
            self._calculate_boundaries()
        return self._count

    def _calculate_boundaries(self) -> None:
        self.boundary_training_validating = int(self._count * 0.7)
        self.boundary_validating_testing = int(self._count * (0.7 + 0.2))

    def fetch(self, fragment: DataFragment) -> Cursor[Row]:
        """
        Fetch individual stock data from a database
        :param fragment:
        :return: a psql cursor
        """
        _ = self.count
        match fragment:
            case DataFragment.TRAINING:
                return self.get_connection().execute(
                    self.query_template.substitute(
                        offset=0,
                        limit=self.boundary_training_validating - 1
                    )
                )
            case DataFragment.VALIDATION:
                return self.get_connection().execute(
                    self.query_template.substitute(
                        offset=self.boundary_training_validating,
                        limit=self.boundary_validating_testing - 1
                    )
                )
            case DataFragment.TESTING:
                return self.get_connection().execute(
                    self.query_template.substitute(
                        offset=self.boundary_validating_testing,
                        limit='all'
                    )
                )
            case DataFragment.ALL:
                return self.get_connection().execute(
                    self.query_template.substitute(
                        offset=0,
                        limit='all'
                    )
                )


def find_start_end():
    paths = list(Path("../data/raw/stocks").iterdir())
    data = []
    for path in paths:
        stock_id = path.stem
        start = head(path, 2)[1].split(',')[1].strip()
        end = tail(path, 1)[0].split(',')[1].strip()
        data.append([stock_id, start, end])
    df = pd.DataFrame(data, columns=['stock_id', 'start', 'end'])
    df.to_clipboard(index=False)


if __name__ == '__main__':
    find_start_end()
