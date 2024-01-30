from enum import Enum
from typing import Optional
from string import Template

from psycopg import Cursor
from psycopg.rows import Row

from model.db import Database


class DataFragment(Enum):
    TRAINING = "training"
    TESTING = "testing"
    VALIDATION = "validation"
    ALL = "all"


class Stocks(Database):
    def __init__(self):
        super().__init__()

    def list_stocks(self) -> list[str]:
        stocks = self.get_connection().execute('select stkcd from stock group by stkcd').fetchall()
        stocks = [row[0] for row in stocks]
        stocks.sort()
        return stocks


class Stock(Database):
    def __init__(self, stk_id: str):
        super().__init__()
        self.stk_id: str = stk_id

        self._count: Optional[int] = None
        self.boundary_training_validating = None
        self.boundary_validating_testing = None

        self.query_template = Template(
            f"select * from stock where stkcd = '000001' order by trddt offset $offset limit $limit")

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
