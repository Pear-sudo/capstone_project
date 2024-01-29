from typing import Tuple, Optional

from model.db import Database


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

    @property
    def count(self) -> int:
        if self._count is None:
            query = (
                f"select count(*) from stock where stkcd = '{self.stk_id}'"
            )
            self._count = self.get_connection().execute(query).fetchone()[0]
        return self._count

    def fetch(self, r: Tuple[int, int] = None):
        """
        Fetch individual stock data from database
        :param r: percentage of stocks to fetch, must be integer between 0 and 100
        :return: a psql cursor
        """
        if r is None:
            return self.get_connection().execute("select * from stock")
        else:
            pass
