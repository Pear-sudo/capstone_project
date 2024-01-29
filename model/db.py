import psycopg


class Database(object):
    def __init__(self):
        self.conn = psycopg.connect("dbname=capstone user=capstone password=123456 host=localhost port=5432")

    def get_connection(self):
        return self.conn
