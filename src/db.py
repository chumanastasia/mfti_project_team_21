import psycopg2
from pydantic import PostgresDsn
from contextlib import contextmanager


class TSDBConnector:
    """A class to connect to the database."""

    def __init__(self, dsn: PostgresDsn) -> None:
        """Initialize a connection to the database."""

        self._dsn = dsn
        self._conn = psycopg2.connect(dsn=self._dsn)

    @contextmanager
    def get_cursor(self):
        """Get a cursor to the database."""

        cursor = self._conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
            self._conn.commit()

    def get_connection(self):
        """Get a connection to the database."""

        return self._conn

