from typing import TypeVar
from src.db import TSDBConnector
from pydantic import PostgresDsn

_T = TypeVar("_T")


class TSRepository(TSDBConnector):
    """A class to interact with the database."""

    def __init__(self, dsn: PostgresDsn) -> None:
        """Initialize a connection to the database."""

        super().__init__(dsn=dsn)

    def insert(self, query: str, values: tuple) -> None:
        """Insert a row into the database."""

        with self.get_cursor() as cursor:
            cursor.execute(query, values)

    def select(self, query: str, values: tuple) -> list:
        """Select rows from the database."""

        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            return cursor.fetchall()

    def select_one(self, query: str, values: tuple) -> _T:
        """Select one row from the database."""

        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            return cursor.fetchone()

    def update(self, query: str, values: tuple) -> None:
        """Update rows in the database."""

        with self.get_cursor() as cursor:
            cursor.execute(query, values)

    def delete(self, query: str, values: tuple) -> None:
        """Delete rows from the database."""

        with self.get_cursor() as cursor:
            cursor.execute(query, values)
