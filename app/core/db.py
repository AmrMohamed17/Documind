import os
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

_pool = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=os.getenv("DATABASE_URL"),
            min_size=1,
            max_size=5,
            configure=register_vector,
            open=True,
        )
    return _pool