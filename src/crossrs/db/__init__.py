from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Generator

from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import Session, sessionmaker

from crossrs.utils.paths import get_app_data_dir
from . import models  # Register models with SQLAlchemy
from .base import Base

__all__ = [
    'get_path',
    'get_session',
    'Session',
]

DB_FILENAME = '{language}.db'
SESSION_MAKERS: dict[str, sessionmaker[Session]] = {}


@event.listens_for(Engine, 'connect')
def set_sqlite_pragma(dbapi_connection: Connection, _connection_record: Any) -> None:
    """Enable foreign key support for SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA foreign_keys=ON')
    cursor.close()


def get_path(language: str) -> Path:
    """Get the path to the database file for the specified language."""
    return get_app_data_dir() / DB_FILENAME.format(language=language)


@contextmanager
def get_session(language: str) -> Generator[Session, None, None]:
    if language not in SESSION_MAKERS:
        engine = create_engine(f'sqlite:///{get_path(language)}')
        Base.metadata.create_all(engine)
        SESSION_MAKERS[language] = sessionmaker(bind=engine)

    session = SESSION_MAKERS[language]()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
