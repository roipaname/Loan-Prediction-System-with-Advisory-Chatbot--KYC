from sqlalchemy import create_engine
from sqlalchemy.orm import session, sessionmaker
from contextlib import contextmanager
from loguru import logger

from config.settings import DB_URL


class Connection:
    def __int__(self):
        self.db_url=DB_URL
        self.engine=self.init_db()
        self.local_session=None

    def init_db(self):
