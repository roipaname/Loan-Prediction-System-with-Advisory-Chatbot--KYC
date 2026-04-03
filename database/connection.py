from sqlalchemy import create_engine, session,sessionmaker

from loguru import logger

from config.settings import DB_URL


class Connection:
    def __int__(self):
        self.db_url=DB_URL
        self.db=self.init_db()