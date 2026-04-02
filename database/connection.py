from sqlalchemy import create_engine, session,sessionmaker

from loguru import logger

from config.settings import DB_URL


class Connection:
    def __int__(self):
        pass