from sqlalchemy import create_engine
from sqlalchemy.orm import session, sessionmaker
from contextlib import contextmanager
from loguru import logger

from config.settings import DB_URL,DB_POOL_SIZE,DB_POOL_TIMEOUT,DB_MAX_OVERFLOW,DB_ECHO,


class Connection:
    def __int__(self):
        self.db_url=DB_URL
        self.engine=self.init_db()
        self.local_session=None

    def init_db(self):
        """Create a database connection and session."""

        try:
            self.engine=create_engine(DB_URL,pool_size=DB_POOL_SIZE,max_overflow=DB_MAX_OVERFLOW,pool_timeout=DB_POOL_TIMEOUT,pool_pre_ping=True)
            self.local_session=sessionmaker(bind=self.engine,expire_on_commit=True)
            from database.schemas import Base
            Base.metadata.create_all(bind=self.engine)
            logger.success("Database Initialized")
            return self.engine
        except Exception as e:
            logger.error(f"Failed to Initialize DB:{e}")
            raise


    @contextmanager
    def get_db(self):
        session=self.local_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database Error : {e}")
            raise
        finally:
            session.close()

if __name__ == "__main__":




