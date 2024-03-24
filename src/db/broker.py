import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session, joinedload
from sqlalchemy_utils import create_database, database_exists, drop_database

from src.domain.domain import Base, Listing
from src.logger import logging


class DbBroker:
    engine: Engine

    def __init__(self):
        self.engine = self.create_engine()
        self.check_connection()
        self.create_database()

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DbBroker, cls).__new__(cls)
        return cls.instance

    def create_engine(self) -> Engine:
        load_dotenv()

        url_object = URL.create(
            drivername="mysql+mysqlconnector",
            username=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),  # type: ignore
            database=os.getenv("DB_NAME"),
        )
        engine = create_engine(url_object)
        return engine

    def check_connection(self):
        try:
            if database_exists(self.engine.url):
                with self.engine.connect():
                    pass
        except Exception as e:
            raise e

    def create_database(self):
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

    def drop_database(self):
        if database_exists(self.engine.url):
            drop_database(self.engine.url)

    def create_schema(self):
        Base.metadata.create_all(self.engine)

    def drop_schema(self):
        Base.metadata.drop_all(self.engine)

    def reset_schema(self):
        self.drop_schema()
        self.create_schema()

    def create_session(self) -> Session:
        return Session(self.engine)

    def save_listing(self, listing: Listing):
        listing_str = str(listing)
        with self.create_session() as session:
            with session.begin():
                session.add(listing)
        logging.info(listing_str)

    def get_all_listings_statement(self):
        with self.create_session() as session:
            query = session.query(Listing).options(
                joinedload(Listing.general_information),
                joinedload(Listing.additional_information),
            )
            return query.statement


if __name__ == "__main__":
    pass
    # DbBroker().reset_schema()
