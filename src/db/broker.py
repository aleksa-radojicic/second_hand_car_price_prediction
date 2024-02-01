from src.domain.domain import Listing
from src.logger import logging

class DbBroker:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DbBroker, cls).__new__(cls)
        return cls.instance
    
    def save_listing(self, listing: Listing):
        # logging.info("Successfully saved listing in database")
        logging.info(str(listing))
        pass