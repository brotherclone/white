from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Release(Base):
    __tablename__ = 'releases'

    id = Column(Integer, primary_key=True)
    discogs_id = Column(Integer, unique=True)
    title = Column(String(255), nullable=False)
    year = Column(Integer)
