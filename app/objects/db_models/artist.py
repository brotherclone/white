from sqlalchemy import Column, Integer, String, ForeignKey, Table, Text, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Artist(Base):
    __tablename__ = 'artists'

    id = Column(Integer, primary_key=True)
    discogs_id = Column(Integer, unique=True)
    name = Column(String(255), nullable=False)
    real_name = Column(String(255))
    profile = Column(Text)
    url = Column(String(255))