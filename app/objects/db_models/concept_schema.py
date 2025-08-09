from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ConceptSchema(Base):
    __tablename__ = 'concept_capture'
    id = Column(Integer, primary_key=True, autoincrement=True)
    concept = Column(Text)
