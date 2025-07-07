#
# from sqlalchemy import Column, Integer, String, ForeignKey, Table, Text, Boolean, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import relationship
# import datetime
#
# Base = declarative_base()
#
# artist_group_association = Table('artist_group_association', Base.metadata,
#     Column('artist_id', Integer, ForeignKey('artists.id')),
#     Column('group_id', Integer, ForeignKey('groups.id'))
# )
#
# artist_release_association = Table('artist_release_association', Base.metadata,
#     Column('artist_id', Integer, ForeignKey('artists.id')),
#     Column('release_id', Integer, ForeignKey('releases.id')),
#     Column('role', String(100))
# )
#
# class Artist(Base):
#     __tablename__ = 'artists'
#
#     id = Column(Integer, primary_key=True)
#     discogs_id = Column(Integer, unique=True)
#     name = Column(String(255), nullable=False)
#     real_name = Column(String(255))
#     profile = Column(Text)
#     url = Column(String(255))
#
#     # Relationships
#     releases = relationship("Release", secondary=artist_release_association, back_populates="artists")
#     groups = relationship("Group", secondary=artist_group_association, back_populates="members")
#
#     created_at = Column(DateTime, default=datetime.datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
#
# class Group(Base):
#     __tablename__ = 'groups'
#
#     id = Column(Integer, primary_key=True)
#     discogs_id = Column(Integer, unique=True)
#     name = Column(String(255), nullable=False)
#
#     # Relationships
#     members = relationship("Artist", secondary=artist_group_association, back_populates="groups")
#
#     created_at = Column(DateTime, default=datetime.datetime.utcnow)
#
# class Release(Base):
#     __tablename__ = 'releases'
#
#     id = Column(Integer, primary_key=True)
#     discogs_id = Column(Integer, unique=True)
#     title = Column(String(255), nullable=False)
#     year = Column(Integer)
#     release_type = Column(String(50))
#     main_release = Column(Boolean, default=False)
#
#     # Relationships
#     artists = relationship("Artist", secondary=artist_release_association, back_populates="releases")
#
#     created_at = Column(DateTime, default=datetime.datetime.utcnow)