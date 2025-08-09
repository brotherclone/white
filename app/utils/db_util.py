import os
import asyncio

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.objects.db_models.artist_schema import ArtistSchema, RainbowArtist
from app.objects.db_models.concept_schema import ConceptSchema
from app.utils.discog_util import search_discogs_artist

# ToDo: Break out to separate files per model

load_dotenv()
engine = create_engine(os.environ['LOCAL_DB_PATH'])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_arist_to_rainbow_artist(db_artist: ArtistSchema) -> RainbowArtist:
    """
    Converts a database artist schema to a RainbowArtist object.
    :param db_artist: ArtistSchema object from the database
    :return: RainbowArtist object
    """
    return RainbowArtist(
        name=db_artist.name,
        id=db_artist.id,
        discogs_id=db_artist.discogs_id,
        profile=db_artist.profile,
    )


async def get_artist_by_name(name):
    db_generator = get_db()
    db = next(db_generator)
    try:
        q = db.query(ArtistSchema).filter(ArtistSchema.name == name).first()
        if q:
            return q
        else:
            print(f"No artist found with name {name}")
            return None
    except Exception as e:
        print(f"Error fetching artist by name {name}: {e}")
        return None
    finally:
        next(db_generator, None)


async def get_local_artist_by_local_id(local_id):
    db_generator = get_db()
    db = next(db_generator)
    try:
        q = db.query(ArtistSchema).filter(ArtistSchema.id == local_id).first()
        if q:
            return q
        else:
            print(f"No artist found with local ID {local_id}")
            return None
    except Exception as e:
        print(f"Error fetching artist by local ID {local_id}: {e}")
        return None
    finally:
        next(db_generator, None)


async def get_local_artist_by_discogs_id(discogs_id):
    db_generator = get_db()
    db = next(db_generator)
    try:
        q = db.query(ArtistSchema).filter(ArtistSchema.discogs_id == discogs_id).first()
        if q:
            return q
        else:
            print(f"No artist found with discogs ID {discogs_id}")
            return None
    except Exception as e:
        print(f"Error fetching artist by discogs ID {discogs_id}: {e}")
        return None
    finally:
        next(db_generator, None)


async def update_artist(artist) -> ArtistSchema | None:
    db_generator = get_db()
    db = next(db_generator)
    try:
        existing_artist = db.query(ArtistSchema).filter(ArtistSchema.id == artist.id).first()
        if not existing_artist:
            print(f"No artist found with ID {artist.id}")
            return None
        await db.merge(artist)
        db.commit()
        return artist
    except Exception as e:
        db.rollback()
        print(f"Error updating artist: {e}")
        return None
    finally:
        next(db_generator, None)


async def create_artist(artist) -> ArtistSchema | None:
    db_generator = get_db()
    db = next(db_generator)
    try:
        db.add(artist)
        db.commit()
        db.refresh(artist)
        return artist
    except Exception as e:
        db.rollback()
        print(f"Error creating artist: {e}")
        return None
    finally:
        next(db_generator, None)


async def get_or_create_artist(artist_schema):
    db_generator = get_db()
    db = next(db_generator)
    try:
        existing_artist = db.query(ArtistSchema).filter(ArtistSchema.name == artist_schema.name).first()
        if existing_artist:
            return existing_artist
        db.add(artist_schema)
        db.commit()
        db.refresh(artist_schema)
        return artist_schema
    except Exception as e:
        db.rollback()
        print(f"Error in get_or_create_artist: {e}")
        return None
    finally:
        next(db_generator, None)  # Properly close the session


async def delete_artist(artist_id) -> bool:
    db_generator = get_db()
    db = next(db_generator)
    try:
        artist = db.query(ArtistSchema).filter(ArtistSchema.id == artist_id).first()
        if not artist:
            print(f"No artist found with ID {artist_id}")
            return False
        db.delete(artist)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting artist: {e}")
        return False
    finally:
        next(db_generator, None)


async def create_concept(concept) -> ConceptSchema | None:
    db_generator = get_db()
    db = next(db_generator)
    try:
        db.add(concept)
        db.commit()
        db.refresh(concept)
        return concept
    except Exception as e:
        db.rollback()
        print(f"Error creating concept: {e}")
        return None
    finally:
        next(db_generator, None)


async def get_random_concept() -> ConceptSchema | None:
    db_generator = get_db()
    db = next(db_generator)
    try:
        concept = db.query(ConceptSchema).order_by(func.random()).first()
        if concept:
            return concept
        else:
            print("No concepts found in the database.")
            return None
    except Exception as e:
        print(f"Error fetching random concept: {e}")
        return None
    finally:
        next(db_generator, None)
