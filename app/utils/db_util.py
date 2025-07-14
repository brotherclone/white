import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base,sessionmaker
from app.objects.db_models.artist import Artist
from app.utils.discog_util import search_discogs_artist

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

def get_artist_by_local_id(local_id):
    db_generator = get_db()
    db = next(db_generator)
    try:
        q = db.query(Artist).filter(Artist.id == local_id).first()
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

def get_artist_by_discogs_id(discogs_id):
    db_generator = get_db()
    db = next(db_generator)
    try:
        q = db.query(Artist).filter(Artist.discogs_id == discogs_id).first()
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

def update_artist(artist)-> Artist | None:
    db_generator = get_db()
    db = next(db_generator)
    try:
        existing_artist = db.query(Artist).filter(Artist.id == artist.id).first()
        if not existing_artist:
            print(f"No artist found with ID {artist.id}")
            return None
        # Merge the updated artist object and commit
        db.merge(artist)
        db.commit()
        return artist
    except Exception as e:
        # Rollback in case of error
        db.rollback()
        print(f"Error updating artist: {e}")
        return None
    finally:
        next(db_generator, None)

def create_artist(artist) -> Artist | None:
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

def delete_artist(artist_id) -> bool:
    db_generator = get_db()
    db = next(db_generator)
    try:
        # Find the artist by ID
        artist = db.query(Artist).filter(Artist.id == artist_id).first()

        if not artist:
            print(f"No artist found with ID {artist_id}")
            return False

        # Delete the artist
        db.delete(artist)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting artist: {e}")
        return False
    finally:
        next(db_generator, None)

if __name__ == '__main__':
    test_artist = Artist(name="The Kinks", discogs_id=0)
    look_up_artist = search_discogs_artist(test_artist.name)
    if look_up_artist:
        test_artist.discogs_id = look_up_artist.id
        test_artist.profile = look_up_artist.profile
        test_artist.name = look_up_artist.name
    created_artist = create_artist(test_artist)
    if created_artist:
        print(f"Created artist: {created_artist.name} with ID {created_artist.id}")
    else:
        print("Failed to create artist")
    patched_artist = created_artist
    patched_artist.profile = "Testing profile update"
    updated_artist = update_artist(patched_artist)
    if updated_artist:
        print(f"Updated artist: {updated_artist.name} with ID {updated_artist.id}")
    else:
        print("Failed to update artist")
    if delete_artist(updated_artist.id):
        print(f"Deleted artist with ID {updated_artist.id}")
