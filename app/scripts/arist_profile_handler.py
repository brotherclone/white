from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.objects.db_models.artist import Artist

def check_arist_profile_exists(db: Session, artist_name=None, discogs_id=None):
    """Check if artist profile exists by name or discogs_id"""
    if discogs_id:
        return db.query(Artist).filter(Artist.discogs_id == discogs_id).first() is not None
    elif artist_name:
        return db.query(Artist).filter(Artist.name == artist_name).first() is not None
    return False

def create_arist_profile(db: Session, artist_data):
    """Create a new artist profile"""
    try:
        artist = Artist(
            discogs_id=artist_data.get('discogs_id'),
            name=artist_data.get('name'),
            real_name=artist_data.get('real_name'),
            profile=artist_data.get('profile'),
            url=artist_data.get('url')
        )
        db.add(artist)
        db.commit()
        db.refresh(artist)
        return artist
    except SQLAlchemyError as e:
        db.rollback()
        raise e

def get_arist_profile(db: Session, artist_id=None, artist_name=None, discogs_id=None):
    """Get artist profile by id, name or discogs_id"""
    if artist_id:
        return db.query(Artist).filter(Artist.id == artist_id).first()
    elif discogs_id:
        return db.query(Artist).filter(Artist.discogs_id == discogs_id).first()
    elif artist_name:
        return db.query(Artist).filter(Artist.name == artist_name).first()
    return None

def update_arist_profile(db: Session, artist_id, update_data):
    """Update an existing artist profile"""
    artist = get_arist_profile(db, artist_id=artist_id)
    if artist:
        try:
            for key, value in update_data.items():
                if hasattr(artist, key):
                    setattr(artist, key, value)
            db.commit()
            db.refresh(artist)
            return artist
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    return None

def delete_arist_profile(db: Session, artist_id):
    """Delete an artist profile"""
    artist = get_arist_profile(db, artist_id=artist_id)
    if artist:
        try:
            db.delete(artist)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    return False