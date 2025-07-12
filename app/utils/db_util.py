from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base,sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"  # Replace with your DB connection string

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == '__main__':
    # Example usage
    db = next(get_db())
    print("Database session created successfully.")
    db.close()
    print("Database session closed.")