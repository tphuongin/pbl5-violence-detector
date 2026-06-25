import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'pbl5')
DB_CHARSET = os.getenv('DB_CHARSET', 'utf8mb4')

# SQLAlchemy Database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset={DB_CHARSET}"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv('API_DEBUG', 'False') == 'True',
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


def get_db():
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
    
    # Auto-seed admin user if none exist
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        from models import User
        import hashlib
        import uuid
        
        if db.query(User).count() == 0:
            print("No users found. Seeding default admin account...")
            admin_user = User(
                UserID=str(uuid.uuid4()),
                Username="admin",
                PasswordHash=hashlib.sha256("admin123".encode()).hexdigest()
            )
            db.add(admin_user)
            db.commit()
            print("✓ Default admin user seeded successfully (admin / admin123)")
    except Exception as e:
        print(f"Error seeding admin user: {e}")
        db.rollback()
    finally:
        db.close()



def drop_db():
    
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped")
