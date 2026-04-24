from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.sql import func
from database import Base
import uuid


class User(Base):
    __tablename__ = "USERS"
    
    UserID = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    Username = Column(String(100), nullable=False, unique=True)
    PasswordHash = Column(String(255), nullable=False)


class Camera(Base):
    __tablename__ = "CAMERAS"
    
    CameraID = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    CameraName = Column(String(255), nullable=False)
    CameraIP = Column(String(50))
    CameraPhoneNum = Column(String(20))
    CameraStatus = Column(Boolean, default=True)
    UserID = Column(String(50), ForeignKey("USERS.UserID"), nullable=False)


class Call(Base):
    __tablename__ = "CALLS"
    
    CallID = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    CallDate = Column(DateTime, nullable=False)
    CameraID = Column(String(50), ForeignKey("CAMERAS.CameraID"), nullable=False)


class ViolenceHistory(Base):
    __tablename__ = "VIOLENCE_HISTORY"
    
    HistoryID = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    Timestamp = Column(DateTime, nullable=False)
    Location = Column(String(255))
    ClipURL = Column(Text)
    Confidence = Column(Float)
    CameraID = Column(String(50), ForeignKey("CAMERAS.CameraID"), nullable=False)