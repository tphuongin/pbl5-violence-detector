#!/usr/bin/env python
"""
Seed dữ liệu mẫu vào database
"""
import sys
from datetime import datetime, timedelta
from database import SessionLocal
from models import User, Camera, Call, ViolenceHistory
import hashlib
import uuid

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def seed_data():
    """Seed dữ liệu mẫu vào database"""
    db = SessionLocal()
    
    try:
        # Xóa dữ liệu cũ (optional)
        db.query(User).delete()
        db.query(Camera).delete()
        db.query(Call).delete()
        db.query(ViolenceHistory).delete()
        
        print("\n--- Seeding Users ---")
        users = [
            User(
                UserID=str(uuid.uuid4()),
                Username="admin",
                PasswordHash=hash_password("admin123")
            ),
            User(
                UserID=str(uuid.uuid4()),
                Username="operator1",
                PasswordHash=hash_password("operator123")
            ),
            User(
                UserID=str(uuid.uuid4()),
                Username="operator2",
                PasswordHash=hash_password("operator456")
            )
        ]
        db.add_all(users)
        db.commit()
        print(f"✓ Added {len(users)} users")
        
        print("\n--- Seeding Cameras ---")
        cameras = [
            Camera(
                CameraID=str(uuid.uuid4()),
                CameraName="Camera Main Hall",
                CameraIP="192.168.1.10",
                CameraPhoneNum="0901234567",
                CameraStatus=True
            ),
            Camera(
                CameraID=str(uuid.uuid4()),
                CameraName="Camera Lobby",
                CameraIP="192.168.1.11",
                CameraPhoneNum="0901234568",
                CameraStatus=True
            ),
            Camera(
                CameraID=str(uuid.uuid4()),
                CameraName="Camera Parking",
                CameraIP="192.168.1.12",
                CameraPhoneNum="0901234569",
                CameraStatus=False
            ),
            Camera(
                CameraID=str(uuid.uuid4()),
                CameraName="Camera Entrance",
                CameraIP="192.168.1.13",
                CameraPhoneNum="0901234570",
                CameraStatus=True
            )
        ]
        db.add_all(cameras)
        db.commit()
        print(f"✓ Added {len(cameras)} cameras")
        
        print("\n--- Seeding Calls ---")
        now = datetime.now()
        calls = [
            Call(
                CallID=str(uuid.uuid4()),
                CallDate=now - timedelta(days=2, hours=5)
            ),
            Call(
                CallID=str(uuid.uuid4()),
                CallDate=now - timedelta(days=1, hours=3, minutes=30)
            ),
            Call(
                CallID=str(uuid.uuid4()),
                CallDate=now - timedelta(hours=12)
            ),
            Call(
                CallID=str(uuid.uuid4()),
                CallDate=now - timedelta(hours=2)
            )
        ]
        db.add_all(calls)
        db.commit()
        print(f"✓ Added {len(calls)} calls")
        
        print("\n--- Seeding Violence History ---")
        violence_records = [
            ViolenceHistory(
                HistoryID=str(uuid.uuid4()),
                Timestamp=now - timedelta(days=2, hours=5, minutes=15),
                Location="Main Hall",
                ClipURL="https://storage.example.com/clips/violence_001.mp4",
                Confidence=0.92
            ),
            ViolenceHistory(
                HistoryID=str(uuid.uuid4()),
                Timestamp=now - timedelta(days=1, hours=3, minutes=45),
                Location="Lobby",
                ClipURL="https://storage.example.com/clips/violence_002.mp4",
                Confidence=0.87
            ),
            ViolenceHistory(
                HistoryID=str(uuid.uuid4()),
                Timestamp=now - timedelta(hours=12, minutes=20),
                Location="Entrance",
                ClipURL="https://storage.example.com/clips/violence_003.mp4",
                Confidence=0.95
            ),
            ViolenceHistory(
                HistoryID=str(uuid.uuid4()),
                Timestamp=now - timedelta(hours=2, minutes=5),
                Location="Parking",
                ClipURL="https://storage.example.com/clips/violence_004.mp4",
                Confidence=0.78
            ),
            ViolenceHistory(
                HistoryID=str(uuid.uuid4()),
                Timestamp=now - timedelta(hours=1),
                Location="Main Hall",
                ClipURL="https://storage.example.com/clips/violence_005.mp4",
                Confidence=0.88
            )
        ]
        db.add_all(violence_records)
        db.commit()
        print(f"✓ Added {len(violence_records)} violence records")
        
        print("\n✓✓✓ Seed data completed successfully! ✓✓✓\n")
        print("Summary:")
        print(f"  - Users: {len(users)}")
        print(f"  - Cameras: {len(cameras)}")
        print(f"  - Calls: {len(calls)}")
        print(f"  - Violence History: {len(violence_records)}")
        
    except Exception as e:
        print(f"\n✗ Error during seeding: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
