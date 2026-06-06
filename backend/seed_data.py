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
        
        # Lưu UserID để sử dụng cho cameras
        user_ids = [user.UserID for user in users]
        
        print("\n--- Seeding Cameras ---")
        cameras = [
            Camera(
                CameraID="jetson-cam-01",
                CameraName="Camera Main Hall",
                CameraIP="192.168.137.2",
                CameraPhoneNum="0901234567",
                CameraStatus=True,
                UserID=user_ids[0]
            )
        ]
        db.add_all(cameras)
        db.commit()
        print(f"✓ Added {len(cameras)} cameras")
        
        # Lưu CameraID để sử dụng cho calls và violence history
        camera_ids = [camera.CameraID for camera in cameras]
        
        
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
