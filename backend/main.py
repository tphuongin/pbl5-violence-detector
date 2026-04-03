from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import engine, Base, SessionLocal, get_db, init_db
from models import User, Camera, Call, ViolenceHistory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize database tables
init_db()

# Create FastAPI app
app = FastAPI(
    title="PBL5 Violence Detector API",
    description="API for violence detection system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

# ============= USER ENDPOINTS =============
@app.get("/api/users")
def get_users(db: Session = Depends(get_db)):
    """Get all users"""
    users = db.query(User).all()
    return {
        "count": len(users),
        "data": [
            {
                "UserID": u.UserID,
                "Username": u.Username
            } for u in users
        ]
    }

# ============= CAMERA ENDPOINTS =============
@app.get("/api/cameras")
def get_cameras(db: Session = Depends(get_db)):
    """Get all cameras"""
    cameras = db.query(Camera).all()
    return {
        "count": len(cameras),
        "data": [
            {
                "CameraID": c.CameraID,
                "CameraName": c.CameraName,
                "CameraIP": c.CameraIP,
                "CameraPhoneNum": c.CameraPhoneNum,
                "CameraStatus": c.CameraStatus
            } for c in cameras
        ]
    }

@app.get("/api/cameras/{camera_id}")
def get_camera_by_id(camera_id: str, db: Session = Depends(get_db)):
    """Get camera by ID"""
    camera = db.query(Camera).filter(Camera.CameraID == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {
        "CameraID": camera.CameraID,
        "CameraName": camera.CameraName,
        "CameraIP": camera.CameraIP,
        "CameraPhoneNum": camera.CameraPhoneNum,
        "CameraStatus": camera.CameraStatus
    }

# ============= VIOLENCE HISTORY ENDPOINTS =============
@app.get("/api/violence-history")
def get_violence_history(db: Session = Depends(get_db)):
    """Get all violence history records"""
    records = db.query(ViolenceHistory).all()
    return {
        "count": len(records),
        "data": [
            {
                "HistoryID": v.HistoryID,
                "Timestamp": v.Timestamp.isoformat() if v.Timestamp else None,
                "Location": v.Location,
                "ClipURL": v.ClipURL,
                "Confidence": v.Confidence
            } for v in records
        ]
    }

@app.get("/api/violence-history/{history_id}")
def get_violence_by_id(history_id: str, db: Session = Depends(get_db)):
    """Get violence record by ID"""
    record = db.query(ViolenceHistory).filter(ViolenceHistory.HistoryID == history_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Violence record not found")
    return {
        "HistoryID": record.HistoryID,
        "Timestamp": record.Timestamp.isoformat() if record.Timestamp else None,
        "Location": record.Location,
        "ClipURL": record.ClipURL,
        "Confidence": record.Confidence
    }

# ============= CALLS ENDPOINTS =============
@app.get("/api/calls")
def get_calls(db: Session = Depends(get_db)):
    """Get all calls"""
    calls = db.query(Call).all()
    return {
        "count": len(calls),
        "data": [
            {
                "CallID": c.CallID,
                "CallDate": c.CallDate.isoformat() if c.CallDate else None
            } for c in calls
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    api_host = os.getenv('API_HOST', 'localhost')
    api_port = int(os.getenv('API_PORT', 8000))
    
    print(f"\n🚀 Starting API on {api_host}:{api_port}")
    uvicorn.run(
        "main:app",
        host=api_host,
        port=api_port,
        reload=os.getenv('API_DEBUG', 'False') == 'True'
    )
