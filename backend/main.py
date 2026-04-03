from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import engine, Base, SessionLocal, get_db, init_db
from models import User, Camera, Call, ViolenceHistory
from upload_service import upload_service
from stream_service import get_stream_buffer, add_frame_to_stream, get_stream_info
import os
from dotenv import load_dotenv
import tempfile

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

# ============= UPLOAD ENDPOINTS =============
@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image to Cloudinary"""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Upload to Cloudinary
        result = upload_service.upload_image(
            tmp_file_path,
            public_id=file.filename,
            folder="images"
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video to Cloudinary"""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Upload to Cloudinary
        result = upload_service.upload_video(
            tmp_file_path,
            public_id=file.filename,
            folder="videos"
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/upload/violence-clip")
async def upload_violence_clip(
    file: UploadFile = File(...),
    location: str = "Unknown",
    timestamp: str = None
):
    """Upload violence detection clip to Cloudinary"""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Upload to Cloudinary with organized naming
        if not timestamp:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
        
        result = upload_service.upload_detection_clip(
            tmp_file_path,
            location=location,
            timestamp=timestamp
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/upload/{public_id}")
def delete_file(public_id: str, resource_type: str = "image"):
    """Delete file from Cloudinary"""
    return upload_service.delete_file(public_id, resource_type)

# ============= VIDEO STREAMING ENDPOINTS =============
@app.post("/api/stream/{camera_id}/frame")
async def receive_frame(camera_id: str, file: UploadFile = File(...)):
    """Receive video frame from camera"""
    try:
        content = await file.read()
        add_frame_to_stream(camera_id, content)
        
        return {
            "success": True,
            "message": f"Frame received for camera {camera_id}",
            "camera_id": camera_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/stream/{camera_id}/live")
async def stream_video(camera_id: str):
    """Get MJPEG stream for camera"""
    stream_buffer = get_stream_buffer(camera_id)
    return StreamingResponse(
        stream_buffer.get_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stream/{camera_id}/info")
def get_camera_stream_info(camera_id: str):
    """Get stream information for camera"""
    return get_stream_info(camera_id)

@app.post("/api/stream/{camera_id}/reset")
def reset_stream(camera_id: str):
    """Reset stream buffer for camera"""
    stream_buffer = get_stream_buffer(camera_id)
    stream_buffer.reset()
    return {"success": True, "message": f"Stream for camera {camera_id} reset"}

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
