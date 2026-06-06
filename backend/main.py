from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from database import engine, Base, SessionLocal, get_db, init_db
from models import User, Camera, Call, ViolenceHistory
from upload_service import upload_service
from stream_service import (
    get_stream_buffer,
    add_frame_to_stream,
    get_stream_info,
    start_recording,
    stop_recording,
)
import os
import cv2
import asyncio
from dotenv import load_dotenv
import logging
import json
from typing import Dict, Set, List
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tempfile
from datetime import datetime

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

# Jetson configuration
JETSON_IP = os.getenv("JETSON_IP", "192.168.137.2")
JETSON_HTTP_PORT = int(os.getenv("JETSON_HTTP_PORT", "8001"))
JETSON_BASE_URL = f"http://{JETSON_IP}:{JETSON_HTTP_PORT}"

BACKEND_HOST_OVERRIDE = os.getenv("BACKEND_HOST")

# Keep latest detection payload per camera in memory for quick access.
latest_detections: dict = {}
violence_states: dict = {}

VIOLENCE_THRESHOLD = float(os.getenv("VIOLENCE_THRESHOLD", "0.6"))
VIOLENCE_MIN_SECONDS = float(os.getenv("VIOLENCE_MIN_SECONDS", "5"))
VIOLENCE_END_GRACE = float(os.getenv("VIOLENCE_END_GRACE", "2"))


class VideoAnalysisHub:
    def __init__(self):
        self._consumers: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def register_consumer(self, job_id: str, ws: WebSocket):
        async with self._lock:
            self._consumers.setdefault(job_id, set()).add(ws)

    async def unregister_consumer(self, job_id: str, ws: WebSocket):
        async with self._lock:
            consumers = self._consumers.get(job_id)
            if not consumers:
                return
            consumers.discard(ws)
            if not consumers:
                self._consumers.pop(job_id, None)

    async def broadcast(self, job_id: str, message: str):
        async with self._lock:
            consumers = list(self._consumers.get(job_id, set()))

        if not consumers:
            return

        stale: List[WebSocket] = []
        for ws in consumers:
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)

        if stale:
            async with self._lock:
                remaining = self._consumers.get(job_id)
                if remaining:
                    for ws in stale:
                        remaining.discard(ws)
                    if not remaining:
                        self._consumers.pop(job_id, None)


hub = VideoAnalysisHub()


def _resolve_camera_location(db: Session, camera_id: str) -> str:
    camera = db.query(Camera).filter(Camera.CameraID == camera_id).first()
    if not camera:
        return camera_id
    return camera.CameraName or camera_id


def _write_video_file(frames, fps: int) -> str:
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    if not height or not width:
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    writer = cv2.VideoWriter(temp_path, fourcc, fps or 30, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return temp_path


def _save_violence_history(camera_id: str, start_time: datetime, end_time: datetime, max_conf: float):
    if not start_time or not end_time:
        return
    recording = stop_recording(camera_id)
    if not recording:
        return

    duration = (end_time - start_time).total_seconds()
    if duration < VIOLENCE_MIN_SECONDS:
        return

    temp_path = _write_video_file(recording["frames"], recording["fps"])
    if not temp_path:
        return

    db = SessionLocal()
    try:
        location = _resolve_camera_location(db, camera_id)
        safe_timestamp = start_time.strftime("%Y%m%dT%H%M%S")
        upload_result = upload_service.upload_detection_clip(
            temp_path,
            location=location,
            timestamp=safe_timestamp,
        )

        if not upload_result.get("success"):
            logger.error(f"Clip upload failed for camera {camera_id}: {upload_result.get('error')}")
            return

        record = ViolenceHistory(
            Timestamp=start_time,
            Location=location,
            ClipURL=upload_result.get("url"),
            Confidence=max_conf,
            CameraID=camera_id,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()
        try:
            os.unlink(temp_path)
        except OSError:
            pass


async def _handle_violence_detection(camera_id: str, payload: dict) -> bool:
    now = datetime.now()
    label = payload.get("label")
    prob = payload.get("prob_smooth")
    is_violent = label == "VIOLENCE" and isinstance(prob, (int, float)) and prob >= VIOLENCE_THRESHOLD

    recording_started = False

    state = violence_states.get(camera_id)
    if not state:
        state = {
            "active": False,
            "start_time": None,
            "last_violent_time": None,
            "max_confidence": 0.0,
        }
        violence_states[camera_id] = state

    if is_violent:
        if not state["active"]:
            start_recording(camera_id)
            state["active"] = True
            state["start_time"] = now
            state["max_confidence"] = float(prob)
            recording_started = True
        else:
            state["max_confidence"] = max(state["max_confidence"], float(prob))
        state["last_violent_time"] = now
        return recording_started

    if state["active"] and state["last_violent_time"]:
        idle_seconds = (now - state["last_violent_time"]).total_seconds()
        if idle_seconds >= VIOLENCE_END_GRACE:
            start_time = state["start_time"]
            max_conf = state["max_confidence"]
            state["active"] = False
            state["start_time"] = None
            state["last_violent_time"] = None
            state["max_confidence"] = 0.0

            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                _save_violence_history,
                camera_id,
                start_time,
                now,
                max_conf,
            )

    return recording_started

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
def get_violence_history(page: int = 1, page_size: int = 12, db: Session = Depends(get_db)):
    """Get paginated violence history records"""
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)

    query = (
        db.query(ViolenceHistory, Camera)
        .outerjoin(Camera, ViolenceHistory.CameraID == Camera.CameraID)
        .order_by(ViolenceHistory.Timestamp.desc())
    )
    total = query.count()
    records = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "data": [
            {
                "HistoryID": v.HistoryID,
                "Timestamp": v.Timestamp.isoformat() if v.Timestamp else None,
                "Location": v.Location,
                "ClipURL": v.ClipURL,
                "Confidence": v.Confidence,
                "CameraID": v.CameraID,
                "CameraName": c.CameraName if c else None,
            } for v, c in records
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
        
        # logger.info(f"Received frame for camera {camera_id}, size: {len(content)} bytes")
        
        return {
            "success": True,
            "message": f"Frame received for camera {camera_id}",
            "camera_id": camera_id
        }
    except Exception as e:
        logger.error(f"Error receiving frame for camera {camera_id}: {e}")
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
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}
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

@app.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for receiving video frames from camera"""
    await websocket.accept()
    logger.info(f"WebSocket connection established for camera {camera_id}")
    
    try:
        while True:
            # Receive binary frame data
            frame_data = await websocket.receive_bytes()
            # Add frame to stream buffer
            add_frame_to_stream(camera_id, frame_data)
            # logger.info(f"Received frame for camera {camera_id}, size: {len(frame_data)} bytes")
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {e}")
    finally:
        logger.info(f"WebSocket connection closed for camera {camera_id}")

@app.websocket("/ws/view/{camera_id}")
async def view_stream(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for viewing video stream"""
    await websocket.accept()
    logger.info(f"WebSocket view connection established for camera {camera_id}")
    
    stream_buffer = get_stream_buffer(camera_id)
    try:
        while True:
            if stream_buffer.frames:
                frame = stream_buffer.frames[-1]
                success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    await websocket.send_bytes(jpeg.tobytes())
            await asyncio.sleep(0.05)
    except Exception as e:
        logger.error(f"WebSocket view error for camera {camera_id}: {e}")
    finally:
        logger.info(f"WebSocket view connection closed for camera {camera_id}")


@app.websocket("/ws/detection/{camera_id}")
async def websocket_detection(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for receiving detection JSON payload from edge devices."""
    await websocket.accept()
    logger.info(f"WebSocket detection connection established for camera {camera_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Invalid detection payload for camera {camera_id}")
                continue

            if isinstance(payload, dict):
                payload.setdefault("camera_id", camera_id)
                recording_started = await _handle_violence_detection(camera_id, payload)
                payload["recording_started"] = recording_started
                latest_detections[camera_id] = payload
                # logger.info(
                #     f"Detection received for camera {camera_id}: "
                #     f"label={payload.get('label')} prob={payload.get('prob_smooth')}"
                # )
    except WebSocketDisconnect:
        logger.info(f"WebSocket detection connection closed for camera {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket detection error for camera {camera_id}: {e}")


@app.get("/api/detection/{camera_id}/latest")
def get_latest_detection(camera_id: str):
    """Get latest detection payload for a camera."""
    data = latest_detections.get(camera_id)
    if not data:
        raise HTTPException(status_code=404, detail="No detection data for this camera")
    return data


def _resolve_backend_host(request: Request) -> str:
    if BACKEND_HOST_OVERRIDE:
        return BACKEND_HOST_OVERRIDE

    forwarded = request.headers.get("x-forwarded-host")
    host_header = forwarded or request.headers.get("host")
    if host_header:
        return host_header.split(":")[0]

    return request.client.host if request.client else "localhost"


@app.post("/api/analyze-video")
async def analyze_video(request: Request, video: UploadFile = File(...)):
    try:
        content = await video.read()
        backend_host = _resolve_backend_host(request)
        files = {
            "video": (
                video.filename,
                content,
                video.content_type or "application/octet-stream",
            )
        }
        data = {"backend_host": backend_host}

        timeout = httpx.Timeout(60.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{JETSON_BASE_URL}/analyze-video",
                data=data,
                files=files,
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Jetson")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Jetson timeout")
    except httpx.HTTPError as exc:
        logger.error(f"Jetson request failed: {exc}")
        raise HTTPException(status_code=503, detail="Jetson request failed")

    if response.status_code == 409:
        try:
            return JSONResponse(status_code=409, content=response.json())
        except json.JSONDecodeError:
            return JSONResponse(status_code=409, content={"detail": response.text})

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


@app.get("/api/video-analysis/{job_id}/status")
async def get_video_analysis_status(job_id: str):
    try:
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{JETSON_BASE_URL}/status")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Jetson")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Jetson timeout")
    except httpx.HTTPError as exc:
        logger.error(f"Jetson request failed: {exc}")
        raise HTTPException(status_code=503, detail="Jetson request failed")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        payload = response.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Invalid status response from Jetson")

    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise HTTPException(status_code=502, detail="Unexpected status payload")

    job = next((item for item in jobs if item.get("job_id") == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@app.post("/api/video-analysis/{job_id}/cancel")
async def cancel_video_analysis(job_id: str):
    payload = {"job_id": job_id}
    try:
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{JETSON_BASE_URL}/cancel", json=payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Jetson")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Jetson timeout")
    except httpx.HTTPError as exc:
        logger.error(f"Jetson request failed: {exc}")
        raise HTTPException(status_code=503, detail="Jetson request failed")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


@app.websocket("/ws/video-analysis/{job_id}")
async def websocket_video_analysis(websocket: WebSocket, job_id: str):
    role = websocket.query_params.get("role", "consumer")

    if role == "producer":
        await websocket.accept()
        logger.info(f"Video analysis producer connected for job {job_id}")
        try:
            while True:
                message = await websocket.receive_text()
                await hub.broadcast(job_id, message)
        except WebSocketDisconnect:
            logger.info(f"Video analysis producer disconnected for job {job_id}")
        except Exception as exc:
            logger.error(f"Video analysis producer error for job {job_id}: {exc}")
        return

    await websocket.accept()
    await hub.register_consumer(job_id, websocket)
    logger.info(f"Video analysis consumer connected for job {job_id}")
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        logger.info(f"Video analysis consumer disconnected for job {job_id}")
    except Exception as exc:
        logger.error(f"Video analysis consumer error for job {job_id}: {exc}")
    finally:
        await hub.unregister_consumer(job_id, websocket)

if __name__ == "__main__":
    import uvicorn
    
    api_host = os.getenv('API_HOST', 'localhost')
    api_port = int(os.getenv('API_PORT', 8000))
    
    print(f"\n🚀 Starting API on {api_host}:{api_port}")
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=api_port,
        reload=os.getenv('API_DEBUG', 'False') == 'True'
    )
