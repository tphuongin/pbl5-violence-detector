"""
Video streaming service for managing camera streams
"""
import os
import threading
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Store active streams: {camera_id: StreamBuffer}
active_streams: dict = {}
STREAM_BUFFER_SIZE = 30  # Keep last 30 frames


class StreamBuffer:
    """Buffer to store video frames from camera"""
    
    def __init__(self, camera_id: str, max_frames: int = STREAM_BUFFER_SIZE):
        self.camera_id = camera_id
        self.frames = deque(maxlen=max_frames)
        self.lock = threading.Lock()
        self.last_update = datetime.now()
        self.is_active = False
        self.frame_count = 0
        self.width = 640
        self.height = 480
        self.fps = 30
        
    def add_frame(self, frame_data: bytes):
        """Add frame to buffer"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                with self.lock:
                    self.frames.append(frame)
                    self.last_update = datetime.now()
                    self.frame_count += 1
                    self.is_active = True
                    
                    # Update dimensions
                    h, w = frame.shape[:2]
                    if h > 0 and w > 0:
                        self.height = h
                        self.width = w
                        
        except Exception as e:
            logger.error(f"Error adding frame for camera {self.camera_id}: {e}")
    
    def get_latest_frame(self) -> bytes:
        """Get latest frame as JPEG bytes"""
        try:
            with self.lock:
                if self.frames:
                    frame = self.frames[-1]
                    success, jpeg = cv2.imencode('.jpg', frame)
                    if success:
                        return jpeg.tobytes()
        except Exception as e:
            logger.error(f"Error getting frame for camera {self.camera_id}: {e}")
        return None
    
    def get_mjpeg_stream(self):
        """Generator for MJPEG stream"""
        try:
            while self.is_active:
                with self.lock:
                    if self.frames:
                        frame = self.frames[-1]
                        success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if success:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' +
                                   jpeg.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Error in MJPEG stream for camera {self.camera_id}: {e}")
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """Check if stream is still active"""
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < timeout_seconds and self.frame_count > 0
    
    def reset(self):
        """Reset stream buffer"""
        with self.lock:
            self.frames.clear()
            self.frame_count = 0
            self.is_active = False


def get_stream_buffer(camera_id: str) -> StreamBuffer:
    """Get or create stream buffer for camera"""
    if camera_id not in active_streams:
        active_streams[camera_id] = StreamBuffer(camera_id)
    return active_streams[camera_id]


def add_frame_to_stream(camera_id: str, frame_data: bytes):
    """Add frame to camera stream"""
    buffer = get_stream_buffer(camera_id)
    buffer.add_frame(frame_data)


def get_stream_info(camera_id: str) -> dict:
    """Get stream info"""
    if camera_id in active_streams:
        stream = active_streams[camera_id]
        return {
            "camera_id": camera_id,
            "is_active": stream.is_alive(),
            "width": stream.width,
            "height": stream.height,
            "fps": stream.fps,
            "frame_count": stream.frame_count,
            "last_update": stream.last_update.isoformat()
        }
    return {
        "camera_id": camera_id,
        "is_active": False,
        "error": "No stream found"
    }


def cleanup_inactive_streams(timeout_seconds: int = 60):
    """Remove inactive streams"""
    inactive = []
    for camera_id, stream in active_streams.items():
        if not stream.is_alive(timeout_seconds):
            inactive.append(camera_id)
    
    for camera_id in inactive:
        logger.info(f"Removing inactive stream for camera {camera_id}")
        del active_streams[camera_id]
    
    return len(inactive)
