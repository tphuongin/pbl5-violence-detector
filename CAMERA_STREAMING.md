# Video Streaming Guide - Jetson Nano Camera

## Architecture

```
Jetson Nano Camera → send frames → Backend Server → receive frames → store in buffer
                                                                              ↓
Frontend Browser ← stream video ← Backend Server ← read from buffer
```

## Setup on Jetson Nano

### 1. Prerequisites

```bash
# SSH into Jetson Nano
ssh jetson@[jetson_ip]

# Install required packages
sudo apt-get update
sudo apt-get install python3-pip python3-cv2
pip3 install requests
```

### 2. Camera Streaming Code (Python)

Create file `camera_stream.py` on Jetson Nano:

```python
import cv2
import requests
import time
import os

# Configuration
SERVER_URL = "http://[your_server_ip]:8000"
CAMERA_ID = "jetson-nano-camera-1"
CAMERA_INDEX = 0  # 0 = USB camera or CSI camera

def stream_camera():
    """Stream camera to server"""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Starting stream from camera {CAMERA_INDEX} to {SERVER_URL}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Encode and send frame
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
            try:
                files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                resp = requests.post(
                    f"{SERVER_URL}/api/stream/{CAMERA_ID}/frame",
                    files=files,
                    timeout=5
                )
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frames sent: {frame_count}")
            except Exception as e:
                print(f"Error: {e}")
        
        # Delay for ~30 FPS
        time.sleep(0.033)
    
    cap.release()

if __name__ == "__main__":
    stream_camera()
```

### 3. Run Stream

```bash
python3 camera_stream.py
```

## Frontend Display

The frontend already handles:
- ✓ Displaying MJPEG stream
- ✓ "No Signal" display when no frames received
- ✓ Auto-refresh frames
- ✓ Camera selection

Just select a camera in CameraPage to view the stream.

## Debugging

### Check Stream Status
```bash
curl http://localhost:8000/api/stream/{camera_id}/info
```

Response example:
```json
{
  "camera_id": "jetson-nano-camera-1",
  "is_active": true,
  "width": 640,
  "height": 480,
  "fps": 30,
  "frame_count": 1500,
  "last_update": "2026-04-04T10:30:00"
}
```

### Reset Stream
```bash
curl -X POST http://localhost:8000/api/stream/{camera_id}/reset
```

### Get Current Frame
```bash
# Get latest frame as JPEG (view in img tag)
GET /api/stream/{camera_id}/live
```

## Testing Locally

Run test script to simulate camera:

```bash
python test_video_stream.py
```

This will:
1. Check if server is running
2. Generate test frames
3. Send them to the server at ~30 FPS
4. Display frame counter

Then open frontend and you'll see video stream!

## Performance Tips

1. **Reduce frame size**: Use 640x480 or lower for better bandwidth
2. **Adjust quality**: Edit stream_service.py to change JPEG quality (currently 80)
3. **Limit FPS**: Send only 10-15 FPS if bandwidth is limited
4. **Use wired connection**: Ethernet is more stable than WiFi

## Troubleshooting

### "No Signal" displayed
- Check if Jetson Nano is connected to network
- Verify `SERVER_URL` is correct (use server's IP, not localhost)
- Check firewall isn't blocking port 8000
- Run test_video_stream.py to verify connection works

### Lag/choppy video
- Reduce resolution to 320x240
- Reduce FPS on Jetson side
- Reduce JPEG quality in stream_service.py
- Check network bandwidth

### Camera not recognized
- Run `ls /dev/video*` on Jetson to find camera device
- Try different indices (0, 1, 2, etc.)
- Check USB camera is properly connected

## Integration with Violence Detection

When violence is detected:
1. Save the frame/clip to Cloudinary
2. Add record to `VIOLENCE_HISTORY` table
3. Display in DetectionHistoryPage
4. ClipURL will link to Cloudinary storage
