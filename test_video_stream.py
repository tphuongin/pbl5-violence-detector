#!/usr/bin/env python
"""
Test script to simulate camera sending frames to the server.
Camera (Jetson Nano) would run similar code to send frames.
"""
import cv2
import requests
import time
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
CAMERA_ID = "test-camera-001"
TEST_IMAGE_PATH = "test_frame.jpg"

def generate_test_frame(width=640, height=480):
    """Generate a test frame (colored noise or pattern)"""
    import numpy as np
    
    # Create a colorful test pattern
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(frame, (50, 50), (590, 150), (0, 255, 0), -1)
    cv2.rectangle(frame, (50, 200), (590, 400), (0, 0, 255), -1)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (20, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Camera: {CAMERA_ID}", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def send_frame(frame, camera_id):
    """Send frame to server"""
    try:
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("✗ Failed to encode frame")
            return False
        
        # Send to server
        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
        response = requests.post(
            f"{API_URL}/api/stream/{camera_id}/frame",
            files=files,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✓ Frame sent ({len(buffer)} bytes)")
            return True
        else:
            print(f"✗ Server error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error sending frame: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Video Stream Test - Simulating Camera")
    print("=" * 60)
    print(f"Camera ID: {CAMERA_ID}")
    print(f"API URL: {API_URL}")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=2)
        print("✓ Server is running")
    except:
        print("✗ Server is not running!")
        print(f"  Start backend: cd backend && python main.py")
        return
    
    print("\n📹 Starting frame streaming...")
    print("Press Ctrl+C to stop\n")
    
    frame_count = 0
    try:
        while True:
            # Generate test frame
            frame = generate_test_frame()
            frame_count += 1
            
            # Send frame
            if send_frame(frame, CAMERA_ID):
                print(f"  Frame #{frame_count} sent successfully")
            
            # Wait a bit before sending next frame (simulate 30 FPS)
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print(f"\n\n✓ Stopped. Sent {frame_count} frames total")

if __name__ == "__main__":
    main()
