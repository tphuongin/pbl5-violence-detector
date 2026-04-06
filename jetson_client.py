import cv2
import asyncio
import websockets
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CẤU HÌNH ============
BACKEND_WS_URL = "ws://192.168.137.1:8000/ws/stream/{CAMERA_ID}"
CAMERA_ID = "jetson-cam-01"
FPS = 15
JPEG_QUALITY = 70
RECONNECT_DELAY = 3  # seconds
# ==================================

def find_camera():
    """Thử lần lượt camera index 1, 2, 3, 4, 0"""
    for index in [1, 2, 3, 4, 0]:
        logger.info(f"🔍 Thử camera index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"✅ Tìm thấy camera tại index {index}")
                return cap, index
            cap.release()
        logger.warning(f"❌ Camera index {index} không dùng được")
    return None, None

async def send_frames():
    cap, index = find_camera()

    if cap is None:
        logger.error("Không tìm thấy camera nào! Đã thử index: 1, 2, 3, 4, 0")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger.info(f"📷 Dùng camera index {index}")
    logger.info(f"🚀 Gửi tới {BACKEND_WS_URL.format(CAMERA_ID=CAMERA_ID)}")

    interval = 1.0 / FPS

    while True:
        try:
            async with websockets.connect(BACKEND_WS_URL.format(CAMERA_ID=CAMERA_ID)) as websocket:
                logger.info("✅ Kết nối WebSocket thành công")

                while True:
                    start = time.time()
                    ret, frame = cap.read()

                    if not ret:
                        logger.warning("Không đọc được frame, thử lại...")
                        await asyncio.sleep(1)
                        continue

                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                    # Send binary frame data
                    await websocket.send(buffer.tobytes())

                    elapsed = time.time() - start
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket bị ngắt, thử kết nối lại sau 3s...")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            logger.error(f"Lỗi: {e}, thử kết nối lại sau 3s...")
            await asyncio.sleep(RECONNECT_DELAY)

    cap.release()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_frames())
