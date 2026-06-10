"""
mobilenet_jetson_client_v3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hai chế độ hoạt động (KHÔNG chạy đồng thời):

  [CAMERA MODE]  — chế độ mặc định
      CameraCapture (thread) → frame_event
          ├── stream_video    (async) → ws://.../ws/stream/{CAMERA_ID}
          └── run_inference   (thread) → DetectionResult + SharedState
                  └── stream_detection (async) → ws://.../ws/detection/{CAMERA_ID}
                          ├── BuzzerController (thread) → GPIO PIN 12
                          └── PhoneDialer      (thread) → SIM AT command

  [VIDEO ANALYSIS MODE]  — khi backend upload video
      Backend POST /analyze-video  →  Jetson HTTP Server (aiohttp)
          └── VideoAnalysisJob (thread)
                  ├── cv2.VideoCapture(file)  đọc từng frame
                  ├── MobileNetTRT.infer()    inference
                  └── WebSocket stream        → ws://.../ws/video-analysis/{job_id}
                          frame payload (real-time)  +  summary payload (cuối)

  Jetson HTTP Server (aiohttp) lắng nghe trên JETSON_HTTP_PORT (mặc định 8001):
      POST /analyze-video       — nhận file video, trả {"job_id": "..."}
      GET  /status              — trả trạng thái hiện tại + job đang chạy
      POST /cancel              — huỷ job đang chạy

JSON detection payload (camera mode, gửi mỗi lần infer):
{
  "camera_id":         "jetson-cam-01",
  "timestamp":         "14:35:22.047",
  "label":             "VIOLENCE" | "Normal",
  "prob_raw":          0.73,
  "conf_thresh":       0.55,
  "alert":             true,
  "alert_until":       1.3,
  "alert_sec":         3.0,
  "infer_fps":         8.5,
  "cam_fps":           29.8,
  "uptime":            142,
  "infer_ms":          118.4,
  "num_frames":        16,
  "violence_duration": 3.7,
  "buzzer_active":     false,
  "phone_calling":     false,
  "thumb_b64":         "<base64 jpeg>"
}

Chạy: python3 mobilenet_jetson_client_v3.py
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import serial
import threading
import asyncio
import websockets
import time
import base64
import json
import logging
import uuid
import os
from collections import deque
from typing import Optional

import aiohttp
from aiohttp import web

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("[Buzzer] Jetson.GPIO không khả dụng — còi và nút bấm sẽ bị tắt.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Backend ──────────────────────────────────────────────
BACKEND_HOST      = "192.168.137.1"
BACKEND_PORT      = 8000
CAMERA_ID         = "jetson-cam-01"

WS_STREAM_URL     = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/ws/stream/{CAMERA_ID}"
WS_DETECTION_URL  = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/ws/detection/{CAMERA_ID}"

RECONNECT_DELAY   = 3       # giây chờ trước khi reconnect WebSocket

# ── Jetson HTTP server (nhận video từ backend) ────────────
JETSON_HTTP_HOST  = "0.0.0.0"
JETSON_HTTP_PORT  = 8001
UPLOAD_TMP_DIR    = "/tmp/jetson_video_upload"
MAX_VIDEO_SIZE_MB = 500

# ── Video analysis WebSocket (Jetson → Backend) ───────────
WS_VIDEO_ANALYSIS_URL_TEMPLATE = (
    f"ws://{{backend_host}}:{BACKEND_PORT}/ws/video-analysis/{{job_id}}?role=producer"
)

# ── Camera ───────────────────────────────────────────────
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
CAMERA_FPS        = 30

# ── Stream video ─────────────────────────────────────────
STREAM_FPS        = 15      # FPS gửi lên backend
JPEG_QUALITY      = 70

# ── TensorRT model ───────────────────────────────────────
ENGINE_PATH       = "mobilenetv3_update.engine"
NUM_FRAMES        = 16
INPUT_SIZE        = 172

# ── Detection logic ───────────────────────────────────────
CONF_THRESH       = 0.55    # ngưỡng xác suất để kích hoạt alert
ALERT_SECONDS     = 3.0     # thời gian duy trì trạng thái alert sau khi detect
INFER_INTERVAL_MS = 80      # khoảng thời gian tối thiểu giữa 2 lần inference (ms)

# ── Video analysis inference ──────────────────────────────
VIDEO_INFER_EVERY_N_FRAMES  = 1   # inference mỗi N frame (1 = mọi frame)
VIDEO_STREAM_EVERY_N_FRAMES = 5   # gửi thumbnail mỗi N frame (giảm băng thông)

# ── Thumbnail gửi kèm payload ────────────────────────────
THUMB_WIDTH       = 320
THUMB_HEIGHT      = 180

# ── Preprocessing (ImageNet normalization) ────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Buzzer (GPIO) ─────────────────────────────────────────
BUZZER_PIN              = 12    # Chân vật lý BOARD
BUTTON_PIN              = 16    # Chân nhận tín hiệu nút bấm (cạnh GND Pin 14)
VIOLENCE_BUZZER_DELAY   = 5.0   # Giây bạo lực liên tục trước khi kêu còi + gọi điện
BUZZER_BEEP_ON_SEC      = 0.5   # Thời gian còi kêu mỗi tiếng beep
BUZZER_BEEP_OFF_SEC     = 0.3   # Khoảng nghỉ giữa các tiếng beep

# ── Phone (SIM module AT command) ────────────────────────
PHONE_NUMBER        = "0961521940"     # Số điện thoại nhận cuộc gọi cảnh báo
PHONE_SERIAL_PORT   = "/dev/ttyUSB2"   # Cổng serial của SIM module
PHONE_BAUD_RATE     = 115200
PHONE_CALL_TIMEOUT  = 30               # Giây duy trì cuộc gọi rồi tự cúp
PHONE_RETRY_DELAY   = 60               # Giây cooldown giữa 2 lần gọi
PHONE_RING_WAIT     = 5                # Giây chờ sau ATD trước khi duy trì cuộc gọi
STARTUP_MUTE_SECONDS = 10.0            # Bỏ qua alert trong N giây đầu sau khi khởi động

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jetson-mobilenet")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JOB REGISTRY  — theo dõi các video analysis job
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JobStatus:
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    CANCELLED  = "cancelled"
    ERROR      = "error"

class VideoJob:
    def __init__(self, job_id: str, video_path: str, filename: str,
                 backend_host: str):
        self.job_id       = job_id
        self.video_path   = video_path
        self.filename     = filename
        self.backend_host = backend_host
        self.status       = JobStatus.QUEUED
        self.created_at   = time.time()
        self.started_at   = None
        self.ended_at     = None
        self.progress     = 0.0        # 0.0 → 1.0
        self.total_frames = 0
        self.processed    = 0
        self.error_msg    = None
        self._cancel_flag = threading.Event()

    def cancel(self):
        self._cancel_flag.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag.is_set()

    def to_dict(self) -> dict:
        return {
            "job_id":       self.job_id,
            "filename":     self.filename,
            "status":       self.status,
            "progress":     round(self.progress, 3),
            "total_frames": self.total_frames,
            "processed":    self.processed,
            "created_at":   self.created_at,
            "started_at":   self.started_at,
            "ended_at":     self.ended_at,
            "error_msg":    self.error_msg,
        }

class JobRegistry:
    """Thread-safe registry lưu tất cả video jobs."""
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, VideoJob] = {}

    def add(self, job: VideoJob):
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def current_running(self) -> Optional[VideoJob]:
        with self._lock:
            for j in self._jobs.values():
                if j.status == JobStatus.RUNNING:
                    return j
            return None

    def list_all(self) -> list:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE  (camera mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SharedState:
    def __init__(self):
        self._lock      = threading.Lock()
        self._raw_frame = None
        self._prob_raw  = 0.0
        self._label     = 'Normal'

    def update(self, raw_frame, prob_raw, label):
        with self._lock:
            self._raw_frame = raw_frame.copy()
            self._prob_raw  = prob_raw
            self._label     = label

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            if self._raw_frame is None:
                return None
            return {
                'raw_frame': self._raw_frame.copy(),
                'prob_raw':  self._prob_raw,
                'label':     self._label,
            }

def render_and_encode_thumb(raw_frame: np.ndarray) -> str:
    thumb = cv2.resize(raw_frame, (THUMB_WIDTH, THUMB_HEIGHT),
                       interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode('utf-8')

def encode_frame_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUZZER CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BuzzerController(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="BuzzerThread")
        self._active   = False
        self._running  = True
        self._lock     = threading.Lock()
        self._cond     = threading.Condition(self._lock)
        self.available = GPIO_AVAILABLE

        if self.available:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.HIGH)
            logger.info(f"[Buzzer] GPIO PIN {BUZZER_PIN} sẵn sàng.")
        else:
            logger.warning("[Buzzer] Chạy ở chế độ giả lập (không có GPIO).")

    def activate(self):
        with self._cond:
            if not self._active:
                self._active = True
                self._cond.notify_all()
                logger.warning("[Buzzer] ⚠  KÍCH HOẠT CÒI CẢNH BÁO!")

    def deactivate(self):
        with self._cond:
            if self._active:
                self._active = False
                self._cond.notify_all()
                logger.info("[Buzzer] Tắt còi.")

    def stop(self):
        with self._cond:
            self._running = False
            self._active  = False
            self._cond.notify_all()

    @property
    def is_active(self):
        with self._lock:
            return self._active

    def run(self):
        try:
            while True:
                with self._cond:
                    while not self._active and self._running:
                        self._cond.wait(timeout=1.0)
                    if not self._running:
                        break
                self._set_gpio(True)
                self._interruptible_sleep(BUZZER_BEEP_ON_SEC)
                self._set_gpio(False)
                self._interruptible_sleep(BUZZER_BEEP_OFF_SEC)
        except Exception as e:
            logger.exception(f"[Buzzer] Lỗi thread: {e}")
        finally:
            self._set_gpio(False)
            if self.available:
                GPIO.cleanup()
                logger.info("[Buzzer] GPIO cleanup xong.")

    def _set_gpio(self, state: bool):
        if self.available:
            GPIO.output(BUZZER_PIN, GPIO.LOW if state else GPIO.HIGH)
        else:
            if state:
                logger.debug("[Buzzer][SIM] BEEP ON")

    def _interruptible_sleep(self, seconds: float):
        deadline = time.time() + seconds
        while time.time() < deadline:
            with self._cond:
                remaining = deadline - time.time()
                if remaining <= 0 or not self._running:
                    break
                self._cond.wait(timeout=min(remaining, 0.05))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHONE DIALER  (SIM module AT command)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PhoneDialer(threading.Thread):
    _STATE_IDLE    = "IDLE"
    _STATE_CALLING = "CALLING"
    _MAX_DIAL_RETRY = 3
    _POST_CALL_SETTLE = 2.0
    _AT_LONG_TIMEOUT = 5.0

    def __init__(self):
        super().__init__(daemon=True, name="PhoneDialerThread")
        self._lock          = threading.Lock()
        self._cond          = threading.Condition(self._lock)
        self._running       = True
        self._requested     = False
        self._cancel        = False
        self._state         = self._STATE_IDLE
        self._serial        = None
        self.available      = False
        self._last_call_end = 0.0
        self._open_serial()

    def _open_serial(self) -> bool:
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass
        self._serial   = None
        self.available = False

        try:
            self._serial = serial.Serial(
                PHONE_SERIAL_PORT,
                baudrate=PHONE_BAUD_RATE,
                timeout=self._AT_LONG_TIMEOUT,
            )
            time.sleep(0.8)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            resp = self._send_at_raw("AT", wait=1.5)
            if "OK" in resp:
                self._send_at_raw("AT+CVHU=0", wait=1.0)
                self._send_at_raw("ATE0", wait=0.5)
                self.available = True
                logger.info(f"[Phone] SIM7600CE sẵn sàng — {PHONE_SERIAL_PORT}.")
                return True
            else:
                logger.warning(f"[Phone] SIM không phản hồi AT: {resp!r}")
                return False

        except Exception as e:
            logger.warning(f"[Phone] Không mở được {PHONE_SERIAL_PORT}: {e}")
            logger.warning("[Phone] Chạy ở chế độ giả lập.")
            return False

    def _ensure_serial(self) -> bool:
        if self._serial is None or not self._serial.is_open:
            logger.warning("[Phone] Serial đóng — thử reopen …")
            return self._open_serial()

        resp = self._send_at_raw("AT", wait=1.0)
        if "OK" in resp:
            return True

        logger.warning(f"[Phone] AT ping thất bại ({resp!r}) — thử reopen …")
        return self._open_serial()

    def _send_at_raw(self, cmd: str, wait: float = 1.0) -> str:
        if self._serial is None or not self._serial.is_open:
            return ""
        try:
            self._serial.reset_input_buffer()
            self._serial.write((cmd + "\r\n").encode())
            time.sleep(wait)
            raw = self._serial.read_all()
            return raw.decode(errors="ignore")
        except Exception as e:
            logger.error(f"[Phone] Lỗi AT '{cmd}': {e}")
            return ""

    def _send_at(self, cmd: str, wait: float = 1.0) -> str:
        if not self.available:
            return ""
        return self._send_at_raw(cmd, wait)

    def _wait_line_clear(self, max_wait: float = 6.0) -> bool:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            resp = self._send_at("AT+CLCC", wait=1.0)
            if "OK" in resp and "+CLCC:" not in resp:
                return True
            logger.debug(f"[Phone] Line chưa rảnh: {resp.strip()!r}")
            time.sleep(0.5)
        logger.warning("[Phone] Timeout chờ line rảnh.")
        return False

    def request_call(self):
        with self._cond:
            if self._state != self._STATE_IDLE:
                return
            now = time.time()
            cooldown_ok = (self._last_call_end == 0.0 or
                           now - self._last_call_end >= PHONE_RETRY_DELAY)
            if cooldown_ok:
                self._requested = True
                self._cond.notify_all()
                logger.warning(f"[Phone] ☎  YÊU CẦU GỌI {PHONE_NUMBER}!")
            else:
                remain = PHONE_RETRY_DELAY - (now - self._last_call_end)
                logger.debug(f"[Phone] Cooldown còn {remain:.0f}s — bỏ qua.")

    def cancel_call(self):
        with self._cond:
            if self._state == self._STATE_CALLING:
                self._cancel = True
                self._cond.notify_all()

    def stop(self):
        with self._cond:
            self._running   = False
            self._cancel    = True
            self._requested = False
            self._cond.notify_all()

    @property
    def is_calling(self) -> bool:
        with self._lock:
            return self._state == self._STATE_CALLING

    def run(self):
        try:
            while True:
                with self._cond:
                    while not self._requested and self._running:
                        self._cond.wait(timeout=1.0)
                    if not self._running:
                        break
                    self._requested = False
                    self._state     = self._STATE_CALLING
                    self._cancel    = False
                self._do_call()
        except Exception as e:
            logger.exception(f"[Phone] Lỗi thread: {e}")
        finally:
            self._hangup()
            try:
                if self._serial and self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass

    def _do_call(self):
        logger.warning(f"[Phone] ☎  Đang quay số {PHONE_NUMBER} …")

        if self.available:
            if not self._ensure_serial():
                logger.error("[Phone] Không thể kết nối lại module SIM — bỏ cuộc gọi.")
                self._finalize_call()
                return

            dialed = False
            for attempt in range(1, self._MAX_DIAL_RETRY + 1):
                resp = self._send_at(
                    f"ATD{PHONE_NUMBER};",
                    wait=self._AT_LONG_TIMEOUT
                )
                logger.info(f"[Phone] ATD attempt {attempt}: {resp.strip()!r}")
                if "OK" in resp or "CONNECT" in resp:
                    dialed = True
                    break
                if "ERROR" in resp or "NO CARRIER" in resp:
                    logger.warning(f"[Phone] Lỗi ATD lần {attempt}: {resp.strip()!r}")
                    time.sleep(1.0)

            if not dialed:
                logger.error("[Phone] Không quay số được sau "
                             f"{self._MAX_DIAL_RETRY} lần thử.")
                self._finalize_call()
                return

            deadline = time.time() + PHONE_CALL_TIMEOUT
            while time.time() < deadline:
                with self._cond:
                    if self._cancel or not self._running:
                        break
                    self._cond.wait(timeout=min(deadline - time.time(), 0.2))

        else:
            logger.warning(f"[Phone][SIM] GIẢ LẬP: ATD{PHONE_NUMBER};")
            deadline = time.time() + PHONE_CALL_TIMEOUT
            while time.time() < deadline:
                with self._cond:
                    if self._cancel or not self._running:
                        break
                    self._cond.wait(timeout=min(deadline - time.time(), 0.2))

        self._hangup()
        self._finalize_call()

    def _hangup(self):
        if self.available:
            logger.info("[Phone] Gửi ATH …")
            self._send_at("ATH", wait=1.5)
            self._wait_line_clear(max_wait=6.0)
            logger.info(f"[Phone] Settle {self._POST_CALL_SETTLE}s …")
            time.sleep(self._POST_CALL_SETTLE)
            logger.info("[Phone] Reopen serial port sau cuộc gọi …")
            self._open_serial()
        else:
            logger.debug("[Phone][SIM] GIẢ LẬP: ATH")

    def _finalize_call(self):
        with self._lock:
            self._last_call_end = time.time()
            self._cancel        = False
            self._state         = self._STATE_IDLE
        logger.info("[Phone] Line rảnh — sẵn sàng cho cuộc gọi tiếp theo.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CAMERA CAPTURE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CameraCapture(threading.Thread):
    def __init__(self, frame_event: threading.Event):
        super().__init__(daemon=True, name="CameraThread")
        self._frame_event = frame_event
        self._frame       = None
        self._lock        = threading.Lock()
        self._running     = True
        self.connected    = False
        self.error_msg    = None
        self.camera_id    = None

    def _try_open(self, idx: int):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return None
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            time.sleep(0.3)
        cap.release()
        return None

    def run(self):
        cap = None
        for idx in range(5):
            logger.info(f"[Camera] Thử index {idx} …")
            cap = self._try_open(idx)
            if cap is not None:
                self.camera_id = idx
                break
        if cap is None:
            self.error_msg = "Không mở được camera nào (đã thử 0–4)"
            logger.error(f"[Camera] {self.error_msg}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        self.connected = True
        logger.info(f"[Camera] OK — index={self.camera_id}")
        while self._running:
            if not cap.grab():
                time.sleep(0.02)
                continue
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._frame_event.set()
        cap.release()

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TENSORRT MODEL  (MobileNetV2-TSM 16-frame)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MobileNetTRT:
    """
    Wrapper TensorRT cho MobileNetV2-TSM.
    Input : (1, 16, 3, INPUT_SIZE, INPUT_SIZE)  float32  ImageNet-normalized
    Output: (1, 2)  logits  [normal, violence]
    """
    def __init__(self, engine_path, cuda_ctx):
        self.cuda_ctx = cuda_ctx
        self.cuda_ctx.push()

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        self.context  = self.engine.create_execution_context()
        self.inputs   = []
        self.outputs  = []
        self.bindings = []

        for b in self.engine:
            shape   = tuple(self.engine.get_binding_shape(b))
            dtype   = trt.nptype(self.engine.get_binding_dtype(b))
            dev_mem = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
            self.bindings.append(int(dev_mem))
            info = {
                "device": dev_mem,
                "host":   cuda.pagelocked_empty(shape, dtype),
            }
            if self.engine.binding_is_input(b):
                self.inputs.append(info)
            else:
                self.outputs.append(info)

        self.stream = cuda.Stream()
        self.cuda_ctx.pop()
        logger.info(f"[TRT] MobileNetV2-TSM loaded — {self.engine.num_bindings} bindings")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        self.cuda_ctx.push()
        try:
            self.inputs[0]["host"].flat[:] = blob.ravel()
            cuda.memcpy_htod_async(
                self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            self.context.execute_async_v2(self.bindings, self.stream.handle)
            cuda.memcpy_dtoh_async(
                self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
            self.stream.synchronize()
            return self.outputs[0]["host"].copy().flatten()
        finally:
            self.cuda_ctx.pop()

    def destroy(self):
        self.cuda_ctx.push()
        del self.context
        del self.engine
        self.cuda_ctx.pop()
        self.cuda_ctx.detach()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREPROCESSING HELPENS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess(frames: list) -> np.ndarray:
    """
    frames: list of NUM_FRAMES BGR numpy arrays
    Returns: (1, NUM_FRAMES, 3, INPUT_SIZE, INPUT_SIZE) float32
    """
    out = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        r   = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        n   = (r.astype(np.float32) / 255.0 - MEAN) / STD
        out.append(n.transpose(2, 0, 1))   # (3, H, W)
    return np.ascontiguousarray(np.stack(out)[np.newaxis], dtype=np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionResult:
    def __init__(self, loop):
        self._lock  = threading.Lock()
        self._data  = None
        self._loop  = loop
        self._event = None

    def put(self, data: dict):
        with self._lock:
            self._data = data
        if self._event is not None:
            self._loop.call_soon_threadsafe(self._event.set)

    async def get_new(self) -> dict:
        if self._event is None:
            self._event = asyncio.Event()
        await self._event.wait()
        self._event.clear()
        with self._lock:
            return self._data

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VIDEO ANALYSIS JOB RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_video_analysis(job: VideoJob,
                       cuda_ctx,
                       loop: asyncio.AbstractEventLoop):
    job.status     = JobStatus.RUNNING
    job.started_at = time.time()
    logger.info(f"[VideoJob:{job.job_id[:8]}] Bắt đầu — {job.filename}")

    ws_url = WS_VIDEO_ANALYSIS_URL_TEMPLATE.format(
        backend_host=job.backend_host,
        job_id=job.job_id,
    )

    future = asyncio.run_coroutine_threadsafe(
        _video_analysis_coroutine(job, cuda_ctx, ws_url), loop
    )
    try:
        future.result()
    except Exception as e:
        logger.exception(f"[VideoJob:{job.job_id[:8]}] Lỗi: {e}")
        job.status    = JobStatus.ERROR
        job.error_msg = str(e)
    finally:
        try:
            if os.path.exists(job.video_path):
                os.remove(job.video_path)
                logger.info(f"[VideoJob:{job.job_id[:8]}] Đã xoá file tạm.")
        except Exception:
            pass

async def _video_analysis_coroutine(job: VideoJob, cuda_ctx, ws_url: str):
    cap = cv2.VideoCapture(job.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {job.video_path}")

    total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s       = total_frames / fps_video
    job.total_frames = total_frames
    
    logger.info(
        f"[VideoJob:{job.job_id[:8]}] "
        f"{total_frames} frames  {fps_video:.1f}fps  {duration_s:.1f}s"
    )

    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(
        None, lambda: MobileNetTRT(ENGINE_PATH, cuda_ctx)
    )

    all_probs    = []
    segments     = []
    seg_start    = None
    seg_max_prob = 0.0
    alert_until  = 0.0

    frame_buffer = deque(maxlen=NUM_FRAMES)

    t_start = time.time()

    async with websockets.connect(ws_url) as ws:
        logger.info(f"[VideoJob:{job.job_id[:8]}] WS OK → {ws_url}")

        frame_idx = 0
        while True:
            if job.is_cancelled:
                job.status = JobStatus.CANCELLED
                logger.info(f"[VideoJob:{job.job_id[:8]}] Đã huỷ.")
                break

            ret, frame = cap.read()
            if not ret:
                break   # hết video

            frame_time_s = frame_idx / fps_video
            frame_buffer.append(frame)

            # Chỉ inference khi đủ NUM_FRAMES và mỗi VIDEO_INFER_EVERY_N_FRAMES
            if (len(frame_buffer) >= NUM_FRAMES and
                    frame_idx % VIDEO_INFER_EVERY_N_FRAMES == 0):

                blob = await loop.run_in_executor(
                    None, lambda fb=frame_buffer: preprocess(list(fb))
                )
                logits = await loop.run_in_executor(None, model.infer, blob)
                probs  = softmax(logits)
                prob_raw = float(probs[1])

                if prob_raw >= CONF_THRESH:
                    alert_until = frame_time_s + ALERT_SECONDS

                is_alert = frame_time_s < alert_until
                label    = "VIOLENCE" if is_alert else "Normal"
                all_probs.append(prob_raw)

                if label == "VIOLENCE":
                    if seg_start is None:
                        seg_start    = frame_time_s
                        seg_max_prob = prob_raw
                    else:
                        seg_max_prob = max(seg_max_prob, prob_raw)
                else:
                    if seg_start is not None:
                        segments.append({
                            "start_s":  round(seg_start, 2),
                            "end_s":    round(frame_time_s, 2),
                            "max_prob": round(seg_max_prob, 3),
                        })
                        seg_start = None

                if frame_idx % VIDEO_STREAM_EVERY_N_FRAMES == 0:
                    thumb_b64 = render_and_encode_thumb(frame)
                    progress  = frame_idx / max(total_frames, 1)
                    job.processed = frame_idx
                    job.progress  = progress

                    ts_str = (time.strftime('%H:%M:%S')
                              + f'.{int((time.time() % 1) * 1000):03d}')

                    frame_payload = {
                        "type":         "frame",
                        "job_id":       job.job_id,
                        "frame_idx":    frame_idx,
                        "frame_time_s": round(frame_time_s, 3),
                        "total_frames": total_frames,
                        "progress":     round(progress, 3),
                        "label":        label,
                        "prob_raw":     round(prob_raw, 4),
                        "thumb_b64":    thumb_b64,
                        "timestamp":    ts_str,
                    }
                    await ws.send(json.dumps(frame_payload, ensure_ascii=False))

            frame_idx += 1

        cap.release()

        if seg_start is not None:
            segments.append({
                "start_s":  round(seg_start, 2),
                "end_s":    round(duration_s, 2),
                "max_prob": round(seg_max_prob, 3),
            })

        if job.status != JobStatus.CANCELLED:
            job.status   = JobStatus.DONE
            job.ended_at = time.time()
            job.progress = 1.0

            n_inferred     = len(all_probs)
            n_violence     = sum(1 for p in all_probs if p >= CONF_THRESH)
            violence_ratio = n_violence / max(n_inferred, 1)
            verdict        = 'VIOLENCE' if violence_ratio >= 0.15 else 'Normal'
            processing_time = job.ended_at - t_start

            summary_payload = {
                "type":               "summary",
                "job_id":             job.job_id,
                "filename":           job.filename,
                "total_frames":       total_frames,
                "frames_inferred":    n_inferred,
                "video_duration_s":   round(duration_s,       2),
                "violence_ratio":     round(violence_ratio,   4),
                "max_prob":           round(max(all_probs, default=0.0), 4),
                "avg_prob":           round(
                    sum(all_probs) / max(len(all_probs), 1), 4),
                "violence_segments":  segments,
                "verdict":            verdict,
                "processing_time_s":  round(processing_time,  2),
            }
            await ws.send(json.dumps(summary_payload, ensure_ascii=False))
            logger.info(
                f"[VideoJob:{job.job_id[:8]}] DONE — "
                f"verdict={verdict}  ratio={violence_ratio:.2%}  "
                f"segments={len(segments)}  "
                f"time={processing_time:.1f}s"
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AIOHTTP HTTP SERVER  — nhận video từ backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_http_app(job_registry: JobRegistry,
                   loop: asyncio.AbstractEventLoop,
                   cuda_ctx) -> web.Application:
    os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)

    async def handle_analyze_video(request: web.Request) -> web.Response:
        running = job_registry.current_running()
        if running:
            return web.json_response(
                {"error": f"Job {running.job_id} đang chạy, vui lòng chờ hoặc /cancel"},
                status=409,
            )

        try:
            reader = await request.multipart()
        except Exception:
            return web.json_response({"error": "Cần gửi multipart/form-data"}, status=400)

        video_path   = None
        filename     = "video.mp4"
        backend_host = BACKEND_HOST

        async for part in reader:
            if part.name == "video":
                filename  = part.filename or "video.mp4"
                ext       = os.path.splitext(filename)[1] or ".mp4"
                tmp_path  = os.path.join(UPLOAD_TMP_DIR, f"{uuid.uuid4().hex}{ext}")
                size      = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = await part.read_chunk(65536)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
                            f.close()
                            os.remove(tmp_path)
                            return web.json_response(
                                {"error": f"Video vượt {MAX_VIDEO_SIZE_MB}MB"},
                                status=413,
                            )
                        f.write(chunk)
                video_path = tmp_path
                logger.info(f"[HTTP] Nhận video: {filename}  {size/1024/1024:.1f}MB")

            elif part.name == "backend_host":
                backend_host = (await part.read()).decode().strip() or BACKEND_HOST

        if video_path is None:
            return web.json_response({"error": "Thiếu field 'video'"}, status=400)

        job = VideoJob(
            job_id=uuid.uuid4().hex,
            video_path=video_path,
            filename=filename,
            backend_host=backend_host,
        )
        job_registry.add(job)

        threading.Thread(
            target=run_video_analysis,
            args=(job, cuda_ctx, loop),
            daemon=True,
            name=f"VideoJob-{job.job_id[:8]}",
        ).start()

        logger.info(f"[HTTP] Job tạo: {job.job_id}  file={filename}")
        return web.json_response({"job_id": job.job_id, "status": "queued"})

    async def handle_status(request: web.Request) -> web.Response:
        running = job_registry.current_running()
        return web.json_response({
            "mode":        "video_analysis" if running else "idle",
            "current_job": running.to_dict() if running else None,
            "jobs":        job_registry.list_all(),
        })

    async def handle_cancel(request: web.Request) -> web.Response:
        try:
            body   = await request.json()
            job_id = body.get("job_id")
        except Exception:
            job_id = None

        if job_id:
            job = job_registry.get(job_id)
        else:
            job = job_registry.current_running()

        if job is None:
            return web.json_response({"error": "Không tìm thấy job"}, status=404)

        job.cancel()
        return web.json_response({"cancelled": True, "job_id": job.job_id})

    app = web.Application(client_max_size=MAX_VIDEO_SIZE_MB * 1024 * 1024 + 1024)
    app.router.add_post("/analyze-video", handle_analyze_video)
    app.router.add_get("/status",         handle_status)
    app.router.add_post("/cancel",        handle_cancel)
    return app

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INFERENCE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_inference(cam: CameraCapture,
                  frame_event: threading.Event,
                  result_queue: DetectionResult,
                  shared_state: SharedState,
                  buzzer: BuzzerController,
                  dialer: PhoneDialer,
                  start_time: float,
                  cuda_ctx):
    """
    Thu thập NUM_FRAMES frame từ camera, chạy TRT inference,
    áp dụng logic alert, rồi đẩy payload vào result_queue.
    Dùng cuda_ctx chia sẻ từ main (không tạo ctx mới).
    """
    cuda_ctx.push()

    logger.info("[Inference] Loading TensorRT engine …")
    model = MobileNetTRT(ENGINE_PATH, cuda_ctx)

    if not frame_event.wait(timeout=30.0):
        logger.error("[Inference] Timeout frame!")
        return

    logger.info("[Inference] Bắt đầu inference.")

    # ── Frame buffer (sliding window NUM_FRAMES frames) ──
    frame_buffer = deque(maxlen=NUM_FRAMES)

    # ── Alert state ───────────────────────────────────────
    alert_until = 0.0
    is_alert    = False

    # ── Biến theo dõi thời gian bạo lực liên tục ─────────
    violence_start_time = None   # None = không có bạo lực hiện tại
    alert_triggered     = False  # đã trigger còi + gọi điện trong đợt này chưa
    last_infer_ms       = 0.0

    # ── [THÊM MỚI] CÀI ĐẶT NÚT BẤM (HARDWARE INTERRUPT) ──────────────────
    manual_reset_event = threading.Event()
    force_mute_until   = 0.0

    if GPIO_AVAILABLE:
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        def button_callback(channel):
            manual_reset_event.set()

        GPIO.add_event_detect(
            BUTTON_PIN, GPIO.FALLING,
            callback=button_callback,
            bouncetime=1000
        )
        logger.info(f"[Button] Hardware interrupt đã kích hoạt — GPIO PIN {BUTTON_PIN}.")
    else:
        logger.warning("[Button] GPIO không khả dụng — nút bấm sẽ không hoạt động.")
    # ────────────────────────────────────────────────────────────────────────

    # ── FPS tracking ─────────────────────────────────────
    cam_times   = deque(maxlen=60)
    infer_times = deque(maxlen=30)

    try:
        while True:
            frame_event.clear()
            frame = cam.get_latest()
            if frame is None:
                frame_event.wait(timeout=0.1)
                continue

            now    = time.time()
            now_ms = now * 1000
            startup_muted = (now - start_time) < STARTUP_MUTE_SECONDS

            if (now_ms - last_infer_ms) < INFER_INTERVAL_MS:
                frame_event.wait(timeout=0.1)
                continue

            # ── [THÊM MỚI] XỬ LÝ KHI NÚT ĐƯỢC BẤM ────────────────────────
            if manual_reset_event.is_set():
                manual_reset_event.clear()
                logger.warning("[Button] 🛑 ĐÃ BẤM NÚT! Tắt còi, hủy gọi điện, ép reset AI.")

                frame_buffer.clear()
                violence_start_time = None
                alert_triggered     = False
                alert_until         = 0.0
                is_alert            = False
                label               = 'Normal'
                violence_duration   = 0.0

                buzzer.deactivate()
                dialer.cancel_call()

                force_mute_until = now + 5.0
                logger.info("[Button] Hệ thống tạm dừng phân tích trong 5 giây.")

            if now < force_mute_until:
                frame_event.wait(timeout=0.1)
                continue
            # ────────────────────────────────────────────────────────────────

            frame_buffer.append(frame)
            cam_times.append(now)

            # Chờ đủ NUM_FRAMES mới bắt đầu infer
            if len(frame_buffer) < NUM_FRAMES:
                logger.info(f"[Inference] Buffering {len(frame_buffer)}/{NUM_FRAMES}…")
                last_infer_ms = now_ms
                frame_event.wait(timeout=0.1)
                continue

            # ── Inference ────────────────────────────────
            t0 = time.time()
            try:
                blob     = preprocess(list(frame_buffer))
                logits   = model.infer(blob)
                probs    = softmax(logits)
                prob_raw = float(probs[1])   # index 1 = violence
            except Exception as e:
                logger.error(f"[Inference] Lỗi infer: {e}")
                last_infer_ms = now_ms
                frame_event.wait(timeout=0.1)
                continue
            t1 = time.time()
            last_infer_ms = now_ms
            infer_times.append(t1)

            # ── Alert logic ───────────────────────────────
            if prob_raw >= CONF_THRESH:
                alert_until = now + ALERT_SECONDS

            is_alert = now < alert_until
            label = "VIOLENCE" if is_alert else "Normal"
            shared_state.update(frame, prob_raw, label)

            # ── Buzzer + Phone logic ──────────────────────
            if label == "VIOLENCE":
                if violence_start_time is None:
                    violence_start_time = now
                    alert_triggered     = False
                    logger.info("[Alert] Bắt đầu theo dõi thời gian bạo lực …")

                violence_duration = now - violence_start_time

                if violence_duration >= VIOLENCE_BUZZER_DELAY:
                    if not startup_muted:
                        buzzer.activate()
                        if not alert_triggered:
                            dialer.request_call()
                            alert_triggered = True
                    else:
                        buzzer.deactivate()
                        dialer.cancel_call()
                else:
                    remaining = VIOLENCE_BUZZER_DELAY - violence_duration
                    logger.debug(
                        f"[Alert] Bạo lực {violence_duration:.1f}s "
                        f"/ {VIOLENCE_BUZZER_DELAY}s "
                        f"(còn {remaining:.1f}s nữa)")
            else:
                if violence_start_time is not None:
                    elapsed = now - violence_start_time
                    logger.info(f"[Alert] Kết thúc bạo lực sau {elapsed:.1f}s — reset.")
                    violence_start_time = None
                    alert_triggered     = False
                buzzer.deactivate()
                dialer.cancel_call()
                violence_duration = 0.0

            # ── FPS ──────────────────────────────────────
            infer_fps = (len(infer_times) - 1) / (infer_times[-1] - infer_times[0]) \
                        if len(infer_times) > 1 else 0.0
            cam_fps   = (len(cam_times) - 1) / (cam_times[-1] - cam_times[0]) \
                        if len(cam_times) > 1 else 0.0

            # ── Thumbnail ────────────────────────────────
            ts_str = (time.strftime("%H:%M:%S") + f".{int((now % 1) * 1000):03d}")

            # ── Payload ───────────────────────────────────
            payload = {
                "camera_id":         CAMERA_ID,
                "timestamp":         ts_str,
                "label":             label,
                "prob_raw":          round(prob_raw,            4),
                "conf_thresh":       CONF_THRESH,
                "alert":             is_alert,
                "alert_until":       round(max(0.0, alert_until - now), 2),
                "alert_sec":         ALERT_SECONDS,
                "infer_fps":         round(infer_fps,           1),
                "cam_fps":           round(cam_fps,             1),
                "uptime":            int(now - start_time),
                "infer_ms":          round((t1 - t0) * 1000,    1),
                "num_frames":        NUM_FRAMES,
                "violence_duration": round(violence_duration,   2),
                "buzzer_active":     buzzer.is_active,
                "phone_calling":     dialer.is_calling,
            }

            result_queue.put(payload)

            logger.info(
                f"prob={prob_raw:.3f}  alert={is_alert}  → {label}  "
                f"dur={violence_duration:.1f}s  "
                f"buzzer={'ON' if buzzer.is_active else 'off'}  "
                f"phone={'CALLING' if dialer.is_calling else 'idle'}  "
                f"({(t1 - t0) * 1000:.0f}ms)"
            )
            frame_event.wait(timeout=0.1)

    except Exception as e:
        logger.exception(f"[Inference] Lỗi nghiêm trọng: {e}")
    finally:
        buzzer.deactivate()
        dialer.cancel_call()
        model.destroy()
        cuda_ctx.pop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC TASK 1 — STREAM VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def stream_video(cam: CameraCapture,
                       loop: asyncio.AbstractEventLoop):
    interval = 1.0 / STREAM_FPS
    while True:
        try:
            async with websockets.connect(WS_STREAM_URL) as ws:
                logger.info(f"[Stream] Kết nối OK → {WS_STREAM_URL}")
                while True:
                    t0    = time.time()
                    frame = cam.get_latest()
                    if frame is not None:
                        jpeg_bytes = await loop.run_in_executor(
                            None, encode_frame_jpeg, frame)
                        await ws.send(jpeg_bytes)
                    sleep_t = interval - (time.time() - t0)
                    if sleep_t > 0:
                        await asyncio.sleep(sleep_t)
                    else:
                        await asyncio.sleep(0)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"[Stream] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …")
        except Exception as e:
            logger.error(f"[Stream] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …")
        await asyncio.sleep(RECONNECT_DELAY)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC TASK 2 — GỬI KẾT QUẢ DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def stream_detection(result_queue: DetectionResult,
                            shared_state: SharedState,
                            loop: asyncio.AbstractEventLoop):
    while True:
        try:
            async with websockets.connect(WS_DETECTION_URL) as ws:
                logger.info(f"[Detection] Kết nối OK → {WS_DETECTION_URL}")
                while True:
                    payload = await result_queue.get_new()
                    snap = shared_state.snapshot()
                    if snap is not None:
                        thumb_b64 = await loop.run_in_executor(
                            None, render_and_encode_thumb, snap['raw_frame'])
                        payload['thumb_b64'] = thumb_b64
                    await ws.send(json.dumps(payload, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"[Detection] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …")
        except Exception as e:
            logger.error(f"[Detection] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …")
        await asyncio.sleep(RECONNECT_DELAY)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    start_time = time.time()
    loop       = asyncio.get_event_loop()

    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    cuda_ctx.pop()   # pop ngay, mỗi thread sẽ push/pop khi cần

    buzzer = BuzzerController()
    buzzer.start()

    dialer = PhoneDialer()
    dialer.start()

    frame_event  = threading.Event()
    shared_state = SharedState()
    job_registry = JobRegistry()

    cam = CameraCapture(frame_event)
    cam.start()
    logger.info("[Main] Chờ camera kết nối …")
    for _ in range(80):
        if cam.connected or cam.error_msg:
            break
        await asyncio.sleep(0.1)

    if cam.error_msg:
        logger.error(f"[Main] {cam.error_msg}")
        buzzer.stop()
        dialer.stop()
        cuda_ctx.detach()
        return

    logger.info(f"[Main] Camera sẵn sàng — index={cam.camera_id}")

    result_queue = DetectionResult(loop)

    infer_thread = threading.Thread(
        target=run_inference,
        args=(cam, frame_event, result_queue, shared_state, buzzer, dialer,
              start_time, cuda_ctx),
        daemon=True,
        name="InferenceThread",
    )
    infer_thread.start()
    logger.info("[Main] Inference thread đã khởi động")

    http_app    = build_http_app(job_registry, loop, cuda_ctx)
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site   = web.TCPSite(http_runner, JETSON_HTTP_HOST, JETSON_HTTP_PORT)
    await http_site.start()

    print("=" * 60)
    print("  Jetson Violence Detection Client — MobileNetV2-TSM  [v5 — MANUAL RESET BUTTON]")
    print(f"  Camera index       : {cam.camera_id}")
    print(f"  Stream URL         : {WS_STREAM_URL}")
    print(f"  Detection URL      : {WS_DETECTION_URL}")
    print(f"  Stream FPS         : {STREAM_FPS}")
    print(f"  Num frames         : {NUM_FRAMES} (sliding window)")
    print(f"  Input size         : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Conf thresh        : {CONF_THRESH}")
    print(f"  Alert window       : {ALERT_SECONDS}s")
    print(f"  ── Video Analysis HTTP Server ─────────────")
    print(f"  Listen             : http://{JETSON_HTTP_HOST}:{JETSON_HTTP_PORT}")
    print(f"  POST               : /analyze-video  (multipart, field: 'video')")
    print(f"  GET                : /status")
    print(f"  POST               : /cancel")
    print(f"  WS result          : ws://BACKEND/ws/video-analysis/{{job_id}}")
    print(f"  Max upload         : {MAX_VIDEO_SIZE_MB}MB")
    print(f"  ── Buzzer ─────────────────────────────────")
    print(f"  GPIO PIN           : {BUZZER_PIN}  (BOARD)")
    print(f"  Kích hoạt sau      : {VIOLENCE_BUZZER_DELAY}s bạo lực liên tục")
    print(f"  Beep ON / OFF      : {BUZZER_BEEP_ON_SEC}s / {BUZZER_BEEP_OFF_SEC}s")
    print(f"  GPIO khả dụng      : {GPIO_AVAILABLE}")
    print(f"  ── Nút Bấm (Manual Reset) ─────────────────")
    print(f"  BUTTON PIN         : {BUTTON_PIN}  (BOARD mode, cạnh GND Pin 14)")
    print(f"  Tác dụng           : Tắt còi + Hủy gọi + Clear Buffer + Mù 5s")
    print(f"  ── Phone ──────────────────────────────────")
    print(f"  Số điện thoại      : {PHONE_NUMBER}")
    print(f"  Serial port        : {PHONE_SERIAL_PORT}  @ {PHONE_BAUD_RATE} baud")
    print(f"  SIM khả dụng       : {dialer.available}")
    print(f"  Thời gian gọi      : {PHONE_CALL_TIMEOUT}s rồi tự cúp")
    print(f"  Cooldown gọi lại   : {PHONE_RETRY_DELAY}s")
    print("=" * 60)

    try:
        await asyncio.gather(
            stream_video(cam, loop),
            stream_detection(result_queue, shared_state, loop),
        )
    finally:
        cam.stop()
        buzzer.stop()
        dialer.stop()
        await http_runner.cleanup()
        cuda_ctx.detach()
        logger.info("[Main] Đã dừng.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("[Main] Dừng chương trình.")
    finally:
        loop.close()
