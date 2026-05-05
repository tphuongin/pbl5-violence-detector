"""
Jetson Violence Detection Client — v5 (Video Analysis Mode)

Hai chế độ hoạt động (KHÔNG chạy đồng thời):

  [CAMERA MODE]
      CameraCapture (thread) → frame_event
          ├── stream_video    (async) → ws://.../ws/stream/{CAMERA_ID}
          └── run_inference   (thread) → SharedState + DetectionResult
                  └── stream_detection (async) → ws://.../ws/detection/{CAMERA_ID}
                          ├── BuzzerController (thread) → GPIO PIN 12
                          └── PhoneDialer      (thread) → SIM AT command

  [VIDEO ANALYSIS MODE]
      Backend POST /analyze-video  →  Jetson HTTP Server (aiohttp)
          └── VideoAnalysisJob (thread)
                  ├── cv2.VideoCapture(file)  đọc từng frame
                  ├── MoViNetTRT.infer()      inference
                  └── WebSocket stream        → ws://.../ws/video-analysis/{job_id}
                          frame payload (real-time)  +  summary payload (cuối)

  Jetson HTTP Server lắng nghe trên JETSON_HTTP_PORT (mặc định 8001):
      POST /analyze-video   — nhận file video, trả {"job_id": "..."}
      GET  /status          — trả trạng thái hiện tại + job đang chạy
      POST /cancel          — huỷ job đang chạy
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from collections import deque
from typing import Optional
import threading
import asyncio
import websockets
import time
import base64
import json
import logging
import serial
import uuid
import os

import aiohttp
from aiohttp import web

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("[Buzzer] Jetson.GPIO không khả dụng — còi sẽ bị tắt.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BACKEND_HOST     = "192.168.137.1"
BACKEND_PORT     = 8000
CAMERA_ID        = "jetson-cam-01"

WS_STREAM_URL    = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/ws/stream/{CAMERA_ID}"
WS_DETECTION_URL = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/ws/detection/{CAMERA_ID}"
RECONNECT_DELAY  = 3

JETSON_HTTP_HOST  = "0.0.0.0"
JETSON_HTTP_PORT  = 8001
UPLOAD_TMP_DIR    = "/tmp/jetson_video_upload"
MAX_VIDEO_SIZE_MB = 500

WS_VIDEO_ANALYSIS_URL_TEMPLATE = (
    f"ws://{{backend_host}}:{BACKEND_PORT}/ws/video-analysis/{{job_id}}?role=producer"
)

CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
CAMERA_FPS       = 30

STREAM_FPS       = 15
JPEG_QUALITY     = 70

ENGINE_PATH      = "movinet_v2.engine"
INPUT_SIZE       = 172

INFER_INTERVAL_MS = 80

VIDEO_INFER_EVERY_N_FRAMES  = 1
VIDEO_STREAM_EVERY_N_FRAMES = 5

EMA_ALPHA    = 0.35
SPIKE_THRESH = 0.12
SPIKE_BOOST  = 1.35

THRESHOLD      = 0.50
CONFIRM_FRAMES = 2
COOLDOWN_SEC   = 0.3   # giảm từ 0.8 — model buffer MoViNet đã có memory nội bộ

THUMB_WIDTH  = 320
THUMB_HEIGHT = 180

BUZZER_PIN            = 12
VIOLENCE_BUZZER_DELAY = 5.0
BUZZER_BEEP_ON_SEC    = 0.5
BUZZER_BEEP_OFF_SEC   = 0.3

PHONE_NUMBER         = "0961521940"
PHONE_SERIAL_PORT    = "/dev/ttyUSB2"
PHONE_BAUD_RATE      = 115200
PHONE_CALL_TIMEOUT   = 30
PHONE_RETRY_DELAY    = 60
PHONE_RING_WAIT      = 5
STARTUP_MUTE_SECONDS = 10.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jetson")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# class 0 = Normal, class 1 = Violence
IMAGE_NAME  = 'serving_default_image:0'
LOGITS_NAME = 'StatefulPartitionedCall:0'

STATE_MAP = {
    'serving_default_state_block4_layer1_pool_frame_count:0': 'StatefulPartitionedCall:37',
    'serving_default_state_block1_layer1_stream_buffer:0':    'StatefulPartitionedCall:8',
    'serving_default_state_block1_layer2_stream_buffer:0':    'StatefulPartitionedCall:11',
    'serving_default_state_block3_layer1_pool_frame_count:0': 'StatefulPartitionedCall:25',
    'serving_default_state_block3_layer2_pool_buffer:0':      'StatefulPartitionedCall:27',
    'serving_default_state_block3_layer3_pool_frame_count:0': 'StatefulPartitionedCall:31',
    'serving_default_state_block4_layer0_pool_buffer:0':      'StatefulPartitionedCall:33',
    'serving_default_state_head_pool_frame_count:0':          'StatefulPartitionedCall:43',
    'serving_default_state_block3_layer0_stream_buffer:0':    'StatefulPartitionedCall:23',
    'serving_default_state_block2_layer2_stream_buffer:0':    'StatefulPartitionedCall:20',
    'serving_default_state_block2_layer1_pool_buffer:0':      'StatefulPartitionedCall:15',
    'serving_default_state_block2_layer1_stream_buffer:0':    'StatefulPartitionedCall:17',
    'serving_default_state_block3_layer1_stream_buffer:0':    'StatefulPartitionedCall:26',
    'serving_default_state_block4_layer0_pool_frame_count:0': 'StatefulPartitionedCall:34',
    'serving_default_state_block3_layer3_stream_buffer:0':    'StatefulPartitionedCall:32',
    'serving_default_state_block1_layer1_pool_buffer:0':      'StatefulPartitionedCall:6',
    'serving_default_state_block3_layer2_pool_frame_count:0': 'StatefulPartitionedCall:28',
    'serving_default_state_block0_layer0_pool_buffer:0':      'StatefulPartitionedCall:1',
    'serving_default_state_block0_layer0_pool_frame_count:0': 'StatefulPartitionedCall:2',
    'serving_default_state_block1_layer0_pool_frame_count:0': 'StatefulPartitionedCall:4',
    'serving_default_state_block3_layer0_pool_buffer:0':      'StatefulPartitionedCall:21',
    'serving_default_state_block2_layer2_pool_frame_count:0': 'StatefulPartitionedCall:19',
    'serving_default_state_block2_layer0_pool_frame_count:0': 'StatefulPartitionedCall:13',
    'serving_default_state_block4_layer2_pool_frame_count:0': 'StatefulPartitionedCall:39',
    'serving_default_state_block1_layer1_pool_frame_count:0': 'StatefulPartitionedCall:7',
    'serving_default_state_block3_layer2_stream_buffer:0':    'StatefulPartitionedCall:29',
    'serving_default_state_block4_layer0_stream_buffer:0':    'StatefulPartitionedCall:35',
    'serving_default_state_block1_layer0_stream_buffer:0':    'StatefulPartitionedCall:5',
    'serving_default_state_block3_layer1_pool_buffer:0':      'StatefulPartitionedCall:24',
    'serving_default_state_block4_layer3_pool_buffer:0':      'StatefulPartitionedCall:40',
    'serving_default_state_block1_layer0_pool_buffer:0':      'StatefulPartitionedCall:3',
    'serving_default_state_block3_layer0_pool_frame_count:0': 'StatefulPartitionedCall:22',
    'serving_default_state_block2_layer0_stream_buffer:0':    'StatefulPartitionedCall:14',
    'serving_default_state_block2_layer2_pool_buffer:0':      'StatefulPartitionedCall:18',
    'serving_default_state_head_pool_buffer:0':               'StatefulPartitionedCall:42',
    'serving_default_state_block4_layer1_pool_buffer:0':      'StatefulPartitionedCall:36',
    'serving_default_state_block1_layer2_pool_frame_count:0': 'StatefulPartitionedCall:10',
    'serving_default_state_block1_layer2_pool_buffer:0':      'StatefulPartitionedCall:9',
    'serving_default_state_block2_layer0_pool_buffer:0':      'StatefulPartitionedCall:12',
    'serving_default_state_block4_layer3_pool_frame_count:0': 'StatefulPartitionedCall:41',
    'serving_default_state_block4_layer2_pool_buffer:0':      'StatefulPartitionedCall:38',
    'serving_default_state_block3_layer3_pool_buffer:0':      'StatefulPartitionedCall:30',
    'serving_default_state_block2_layer1_pool_frame_count:0': 'StatefulPartitionedCall:16',
}


def _smooth_update(prob_smooth: float, prob_raw: float,
                   prev_prob_raw: float) -> float:
    """
    EMA + spike/drop boost đối xứng.
    - Tăng đột ngột (delta >  SPIKE_THRESH): nhân SPIKE_BOOST  → phát hiện nhanh
    - Giảm đột ngột (delta < -SPIKE_THRESH): chia SPIKE_BOOST  → hồi phục nhanh
    """
    smooth = EMA_ALPHA * prob_raw + (1.0 - EMA_ALPHA) * prob_smooth
    delta  = prob_raw - prev_prob_raw
    if delta > SPIKE_THRESH:
        smooth = min(1.0, smooth * SPIKE_BOOST)
    elif delta < -SPIKE_THRESH:
        smooth = max(0.0, smooth / SPIKE_BOOST)
    return smooth


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JOB REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JobStatus:
    QUEUED    = "queued"
    RUNNING   = "running"
    DONE      = "done"
    CANCELLED = "cancelled"
    ERROR     = "error"


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
        self.progress     = 0.0
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
        self._lock        = threading.Lock()
        self._raw_frame   = None
        self._prob_raw    = 0.0
        self._prob_smooth = 0.0
        self._label       = 'Normal'

    def update(self, raw_frame: np.ndarray, prob_raw: float,
               prob_smooth: float, label: str) -> None:
        with self._lock:
            self._raw_frame   = raw_frame.copy()
            self._prob_raw    = prob_raw
            self._prob_smooth = prob_smooth
            self._label       = label

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            if self._raw_frame is None:
                return None
            return {
                'raw_frame':   self._raw_frame.copy(),
                'prob_raw':    self._prob_raw,
                'prob_smooth': self._prob_smooth,
                'label':       self._label,
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

    def _set_gpio(self, state: bool):
        if self.available:
            GPIO.output(BUZZER_PIN, GPIO.LOW if state else GPIO.HIGH)
        elif state:
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
#  PHONE DIALER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PhoneDialer(threading.Thread):
    _STATE_IDLE    = "IDLE"
    _STATE_CALLING = "CALLING"

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

    def _open_serial(self):
        try:
            self._serial = serial.Serial(
                PHONE_SERIAL_PORT, baudrate=PHONE_BAUD_RATE, timeout=3)
            time.sleep(0.5)
            resp = self._send_at("AT")
            if "OK" in resp:
                self.available = True
                logger.info(f"[Phone] SIM module sẵn sàng — {PHONE_SERIAL_PORT}.")
            else:
                logger.warning(f"[Phone] SIM không phản hồi AT: {resp!r}")
        except Exception as e:
            logger.warning(f"[Phone] Không mở được {PHONE_SERIAL_PORT}: {e}")
            logger.warning("[Phone] Chạy ở chế độ giả lập.")

    def _send_at(self, cmd: str, wait: float = 1.0) -> str:
        if self._serial is None or not self._serial.is_open:
            return ""
        try:
            self._serial.reset_input_buffer()
            self._serial.write((cmd + "\r\n").encode())
            time.sleep(wait)
            return self._serial.read_all().decode(errors="ignore")
        except Exception as e:
            logger.error(f"[Phone] Lỗi AT '{cmd}': {e}")
            return ""

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
            if self._serial and self._serial.is_open:
                self._serial.close()

    def _do_call(self):
        logger.warning(f"[Phone] ☎  Đang quay số {PHONE_NUMBER} …")
        if self.available:
            self._send_at("ATE0")
            self._send_at(f"ATD{PHONE_NUMBER};", wait=PHONE_RING_WAIT)
        else:
            logger.warning(f"[Phone][SIM] GIẢ LẬP: ATD{PHONE_NUMBER};")
        deadline = time.time() + PHONE_CALL_TIMEOUT
        while time.time() < deadline:
            with self._cond:
                if self._cancel or not self._running:
                    break
                self._cond.wait(timeout=min(deadline - time.time(), 0.2))
        self._hangup()
        with self._lock:
            self._last_call_end = time.time()
            self._cancel        = False
            self._state         = self._STATE_IDLE

    def _hangup(self):
        if self.available:
            self._send_at("ATH", wait=0.5)
        else:
            logger.debug("[Phone][SIM] GIẢ LẬP: ATH")


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
#  TENSORRT MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MoViNetTRT:
    """
    Wrapper TensorRT cho MoViNet-A2 Stream (fp16).
    class 0 → Normal, class 1 → Violence.
    Stream state tự quản lý temporal context; reset_states() chỉ gọi
    khi khởi tạo hoặc camera reconnect.
    """

    def __init__(self, engine_path: str, cuda_ctx):
        self.cuda_ctx = cuda_ctx
        self.cuda_ctx.push()
        self._runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.hbuf    = {}
        self.dbuf    = {}
        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size  = max(1, int(np.prod(shape)))
            h = cuda.pagelocked_empty(size, dtype)
            d = cuda.mem_alloc(h.nbytes)
            self.hbuf[name] = h
            self.dbuf[name] = d
        self.reset_states()
        self.cuda_ctx.pop()
        logger.info(f"[TRT] Loaded — {self.engine.num_bindings} bindings  "
                    f"states={len(STATE_MAP)} pairs")

    def reset_states(self):
        for name in STATE_MAP:
            self.hbuf[name].fill(0)
            cuda.memcpy_htod(self.dbuf[name], self.hbuf[name])

    def partial_reset(self, strength: float = 0.5):
        """
        Giảm các stream buffer xuống còn `strength` lần giá trị hiện tại.
        Không reset hoàn toàn để tránh mất context đột ngột (gây warm-up lag).
        Chỉ reset pool_buffer (lưu temporal average) — đây là nguyên nhân chính
        khiến prob_raw giảm chậm. Stream_buffer giữ nguyên vì nó ngắn hạn hơn.
        """
        self.cuda_ctx.push()
        try:
            for name in STATE_MAP:
                if 'pool_buffer' in name:   # chỉ reset pool buffer
                    self.hbuf[name] *= strength
                    cuda.memcpy_htod(self.dbuf[name], self.hbuf[name])
        finally:
            self.cuda_ctx.pop()

    def infer(self, frame_bgr: np.ndarray) -> float:
        """Trả về prob_violence (class 1, float32 sau softmax)."""
        self.cuda_ctx.push()
        try:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            img = (frame.astype(np.float32) / 255.0).reshape(
                1, 1, INPUT_SIZE, INPUT_SIZE, 3)
            np.copyto(self.hbuf[IMAGE_NAME], img.ravel())
            cuda.memcpy_htod(self.dbuf[IMAGE_NAME], self.hbuf[IMAGE_NAME])
            for in_name in STATE_MAP:
                cuda.memcpy_htod(self.dbuf[in_name], self.hbuf[in_name])
            bindings = [
                int(self.dbuf[self.engine.get_binding_name(i)])
                for i in range(self.engine.num_bindings)
            ]
            self.context.execute_v2(bindings)
            cuda.memcpy_dtoh(self.hbuf[LOGITS_NAME], self.dbuf[LOGITS_NAME])
            logits = self.hbuf[LOGITS_NAME].copy().astype(np.float32)
            exp    = np.exp(logits - logits.max())
            probs  = exp / exp.sum()   # probs[0]=Normal, probs[1]=Violence
            for in_name, out_name in STATE_MAP.items():
                cuda.memcpy_dtoh(self.hbuf[in_name], self.dbuf[out_name])
            return float(probs[1])
        finally:
            self.cuda_ctx.pop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DETECTION RESULT  (camera mode — async event bridge)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionResult:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._lock  = threading.Lock()
        self._data  = None
        self._loop  = loop
        self._event = asyncio.Event()

    def put(self, data: dict):
        with self._lock:
            self._data = data
        self._loop.call_soon_threadsafe(self._event.set)

    async def get_new(self) -> dict:
        await self._event.wait()
        self._event.clear()
        with self._lock:
            return self._data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VIDEO ANALYSIS JOB RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_video_analysis(job: VideoJob, cuda_ctx,
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
    """
    Coroutine async: mở WebSocket → inference từng frame → stream → summary.

    Payload mỗi frame (type: "frame"):
    {
        "type":          "frame",
        "job_id":        "...",
        "frame_idx":     42,
        "frame_time_s":  1.4,
        "total_frames":  300,
        "progress":      0.14,
        "label":         "VIOLENCE" | "Normal",
        "prob_raw":      0.73,
        "prob_smooth":   0.68,
        "thumb_b64":     "<base64 jpeg>",
        "timestamp":     "14:35:22.047"
    }

    Payload cuối (type: "summary"):
    {
        "type":               "summary",
        "job_id":             "...",
        "filename":           "clip.mp4",
        "total_frames":       300,
        "frames_inferred":    300,
        "video_duration_s":   10.0,
        "violence_ratio":     0.23,
        "max_prob":           0.91,
        "avg_prob":           0.34,
        "violence_segments":  [{"start_s": 2.1, "end_s": 4.7, "max_prob": 0.91}],
        "verdict":            "VIOLENCE" | "Normal",
        "processing_time_s":  5.2
    }
    """
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

    loop  = asyncio.get_event_loop()
    model = await loop.run_in_executor(
        None, lambda: MoViNetTRT(ENGINE_PATH, cuda_ctx)
    )

    prob_smooth   = 0.0
    prev_prob_raw = 0.0
    confirm_count = 0
    violence_until = 0.0

    all_probs  = []
    segments   = []
    seg_start  = None
    seg_max_prob = 0.0
    t_start    = time.time()

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
                break

            frame_time_s = frame_idx / fps_video

            if frame_idx % VIDEO_INFER_EVERY_N_FRAMES == 0:
                prob_raw = await loop.run_in_executor(None, model.infer, frame)

                # EMA + spike/drop boost đối xứng
                prob_smooth   = _smooth_update(prob_smooth, prob_raw, prev_prob_raw)
                prev_prob_raw = prob_raw

                # Confirm + cooldown
                # Chỉ cộng dồn violence_until khi prob_raw thực sự còn cao,
                # tránh EMA quán tính kéo dài nhãn VIOLENCE.
                now_virtual = frame_time_s
                if prob_smooth >= THRESHOLD:
                    confirm_count += 1
                else:
                    confirm_count = 0
                if confirm_count >= CONFIRM_FRAMES and prob_raw >= THRESHOLD:
                    violence_until = now_virtual + COOLDOWN_SEC

                # Thoát VIOLENCE khi hết cooldown
                if now_virtual < violence_until:
                    label = 'VIOLENCE'
                else:
                    label = 'Normal'

                # Flush pool_buffer liên tục khi đang VIOLENCE nhưng prob_raw đang giảm
                # → tăng tốc độ giảm từ đầu thay vì chờ đến khi gần ngưỡng
                if label == 'VIOLENCE' and prob_raw < prev_prob_raw:
                    model.partial_reset(strength=0.7)

                all_probs.append(prob_raw)

                # Theo dõi segments bạo lực
                if label == 'VIOLENCE':
                    if seg_start is None:
                        seg_start    = frame_time_s
                        seg_max_prob = prob_smooth
                    else:
                        seg_max_prob = max(seg_max_prob, prob_smooth)
                else:
                    if seg_start is not None:
                        segments.append({
                            "start_s":  round(seg_start,    2),
                            "end_s":    round(frame_time_s, 2),
                            "max_prob": round(seg_max_prob, 3),
                        })
                        seg_start = None

                if frame_idx % VIDEO_STREAM_EVERY_N_FRAMES == 0:
                    thumb_b64     = render_and_encode_thumb(frame)
                    progress      = frame_idx / max(total_frames, 1)
                    job.processed = frame_idx
                    job.progress  = progress
                    ts_str = (time.strftime('%H:%M:%S')
                              + f'.{int((time.time() % 1) * 1000):03d}')
                    await ws.send(json.dumps({
                        "type":         "frame",
                        "job_id":       job.job_id,
                        "frame_idx":    frame_idx,
                        "frame_time_s": round(frame_time_s, 3),
                        "total_frames": total_frames,
                        "progress":     round(progress,    3),
                        "label":        label,
                        "prob_raw":     round(prob_raw,    4),
                        "prob_smooth":  round(prob_smooth, 4),
                        "thumb_b64":    thumb_b64,
                        "timestamp":    ts_str,
                    }, ensure_ascii=False))

            frame_idx += 1

        cap.release()

        # Đóng segment cuối nếu còn dở
        if seg_start is not None:
            segments.append({
                "start_s":  round(seg_start,   2),
                "end_s":    round(duration_s,  2),
                "max_prob": round(seg_max_prob, 3),
            })

        if job.status != JobStatus.CANCELLED:
            job.status   = JobStatus.DONE
            job.ended_at = time.time()
            job.progress = 1.0

            n_inferred      = len(all_probs)
            n_violence      = sum(1 for p in all_probs if p >= THRESHOLD)
            violence_ratio  = n_violence / max(n_inferred, 1)
            verdict         = 'VIOLENCE' if violence_ratio >= 0.15 else 'Normal'
            processing_time = job.ended_at - t_start

            await ws.send(json.dumps({
                "type":               "summary",
                "job_id":             job.job_id,
                "filename":           job.filename,
                "total_frames":       total_frames,
                "frames_inferred":    n_inferred,
                "video_duration_s":   round(duration_s,      2),
                "violence_ratio":     round(violence_ratio,  4),
                "max_prob":           round(max(all_probs, default=0.0), 4),
                "avg_prob":           round(sum(all_probs) / max(n_inferred, 1), 4),
                "violence_segments":  segments,
                "verdict":            verdict,
                "processing_time_s":  round(processing_time, 2),
            }, ensure_ascii=False))

            logger.info(
                f"[VideoJob:{job.job_id[:8]}] DONE — "
                f"verdict={verdict}  ratio={violence_ratio:.2%}  "
                f"segments={len(segments)}  time={processing_time:.1f}s"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CAMERA MODE — INFERENCE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_inference(cam: CameraCapture,
                  frame_event: threading.Event,
                  shared_state: SharedState,
                  result_queue: DetectionResult,
                  buzzer: BuzzerController,
                  dialer: PhoneDialer,
                  start_time: float):
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()

    logger.info("[Inference] Loading TensorRT engine …")
    model = MoViNetTRT(ENGINE_PATH, cuda_ctx)

    if not frame_event.wait(timeout=30.0):
        logger.error("[Inference] Timeout frame!")
        cuda_ctx.pop()
        cuda_ctx.detach()
        return

    logger.info("[Inference] Bắt đầu inference.")

    prob_smooth         = 0.0
    prev_prob_raw       = 0.0
    confirm_count       = 0
    violence_until      = 0.0
    violence_start_time = None
    alert_triggered     = False
    last_infer_ms       = 0.0

    cam_times   = deque(maxlen=60)
    infer_times = deque(maxlen=30)

    try:
        while True:
            frame_event.clear()
            frame = cam.get_latest()
            if frame is None:
                frame_event.wait(timeout=0.1)
                continue

            now           = time.time()
            now_ms        = now * 1000
            startup_muted = (now - start_time) < STARTUP_MUTE_SECONDS

            if (now_ms - last_infer_ms) < INFER_INTERVAL_MS:
                frame_event.wait(timeout=0.1)
                continue

            t0       = time.time()
            prob_raw = model.infer(frame)
            t1       = time.time()
            last_infer_ms = now_ms
            infer_times.append(t1)

            # EMA + spike/drop boost đối xứng
            prob_smooth   = _smooth_update(prob_smooth, prob_raw, prev_prob_raw)
            prev_prob_raw = prob_raw

            # Confirm + cooldown
            # Chỉ cộng dồn violence_until khi prob_raw thực sự còn cao,
            # tránh EMA quán tính kéo dài nhãn VIOLENCE.
            if prob_smooth >= THRESHOLD:
                confirm_count += 1
            else:
                confirm_count = 0
            if confirm_count >= CONFIRM_FRAMES and prob_raw >= THRESHOLD:
                violence_until = now + COOLDOWN_SEC

            # Thoát VIOLENCE khi hết cooldown
            if now < violence_until:
                label = 'VIOLENCE'
            else:
                label = 'Normal'

            # Flush pool_buffer liên tục khi đang VIOLENCE nhưng prob_raw đang giảm
            # → tăng tốc độ giảm từ đầu thay vì chờ đến khi gần ngưỡng
            if label == 'VIOLENCE' and prob_raw < prev_prob_raw:
                model.partial_reset(strength=0.5)

            shared_state.update(frame, prob_raw, prob_smooth, label)

            # Buzzer + Phone logic
            if label == 'VIOLENCE':
                if violence_start_time is None:
                    violence_start_time = now
                    alert_triggered     = False
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
                if violence_start_time is not None:
                    violence_start_time = None
                    alert_triggered     = False
                buzzer.deactivate()
                dialer.cancel_call()
                violence_duration = 0.0

            cam_times.append(now)
            infer_fps = (
                (len(infer_times) - 1) / (infer_times[-1] - infer_times[0])
                if len(infer_times) > 1 else 0.0
            )
            cam_fps = (
                (len(cam_times) - 1) / (cam_times[-1] - cam_times[0])
                if len(cam_times) > 1 else 0.0
            )

            ts_str = time.strftime('%H:%M:%S') + f'.{int((now % 1) * 1000):03d}'

            result_queue.put({
                "camera_id":         CAMERA_ID,
                "timestamp":         ts_str,
                "label":             label,
                "prob_raw":          round(prob_raw,          4),
                "prob_smooth":       round(prob_smooth,        4),
                "confirm_count":     confirm_count,
                "confirm_needed":    CONFIRM_FRAMES,
                "threshold":         THRESHOLD,
                "infer_fps":         round(infer_fps,         1),
                "cam_fps":           round(cam_fps,           1),
                "uptime":            int(now - start_time),
                "infer_ms":          round((t1 - t0) * 1000,  1),
                "violence_duration": round(violence_duration,  2),
                "buzzer_active":     buzzer.is_active,
                "phone_calling":     dialer.is_calling,
            })

            logger.info(
                f"raw={prob_raw:.3f}  smooth={prob_smooth:.3f}  "
                f"confirm={confirm_count}/{CONFIRM_FRAMES}  → {label}  "
                f"dur={violence_duration:.1f}s  "
                f"buzzer={'ON' if buzzer.is_active else 'off'}  "
                f"phone={'CALLING' if dialer.is_calling else 'idle'}  "
                f"({(t1 - t0) * 1000:.0f}ms)"
            )

            frame_event.wait(timeout=0.1)

    except Exception as e:
        logger.exception(f"[Inference] Lỗi: {e}")
    finally:
        buzzer.deactivate()
        dialer.cancel_call()
        cuda_ctx.pop()
        cuda_ctx.detach()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC — STREAM VIDEO  (camera mode)
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
                    await asyncio.sleep(max(0.0, interval - (time.time() - t0)))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"[Stream] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …")
        except Exception as e:
            logger.error(f"[Stream] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …")
        await asyncio.sleep(RECONNECT_DELAY)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC — STREAM DETECTION  (camera mode)
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
                    else:
                        payload['thumb_b64'] = ''
                    await ws.send(json.dumps(payload, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"[Detection] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …")
        except Exception as e:
            logger.error(f"[Detection] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …")
        await asyncio.sleep(RECONNECT_DELAY)


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
            return web.json_response(
                {"error": "Cần gửi multipart/form-data"}, status=400)

        video_path   = None
        filename     = "video.mp4"
        backend_host = BACKEND_HOST

        async for part in reader:
            if part.name == "video":
                filename = part.filename or "video.mp4"
                ext      = os.path.splitext(filename)[1] or ".mp4"
                tmp_path = os.path.join(UPLOAD_TMP_DIR, f"{uuid.uuid4().hex}{ext}")
                size     = 0
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
                                status=413)
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

        job = job_registry.get(job_id) if job_id else job_registry.current_running()
        if job is None:
            return web.json_response({"error": "Không tìm thấy job"}, status=404)
        job.cancel()
        return web.json_response({"cancelled": True, "job_id": job.job_id})

    app = web.Application(
        client_max_size=MAX_VIDEO_SIZE_MB * 1024 * 1024 + 1024)
    app.router.add_post("/analyze-video", handle_analyze_video)
    app.router.add_get("/status",         handle_status)
    app.router.add_post("/cancel",        handle_cancel)
    return app


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    start_time = time.time()
    loop       = asyncio.get_event_loop()

    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    cuda_ctx.pop()

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
    for _ in range(300):
        if cam.connected or cam.error_msg:
            break
        await asyncio.sleep(0.1)

    if not cam.connected:
        logger.error(f"[Main] Không thể kết nối camera: {cam.error_msg}")
        buzzer.stop()
        dialer.stop()
        cuda_ctx.detach()
        return

    logger.info(f"[Main] Camera sẵn sàng — index={cam.camera_id}")

    result_queue = DetectionResult(loop)

    threading.Thread(
        target=run_inference,
        args=(cam, frame_event, shared_state, result_queue,
              buzzer, dialer, start_time),
        daemon=True,
        name="InferenceThread",
    ).start()

    http_app    = build_http_app(job_registry, loop, cuda_ctx)
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site   = web.TCPSite(http_runner, JETSON_HTTP_HOST, JETSON_HTTP_PORT)
    await http_site.start()

    print("=" * 60)
    print("  Jetson Violence Detection Client  [v5]")
    print(f"  Camera index   : {cam.camera_id}")
    print(f"  Stream URL     : {WS_STREAM_URL}")
    print(f"  Detection URL  : {WS_DETECTION_URL}")
    print(f"  ── Video Analysis HTTP Server ─────────────")
    print(f"  Listen         : http://{JETSON_HTTP_HOST}:{JETSON_HTTP_PORT}")
    print(f"  POST           : /analyze-video  (multipart, field: 'video')")
    print(f"  GET            : /status")
    print(f"  POST           : /cancel")
    print(f"  Max upload     : {MAX_VIDEO_SIZE_MB}MB")
    print(f"  ── Inference ──────────────────────────────")
    print(f"  EMA alpha      : {EMA_ALPHA}  spike/drop boost x{SPIKE_BOOST}")
    print(f"  Threshold      : {THRESHOLD}  confirm {CONFIRM_FRAMES}f")
    print(f"  Cooldown       : {COOLDOWN_SEC}s")
    print(f"  ── Buzzer ─────────────────────────────────")
    print(f"  GPIO PIN       : {BUZZER_PIN}  delay {VIOLENCE_BUZZER_DELAY}s")
    print(f"  ── Phone ──────────────────────────────────")
    print(f"  Số             : {PHONE_NUMBER}  port {PHONE_SERIAL_PORT}")
    print(f"  SIM available  : {dialer.available}")
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