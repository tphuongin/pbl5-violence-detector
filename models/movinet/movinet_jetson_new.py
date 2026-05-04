"""
Bản mới

Luồng hoạt động:
  CameraCapture (thread) → frame_event
      ├── stream_video    (async) → ws://.../ws/stream/{CAMERA_ID}
      └── run_inference   (thread) → SharedState + DetectionResult
              └── stream_detection (async) → ws://.../ws/detection/{CAMERA_ID}
                      └── BuzzerController (thread) → GPIO PIN 12
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

CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
CAMERA_FPS       = 30

STREAM_FPS       = 15
JPEG_QUALITY     = 70

ENGINE_PATH      = "movinet_v2.engine"
INPUT_SIZE       = 172

INFER_INTERVAL_MS = 80

EMA_ALPHA        = 0.35
SPIKE_THRESH     = 0.12
SPIKE_BOOST      = 1.35

THRESHOLD        = 0.50
CONFIRM_FRAMES   = 2
COOLDOWN_SEC     = 0.8

THUMB_WIDTH      = 120
THUMB_HEIGHT     = 68

BUZZER_PIN             = 12
VIOLENCE_BUZZER_DELAY  = 5.0
BUZZER_BEEP_ON_SEC     = 0.5
BUZZER_BEEP_OFF_SEC    = 0.3

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
    'serving_default_state_block1_layer1_stream_buffer:0': 'StatefulPartitionedCall:8',
    'serving_default_state_block1_layer2_stream_buffer:0': 'StatefulPartitionedCall:11',
    'serving_default_state_block3_layer1_pool_frame_count:0': 'StatefulPartitionedCall:25',
    'serving_default_state_block3_layer2_pool_buffer:0': 'StatefulPartitionedCall:27',
    'serving_default_state_block3_layer3_pool_frame_count:0': 'StatefulPartitionedCall:31',
    'serving_default_state_block4_layer0_pool_buffer:0': 'StatefulPartitionedCall:33',
    'serving_default_state_head_pool_frame_count:0': 'StatefulPartitionedCall:43',
    'serving_default_state_block3_layer0_stream_buffer:0': 'StatefulPartitionedCall:23',
    'serving_default_state_block2_layer2_stream_buffer:0': 'StatefulPartitionedCall:20',
    'serving_default_state_block2_layer1_pool_buffer:0': 'StatefulPartitionedCall:15',
    'serving_default_state_block2_layer1_stream_buffer:0': 'StatefulPartitionedCall:17',
    'serving_default_state_block3_layer1_stream_buffer:0': 'StatefulPartitionedCall:26',
    'serving_default_state_block4_layer0_pool_frame_count:0': 'StatefulPartitionedCall:34',
    'serving_default_state_block3_layer3_stream_buffer:0': 'StatefulPartitionedCall:32',
    'serving_default_state_block1_layer1_pool_buffer:0': 'StatefulPartitionedCall:6',
    'serving_default_state_block3_layer2_pool_frame_count:0': 'StatefulPartitionedCall:28',
    'serving_default_state_block0_layer0_pool_buffer:0': 'StatefulPartitionedCall:1',
    'serving_default_state_block0_layer0_pool_frame_count:0': 'StatefulPartitionedCall:2',
    'serving_default_state_block1_layer0_pool_frame_count:0': 'StatefulPartitionedCall:4',
    'serving_default_state_block3_layer0_pool_buffer:0': 'StatefulPartitionedCall:21',
    'serving_default_state_block2_layer2_pool_frame_count:0': 'StatefulPartitionedCall:19',
    'serving_default_state_block2_layer0_pool_frame_count:0': 'StatefulPartitionedCall:13',
    'serving_default_state_block4_layer2_pool_frame_count:0': 'StatefulPartitionedCall:39',
    'serving_default_state_block1_layer1_pool_frame_count:0': 'StatefulPartitionedCall:7',
    'serving_default_state_block3_layer2_stream_buffer:0': 'StatefulPartitionedCall:29',
    'serving_default_state_block4_layer0_stream_buffer:0': 'StatefulPartitionedCall:35',
    'serving_default_state_block1_layer0_stream_buffer:0': 'StatefulPartitionedCall:5',
    'serving_default_state_block3_layer1_pool_buffer:0': 'StatefulPartitionedCall:24',
    'serving_default_state_block4_layer3_pool_buffer:0': 'StatefulPartitionedCall:40',
    'serving_default_state_block1_layer0_pool_buffer:0': 'StatefulPartitionedCall:3',
    'serving_default_state_block3_layer0_pool_frame_count:0': 'StatefulPartitionedCall:22',
    'serving_default_state_block2_layer0_stream_buffer:0': 'StatefulPartitionedCall:14',
    'serving_default_state_block2_layer2_pool_buffer:0': 'StatefulPartitionedCall:18',
    'serving_default_state_head_pool_buffer:0': 'StatefulPartitionedCall:42',
    'serving_default_state_block4_layer1_pool_buffer:0': 'StatefulPartitionedCall:36',
    'serving_default_state_block1_layer2_pool_frame_count:0': 'StatefulPartitionedCall:10',
    'serving_default_state_block1_layer2_pool_buffer:0': 'StatefulPartitionedCall:9',
    'serving_default_state_block2_layer0_pool_buffer:0': 'StatefulPartitionedCall:12',
    'serving_default_state_block4_layer3_pool_frame_count:0': 'StatefulPartitionedCall:41',
    'serving_default_state_block4_layer2_pool_buffer:0': 'StatefulPartitionedCall:38',
    'serving_default_state_block3_layer3_pool_buffer:0': 'StatefulPartitionedCall:30',
    'serving_default_state_block2_layer1_pool_frame_count:0': 'StatefulPartitionedCall:16'
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE
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
        self._active  = False
        self._running = True
        self._lock    = threading.Lock()
        self._cond    = threading.Condition(self._lock)
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
        logger.info(
            f"[Camera] OK — index={self.camera_id}  "
            f"{int(cap.get(3))}x{int(cap.get(4))}  {cap.get(5):.0f}fps"
        )

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
        logger.info("[Camera] Đã đóng camera.")

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
    - class 0 → Normal, class 1 → Violence
    - Stream state tự quản lý temporal context; reset_states() chỉ gọi
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

        logger.info(
            f"[TRT] Loaded — {self.engine.num_bindings} bindings  "
            f"states={len(STATE_MAP)} pairs"
        )

    def reset_states(self):
        for name in STATE_MAP:
            self.hbuf[name].fill(0)
            cuda.memcpy_htod(self.dbuf[name], self.hbuf[name])

    def infer(self, frame_bgr: np.ndarray) -> float:
        """
        Chạy inference 1 frame.
        Trả về prob_violence (class 1, float32 sau softmax).
        """
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
            probs  = exp / exp.sum()  # probs[0]=Normal, probs[1]=Violence

            for in_name, out_name in STATE_MAP.items():
                cuda.memcpy_dtoh(self.hbuf[in_name], self.dbuf[out_name])

            return float(probs[1])
        finally:
            self.cuda_ctx.pop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DETECTION RESULT  (async event bridge)
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
#  INFERENCE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_inference(cam: CameraCapture,
                  frame_event: threading.Event,
                  shared_state: SharedState,
                  result_queue: DetectionResult,
                  buzzer: BuzzerController,
                  start_time: float):
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()

    logger.info("[Inference] Loading TensorRT engine …")
    model = MoViNetTRT(ENGINE_PATH, cuda_ctx)

    logger.info("[Inference] Chờ frame đầu tiên …")
    if not frame_event.wait(timeout=30.0):
        logger.error("[Inference] Timeout: không nhận được frame sau 30s!")
        cuda_ctx.pop()
        cuda_ctx.detach()
        return

    logger.info("[Inference] Đã nhận frame — bắt đầu inference.")

    prob_smooth         = 0.0
    prev_prob_raw       = 0.0
    confirm_count       = 0
    violence_until      = 0.0
    violence_start_time = None
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

            now    = time.time()
            now_ms = now * 1000

            if (now_ms - last_infer_ms) < INFER_INTERVAL_MS:
                frame_event.wait(timeout=0.1)
                continue

            t0        = time.time()
            prob_raw  = model.infer(frame)
            t1        = time.time()
            last_infer_ms = now_ms
            infer_times.append(t1)

            # EMA smoothing
            prob_smooth = EMA_ALPHA * prob_raw + (1.0 - EMA_ALPHA) * prob_smooth

            # Spike boost
            if (prob_raw - prev_prob_raw) > SPIKE_THRESH:
                prob_smooth = min(1.0, prob_smooth * SPIKE_BOOST)
            prev_prob_raw = prob_raw

            # Confirm + cooldown
            if prob_smooth >= THRESHOLD:
                confirm_count += 1
            else:
                confirm_count = 0

            if confirm_count >= CONFIRM_FRAMES:
                violence_until = now + COOLDOWN_SEC

            label = 'VIOLENCE' if now < violence_until else 'Normal'

            shared_state.update(frame, prob_raw, prob_smooth, label)

            # Buzzer logic
            if label == 'VIOLENCE':
                if violence_start_time is None:
                    violence_start_time = now
                    logger.info("[Buzzer] Bắt đầu theo dõi thời gian bạo lực …")
                violence_duration = now - violence_start_time
                if violence_duration >= VIOLENCE_BUZZER_DELAY:
                    buzzer.activate()
                else:
                    logger.debug(
                        f"[Buzzer] {violence_duration:.1f}s "
                        f"/ {VIOLENCE_BUZZER_DELAY}s"
                    )
            else:
                if violence_start_time is not None:
                    logger.info(
                        f"[Buzzer] Kết thúc bạo lực sau "
                        f"{now - violence_start_time:.1f}s — reset."
                    )
                    violence_start_time = None
                buzzer.deactivate()
                violence_duration = 0.0

            # FPS
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

            payload = {
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
            }

            result_queue.put(payload)

            logger.info(
                f"raw={prob_raw:.3f}  smooth={prob_smooth:.3f}  "
                f"confirm={confirm_count}/{CONFIRM_FRAMES}  → {label}  "
                f"dur={violence_duration:.1f}s  "
                f"buzzer={'ON' if buzzer.is_active else 'off'}  "
                f"({(t1 - t0) * 1000:.0f}ms)"
            )

            frame_event.wait(timeout=0.1)

    except Exception as e:
        logger.exception(f"[Inference] Lỗi: {e}")
    finally:
        buzzer.deactivate()
        cuda_ctx.pop()
        cuda_ctx.detach()


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
                            None, encode_frame_jpeg, frame
                        )
                        await ws.send(jpeg_bytes)
                    await asyncio.sleep(max(0.0, interval - (time.time() - t0)))
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
                            None, render_and_encode_thumb, snap['raw_frame']
                        )
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
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    start_time = time.time()
    loop       = asyncio.get_event_loop()

    buzzer = BuzzerController()
    buzzer.start()

    frame_event  = threading.Event()
    shared_state = SharedState()

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
        return

    logger.info(f"[Main] Camera sẵn sàng — index={cam.camera_id}")

    result_queue = DetectionResult(loop)

    infer_thread = threading.Thread(
        target=run_inference,
        args=(cam, frame_event, shared_state, result_queue, buzzer, start_time),
        daemon=True,
        name="InferenceThread",
    )
    infer_thread.start()
    logger.info("[Main] Inference thread đã khởi động.")

    print("=" * 60)
    print("  Jetson Violence Detection Client  [v4]")
    print(f"  Camera index   : {cam.camera_id}")
    print(f"  Stream URL     : {WS_STREAM_URL}")
    print(f"  Detection URL  : {WS_DETECTION_URL}")
    print(f"  Stream FPS     : {STREAM_FPS}")
    print(f"  Infer rate     : mỗi {INFER_INTERVAL_MS}ms (~{1000 // INFER_INTERVAL_MS}/s)")
    print(f"  EMA alpha      : {EMA_ALPHA}")
    print(f"  Spike boost    : x{SPIKE_BOOST} khi delta > {SPIKE_THRESH}")
    print(f"  Threshold      : {THRESHOLD}")
    print(f"  Confirm        : {CONFIRM_FRAMES} frames liên tiếp")
    print(f"  Cooldown       : {COOLDOWN_SEC}s")
    print(f"  GPIO PIN       : {BUZZER_PIN}  (BOARD)")
    print(f"  Buzzer delay   : {VIOLENCE_BUZZER_DELAY}s bạo lực liên tục")
    print(f"  Beep ON / OFF  : {BUZZER_BEEP_ON_SEC}s / {BUZZER_BEEP_OFF_SEC}s")
    print(f"  GPIO available : {GPIO_AVAILABLE}")
    print("=" * 60)

    try:
        await asyncio.gather(
            stream_video(cam, loop),
            stream_detection(result_queue, shared_state, loop),
        )
    finally:
        cam.stop()
        buzzer.stop()
        logger.info("[Main] Đã dừng.")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("[Main] Dừng chương trình.")
    finally:
        loop.close()