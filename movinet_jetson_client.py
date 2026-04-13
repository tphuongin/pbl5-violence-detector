"""
movinet_jetson_client_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cập nhật từ v1: thêm cảnh báo còi (Buzzer) khi phát hiện
bạo lực liên tục quá VIOLENCE_BUZZER_DELAY giây (mặc định 5s).

Luồng hoạt động:
  ┌─────────────────────────────────────┐
  │  CameraCapture (thread)             │  ← grab frame liên tục
  └──────────────┬──────────────────────┘
                 │ frame
        ┌────────┴────────┐
        ▼                 ▼
  VideoStreamer      InferenceWorker (thread)
  (async task)         TensorRT MoViNet
        │                 │
        │ JPEG binary      │ JSON result
        ▼                 ▼
  ws://.../ws/stream/   ws://.../ws/detection/
  {CAMERA_ID}           {CAMERA_ID}
                         │
                         ▼
                   BuzzerController (thread)
                   GPIO PIN 12 — kêu khi bạo lực ≥ 5s

JSON detection payload (gửi mỗi lần infer):
{
  "camera_id":         "jetson-cam-01",
  "timestamp":         "14:35:22.047",
  "label":             "VIOLENCE" | "Normal",
  "prob_raw":          0.73,
  "prob_smooth":       0.61,
  "confirm_count":     2,
  "infer_fps":         12.3,
  "cam_fps":           29.8,
  "uptime":            142,
  "window_countdown":  1.3,
  "violence_duration": 3.7,   // giây bạo lực liên tục (0 nếu Normal)
  "buzzer_active":     false,  // còi đang kêu hay không
  "thumb_b64":         "<base64 jpeg 120x68>"
}

Chạy: python3 movinet_jetson_client_v2.py
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import threading
import asyncio
import websockets
import time
import base64
import json
import logging
from collections import deque

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("[Buzzer] Jetson.GPIO không khả dụng — còi sẽ bị tắt.")

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

# ── Camera ───────────────────────────────────────────────
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
CAMERA_FPS        = 30

# ── Stream video ─────────────────────────────────────────
STREAM_FPS        = 15      # FPS gửi lên backend
JPEG_QUALITY      = 70

# ── TensorRT model ───────────────────────────────────────
ENGINE_PATH       = "movinet_stream.engine"
INPUT_SIZE        = 172

# ── Inference timing ─────────────────────────────────────
INFER_INTERVAL_MS = 80      # ~12 lần/giây

# ── Sliding window ───────────────────────────────────────
WINDOW_SEC        = 2.0

# ── EMA smoothing ────────────────────────────────────────
EMA_ALPHA         = 0.35

# ── Peak boost ───────────────────────────────────────────
SPIKE_THRESH      = 0.12
SPIKE_BOOST       = 1.35

# ── Decision ─────────────────────────────────────────────
THRESHOLD         = 0.50
CONFIRM_FRAMES    = 2
COOLDOWN_SEC      = 0.8

# ── Thumbnail gửi kèm payload ────────────────────────────
THUMB_WIDTH       = 120
THUMB_HEIGHT      = 68

# ── Buzzer (GPIO) ─────────────────────────────────────────
BUZZER_PIN              = 12    # Chân vật lý BOARD
VIOLENCE_BUZZER_DELAY   = 5.0   # Giây bạo lực liên tục trước khi kêu còi
BUZZER_BEEP_ON_SEC      = 0.5   # Thời gian còi kêu mỗi tiếng beep
BUZZER_BEEP_OFF_SEC     = 0.3   # Khoảng nghỉ giữa các tiếng beep

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jetson")

# ── STATE_MAP (MoViNet A2 streaming states) ───────────────
STATE_MAP = {
    'call_state_block4_layer1_pool_frame_count:0': 'StatefulPartitionedCall:37',
    'call_state_block1_layer1_stream_buffer:0':    'StatefulPartitionedCall:8',
    'call_state_block1_layer2_stream_buffer:0':    'StatefulPartitionedCall:11',
    'call_state_block3_layer1_pool_frame_count:0': 'StatefulPartitionedCall:25',
    'call_state_block3_layer2_pool_buffer:0':      'StatefulPartitionedCall:27',
    'call_state_block3_layer3_pool_frame_count:0': 'StatefulPartitionedCall:31',
    'call_state_block4_layer0_pool_buffer:0':      'StatefulPartitionedCall:33',
    'call_state_head_pool_frame_count:0':          'StatefulPartitionedCall:43',
    'call_state_block3_layer0_stream_buffer:0':    'StatefulPartitionedCall:23',
    'call_state_block2_layer2_stream_buffer:0':    'StatefulPartitionedCall:20',
    'call_state_block2_layer1_pool_buffer:0':      'StatefulPartitionedCall:15',
    'call_state_block2_layer1_stream_buffer:0':    'StatefulPartitionedCall:17',
    'call_state_block3_layer1_stream_buffer:0':    'StatefulPartitionedCall:26',
    'call_state_block4_layer0_pool_frame_count:0': 'StatefulPartitionedCall:34',
    'call_state_block3_layer3_stream_buffer:0':    'StatefulPartitionedCall:32',
    'call_state_block1_layer1_pool_buffer:0':      'StatefulPartitionedCall:6',
    'call_state_block3_layer2_pool_frame_count:0': 'StatefulPartitionedCall:28',
    'call_state_block0_layer0_pool_buffer:0':      'StatefulPartitionedCall:1',
    'call_state_block0_layer0_pool_frame_count:0': 'StatefulPartitionedCall:2',
    'call_state_block1_layer0_pool_frame_count:0': 'StatefulPartitionedCall:4',
    'call_state_block3_layer0_pool_buffer:0':      'StatefulPartitionedCall:21',
    'call_state_block2_layer2_pool_frame_count:0': 'StatefulPartitionedCall:19',
    'call_state_block2_layer0_pool_frame_count:0': 'StatefulPartitionedCall:13',
    'call_state_block4_layer2_pool_frame_count:0': 'StatefulPartitionedCall:39',
    'call_state_block1_layer1_pool_frame_count:0': 'StatefulPartitionedCall:7',
    'call_state_block3_layer2_stream_buffer:0':    'StatefulPartitionedCall:29',
    'call_state_block4_layer0_stream_buffer:0':    'StatefulPartitionedCall:35',
    'call_state_block1_layer0_stream_buffer:0':    'StatefulPartitionedCall:5',
    'call_state_block3_layer1_pool_buffer:0':      'StatefulPartitionedCall:24',
    'call_state_block4_layer3_pool_buffer:0':      'StatefulPartitionedCall:40',
    'call_state_block1_layer0_pool_buffer:0':      'StatefulPartitionedCall:3',
    'call_state_block3_layer0_pool_frame_count:0': 'StatefulPartitionedCall:22',
    'call_state_block2_layer0_stream_buffer:0':    'StatefulPartitionedCall:14',
    'call_state_block2_layer2_pool_buffer:0':      'StatefulPartitionedCall:18',
    'call_state_head_pool_buffer:0':               'StatefulPartitionedCall:42',
    'call_state_block4_layer1_pool_buffer:0':      'StatefulPartitionedCall:36',
    'call_state_block1_layer2_pool_frame_count:0': 'StatefulPartitionedCall:10',
    'call_state_block1_layer2_pool_buffer:0':      'StatefulPartitionedCall:9',
    'call_state_block2_layer0_pool_buffer:0':      'StatefulPartitionedCall:12',
    'call_state_block4_layer3_pool_frame_count:0': 'StatefulPartitionedCall:41',
    'call_state_block4_layer2_pool_buffer:0':      'StatefulPartitionedCall:38',
    'call_state_block3_layer3_pool_buffer:0':      'StatefulPartitionedCall:30',
    'call_state_block2_layer1_pool_frame_count:0': 'StatefulPartitionedCall:16',
}
LOGITS_NAME = 'StatefulPartitionedCall:0'
IMAGE_NAME  = 'call_image:0'
TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUZZER CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BuzzerController(threading.Thread):
    """
    Thread quản lý còi GPIO độc lập.

    Cách dùng:
        buzzer = BuzzerController()
        buzzer.start()
        buzzer.activate()    # bắt đầu kêu (beep lặp lại)
        buzzer.deactivate()  # tắt còi
        buzzer.stop()        # dừng thread + cleanup GPIO
    """

    def __init__(self):
        super().__init__(daemon=True, name="BuzzerThread")
        self._active   = False          # còi có được yêu cầu kêu không
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

    # ── public API ───────────────────────────────────────

    def activate(self):
        """Yêu cầu còi bắt đầu kêu (gọi được từ bất kỳ thread nào)."""
        with self._cond:
            if not self._active:
                self._active = True
                self._cond.notify_all()
                logger.warning("[Buzzer] ⚠  KÍCH HOẠT CÒI CẢNH BÁO!")

    def deactivate(self):
        """Tắt còi (gọi được từ bất kỳ thread nào)."""
        with self._cond:
            if self._active:
                self._active = False
                self._cond.notify_all()
                logger.info("[Buzzer] Tắt còi.")

    def stop(self):
        """Dừng thread và cleanup GPIO."""
        with self._cond:
            self._running = False
            self._active  = False
            self._cond.notify_all()

    @property
    def is_active(self):
        with self._lock:
            return self._active

    # ── thread body ──────────────────────────────────────

    def run(self):
        try:
            while True:
                # Chờ cho đến khi được kích hoạt hoặc dừng
                with self._cond:
                    while not self._active and self._running:
                        self._cond.wait(timeout=1.0)
                    if not self._running:
                        break

                # Beep một tiếng, sau đó kiểm tra lại trạng thái
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

    # ── helpers ──────────────────────────────────────────

    def _set_gpio(self, state: bool):
        if self.available:
            GPIO.output(BUZZER_PIN, GPIO.LOW if state else GPIO.HIGH)
        else:
            # Giả lập — in log để debug khi không có phần cứng
            if state:
                logger.debug("[Buzzer][SIM] BEEP ON")

    def _interruptible_sleep(self, seconds: float):
        """Ngủ nhưng có thể bị ngắt sớm khi trạng thái thay đổi."""
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
    def __init__(self):
        super().__init__(daemon=True)
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        self.connected = False
        self.error_msg = None
        self.camera_id = None

    def _try_open(self, idx):
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
        for idx in [0, 1, 2, 3, 4]:
            logger.info(f"[Camera] Thử index {idx} …")
            cap = self._try_open(idx)
            if cap is not None:
                self.camera_id = idx
                break

        if cap is None:
            self.error_msg = "Không mở được camera nào (đã thử 0,1,2,3,4)"
            logger.error(f"[Camera] {self.error_msg}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,   CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,           CAMERA_FPS)
        self.connected = True
        logger.info(f"[Camera] OK — index={self.camera_id}  "
                    f"{int(cap.get(3))}x{int(cap.get(4))}  {cap.get(5):.0f}fps")

        while self._running:
            if not cap.grab():
                time.sleep(0.02)
                continue
            ret, frame = cap.retrieve()
            if ret:
                with self._lock:
                    self._frame = frame

        cap.release()

    def get_latest(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TENSORRT MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MoViNetTRT:
    def __init__(self, engine_path, cuda_ctx):
        self.cuda_ctx = cuda_ctx
        self.cuda_ctx.push()
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.hbuf, self.dbuf = {}, {}
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
        logger.info(f"[TRT] Loaded — {self.engine.num_bindings} bindings")

    def reset_states(self):
        for name in STATE_MAP:
            self.hbuf[name].fill(0)
            cuda.memcpy_htod(self.dbuf[name], self.hbuf[name])

    def infer(self, frame_bgr):
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
            logits = self.hbuf[LOGITS_NAME].copy()
            exp    = np.exp(logits - logits.max())
            probs  = exp / exp.sum()
            for in_name, out_name in STATE_MAP.items():
                cuda.memcpy_dtoh(self.hbuf[in_name], self.dbuf[out_name])
            return float(probs[0]), float(probs[1])
        finally:
            self.cuda_ctx.pop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionResult:
    """
    Thread-safe container cho kết quả detection mới nhất.
    asyncio.Event được tạo lazy để đảm bảo nằm trong đúng event loop.
    """
    def __init__(self, loop):
        self._lock   = threading.Lock()
        self._data   = None
        self._loop   = loop
        self._event  = None

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
#  INFERENCE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_inference(cam: CameraCapture,
                  result_queue: DetectionResult,
                  buzzer: BuzzerController,
                  start_time: float):
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()

    logger.info("[Inference] Loading TensorRT engine …")
    model = MoViNetTRT(ENGINE_PATH, cuda_ctx)

    logger.info("[Inference] Chờ frame đầu tiên …")
    for _ in range(300):
        if cam.get_latest() is not None:
            break
        time.sleep(0.1)
    else:
        logger.error("[Inference] Timeout: không nhận được frame sau 30s!")
        cuda_ctx.pop()
        cuda_ctx.detach()
        return

    logger.info("[Inference] Đã nhận frame — bắt đầu inference.")

    prob_smooth    = 0.0
    prev_prob_raw  = 0.0
    confirm_count  = 0
    violence_until = 0.0
    window_start   = time.time()

    # ── Biến theo dõi thời gian bạo lực liên tục ─────────
    violence_start_time = None   # None = không có bạo lực hiện tại

    cam_times   = deque(maxlen=60)
    infer_times = deque(maxlen=30)
    last_infer_ms = 0.0

    try:
        while True:
            frame = cam.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            now    = time.time()
            now_ms = now * 1000

            # ── Sliding window reset ──────────────────────────────
            elapsed_window = now - window_start
            if elapsed_window >= WINDOW_SEC:
                model.cuda_ctx.push()
                model.reset_states()
                model.cuda_ctx.pop()
                prob_smooth   = 0.0
                prev_prob_raw = 0.0
                confirm_count = 0
                window_start  = now
                elapsed_window = 0.0
                logger.info(f"[Window] Reset @ {time.strftime('%H:%M:%S')}")

            countdown = WINDOW_SEC - elapsed_window

            # ── Time-based inference ──────────────────────────────
            if (now_ms - last_infer_ms) >= INFER_INTERVAL_MS:
                t0 = time.time()
                prob_raw, _ = model.infer(frame)
                t1 = time.time()
                last_infer_ms = now_ms
                infer_times.append(t1)

                # EMA smoothing
                prob_smooth = EMA_ALPHA * prob_raw + (1.0 - EMA_ALPHA) * prob_smooth

                # Spike boost
                delta = prob_raw - prev_prob_raw
                if delta > SPIKE_THRESH:
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

                # ── Buzzer logic ──────────────────────────────────
                # Bắt đầu đếm thời gian khi nhãn là VIOLENCE
                if label == 'VIOLENCE':
                    if violence_start_time is None:
                        violence_start_time = now
                        logger.info("[Buzzer] Bắt đầu theo dõi thời gian bạo lực …")

                    violence_duration = now - violence_start_time

                    # Kích hoạt còi nếu bạo lực liên tục ≥ ngưỡng
                    if violence_duration >= VIOLENCE_BUZZER_DELAY:
                        buzzer.activate()
                    else:
                        remaining_to_alarm = VIOLENCE_BUZZER_DELAY - violence_duration
                        logger.debug(
                            f"[Buzzer] Bạo lực {violence_duration:.1f}s "
                            f"/ {VIOLENCE_BUZZER_DELAY}s "
                            f"(còn {remaining_to_alarm:.1f}s nữa)"
                        )
                else:
                    # Về Normal → reset đếm + tắt còi
                    if violence_start_time is not None:
                        elapsed = now - violence_start_time
                        logger.info(
                            f"[Buzzer] Kết thúc bạo lực sau {elapsed:.1f}s — reset."
                        )
                        violence_start_time = None
                    buzzer.deactivate()
                    violence_duration = 0.0

                # FPS
                infer_fps = (len(infer_times) - 1) / (infer_times[-1] - infer_times[0]) \
                            if len(infer_times) > 1 else 0.0
                cam_times.append(now)
                cam_fps = (len(cam_times) - 1) / (cam_times[-1] - cam_times[0]) \
                          if len(cam_times) > 1 else 0.0

                # Thumbnail
                thumb = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                thumb_b64 = base64.b64encode(buf).decode('utf-8')

                ts_str = time.strftime('%H:%M:%S') + f'.{int((now % 1)*1000):03d}'

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
                    "window_countdown":  round(countdown,         2),
                    "window_sec":        WINDOW_SEC,
                    "infer_ms":          round((t1 - t0) * 1000,  1),
                    "violence_duration": round(violence_duration,  2),
                    "buzzer_active":     buzzer.is_active,
                    "thumb_b64":         thumb_b64,
                }

                result_queue.put(payload)

                logger.info(
                    f"raw={prob_raw:.3f}  smooth={prob_smooth:.3f}  "
                    f"confirm={confirm_count}/{CONFIRM_FRAMES}  → {label}  "
                    f"dur={violence_duration:.1f}s  "
                    f"buzzer={'ON' if buzzer.is_active else 'off'}  "
                    f"({(t1-t0)*1000:.0f}ms)"
                )

            time.sleep(0.005)

    except Exception as e:
        logger.exception(f"[Inference] Lỗi: {e}")
    finally:
        buzzer.deactivate()     # Đảm bảo tắt còi khi inference dừng
        cuda_ctx.pop()
        cuda_ctx.detach()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC TASK 1 — STREAM VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def stream_video(cam: CameraCapture):
    """Gửi JPEG frame liên tục lên ws://.../ws/stream/{CAMERA_ID}"""
    interval = 1.0 / STREAM_FPS
    while True:
        try:
            async with websockets.connect(WS_STREAM_URL) as ws:
                logger.info(f"[Stream] Kết nối OK → {WS_STREAM_URL}")
                while True:
                    t0    = time.time()
                    frame = cam.get_latest()
                    if frame is not None:
                        _, buf = cv2.imencode(
                            '.jpg', frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                        )
                        await ws.send(buf.tobytes())
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

async def stream_detection(result_queue: DetectionResult):
    """Gửi JSON detection result lên ws://.../ws/detection/{CAMERA_ID}"""
    while True:
        try:
            async with websockets.connect(WS_DETECTION_URL) as ws:
                logger.info(f"[Detection] Kết nối OK → {WS_DETECTION_URL}")
                while True:
                    payload = await result_queue.get_new()
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

    # ── Khởi động Buzzer ──────────────────────────────────
    buzzer = BuzzerController()
    buzzer.start()

    # ── Khởi động camera ─────────────────────────────────
    cam = CameraCapture()
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

    # ── Tạo DetectionResult ───────────────────────────────
    loop = asyncio.get_event_loop()
    result_queue = DetectionResult(loop)

    # ── Khởi động inference thread ────────────────────────
    infer_thread = threading.Thread(
        target=run_inference,
        args=(cam, result_queue, buzzer, start_time),
        daemon=True,
    )
    infer_thread.start()
    logger.info("[Main] Inference thread đã khởi động")

    # ── In thông tin cấu hình ─────────────────────────────
    print("=" * 60)
    print("  Jetson Violence Detection Client  [v2]")
    print(f"  Camera index       : {cam.camera_id}")
    print(f"  Stream URL         : {WS_STREAM_URL}")
    print(f"  Detection URL      : {WS_DETECTION_URL}")
    print(f"  Stream FPS         : {STREAM_FPS}")
    print(f"  Infer rate         : mỗi {INFER_INTERVAL_MS}ms (~{1000//INFER_INTERVAL_MS}/s)")
    print(f"  Window reset       : mỗi {WINDOW_SEC}s")
    print(f"  EMA alpha          : {EMA_ALPHA}")
    print(f"  Threshold          : {THRESHOLD}  (áp lên smooth prob)")
    print(f"  Confirm            : {CONFIRM_FRAMES} lần liên tiếp")
    print(f"  Cooldown           : {COOLDOWN_SEC}s")
    print(f"  ── Buzzer ─────────────────────────────────")
    print(f"  GPIO PIN           : {BUZZER_PIN}  (BOARD)")
    print(f"  Kích hoạt sau      : {VIOLENCE_BUZZER_DELAY}s bạo lực liên tục")
    print(f"  Beep ON / OFF      : {BUZZER_BEEP_ON_SEC}s / {BUZZER_BEEP_OFF_SEC}s")
    print(f"  GPIO khả dụng      : {GPIO_AVAILABLE}")
    print("=" * 60)

    try:
        await asyncio.gather(
            stream_video(cam),
            stream_detection(result_queue),
        )
    finally:
        buzzer.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("[Main] Dừng chương trình.")
    finally:
        loop.close()