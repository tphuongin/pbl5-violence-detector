"""
mobilenet_jetson_client_v4.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cập nhật từ v3: thêm cảnh báo còi (Buzzer) khi phát hiện
bạo lực liên tục quá VIOLENCE_BUZZER_DELAY giây (mặc định 5s).
(Ported từ movinet_jetson_client_v1.py)

Luồng hoạt động:
  ┌─────────────────────────────────────┐
  │  CameraCapture (thread)             │  ← grab frame liên tục
  └──────────────┬──────────────────────┘
                 │ frame
        ┌────────┴────────┐
        ▼                 ▼
  VideoStreamer      InferenceWorker (thread)
  (async task)         TensorRT MobileNetV2-TSM
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
  "conf_thresh":       0.55,
  "alert":             true,
  "alert_until":       1.3,           // giây còn lại trong alert window
  "alert_sec":         3.0,
  "infer_fps":         8.5,
  "cam_fps":           29.8,
  "uptime":            142,
  "infer_ms":          118.4,
  "num_frames":        16,
  "violence_duration": 3.7,   // giây bạo lực liên tục (0 nếu Normal)
  "buzzer_active":     false,  // còi đang kêu hay không
  "thumb_b64":         "<base64 jpeg 120x68>"
}

Chạy: python3 mobilenet_jetson_client_v4.py
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
ENGINE_PATH       = "mobilenetv2_tsm_16frames.engine"
NUM_FRAMES        = 16
INPUT_SIZE        = 172

# ── Detection logic (giữ nguyên từ mobilenet_v3.py) ──────
CONF_THRESH       = 0.55    # ngưỡng xác suất để kích hoạt alert
ALERT_SECONDS     = 3.0     # thời gian duy trì trạng thái alert sau khi detect

# ── Thumbnail gửi kèm payload ────────────────────────────
THUMB_WIDTH       = 120
THUMB_HEIGHT      = 68

# ── Preprocessing (ImageNet normalization) ────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
logger = logging.getLogger("jetson-mobilenet")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUZZER CONTROLLER  (ported từ movinet_jetson_client_v1.py)
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
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        return cap

    def run(self):
        cap = None
        for idx in [1, 2, 3, 4, 0]:
            logger.info(f"[Camera] Thử index {idx} …")
            cap = self._try_open(idx)
            if cap is not None:
                self.camera_id = idx
                break

        if cap is None:
            self.error_msg = "Không mở được camera nào (đã thử 1,2,3,4,0)"
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

        self.context = self.engine.create_execution_context()
        self.inputs  = []
        self.outputs = []
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
        logger.info(f"[TRT] MobileNetV2-TSM loaded — "
                    f"{self.engine.num_bindings} bindings")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """
        blob: numpy array (1, 16, 3, H, W) float32, already normalized.
        Returns logits array shape (2,).
        """
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
#  PREPROCESSING HELPERS  (giữ nguyên từ mobilenet_v3.py)
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
    # stack → (NUM_FRAMES, 3, H, W) → add batch → (1, NUM_FRAMES, 3, H, W)
    return np.ascontiguousarray(np.stack(out)[np.newaxis], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE  (giữa inference thread ↔ async tasks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionResult:
    """
    Thread-safe container cho kết quả detection mới nhất.
    asyncio.Event được tạo lazy để đảm bảo nằm đúng event loop.
    """
    def __init__(self, loop):
        self._lock  = threading.Lock()
        self._data  = None
        self._loop  = loop
        self._event = None   # lazy init trong coroutine

    def put(self, data: dict):
        """Gọi từ inference thread."""
        with self._lock:
            self._data = data
        if self._event is not None:
            self._loop.call_soon_threadsafe(self._event.set)

    async def get_new(self) -> dict:
        """Chờ đến khi có kết quả mới, trả về rồi clear event."""
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
    """
    Thu thập NUM_FRAMES frame từ camera, chạy TRT inference,
    áp dụng logic alert giống mobilenet_v3.py, rồi đẩy payload
    vào result_queue để async task gửi lên backend.

    Bổ sung so với v3: BuzzerController kích hoạt còi khi
    bạo lực liên tục ≥ VIOLENCE_BUZZER_DELAY giây.
    """
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()

    logger.info("[Inference] Loading TensorRT engine …")
    model = MobileNetTRT(ENGINE_PATH, cuda_ctx)

    logger.info("[Inference] Chờ frame đầu tiên …")
    for _ in range(100):
        if cam.get_latest() is not None:
            break
        time.sleep(0.1)

    # ── Frame buffer (sliding window NUM_FRAMES frames) ──
    frame_buffer = deque(maxlen=NUM_FRAMES)

    # ── Alert state (giữ nguyên logic từ mobilenet_v3.py) ─
    alert_until = 0.0
    is_alert    = False

    # ── Biến theo dõi thời gian bạo lực liên tục ─────────
    violence_start_time = None   # None = không có bạo lực hiện tại

    # ── FPS tracking ─────────────────────────────────────
    cam_times   = deque(maxlen=60)
    infer_times = deque(maxlen=30)

    try:
        while True:
            frame = cam.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            frame_buffer.append(frame)
            cam_times.append(now)

            # Chờ đủ NUM_FRAMES mới bắt đầu infer
            if len(frame_buffer) < NUM_FRAMES:
                logger.info(
                    f"[Inference] Buffering {len(frame_buffer)}/{NUM_FRAMES}…")
                time.sleep(0.05)
                continue

            # ── Inference ────────────────────────────────
            t0 = time.time()
            try:
                blob       = preprocess(list(frame_buffer))
                logits     = model.infer(blob)
                probs      = softmax(logits)
                prob_raw   = float(probs[1])   # index 1 = violence
            except Exception as e:
                logger.error(f"[Inference] Lỗi infer: {e}")
                time.sleep(0.05)
                continue
            t1 = time.time()
            infer_times.append(t1)

            # ── Alert logic (giữ nguyên từ mobilenet_v3.py) ──
            if prob_raw >= CONF_THRESH:
                is_alert    = True
                alert_until = now + ALERT_SECONDS
            elif now > alert_until:
                is_alert = False

            label = "VIOLENCE" if is_alert else "Normal"

            # ── Buzzer logic (ported từ movinet_jetson_client_v1.py) ──
            if label == "VIOLENCE":
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

            # ── FPS ──────────────────────────────────────
            infer_fps = (len(infer_times) - 1) / (infer_times[-1] - infer_times[0]) \
                        if len(infer_times) > 1 else 0.0
            cam_fps   = (len(cam_times) - 1) / (cam_times[-1] - cam_times[0]) \
                        if len(cam_times) > 1 else 0.0

            # ── Thumbnail ────────────────────────────────
            thumb = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)
            _, buf = cv2.imencode(
                ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            thumb_b64 = base64.b64encode(buf).decode("utf-8")

            ts_str = (time.strftime("%H:%M:%S")
                      + f".{int((now % 1) * 1000):03d}")

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
                "thumb_b64":         thumb_b64,
            }

            result_queue.put(payload)

            logger.info(
                f"prob={prob_raw:.3f}  alert={is_alert}  → {label}  "
                f"dur={violence_duration:.1f}s  "
                f"buzzer={'ON' if buzzer.is_active else 'off'}  "
                f"({(t1 - t0) * 1000:.0f}ms)"
            )

    except Exception as e:
        logger.exception(f"[Inference] Lỗi nghiêm trọng: {e}")
    finally:
        buzzer.deactivate()     # Đảm bảo tắt còi khi inference dừng
        model.destroy()


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
                            ".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                        )
                        await ws.send(buf.tobytes())
                    sleep_t = interval - (time.time() - t0)
                    if sleep_t > 0:
                        await asyncio.sleep(sleep_t)
                    else:
                        await asyncio.sleep(0)

        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                f"[Stream] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …")
        except Exception as e:
            logger.error(
                f"[Stream] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …")
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
            logger.warning(
                f"[Detection] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …")
        except Exception as e:
            logger.error(
                f"[Detection] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …")
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
    for _ in range(80):
        if cam.connected or cam.error_msg:
            break
        await asyncio.sleep(0.1)

    if cam.error_msg:
        logger.error(f"[Main] {cam.error_msg}")
        buzzer.stop()
        return

    logger.info(f"[Main] Camera sẵn sàng — index={cam.camera_id}")

    # ── Tạo DetectionResult với event loop hiện tại ──────
    loop = asyncio.get_event_loop()
    result_queue = DetectionResult(loop)

    # ── Khởi động inference thread ───────────────────────
    infer_thread = threading.Thread(
        target=run_inference,
        args=(cam, result_queue, buzzer, start_time),
        daemon=True,
    )
    infer_thread.start()
    logger.info("[Main] Inference thread đã khởi động")

    # ── In thông tin cấu hình ─────────────────────────────
    print("=" * 60)
    print("  Jetson Violence Detection Client — MobileNetV2-TSM  [v4]")
    print(f"  Camera index       : {cam.camera_id}")
    print(f"  Stream URL         : {WS_STREAM_URL}")
    print(f"  Detection URL      : {WS_DETECTION_URL}")
    print(f"  Stream FPS         : {STREAM_FPS}")
    print(f"  Num frames         : {NUM_FRAMES} (sliding window)")
    print(f"  Input size         : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Conf thresh        : {CONF_THRESH}")
    print(f"  Alert window       : {ALERT_SECONDS}s")
    print(f"  ── Buzzer ─────────────────────────────────")
    print(f"  GPIO PIN           : {BUZZER_PIN}  (BOARD)")
    print(f"  Kích hoạt sau      : {VIOLENCE_BUZZER_DELAY}s bạo lực liên tục")
    print(f"  Beep ON / OFF      : {BUZZER_BEEP_ON_SEC}s / {BUZZER_BEEP_OFF_SEC}s")
    print(f"  GPIO khả dụng      : {GPIO_AVAILABLE}")
    print("=" * 60)

    # ── Chạy 2 async task song song ───────────────────────
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