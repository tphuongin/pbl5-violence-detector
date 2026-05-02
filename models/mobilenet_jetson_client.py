# mobilenet_jetson_client_v6.py
#
# Cập nhật từ v5:
#   - MODEL THAY ĐỔI: từ Offline TSM (16-frame batch) → Online TSM (1-frame streaming)
#     * Tên engine: mobilenetv4.engine  (thay vì mobilenetv2_tsm_16frames.engine)
#     * Input:  'input' (1,3,172,172) + 17 state buffers  buffer_0..buffer_16
#     * Output: 'output' (1,2)        + 17 new_buffer_0..new_buffer_16
#     * Không cần frame_buffer/deque NUM_FRAMES nữa — infer từng frame một
#   - MobileNetTRT được viết lại hoàn toàn để quản lý buffers liên tục
#   - NUM_FRAMES config vẫn giữ lại cho phần logging/payload (giá trị = 1)
#
# Buffer shapes (float32):
#   buffer_0  : (1, 4,  86, 86)    new_buffer_0  : (1, 4,  86, 86)
#   buffer_1  : (1, 2,  86, 86)    new_buffer_1  : (1, 2,  86, 86)
#   buffer_2  : (1, 3,  43, 43)    new_buffer_2  : (1, 3,  43, 43)
#   buffer_3  : (1, 3,  43, 43)    new_buffer_3  : (1, 3,  43, 43)
#   buffer_4  : (1, 4,  22, 22)    new_buffer_4  : (1, 4,  22, 22)
#   buffer_5  : (1, 4,  22, 22)    new_buffer_5  : (1, 4,  22, 22)
#   buffer_6  : (1, 4,  22, 22)    new_buffer_6  : (1, 4,  22, 22)
#   buffer_7  : (1, 8,  11, 11)    new_buffer_7  : (1, 8,  11, 11)
#   buffer_8  : (1, 8,  11, 11)    new_buffer_8  : (1, 8,  11, 11)
#   buffer_9  : (1, 8,  11, 11)    new_buffer_9  : (1, 8,  11, 11)
#   buffer_10 : (1, 8,  11, 11)    new_buffer_10 : (1, 8,  11, 11)
#   buffer_11 : (1, 12, 11, 11)    new_buffer_11 : (1, 12, 11, 11)
#   buffer_12 : (1, 12, 11, 11)    new_buffer_12 : (1, 12, 11, 11)
#   buffer_13 : (1, 12, 11, 11)    new_buffer_13 : (1, 12, 11, 11)
#   buffer_14 : (1, 20,  6,  6)    new_buffer_14 : (1, 20,  6,  6)
#   buffer_15 : (1, 20,  6,  6)    new_buffer_15 : (1, 20,  6,  6)
#   buffer_16 : (1, 20,  6,  6)    new_buffer_16 : (1, 20,  6,  6)
#
# Luồng hoạt động:
#   ┌─────────────────────────────────────┐
#   │  CameraCapture (thread)             │  ← grab frame liên tục
#   └──────────────┬──────────────────────┘
#                  │ frame
#         ┌────────┴────────┐
#         ▼                 ▼
#   VideoStreamer      InferenceWorker (thread)
#   (async task)         TensorRT MobileNetV2-TSM ONLINE
#         │               1 frame/infer + 17 state buffers
#         │ JPEG binary      │ JSON result
#         ▼                 ▼
#   ws://.../ws/stream/   ws://.../ws/detection/
#   {CAMERA_ID}           {CAMERA_ID}
#                          │
#                     ┌────┴────┐
#                     ▼         ▼
#              BuzzerController  PhoneDialer
#              GPIO PIN 12        SIM module
#              beep khi bạo lực   ATD0961521940;
#
# JSON detection payload (gửi mỗi lần infer):
# {
#   "camera_id":         "jetson-cam-01",
#   "timestamp":         "14:35:22.047",
#   "label":             "VIOLENCE" | "Normal",
#   "prob_raw":          0.73,
#   "conf_thresh":       0.55,
#   "alert":             true,
#   "alert_until":       1.3,
#   "alert_sec":         3.0,
#   "infer_fps":         8.5,
#   "cam_fps":           29.8,
#   "uptime":            142,
#   "infer_ms":          118.4,
#   "violence_duration": 3.7,
#   "buzzer_active":     false,
#   "phone_calling":     false,
#   "thumb_b64":         "<base64 jpeg 120x68>"
# }
#
# Chạy: python3 mobilenet_jetson_client_v6.py

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
ENGINE_PATH       = "mobilenetv4.engine"   # ← tên file engine thực tế
INPUT_SIZE        = 172                     # ← từ ONNX input shape (1,3,172,172)
NUM_BUFFERS       = 17                      # ← số state buffer của ONLINE TSM

# Shapes của 17 buffer (dùng để khởi tạo zero state)
BUFFER_SHAPES = [
    (1, 4,  86, 86),   # buffer_0
    (1, 2,  86, 86),   # buffer_1
    (1, 3,  43, 43),   # buffer_2
    (1, 3,  43, 43),   # buffer_3
    (1, 4,  22, 22),   # buffer_4
    (1, 4,  22, 22),   # buffer_5
    (1, 4,  22, 22),   # buffer_6
    (1, 8,  11, 11),   # buffer_7
    (1, 8,  11, 11),   # buffer_8
    (1, 8,  11, 11),   # buffer_9
    (1, 8,  11, 11),   # buffer_10
    (1, 12, 11, 11),   # buffer_11
    (1, 12, 11, 11),   # buffer_12
    (1, 12, 11, 11),   # buffer_13
    (1, 20,  6,  6),   # buffer_14
    (1, 20,  6,  6),   # buffer_15
    (1, 20,  6,  6),   # buffer_16
]

# ── Detection logic ───────────────────────────────────────
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
VIOLENCE_BUZZER_DELAY   = 5.0   # Giây bạo lực liên tục trước khi kêu còi + gọi điện
BUZZER_BEEP_ON_SEC      = 0.5   # Thời gian còi kêu mỗi tiếng beep
BUZZER_BEEP_OFF_SEC     = 0.3   # Khoảng nghỉ giữa các tiếng beep

# ── Phone (SIM module AT command) ────────────────────────
PHONE_NUMBER        = "0961521940"
PHONE_SERIAL_PORT   = "/dev/ttyUSB2"   # Đổi thành ttyUSB1 nếu cần
PHONE_BAUD_RATE     = 115200
PHONE_CALL_TIMEOUT  = 30               # Giây duy trì cuộc gọi rồi tự cúp
PHONE_RETRY_DELAY   = 60               # Giây cooldown giữa 2 lần gọi
PHONE_RING_WAIT     = 5                # Giây chờ sau ATD trước khi duy trì cuộc gọi
STARTUP_MUTE_SECONDS = 10.0            # Tắt còi + gọi trong 10 giây đầu khi khởi động

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jetson-mobilenet")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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
                logger.info(
                    f"[Phone] SIM module sẵn sàng — {PHONE_SERIAL_PORT} "
                    f"@ {PHONE_BAUD_RATE} baud.")
            else:
                logger.warning(f"[Phone] SIM không phản hồi AT: {resp!r}")
        except Exception as e:
            logger.warning(f"[Phone] Không mở được {PHONE_SERIAL_PORT}: {e}")
            logger.warning("[Phone] Chạy ở chế độ giả lập (không có SIM module).")

    def _send_at(self, cmd: str, wait: float = 1.0) -> str:
        if self._serial is None or not self._serial.is_open:
            return ""
        try:
            self._serial.reset_input_buffer()
            self._serial.write((cmd + "\r\n").encode())
            time.sleep(wait)
            resp = self._serial.read_all().decode(errors="ignore")
            logger.debug(f"[Phone] >> {cmd!r}  << {resp.strip()!r}")
            return resp
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
                logger.info("[Phone] Yêu cầu cúp máy (về Normal).")

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
            logger.info("[Phone] Thread dừng, serial đóng.")

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
        logger.info("[Phone] Cuộc gọi kết thúc.")

    def _hangup(self):
        if self.available:
            self._send_at("ATH", wait=0.5)
            logger.info("[Phone] ATH — Cúp máy.")
        else:
            logger.debug("[Phone][SIM] GIẢ LẬP: ATH")


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
#  TENSORRT MODEL  — MobileNetV2-TSM ONLINE (1-frame streaming)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MobileNetTRT:
    """
    Wrapper TensorRT cho MobileNetV2-TSM ONLINE mode.

    Binding order (từ engine):
        Input  bindings: 'input', 'buffer_0' … 'buffer_16'   (18 inputs)
        Output bindings: 'output', 'new_buffer_0' … 'new_buffer_16'  (18 outputs)

    Workflow mỗi frame:
        1. Copy frame blob → input device memory
        2. Copy current state buffers → buffer_0..16 device memory
        3. Execute inference
        4. Read output logits + new state buffers từ device
        5. Cập nhật state buffers cho lần infer tiếp theo

    State buffers được khởi tạo bằng zeros và tự động cập nhật sau mỗi infer.
    """

    # Thứ tự binding theo tên — phải khớp với engine
    INPUT_NAMES  = ["input"] + [f"buffer_{i}"     for i in range(NUM_BUFFERS)]
    OUTPUT_NAMES = ["output"] + [f"new_buffer_{i}" for i in range(NUM_BUFFERS)]

    def __init__(self, engine_path: str, cuda_ctx):
        self.cuda_ctx = cuda_ctx
        self.cuda_ctx.push()

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        self.context  = self.engine.create_execution_context()
        self.stream   = cuda.Stream()

        # ── Phân loại bindings theo tên ──────────────────
        self._in_bufs  = {}   # name → {"device": ..., "host": ...}
        self._out_bufs = {}   # name → {"device": ..., "host": ...}
        self._bindings = [None] * self.engine.num_bindings

        for b in self.engine:
            idx   = self.engine.get_binding_index(b)
            shape = tuple(self.engine.get_binding_shape(b))
            dtype = trt.nptype(self.engine.get_binding_dtype(b))
            size  = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dev   = cuda.mem_alloc(size)
            host  = cuda.pagelocked_empty(shape, dtype)
            info  = {"device": dev, "host": host, "shape": shape, "dtype": dtype}
            self._bindings[idx] = int(dev)
            if self.engine.binding_is_input(b):
                self._in_bufs[b]  = info
            else:
                self._out_bufs[b] = info

        # ── Kiểm tra đủ bindings ──────────────────────────
        missing_in  = [n for n in self.INPUT_NAMES  if n not in self._in_bufs]
        missing_out = [n for n in self.OUTPUT_NAMES if n not in self._out_bufs]
        if missing_in or missing_out:
            raise RuntimeError(
                f"[TRT] Engine thiếu bindings!\n"
                f"  Inputs missing : {missing_in}\n"
                f"  Outputs missing: {missing_out}"
            )

        # ── Khởi tạo state buffers = zeros ───────────────
        for i in range(NUM_BUFFERS):
            name = f"buffer_{i}"
            self._in_bufs[name]["host"].fill(0)
            cuda.memcpy_htod_async(
                self._in_bufs[name]["device"],
                self._in_bufs[name]["host"],
                self.stream,
            )
        self.stream.synchronize()

        self.cuda_ctx.pop()
        logger.info(
            f"[TRT] MobileNetV2-TSM ONLINE loaded from '{engine_path}' — "
            f"{self.engine.num_bindings} bindings  "
            f"({len(self._in_bufs)} inputs / {len(self._out_bufs)} outputs)"
        )

    def infer(self, frame_blob: np.ndarray) -> np.ndarray:
        """
        frame_blob: (1, 3, INPUT_SIZE, INPUT_SIZE) float32, ImageNet-normalized
        Returns   : (2,) float32  logits  [normal, violence]
        """
        self.cuda_ctx.push()
        try:
            # 1. Copy frame vào input
            inp = self._in_bufs["input"]
            inp["host"].flat[:] = frame_blob.ravel()
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

            # 2. State buffers đã nằm trên device từ lần trước (hoặc zeros)
            #    → không cần copy lại, chỉ cần đảm bảo bindings trỏ đúng

            # 3. Execute
            self.context.execute_async_v2(self._bindings, self.stream.handle)

            # 4. Copy output → host
            out_info = self._out_bufs["output"]
            cuda.memcpy_dtoh_async(out_info["host"], out_info["device"], self.stream)

            # 5. Copy new_buffer_* về host rồi ngay lập tức copy vào buffer_* device
            for i in range(NUM_BUFFERS):
                new_buf = self._out_bufs[f"new_buffer_{i}"]
                cur_buf = self._in_bufs[f"buffer_{i}"]
                cuda.memcpy_dtoh_async(new_buf["host"], new_buf["device"], self.stream)

            self.stream.synchronize()

            # 6. Cập nhật state: host new_buffer_i → device buffer_i
            for i in range(NUM_BUFFERS):
                new_buf = self._out_bufs[f"new_buffer_{i}"]
                cur_buf = self._in_bufs[f"buffer_{i}"]
                np.copyto(cur_buf["host"], new_buf["host"])
                cuda.memcpy_htod_async(cur_buf["device"], cur_buf["host"], self.stream)

            self.stream.synchronize()

            return out_info["host"].copy().flatten()

        finally:
            self.cuda_ctx.pop()

    def reset_state(self):
        """Reset toàn bộ state buffers về zero (dùng khi muốn bắt đầu lại)."""
        self.cuda_ctx.push()
        try:
            for i in range(NUM_BUFFERS):
                buf = self._in_bufs[f"buffer_{i}"]
                buf["host"].fill(0)
                cuda.memcpy_htod_async(buf["device"], buf["host"], self.stream)
            self.stream.synchronize()
            logger.info("[TRT] State buffers đã reset về zero.")
        finally:
            self.cuda_ctx.pop()

    def destroy(self):
        self.cuda_ctx.push()
        del self.context
        del self.engine
        self.cuda_ctx.pop()
        self.cuda_ctx.detach()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREPROCESSING HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    frame: BGR numpy array (H, W, 3) từ OpenCV
    Returns: (1, 3, INPUT_SIZE, INPUT_SIZE) float32  ImageNet-normalized
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r   = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                     interpolation=cv2.INTER_LINEAR)
    n   = (r.astype(np.float32) / 255.0 - MEAN) / STD
    chw = n.transpose(2, 0, 1)                        # (3, H, W)
    return np.ascontiguousarray(chw[np.newaxis], dtype=np.float32)  # (1,3,H,W)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE
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
#  INFERENCE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_inference(cam: CameraCapture,
                  result_queue: DetectionResult,
                  buzzer: BuzzerController,
                  dialer: PhoneDialer,
                  start_time: float):
    """
    ONLINE TSM mode: infer từng frame một.
    State 17 buffers được duy trì tự động bên trong MobileNetTRT.

    So với v5:
      - Bỏ frame_buffer (deque NUM_FRAMES) — không còn cần tích lũy 16 frames
      - Bỏ bước "Buffering X/NUM_FRAMES" — infer ngay từ frame đầu tiên
      - preprocess_frame() thay cho preprocess(list_of_frames)
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

    # ── Alert state ───────────────────────────────────────
    alert_until = 0.0
    is_alert    = False

    # ── Biến theo dõi thời gian bạo lực liên tục ─────────
    violence_start_time = None
    alert_triggered     = False

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
            startup_muted = (now - start_time) < STARTUP_MUTE_SECONDS
            cam_times.append(now)

            # ── Inference (1 frame, ONLINE mode) ─────────
            t0 = time.time()
            try:
                blob   = preprocess_frame(frame)   # (1,3,172,172)
                logits = model.infer(blob)          # (2,)  [normal, violence]
                probs  = softmax(logits)
                prob_raw = float(probs[1])          # index 1 = violence
            except Exception as e:
                logger.error(f"[Inference] Lỗi infer: {e}")
                time.sleep(0.05)
                continue
            t1 = time.time()
            infer_times.append(t1)

            # ── Alert logic ───────────────────────────────
            if prob_raw >= CONF_THRESH:
                is_alert    = True
                alert_until = now + ALERT_SECONDS
            elif now > alert_until:
                is_alert = False

            label = "VIOLENCE" if is_alert else "Normal"

            # ── Buzzer + Phone logic ──────────────────────
            if label == "VIOLENCE":
                if violence_start_time is None:
                    violence_start_time = now
                    alert_triggered     = False
                    logger.info("[Alert] Bắt đầu theo dõi thời gian bạo lực …")

                violence_duration = now - violence_start_time

                if violence_duration >= VIOLENCE_BUZZER_DELAY:
                    if startup_muted:
                        buzzer.deactivate()
                        dialer.cancel_call()
                    else:
                        buzzer.activate()
                        if not alert_triggered:
                            dialer.request_call()
                            alert_triggered = True
                else:
                    remaining = VIOLENCE_BUZZER_DELAY - violence_duration
                    logger.debug(
                        f"[Alert] Bạo lực {violence_duration:.1f}s "
                        f"/ {VIOLENCE_BUZZER_DELAY}s "
                        f"(còn {remaining:.1f}s nữa)")
            else:
                if violence_start_time is not None:
                    elapsed = now - violence_start_time
                    logger.info(
                        f"[Alert] Kết thúc bạo lực sau {elapsed:.1f}s — reset.")
                    violence_start_time = None
                    alert_triggered     = False
                buzzer.deactivate()
                dialer.cancel_call()
                violence_duration = 0.0

            # ── FPS ──────────────────────────────────────
            infer_fps = (len(infer_times) - 1) / (infer_times[-1] - infer_times[0]) \
                        if len(infer_times) > 1 else 0.0
            cam_fps   = (len(cam_times)   - 1) / (cam_times[-1]   - cam_times[0]) \
                        if len(cam_times)   > 1 else 0.0

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
                "violence_duration": round(violence_duration,   2),
                "buzzer_active":     buzzer.is_active,
                "phone_calling":     dialer.is_calling,
                "thumb_b64":         thumb_b64,
            }

            result_queue.put(payload)

            logger.info(
                f"prob={prob_raw:.3f}  alert={is_alert}  → {label}  "
                f"dur={violence_duration:.1f}s  "
                f"buzzer={'ON' if buzzer.is_active else 'off'}  "
                f"phone={'CALLING' if dialer.is_calling else 'idle'}  "
                f"({(t1 - t0) * 1000:.0f}ms)"
            )

    except Exception as e:
        logger.exception(f"[Inference] Lỗi nghiêm trọng: {e}")
    finally:
        buzzer.deactivate()
        dialer.cancel_call()
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

    buzzer = BuzzerController()
    buzzer.start()

    dialer = PhoneDialer()
    dialer.start()

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
        dialer.stop()
        return

    logger.info(f"[Main] Camera sẵn sàng — index={cam.camera_id}")

    loop = asyncio.get_event_loop()
    result_queue = DetectionResult(loop)

    infer_thread = threading.Thread(
        target=run_inference,
        args=(cam, result_queue, buzzer, dialer, start_time),
        daemon=True,
    )
    infer_thread.start()
    logger.info("[Main] Inference thread đã khởi động")

    print("=" * 60)
    print("  Jetson Violence Detection Client — MobileNetV2-TSM  [v6]")
    print(f"  MODEL MODE         : ONLINE (1-frame streaming)")
    print(f"  Engine file        : {ENGINE_PATH}")
    print(f"  State buffers      : {NUM_BUFFERS} buffers (tự động)")
    print(f"  Camera index       : {cam.camera_id}")
    print(f"  Stream URL         : {WS_STREAM_URL}")
    print(f"  Detection URL      : {WS_DETECTION_URL}")
    print(f"  Stream FPS         : {STREAM_FPS}")
    print(f"  Input size         : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Conf thresh        : {CONF_THRESH}")
    print(f"  Alert window       : {ALERT_SECONDS}s")
    print(f"  ── Buzzer ─────────────────────────────────")
    print(f"  GPIO PIN           : {BUZZER_PIN}  (BOARD)")
    print(f"  Kích hoạt sau      : {VIOLENCE_BUZZER_DELAY}s bạo lực liên tục")
    print(f"  Beep ON / OFF      : {BUZZER_BEEP_ON_SEC}s / {BUZZER_BEEP_OFF_SEC}s")
    print(f"  GPIO khả dụng      : {GPIO_AVAILABLE}")
    print(f"  ── Phone ──────────────────────────────────")
    print(f"  Số điện thoại      : {PHONE_NUMBER}")
    print(f"  Serial port        : {PHONE_SERIAL_PORT}  @ {PHONE_BAUD_RATE} baud")
    print(f"  SIM khả dụng       : {dialer.available}")
    print(f"  Thời gian gọi      : {PHONE_CALL_TIMEOUT}s rồi tự cúp")
    print(f"  Cooldown gọi lại   : {PHONE_RETRY_DELAY}s")
    print("=" * 60)

    try:
        await asyncio.gather(
            stream_video(cam),
            stream_detection(result_queue),
        )
    finally:
        buzzer.stop()
        dialer.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("[Main] Dừng chương trình.")
    finally:
        loop.close()