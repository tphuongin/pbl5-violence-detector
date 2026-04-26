"""
movinet_jetson_client_v4.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Luồng hoạt động:
  ┌─────────────────────────────────────┐
  │  CameraCapture (thread)             │  ← grab frame liên tục
  │  → set frame_event sau mỗi frame    │     (event-based sync)
  └──────────────┬──────────────────────┘
                 │ frame_event.wait()
        ┌────────┴────────┐
        ▼                 ▼
  stream_video        run_inference (thread)
  (async task)          TensorRT MoViNet
        │                 │ lưu frame gốc + kết quả
        │ JPEG binary      │ vào SharedState
        ▼                 ▼
  ws://.../ws/stream/   ws://.../ws/detection/
  {CAMERA_ID}           {CAMERA_ID}
                         │
                         ▼
                   BuzzerController (thread)
                   GPIO PIN 12 — kêu khi bạo lực ≥ 5s

Thay đổi so với v3:
  [v4] Auto binding mapping trong MoViNetTRT:
      - Bỏ hoàn toàn STATE_MAP, LOGITS_NAME, IMAGE_NAME hard-code
      - _auto_map_bindings() tự detect sau khi load engine dựa vào shape:
          · Input  shape [1,1,H,W,3]      → IMAGE_NAME
          · Output shape [1, num_classes]  → LOGITS_NAME
          · Phần còn lại → state pairs, ghép input↔output bằng:
              1. shape matching
              2. suffix tên sau "block..." / "head..." để phân biệt
                 khi nhiều tensor cùng shape
      - Tương thích cả 2 version ONNX export:
          · Bản cũ: prefix "call_state_..." / "call_image:0"
          · Bản mới: prefix "serving_default_state_..." /
                     "serving_default_image:0"
      - Chỉ cần đổi ENGINE_PATH khi dùng engine mới

Changelog đầy đủ (v1 → v4):
  [v2-1] Tách Inference ↔ Rendering — encode JPEG/thumbnail
         chỉ trong stream_detection(), không trong inference loop
  [v2-2] Event-based sync thay polling — frame_event.wait/clear/set
  [v3-1] Bỏ Sliding Window Reset — MoViNet Stream tự quản lý context
  [v3-2] Fix race condition DetectionResult — _event tạo sẵn trong __init__
  [v3-3] Fix snapshot() deep copy raw_frame
  [v3-4] Fix thứ tự frame_event: clear → get_latest → wait
  [v3-5] Fix stream_video block event loop — encode qua run_in_executor
  [v3-6] Fix trt.Runtime lifetime — giữ làm instance variable
  [v4]   Auto binding mapping — không còn hard-code tên binding

JSON detection payload:
{
  "camera_id":         "jetson-cam-01",
  "timestamp":         "14:35:22.047",
  "label":             "VIOLENCE" | "Normal",
  "prob_raw":          0.73,
  "prob_smooth":       0.61,
  "confirm_count":     2,
  "confirm_needed":    2,
  "threshold":         0.5,
  "infer_fps":         12.3,
  "cam_fps":           29.8,
  "uptime":            142,
  "infer_ms":          48.2,
  "violence_duration": 3.7,
  "buzzer_active":     false,
  "thumb_b64":         "<base64 jpeg 120x68>"
}

Chạy: python3 movinet_jetson_client_v4.py
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
STREAM_FPS        = 15
JPEG_QUALITY      = 70

# ── TensorRT model ───────────────────────────────────────
ENGINE_PATH       = "movinet_v2.engine"
INPUT_SIZE        = 172      # dùng để validate IMAGE binding khi auto-map

# ── Inference timing ─────────────────────────────────────
INFER_INTERVAL_MS = 80       # ~12 lần/giây

# ── EMA smoothing ────────────────────────────────────────
EMA_ALPHA         = 0.35

# ── Peak boost ───────────────────────────────────────────
SPIKE_THRESH      = 0.12
SPIKE_BOOST       = 1.35

# ── Decision ─────────────────────────────────────────────
THRESHOLD         = 0.50
CONFIRM_FRAMES    = 2
COOLDOWN_SEC      = 0.8

# ── Thumbnail ────────────────────────────────────────────
THUMB_WIDTH       = 120
THUMB_HEIGHT      = 68

# ── Buzzer (GPIO) ─────────────────────────────────────────
BUZZER_PIN              = 12
VIOLENCE_BUZZER_DELAY   = 5.0
BUZZER_BEEP_ON_SEC      = 0.5
BUZZER_BEEP_OFF_SEC     = 0.3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jetson")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SharedState:
    """
    Thread-safe container lưu frame gốc numpy và kết quả inference
    mới nhất. Tách hoàn toàn khỏi việc encode JPEG/thumbnail.

    update()   — gọi từ inference thread
    snapshot() — gọi từ bất kỳ thread/coroutine nào,
                 trả về deep copy của raw_frame để tránh race condition
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._raw_frame   = None
        self._prob_raw    = 0.0
        self._prob_smooth = 0.0
        self._label       = 'Normal'

    def update(self,
               raw_frame: np.ndarray,
               prob_raw: float,
               prob_smooth: float,
               label: str) -> None:
        with self._lock:
            self._raw_frame   = raw_frame.copy()
            self._prob_raw    = prob_raw
            self._prob_smooth = prob_smooth
            self._label       = label

    def snapshot(self) -> Optional[dict]:
        """
        Trả về dict hoặc None nếu chưa có dữ liệu.
        raw_frame là deep copy — caller tự do thao tác mà không
        ảnh hưởng buffer nội bộ hay lần infer kế tiếp.
        """
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
    """
    Thu nhỏ frame thành thumbnail, trả về chuỗi base64 JPEG.
    Gọi từ stream_detection() qua run_in_executor() —
    KHÔNG gọi từ run_inference().
    """
    thumb = cv2.resize(raw_frame, (THUMB_WIDTH, THUMB_HEIGHT),
                       interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode('utf-8')


def encode_frame_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """
    Encode frame thành JPEG bytes.
    Chạy qua run_in_executor() trong stream_video() —
    tránh block asyncio event loop.
    """
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUZZER CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BuzzerController(threading.Thread):
    """
    Thread quản lý còi GPIO độc lập.

    activate()   — bắt đầu beep lặp lại
    deactivate() — tắt còi
    stop()       — dừng thread + cleanup GPIO
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
#  TENSORRT MODEL  [v4: auto binding mapping]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MoViNetTRT:
    """
    Wrapper TensorRT cho MoViNet-A2 Stream.

    [v4] Auto binding mapping:
      Không còn hard-code STATE_MAP / IMAGE_NAME / LOGITS_NAME.
      _auto_map_bindings() tự detect toàn bộ sau khi load engine,
      tương thích với mọi version ONNX export bất kể prefix tên.

    Thuật toán mapping:
      1. IMAGE input   : input duy nhất có ndim=5 và dim[-1]=3
      2. LOGITS output : output duy nhất có ndim=2 và dim[-1] > 1
      3. State pairs   : các input/output còn lại, ghép theo:
           a. shape matching (grouping outputs theo shape)
           b. suffix sau "block..." hoặc "head..." để phân biệt
              chính xác khi nhiều tensor cùng shape
              vd: "serving_default_state_block0_layer0_pool_buffer:0"
                  → suffix = "block0_layer0_pool_buffer:0"

    Không có sliding window reset:
      MoViNet Stream buffer tự quản lý temporal context qua EMA
      nội bộ — không tích lũy nhiễu theo thời gian.
      reset_states() chỉ gọi khi camera reconnect thật sự.
    """

    def __init__(self, engine_path: str, cuda_ctx):
        self.cuda_ctx = cuda_ctx
        self.cuda_ctx.push()

        # Giữ runtime sống cùng engine (tránh dangling reference)
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

        # [v4] Auto-detect tất cả tên binding — không hard-code
        self.IMAGE_NAME  = None
        self.LOGITS_NAME = None
        self.STATE_MAP   = {}   # input_name → output_name

        self._auto_map_bindings()
        self.reset_states()
        self.cuda_ctx.pop()

        logger.info(
            f"[TRT] Loaded — {self.engine.num_bindings} bindings\n"
            f"      IMAGE  : {self.IMAGE_NAME}\n"
            f"      LOGITS : {self.LOGITS_NAME}\n"
            f"      States : {len(self.STATE_MAP)} pairs"
        )

    # ── Auto mapping ──────────────────────────────────────

    def _auto_map_bindings(self):
        """
        Tự động xây dựng IMAGE_NAME, LOGITS_NAME, STATE_MAP
        từ thông tin binding của engine — không phụ thuộc prefix tên.

        Tại sao dùng BINDING INDEX thay suffix matching:
          - Output state tên "StatefulPartitionedCall:X" không chứa
            "block"/"head" → regex suffix không extract được gì có nghĩa
          - Fallback candidates[0] theo dict order tình cờ đúng với
            engine cũ nhưng sai với engine mới do thứ tự binding khác
          - TensorRT đảm bảo thứ tự binding cố định và tương ứng:
            state_input[i] trong graph → state_output[i] trong graph
            → map theo index là chính xác và ổn định với mọi engine

        Tương thích fp16:
          - hbuf dtype = trt.nptype() của binding (có thể float16)
          - infer() cast logits sang float32 trước softmax để tránh
            overflow → NaN với logits lớn trên fp16
        """
        # Thu thập binding theo đúng thứ tự index
        inputs_ordered  = []   # [(index, name, shape), ...]
        outputs_ordered = []   # [(index, name, shape), ...]

        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                inputs_ordered.append((i, name, shape))
            else:
                outputs_ordered.append((i, name, shape))

        # Giữ dict để lookup shape nhanh
        inputs  = {name: shape for _, name, shape in inputs_ordered}
        outputs = {name: shape for _, name, shape in outputs_ordered}

        # ── 1. Detect IMAGE input ────────────────────────────────────────
        # Shape [1, 1, H, W, 3] — ndim=5, last dim=3 (RGB channels)
        for _, name, shape in inputs_ordered:
            if len(shape) == 5 and shape[-1] == 3:
                self.IMAGE_NAME = name
                break

        if self.IMAGE_NAME is None:
            raise RuntimeError(
                "[TRT] Auto-map thất bại: không tìm được IMAGE input "
                "(cần shape [1,1,H,W,3]). Kiểm tra lại engine."
            )

        img_h = inputs[self.IMAGE_NAME][2]
        img_w = inputs[self.IMAGE_NAME][3]
        if img_h != INPUT_SIZE or img_w != INPUT_SIZE:
            logger.warning(
                f"[TRT] IMAGE shape={inputs[self.IMAGE_NAME]} "
                f"khác INPUT_SIZE={INPUT_SIZE} — "
                f"sẽ resize về {img_h}x{img_w}."
            )
        self._infer_h = img_h
        self._infer_w = img_w

        # ── 2. Detect LOGITS output ──────────────────────────────────────
        # Shape [1, num_classes] — ndim=2, last dim > 1
        for _, name, shape in outputs_ordered:
            if len(shape) == 2 and shape[-1] > 1:
                self.LOGITS_NAME = name
                break

        if self.LOGITS_NAME is None:
            raise RuntimeError(
                "[TRT] Auto-map thất bại: không tìm được LOGITS output "
                "(cần shape [1, N] với N>1). Kiểm tra lại engine."
            )

        # ── 3. Build STATE_MAP theo binding index ────────────────────────
        # Lọc state inputs/outputs, GIỮ NGUYÊN thứ tự binding index
        state_inputs_ordered = [
            (idx, name, shape)
            for idx, name, shape in inputs_ordered
            if name != self.IMAGE_NAME
        ]
        state_outputs_ordered = [
            (idx, name, shape)
            for idx, name, shape in outputs_ordered
            if name != self.LOGITS_NAME
        ]

        if len(state_inputs_ordered) > len(state_outputs_ordered):
            raise RuntimeError(
                f"[TRT] state input ({len(state_inputs_ordered)}) "
                f"> state output ({len(state_outputs_ordered)}). "
                f"Engine thiếu output state?"
            )

        if len(state_inputs_ordered) != len(state_outputs_ordered):
            logger.warning(
                f"[TRT] state input ({len(state_inputs_ordered)}) ≠ "
                f"state output ({len(state_outputs_ordered)}) — "
                f"bỏ qua {len(state_outputs_ordered) - len(state_inputs_ordered)} "
                f"output(s) dư."
            )

        # Map input[i] → output[i] theo thứ tự index
        # Validate shape phải khớp — nếu không khớp là engine lỗi
        for (_, in_name, in_shape), (_, out_name, out_shape) in zip(
            state_inputs_ordered, state_outputs_ordered
        ):
            if in_shape != out_shape:
                raise RuntimeError(
                    f"[TRT] Shape mismatch tại vị trí map:\n"
                    f"  input  '{in_name}' shape={in_shape}\n"
                    f"  output '{out_name}' shape={out_shape}\n"
                    f"Engine không phải MoViNet Stream hoặc bị corrupt."
                )
            self.STATE_MAP[in_name] = out_name

        logger.info(
            f"[TRT] Auto-mapped {len(self.STATE_MAP)} state pairs "
            f"(index-based)."
        )

    # ── Model operations ──────────────────────────────────

    def reset_states(self):
        """
        Zero toàn bộ stream state của MoViNet.
        Chỉ gọi khi khởi tạo hoặc camera reconnect —
        KHÔNG gọi định kỳ vì sẽ làm mất temporal context.
        """
        for name in self.STATE_MAP:
            self.hbuf[name].fill(0)
            cuda.memcpy_htod(self.dbuf[name], self.hbuf[name])

    def infer(self, frame_bgr: np.ndarray):
        """
        Chạy inference 1 frame, trả về (prob_violence, prob_normal).
        Stream state được cập nhật nội bộ sau mỗi lần infer.
        """
        self.cuda_ctx.push()
        try:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self._infer_w, self._infer_h),
                               interpolation=cv2.INTER_LINEAR)
            img = (frame.astype(np.float32) / 255.0).reshape(
                1, 1, self._infer_h, self._infer_w, 3)

            np.copyto(self.hbuf[self.IMAGE_NAME], img.ravel())
            cuda.memcpy_htod(self.dbuf[self.IMAGE_NAME],
                             self.hbuf[self.IMAGE_NAME])

            for in_name in self.STATE_MAP:
                cuda.memcpy_htod(self.dbuf[in_name], self.hbuf[in_name])

            bindings = [
                int(self.dbuf[self.engine.get_binding_name(i)])
                for i in range(self.engine.num_bindings)
            ]
            self.context.execute_v2(bindings)

            cuda.memcpy_dtoh(self.hbuf[self.LOGITS_NAME],
                             self.dbuf[self.LOGITS_NAME])
            # Cast sang float32 trước khi tính softmax —
            # fp16 engine trả về logits float16, np.exp() trên float16
            # overflow khi logit > ~89 → inf → inf/inf = NaN
            logits = self.hbuf[self.LOGITS_NAME].copy().astype(np.float32)
            exp    = np.exp(logits - logits.max())
            probs  = exp / exp.sum()

            # Cập nhật stream state cho frame tiếp theo
            for in_name, out_name in self.STATE_MAP.items():
                cuda.memcpy_dtoh(self.hbuf[in_name], self.dbuf[out_name])

            return float(probs[0]), float(probs[1])
        finally:
            self.cuda_ctx.pop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DETECTION RESULT  (async event bridge)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionResult:
    """
    Thread-safe bridge giữa inference thread (put) và
    asyncio coroutine (get_new).

    _event tạo sẵn trong __init__ để tránh race condition
    khi put() được gọi trước get_new() lần đầu.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._lock  = threading.Lock()
        self._data  = None
        self._loop  = loop
        self._event = asyncio.Event()   # tạo sẵn, không lazy

    def put(self, data: dict):
        """Gọi từ inference thread."""
        with self._lock:
            self._data = data
        self._loop.call_soon_threadsafe(self._event.set)

    async def get_new(self) -> dict:
        """Chờ result mới. Gọi từ asyncio coroutine."""
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
    """
    Inference loop:
      1. Chờ frame mới qua frame_event (event-based, không busy-poll)
      2. model.infer() → prob_raw, prob_smooth, label
      3. shared_state.update() — lưu frame gốc numpy + kết quả
      4. result_queue.put() — payload KHÔNG chứa thumbnail

    Thumbnail encode được thực hiện bởi stream_detection()
    qua render_and_encode_thumb() + run_in_executor().

    Không có sliding window reset:
      MoViNet Stream buffer tự quản lý temporal context —
      reset định kỳ làm mất context và gây warm-up lag ~1s.
    """
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

    cam_times     = deque(maxlen=60)
    infer_times   = deque(maxlen=30)
    last_infer_ms = 0.0

    try:
        while True:
            # Thứ tự: clear → get_latest → (xử lý) → wait
            # clear trước để không bỏ sót frame mới ghi sau get_latest
            frame_event.clear()
            frame = cam.get_latest()

            if frame is None:
                frame_event.wait(timeout=0.1)
                continue

            now    = time.time()
            now_ms = now * 1000

            # Throttle: bỏ qua frame nếu chưa đến interval tiếp theo
            if (now_ms - last_infer_ms) < INFER_INTERVAL_MS:
                frame_event.wait(timeout=0.1)
                continue

            t0 = time.time()
            prob_raw, _ = model.infer(frame)
            t1 = time.time()
            last_infer_ms = now_ms
            infer_times.append(t1)

            # ── EMA smoothing ─────────────────────────────────────────────
            prob_smooth = EMA_ALPHA * prob_raw + (1.0 - EMA_ALPHA) * prob_smooth

            # ── Spike boost ───────────────────────────────────────────────
            delta = prob_raw - prev_prob_raw
            if delta > SPIKE_THRESH:
                prob_smooth = min(1.0, prob_smooth * SPIKE_BOOST)
            prev_prob_raw = prob_raw

            # ── Confirm + cooldown ────────────────────────────────────────
            if prob_smooth >= THRESHOLD:
                confirm_count += 1
            else:
                confirm_count = 0

            if confirm_count >= CONFIRM_FRAMES:
                violence_until = now + COOLDOWN_SEC

            label = 'VIOLENCE' if now < violence_until else 'Normal'

            # Lưu frame gốc numpy — KHÔNG encode JPEG ở đây
            shared_state.update(frame, prob_raw, prob_smooth, label)

            # ── Buzzer logic ──────────────────────────────────────────────
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
                        f"/ {VIOLENCE_BUZZER_DELAY}s "
                        f"(còn {VIOLENCE_BUZZER_DELAY - violence_duration:.1f}s)"
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

            # ── FPS ───────────────────────────────────────────────────────
            infer_fps = (
                (len(infer_times) - 1) / (infer_times[-1] - infer_times[0])
                if len(infer_times) > 1 else 0.0
            )
            cam_times.append(now)
            cam_fps = (
                (len(cam_times) - 1) / (cam_times[-1] - cam_times[0])
                if len(cam_times) > 1 else 0.0
            )

            ts_str = time.strftime('%H:%M:%S') + f'.{int((now % 1) * 1000):03d}'

            # thumb_b64 được thêm bởi stream_detection()
            payload = {
                "camera_id":         CAMERA_ID,
                "timestamp":         ts_str,
                "label":             label,
                "prob_raw":          round(prob_raw,           4),
                "prob_smooth":       round(prob_smooth,         4),
                "confirm_count":     confirm_count,
                "confirm_needed":    CONFIRM_FRAMES,
                "threshold":         THRESHOLD,
                "infer_fps":         round(infer_fps,          1),
                "cam_fps":           round(cam_fps,            1),
                "uptime":            int(now - start_time),
                "infer_ms":          round((t1 - t0) * 1000,   1),
                "violence_duration": round(violence_duration,   2),
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
    """
    Gửi JPEG frame liên tục lên ws://.../ws/stream/{CAMERA_ID}.
    cv2.imencode() chạy trong run_in_executor() — không block event loop.
    """
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
            logger.warning(
                f"[Stream] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …"
            )
        except Exception as e:
            logger.error(
                f"[Stream] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …"
            )
        await asyncio.sleep(RECONNECT_DELAY)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC TASK 2 — GỬI KẾT QUẢ DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def stream_detection(result_queue: DetectionResult,
                            shared_state: SharedState,
                            loop: asyncio.AbstractEventLoop):
    """
    Gửi JSON detection result lên ws://.../ws/detection/{CAMERA_ID}.

    Encode thumbnail tại đây (tách khỏi inference loop):
      1. Nhận payload từ result_queue (không có thumb_b64)
      2. Lấy frame gốc từ shared_state.snapshot()
      3. Gọi render_and_encode_thumb() qua run_in_executor()
      4. Thêm thumb_b64 vào payload rồi gửi
    """
    while True:
        try:
            async with websockets.connect(WS_DETECTION_URL) as ws:
                logger.info(f"[Detection] Kết nối OK → {WS_DETECTION_URL}")
                while True:
                    payload = await result_queue.get_new()

                    snap = shared_state.snapshot()
                    if snap is not None:
                        thumb_b64 = await loop.run_in_executor(
                            None,
                            render_and_encode_thumb,
                            snap['raw_frame'],
                        )
                        payload['thumb_b64'] = thumb_b64
                    else:
                        payload['thumb_b64'] = ''

                    await ws.send(json.dumps(payload, ensure_ascii=False))

        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                f"[Detection] Mất kết nối, thử lại sau {RECONNECT_DELAY}s …"
            )
        except Exception as e:
            logger.error(
                f"[Detection] Lỗi: {e}, thử lại sau {RECONNECT_DELAY}s …"
            )
        await asyncio.sleep(RECONNECT_DELAY)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    start_time = time.time()
    loop       = asyncio.get_event_loop()   # Python 3.6 compatible

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

    print("=" * 62)
    print("  Jetson Violence Detection Client  [v4]")
    print(f"  Camera index       : {cam.camera_id}")
    print(f"  Stream URL         : {WS_STREAM_URL}")
    print(f"  Detection URL      : {WS_DETECTION_URL}")
    print(f"  Stream FPS         : {STREAM_FPS}")
    print(f"  Infer rate         : mỗi {INFER_INTERVAL_MS}ms (~{1000 // INFER_INTERVAL_MS}/s)")
    print(f"  Binding mapping    : TỰ ĐỘNG — không hard-code tên")
    print(f"  Sliding window     : TẮT — MoViNet Stream tự quản lý context")
    print(f"  EMA alpha          : {EMA_ALPHA}")
    print(f"  Spike boost        : x{SPIKE_BOOST} khi delta > {SPIKE_THRESH}")
    print(f"  Threshold          : {THRESHOLD}  (áp lên smooth prob)")
    print(f"  Confirm            : {CONFIRM_FRAMES} lần liên tiếp")
    print(f"  Cooldown           : {COOLDOWN_SEC}s")
    print(f"  ── Buzzer ──────────────────────────────────────")
    print(f"  GPIO PIN           : {BUZZER_PIN}  (BOARD)")
    print(f"  Kích hoạt sau      : {VIOLENCE_BUZZER_DELAY}s bạo lực liên tục")
    print(f"  Beep ON / OFF      : {BUZZER_BEEP_ON_SEC}s / {BUZZER_BEEP_OFF_SEC}s")
    print(f"  GPIO khả dụng      : {GPIO_AVAILABLE}")
    print("=" * 62)

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