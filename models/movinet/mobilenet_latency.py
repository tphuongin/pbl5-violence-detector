"""
mobilenet_latency.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Đo latency phát hiện bạo lực của MobileNetV2-TSM Jetson Client.

Workflow:
  1. Script này chạy trên máy tính (backend), lắng nghe WebSocket
     tại ws://0.0.0.0:8000/ws/video-analysis/{job_id}
  2. Gửi từng video lên Jetson qua POST /analyze-video
     (kèm backend_host = IP máy tính này)
  3. Jetson (mobilenet_jetson_client_v4.py) phân tích video rồi đẩy kết quả
     về WS server của script qua ws://{backend}:8000/ws/video-analysis/{job_id}?role=producer
  4. Khi nhận summary → tính latency = segments[0]['start_s'] - gt_start_s
  5. In bảng kết quả + thống kê tổng hợp

Cài đặt:
  pip install websockets aiohttp tabulate

Chạy:
  python3 mobilenet_latency.py --jetson 192.168.137.2 --config clips_config.json

Lưu ý cấu trúc thư mục:
  models/
    clips/
      vid1.mp4 ... vid20.mp4
    mobilenet/
      clips_config.json      ← path dùng "../clips/vid1.mp4"
      mobilenet_latency.py
"""

import asyncio
import aiohttp
import websockets
import websockets.server
import json
import time
import argparse
import os
import sys
import socket
from tabulate import tabulate

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CẤU HÌNH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JETSON_HOST   = "192.168.137.2"  # IP Jetson Nano — đổi cho phù hợp
JETSON_PORT   = 8001             # HTTP port Jetson (aiohttp, POST /analyze-video)

# Port WS server chạy trên máy này để nhận kết quả từ Jetson đẩy về
LOCAL_WS_PORT = 8000

TEST_CLIPS = [
    {"path": "../clips/vid1.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 1"},
    {"path": "../clips/vid2.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 2"},
    {"path": "../clips/vid3.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 3"},
    {"path": "../clips/vid4.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 4"},
    {"path": "../clips/vid5.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 5"},
    {"path": "../clips/vid6.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 6"},
    {"path": "../clips/vid7.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 7"},
    {"path": "../clips/vid8.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 8"},
    {"path": "../clips/vid9.mp4",  "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 9"},
    {"path": "../clips/vid10.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 10"},
    {"path": "../clips/vid11.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 11"},
    {"path": "../clips/vid12.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 12"},
    {"path": "../clips/vid13.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 13"},
    {"path": "../clips/vid14.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 14"},
    {"path": "../clips/vid15.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 15"},
    {"path": "../clips/vid16.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 16"},
    {"path": "../clips/vid17.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 17"},
    {"path": "../clips/vid18.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 18"},
    {"path": "../clips/vid19.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 19"},
    {"path": "../clips/vid20.mp4", "gt_start_s": 0.0, "gt_end_s": 5.0, "note": "clip 20"},
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LẤY IP MÁY TÍNH (backend host gửi cho Jetson)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_local_ip() -> str:
    """Lấy IP của máy tính trong LAN (để Jetson biết gửi WS về đâu)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WS SERVER — nhận kết quả từ Jetson đẩy về
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WsResultServer:
    """
    WebSocket server nhỏ lắng nghe tại:
        ws://0.0.0.0:{LOCAL_WS_PORT}/ws/video-analysis/{job_id}

    MobileNet Jetson client sẽ kết nối vào đây với
        ?role=producer
    và stream frame payload + summary payload.

    Mỗi job_id được đăng ký trước qua register(job_id),
    summary nhận được sẽ được đưa vào asyncio.Queue tương ứng.

    Ghi chú về payload MobileNet:
        frame payload:
            {
              "type":          "frame",
              "progress":      0.0–1.0,
              "frame_time_s":  <float>,
              "label":         "VIOLENCE" | "Normal",
              "prob_raw":      <float>,
              "prob_smooth":   <float>,   # EMA smoothed
              "infer_ms":      <float>
            }
        summary payload:
            {
              "type":               "summary",
              "verdict":            "VIOLENCE" | "Normal",
              "violence_ratio":     <float>,
              "violence_segments":  [{"start_s": .., "end_s": .., "max_prob": ..}, ...],
              "video_duration_s":   <float>,
              "processing_time_s":  <float>,
              "total_frames":       <int>,
              "engine":             "mobilenet"
            }
    """

    def __init__(self, port: int):
        self._port    = port
        self._waiters: dict[str, asyncio.Queue] = {}
        self._server  = None

    def register(self, job_id: str) -> asyncio.Queue:
        """Đăng ký job_id và trả về Queue sẽ nhận summary."""
        q = asyncio.Queue(maxsize=1)
        self._waiters[job_id] = q
        return q

    def unregister(self, job_id: str):
        self._waiters.pop(job_id, None)

    async def _handler(self, ws):
        """Xử lý từng WebSocket connection từ Jetson MobileNet client."""
        # Path: /ws/video-analysis/{job_id}  (có thể kèm ?role=producer)
        path   = ws.request.path if hasattr(ws, 'request') else getattr(ws, 'path', '/')
        job_id = path.rstrip('/').split('/')[-1].split('?')[0]

        queue = self._waiters.get(job_id)
        if queue is None:
            await ws.close(1008, f"Unknown job_id: {job_id}")
            return

        print(f"    [WS←MobileNet] Kết nối: job={job_id[:8]}")
        try:
            async for raw in ws:
                pkt = json.loads(raw)

                if pkt.get("type") == "frame":
                    pct      = pkt.get("progress", 0) * 100
                    label    = pkt.get("label", "?")
                    t_s      = pkt.get("frame_time_s", 0)
                    prob_raw = pkt.get("prob_raw", 0)
                    smooth   = pkt.get("prob_smooth", prob_raw)  # fallback nếu không có EMA
                    infer_ms = pkt.get("infer_ms", 0)
                    icon     = "🔴" if label == "VIOLENCE" else "🟢"
                    print(
                        f"    {icon} [{pct:5.1f}%] t={t_s:.2f}s  "
                        f"raw={prob_raw:.3f}  smooth={smooth:.3f}  "
                        f"{label}  ({infer_ms:.0f}ms)",
                        end="\r", flush=True,
                    )

                elif pkt.get("type") == "summary":
                    print()   # newline sau \r
                    await queue.put(pkt)
                    break     # summary là packet cuối

        except websockets.exceptions.ConnectionClosed:
            pass

    async def start(self):
        self._server = await websockets.server.serve(
            self._handler, "0.0.0.0", self._port
        )
        print(f"[WS Server] Lắng nghe tại "
              f"ws://0.0.0.0:{self._port}/ws/video-analysis/{{job_id}}")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(
        description="Đánh giá latency MobileNetV2-TSM Jetson Violence Detection"
    )
    p.add_argument("--jetson",  default=JETSON_HOST,
                   help=f"IP Jetson (mặc định: {JETSON_HOST})")
    p.add_argument("--port",    type=int, default=JETSON_PORT,
                   help=f"HTTP port Jetson (mặc định: {JETSON_PORT})")
    p.add_argument("--ws-port", type=int, default=LOCAL_WS_PORT,
                   help=f"Port WS server local (mặc định: {LOCAL_WS_PORT})")
    p.add_argument("--config",  default=None,
                   help="File JSON chứa danh sách clip + ground truth")
    p.add_argument("--output",  default="mobilenet_latency_results.json",
                   help="File JSON lưu kết quả (mặc định: mobilenet_latency_results.json)")
    p.add_argument("--delay",   type=float, default=3.0,
                   help="Giây chờ giữa các clip (mặc định: 3.0)")
    return p.parse_args()


def load_clips_from_config(path: str) -> list:
    """
    Load danh sách clip từ file JSON. Format:
    [
        {
            "path":       "clips/clip_01.mp4",
            "gt_start_s": 2.0,
            "gt_end_s":   8.5,
            "note":       "..."
        },
        ...
    ]
    """
    with open(path, encoding="utf-8") as f:
        clips = json.load(f)
    print(f"[Config] Loaded {len(clips)} clips từ {path}")
    return clips


async def upload_video(session: aiohttp.ClientSession,
                       jetson_url: str,
                       video_path: str,
                       backend_host: str) -> dict:
    """
    Gửi video lên Jetson MobileNet HTTP server qua POST /analyze-video.
    Field multipart:
        video        — file mp4
        backend_host — IP máy tính này (Jetson sẽ gửi WS về đây)
    Trả về { "job_id": "...", "status": "queued" }.
    """
    upload_url = f"{jetson_url}/analyze-video"
    with open(video_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field(
            "video", f,
            filename=os.path.basename(video_path),
            content_type="video/mp4",
        )
        data.add_field("backend_host", backend_host)
        async with session.post(upload_url, data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Upload thất bại [{resp.status}]: {text}")
            return await resp.json()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ĐÁNH GIÁ TỪNG CLIP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def evaluate_clip(session:       aiohttp.ClientSession,
                         jetson_url:    str,
                         ws_server:     WsResultServer,
                         backend_host:  str,
                         clip:          dict,
                         clip_no:       int,
                         total:         int) -> dict:
    """Chạy đánh giá 1 clip. Trả về dict kết quả."""
    path       = clip["path"]
    gt_start_s = clip["gt_start_s"]
    gt_end_s   = clip.get("gt_end_s")
    note       = clip.get("note", "")
    filename   = os.path.basename(path)

    print(f"\n{'─'*60}")
    print(f"[{clip_no}/{total}] {filename}  [MobileNetV2-TSM]")
    print(f"  Ground truth  : bạo lực bắt đầu @ {gt_start_s:.2f}s", end="")
    if gt_end_s is not None:
        print(f"  →  {gt_end_s:.2f}s  ({gt_end_s - gt_start_s:.1f}s)", end="")
    print()
    if note:
        print(f"  Ghi chú       : {note}")

    if not os.path.exists(path):
        print(f"  ❌ File không tồn tại: {path}")
        return {
            "filename":  filename,
            "gt_start_s": gt_start_s,
            "error":     "File không tồn tại",
            "verdict":   None,
            "latency_s": None,
            "detected":  False,
        }

    # ── Upload lên Jetson ─────────────────────────────────
    print(f"  Đang upload …")
    t_upload = time.time()
    try:
        resp = await upload_video(session, jetson_url, path, backend_host)
    except Exception as e:
        print(f"  ❌ Upload lỗi: {e}")
        return {
            "filename":  filename,
            "gt_start_s": gt_start_s,
            "error":     str(e),
            "verdict":   None,
            "latency_s": None,
            "detected":  False,
        }

    job_id      = resp["job_id"]
    upload_time = time.time() - t_upload
    print(f"  Job ID        : {job_id}")
    print(f"  Upload time   : {upload_time:.2f}s")
    print(f"  Đang phân tích (MobileNetV2-TSM) …")

    # ── Đăng ký nhận summary từ WS server ────────────────
    summary_queue = ws_server.register(job_id)
    try:
        summary = await asyncio.wait_for(summary_queue.get(), timeout=300.0)
    except asyncio.TimeoutError:
        ws_server.unregister(job_id)
        print(f"\n  ❌ Timeout — không nhận được summary sau 300s")
        return {
            "filename":  filename,
            "gt_start_s": gt_start_s,
            "job_id":    job_id,
            "error":     "Timeout",
            "verdict":   None,
            "latency_s": None,
            "detected":  False,
        }
    finally:
        ws_server.unregister(job_id)

    # ── Parse summary ─────────────────────────────────────
    verdict      = summary.get("verdict", "Normal")
    segments     = summary.get("violence_segments", [])
    duration_s   = summary.get("video_duration_s", 0)
    proc_time    = summary.get("processing_time_s", 0)
    ratio        = summary.get("violence_ratio", 0)
    total_frames = summary.get("total_frames", 0)
    engine_tag   = summary.get("engine", "mobilenet")

    # ── Tính latency ──────────────────────────────────────
    # latency = thời điểm phát hiện đầu tiên - ground truth start
    # Âm   = phát hiện sớm hơn GT (EMA/smoothing lan sang)
    # Dương = phát hiện trễ hơn GT (latency thực sự)
    detected      = False
    latency_s     = None
    detected_at_s = None

    if segments:
        detected_at_s = segments[0]["start_s"]
        latency_s     = round(detected_at_s - gt_start_s, 3)
        detected      = True

    # ── In kết quả ────────────────────────────────────────
    print(f"\n  ── Kết quả ({engine_tag}) ─────────────────────────")
    print(f"  Verdict       : {'🔴 VIOLENCE' if verdict == 'VIOLENCE' else '🟢 Normal'}")
    print(f"  Violence ratio: {ratio:.1%}")
    print(f"  Segments      : {len(segments)}")
    for i, seg in enumerate(segments):
        dur = seg['end_s'] - seg['start_s']
        print(f"    [{i+1}] {seg['start_s']:.2f}s → {seg['end_s']:.2f}s  "
              f"({dur:.2f}s)  max_prob={seg['max_prob']:.3f}")

    print(f"\n  ── Latency ───────────────────────────────────")
    if detected:
        print(f"  GT bắt đầu         : {gt_start_s:.2f}s")
        print(f"  Phát hiện đầu tiên : {detected_at_s:.2f}s")
        if latency_s >= 0:
            print(f"  ⏱ Latency          : +{latency_s:.3f}s  ← trễ so với GT")
        else:
            print(f"  ⏱ Latency          : {latency_s:.3f}s  ← phát hiện sớm hơn GT")
    else:
        print(f"  ❌ Không phát hiện được segment bạo lực nào!")
        if verdict == "VIOLENCE":
            print(f"     (verdict=VIOLENCE nhưng không có segment — bất thường)")

    print(f"  Xử lý         : {proc_time:.2f}s / {duration_s:.1f}s video "
          f"({proc_time / max(duration_s, 0.01) * 100:.0f}% realtime)  "
          f"| {total_frames} frames")

    return {
        "filename":          filename,
        "path":              path,
        "job_id":            job_id,
        "gt_start_s":        gt_start_s,
        "gt_end_s":          gt_end_s,
        "note":              note,
        "engine":            engine_tag,
        "verdict":           verdict,
        "violence_ratio":    ratio,
        "segments":          segments,
        "detected":          detected,
        "detected_at_s":     detected_at_s,
        "latency_s":         latency_s,
        "video_duration_s":  duration_s,
        "processing_time_s": proc_time,
        "total_frames":      total_frames,
        "upload_time_s":     round(upload_time, 3),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BẢNG TỔNG HỢP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_summary_table(results: list):
    """In bảng tổng hợp tất cả kết quả và thống kê."""
    rows = []
    for r in results:
        if r.get("error"):
            rows.append([r["filename"], "—", "—", "ERROR", r["error"][:30]])
            continue
        latency = r.get("latency_s")
        lat_str = (
            f"+{latency:.3f}s" if latency is not None and latency >= 0
            else f"{latency:.3f}s" if latency is not None
            else "NOT DETECTED"
        )
        rows.append([
            r["filename"],
            f"{r.get('gt_start_s', 0):.2f}s",
            f"{r['detected_at_s']:.2f}s" if r.get("detected_at_s") is not None else "—",
            lat_str,
            r.get("verdict", "?"),
        ])

    print(f"\n{'═'*60}")
    print("  BẢNG TỔNG HỢP LATENCY  [MobileNetV2-TSM]")
    print('═'*60)
    print(tabulate(
        rows,
        headers=["File", "GT start", "Detected", "Latency", "Verdict"],
        tablefmt="rounded_outline",
    ))

    # ── Thống kê ──────────────────────────────────────────
    valid    = [r for r in results if r.get("latency_s") is not None]
    detected = [r for r in results if r.get("detected")]
    missed   = [r for r in results if not r.get("error") and not r.get("detected")]
    errors   = [r for r in results if r.get("error")]

    if not valid:
        print("\n  Không có kết quả hợp lệ để thống kê.")
        return

    latencies = [r["latency_s"] for r in valid]
    avg_lat   = sum(latencies) / len(latencies)
    min_lat   = min(latencies)
    max_lat   = max(latencies)
    std_lat   = (sum((l - avg_lat) ** 2 for l in latencies) / len(latencies)) ** 0.5

    print(f"\n  Tổng clip       : {len(results)}")
    print(f"  Phát hiện đúng  : {len(detected)}/{len(results) - len(errors)}")
    print(f"  Bỏ sót          : {len(missed)}")
    print(f"  Lỗi             : {len(errors)}")
    print(f"\n  Latency (clip phát hiện được):")
    print(f"    Trung bình    : {avg_lat:+.3f}s")
    print(f"    Nhỏ nhất      : {min_lat:+.3f}s")
    print(f"    Lớn nhất      : {max_lat:+.3f}s")
    print(f"    Độ lệch chuẩn : {std_lat:.3f}s")

    # Thống kê thời gian xử lý (realtime ratio)
    proc_times = [r["processing_time_s"] for r in results
                  if r.get("processing_time_s") and r.get("video_duration_s")]
    if proc_times:
        durations  = [r["video_duration_s"] for r in results
                      if r.get("processing_time_s") and r.get("video_duration_s")]
        ratios     = [p / max(d, 0.01) * 100 for p, d in zip(proc_times, durations)]
        avg_ratio  = sum(ratios) / len(ratios)
        print(f"\n  Tốc độ xử lý (trung bình): {avg_ratio:.0f}% realtime")

    if missed:
        print(f"\n  Clip bị bỏ sót:")
        for r in missed:
            print(f"    - {r['filename']}  (verdict={r.get('verdict')})")

    print('═'*60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    args = parse_args()

    local_ip   = get_local_ip()
    jetson_url = f"http://{args.jetson}:{args.port}"

    clips = load_clips_from_config(args.config) if args.config else TEST_CLIPS
    if not clips:
        print("❌ Không có clip nào.")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  MobileNetV2-TSM Violence Detection — Latency Evaluator")
    print(f"  Jetson HTTP    : {jetson_url}")
    print(f"  Local IP       : {local_ip}  (Jetson sẽ gửi WS về đây)")
    print(f"  WS Server      : ws://0.0.0.0:{args.ws_port}"
          f"/ws/video-analysis/{{job_id}}")
    print(f"  Tổng clip      : {len(clips)}")
    print(f"  Delay giữa clip: {args.delay}s")
    print('═'*60)

    # Khởi động WS server nhận kết quả từ Jetson
    ws_server = WsResultServer(args.ws_port)
    await ws_server.start()

    results = []
    async with aiohttp.ClientSession() as session:
        for i, clip in enumerate(clips, 1):
            result = await evaluate_clip(
                session, jetson_url, ws_server,
                local_ip, clip, i, len(clips),
            )
            results.append(result)
            if i < len(clips):
                print(f"\n  Chờ {args.delay:.0f}s trước clip tiếp theo …")
                await asyncio.sleep(args.delay)

    await ws_server.stop()

    print_summary_table(results)

    output_data = {
        "model":     "MobileNetV2-TSM",
        "jetson":    args.jetson,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results":   results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Kết quả đã lưu → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())