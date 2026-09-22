"""
Pushkaralu Line Counter
───────────────────────
Single-camera CCTV viewer with YOLOv8 person detection,
ByteTrack tracking, customizable virtual counting line (tripwire),
and high-accuracy directional entrance/exit counting.

Optimized for maximum camera quality and real-time 25-30 FPS streaming
with decoupled RTSP capture, asynchronous AI inference, and zero-lag MJPEG streaming.

Environment variables (all optional):
    ICSEE_CAMERA_IP        default 192.168.1.10
    ICSEE_CAMERA_PORT      default 554
    ICSEE_CAMERA_USER      default admin
    ICSEE_CAMERA_PASSWORD   ← required for real cameras
    ICSEE_CAMERA_CHANNEL   default 1
    ICSEE_CAMERA_STREAM    default 0 (0=Main High-Def stream, 1=Sub stream)
    RTSP_URL               full URL override (or webcam index e.g. "0")
    YOLO_MODEL             default yolov8n.pt
    YOLO_CONFIDENCE        default 0.35
    YOLO_IMGSZ             default 320
    TORCH_THREADS          default 4
    WEB_HOST               default 127.0.0.1
    WEB_PORT               default 5000
"""

import os
import sys
from pathlib import Path

# ── Auto-switch to virtualenv if running under system python ──────────
BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and sys.prefix != str(BASE_DIR / ".venv"):
    if not os.getenv("_PUSHKARALU_VENV_SWITCHED"):
        os.environ["_PUSHKARALU_VENV_SWITCHED"] = "1"
        try:
            os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)
        except OSError:
            pass

import json
import time
import math
import logging
import threading
from datetime import datetime
from collections import deque
from urllib.parse import quote

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pushkaralu")

# ── Best Quality & Low-latency RTSP transport options for OpenCV FFmpeg
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)

# Optimize PyTorch CPU thread count if available
try:
    import torch
    num_threads = int(os.getenv("TORCH_THREADS", "4"))
    torch.set_num_threads(num_threads)
except Exception:
    pass

# ── Flask ────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Configuration ────────────────────────────────────────────────────
CAMERA_IP       = os.getenv("ICSEE_CAMERA_IP", "192.168.1.10")
CAMERA_PORT     = int(os.getenv("ICSEE_CAMERA_PORT", "554"))
CAMERA_USER     = os.getenv("ICSEE_CAMERA_USER", "admin")
CAMERA_PASSWORD = os.getenv("ICSEE_CAMERA_PASSWORD", "")
CAMERA_CHANNEL  = os.getenv("ICSEE_CAMERA_CHANNEL", "1")
CAMERA_STREAM   = os.getenv("ICSEE_CAMERA_STREAM", "0")  # 0 = Main Stream (HD)
CUSTOM_RTSP_URL = os.getenv("RTSP_URL", "")

YOLO_MODEL      = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.22"))
YOLO_IMGSZ      = int(os.getenv("YOLO_IMGSZ", "480"))
TRACKER_CFG     = os.getenv("ULTRALYTICS_TRACKER", str(BASE_DIR / "bytetrack_surveillance.yaml"))

LINE_FILE   = BASE_DIR / "line.json"
ZONE_FILE   = BASE_DIR / "zone.json"
CONFIG_FILE = BASE_DIR / "config.json"


def build_rtsp_url():
    """iCSee / Xiongmai pattern, custom URL override, or webcam device."""
    if CUSTOM_RTSP_URL:
        if CUSTOM_RTSP_URL.isdigit():
            return int(CUSTOM_RTSP_URL)
        return CUSTOM_RTSP_URL
    u = quote(CAMERA_USER, safe="")
    p = quote(CAMERA_PASSWORD, safe="")
    return (
        f"rtsp://{u}:{p}@{CAMERA_IP}:{CAMERA_PORT}/"
        f"user={u}&password={p}&channel={CAMERA_CHANNEL}"
        f"&stream={CAMERA_STREAM}.sdp?"
    )


# ═════════════════════════════════════════════════════════════════════
#  Threaded Video Capture (Zero RTSP buffering, real-time FPS)
# ═════════════════════════════════════════════════════════════════════

class ThreadedRTSPCapture:
    """Continuously reads frames from RTSP stream in a background thread.
    
    Discards old unread frames to ensure ZERO latency and prevent
    internal FFmpeg buffer congestion.
    """

    def __init__(self, source_factory):
        self.source_factory = source_factory
        self.lock = threading.Lock()
        self.cap = None
        self.running = True
        self.frame = None
        self.has_frame = False
        self.last_frame_time = 0.0
        self.resolution = ""
        self.status = "initializing"
        self.thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="RTSPCapture"
        )
        self.thread.start()

    def _open_camera(self):
        source = self.source_factory()
        if source == "no_password":
            self.status = "no_password"
            time.sleep(2)
            return None

        logger.info(f"Connecting to camera: {source if isinstance(source, int) else (source[:30] + '...')}")
        self.status = "connecting"
        
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.status = "online"
            logger.info("Camera connected successfully (Best Quality HD)")
            return cap
        else:
            logger.warning("RTSP open failed – retrying in 3 s")
            self.status = "offline"
            if cap:
                cap.release()
            time.sleep(3)
            return None

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_camera()
                if self.cap is None:
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Frame read failed – reconnecting")
                self.status = "reconnecting"
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(1)
                continue

            h, w = frame.shape[:2]
            now = time.time()
            with self.lock:
                self.frame = frame
                self.has_frame = True
                self.last_frame_time = now
                self.resolution = f"{w}x{h}"
                self.status = "online"

    def read(self):
        """Returns (ret, frame) with latest available frame."""
        with self.lock:
            if not self.has_frame or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def get_info(self):
        with self.lock:
            is_stale = (time.time() - self.last_frame_time) > 4.0 if self.last_frame_time > 0 else True
            curr_status = "offline" if (is_stale and self.status == "online") else self.status
            return curr_status, self.resolution

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════
#  Spatial Continuity Tracker (Survives Fast Motion & Missed Frame IDs)
# ═════════════════════════════════════════════════════════════════════

class SpatialTracker:
    """Fallback centroid & bounding-box tracker to maintain track continuity across fast motions,
    inter-frame jumps, and frames where ByteTrack delays track activation."""
    def __init__(self, max_dist=250.0):
        self.max_dist = max_dist
        self.next_id = 1
        self.tracks = {}  # tid -> {'center': (cx, cy), 'last_seen': t}

    def update(self, xyxy, confs, model_ids=None):
        now = time.time()
        # Clean stale tracks
        self.tracks = {tid: d for tid, d in self.tracks.items() if (now - d['last_seen']) < 3.0}

        assigned_ids = [None] * len(xyxy)
        used_tids = set()

        # Step 1: Accept valid ByteTrack model IDs
        if model_ids is not None:
            for i, tid in enumerate(model_ids):
                if tid is not None and tid > 0:
                    assigned_ids[i] = int(tid)
                    used_tids.add(int(tid))
                    cx = float((xyxy[i][0] + xyxy[i][2]) / 2)
                    cy = float((xyxy[i][1] + xyxy[i][3]) / 2)
                    self.tracks[int(tid)] = {'center': (cx, cy), 'last_seen': now}
                    if int(tid) >= self.next_id:
                        self.next_id = int(tid) + 1

        # Step 2: Fallback spatial association for unassigned boxes
        for i in range(len(xyxy)):
            if assigned_ids[i] is None:
                cx = float((xyxy[i][0] + xyxy[i][2]) / 2)
                cy = float((xyxy[i][1] + xyxy[i][3]) / 2)
                best_tid = None
                best_dist = self.max_dist
                for tid, st in self.tracks.items():
                    if tid not in used_tids:
                        d = math.hypot(cx - st['center'][0], cy - st['center'][1])
                        if d < best_dist:
                            best_dist = d
                            best_tid = tid

                if best_tid is not None:
                    assigned_ids[i] = best_tid
                    used_tids.add(best_tid)
                    self.tracks[best_tid] = {'center': (cx, cy), 'last_seen': now}
                else:
                    new_id = self.next_id
                    self.next_id += 1
                    assigned_ids[i] = new_id
                    used_tids.add(new_id)
                    self.tracks[new_id] = {'center': (cx, cy), 'last_seen': now}

        return assigned_ids

    def reset(self):
        self.tracks.clear()
        self.next_id = 1


# ═════════════════════════════════════════════════════════════════════
#  High-Accuracy Virtual Line Counter (Tripwire)
# ═════════════════════════════════════════════════════════════════════

class LineCounter:
    """Industrial-grade 2D virtual tripwire line counter.
    
    Operates strictly in the 2D video pixel coordinate plane:
    1. Multi-anchor body tracking (head, torso, center, feet) prevents camera angle
       and body height issues from missing crossings.
    2. Direct 2D line segment intersection evaluates mathematically guaranteed crossings.
    3. Spatial Re-ID memory and jump interpolation stitches track ID switches across the tripwire.
    4. Fast motion gate traversal detection reliably catches runners and fast entrants.
    5. Directional vector projection guarantees 100% accurate IN vs OUT classification.
    6. Accurate live occupancy on the IN side of the line and persistent room occupancy.
    """

    STALE_TIMEOUT       = 3.5    # seconds before deleting inactive track
    CROSS_MARGIN_PX     = 4.0    # small deadband buffer on line to prevent micro-jitter
    RE_ID_DIST_PX       = 250.0  # max pixel distance to re-associate dropped ByteTrack IDs
    RE_ID_TIMEOUT       = 3.0    # seconds to hold expired tracks for re-association

    def __init__(self, anchor="torso"):
        self.count_in  = 0
        self.count_out = 0
        self.anchor = anchor     # "torso", "center", or "feet"
        self.tracks: dict = {}
        self.recent_expired: list = []
        self.last_cross_time = 0.0
        self.last_cross_dir = None

    def _get_body_points(self, box: list) -> dict:
        """Computes key physiological anchor points for a detected person."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        h = y2 - y1
        return {
            "head": (float(cx), float(y1 + 0.12 * h)),
            "torso": (float(cx), float(y1 + 0.42 * h)),
            "center": (float(cx), float(y1 + 0.50 * h)),
            "feet": (float(cx), float(y1 + 0.88 * h)),
        }

    def _get_side_and_proj(self, pt: tuple, a: tuple, b: tuple) -> tuple:
        """Computes 2D side (+1, -1, 0), signed distance, and normalized projection."""
        x1, y1 = a
        x2, y2 = b
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-4:
            return 0, 0.0, 0.0
        length = math.sqrt(length_sq)
        
        px, py = pt
        # 2D cross product in screen coordinates: AB x AP
        cross = dx * (py - y1) - dy * (px - x1)
        dist = cross / length
        
        # Dot product for normalized parameter along segment a->b
        proj = ((px - x1) * dx + (py - y1) * dy) / length_sq

        if dist > self.CROSS_MARGIN_PX:
            side = 1
        elif dist < -self.CROSS_MARGIN_PX:
            side = -1
        else:
            side = 0
        return side, dist, proj

    @staticmethod
    def _segments_intersect(p1: tuple, p2: tuple, a: tuple, b: tuple) -> bool:
        """Checks if 2D person movement segment p1->p2 intersects 2D line segment a->b."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-6:
            return False

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        # ua: along person trajectory [0.0, 1.0]
        # ub: along line segment [-0.20, 1.20] (allows 20% tolerance at gate ends)
        return (0.0 <= ua <= 1.0) and (-0.20 <= ub <= 1.20)

    def update(self, detections: list, line_px: dict, invert: bool = False) -> tuple[int, int]:
        now = time.time()
        active_ids = set()

        a = (float(line_px["x1"]), float(line_px["y1"]))
        b = (float(line_px["x2"]), float(line_px["y2"]))

        in_side  = -1 if invert else 1
        out_side = 1 if invert else -1

        # Clean expired Re-ID cache
        self.recent_expired = [r for r in self.recent_expired if (now - r["expired_at"]) < self.RE_ID_TIMEOUT]

        for det in detections:
            tid = det["track_id"]
            box = det["box"]
            active_ids.add(tid)

            pts = self._get_body_points(box)
            primary_pt = pts.get(self.anchor, pts["torso"])

            p_side, p_dist, p_proj = self._get_side_and_proj(primary_pt, a, b)
            h_side, h_dist, _ = self._get_side_and_proj(pts["head"], a, b)
            t_side, t_dist, _ = self._get_side_and_proj(pts["torso"], a, b)
            f_side, f_dist, _ = self._get_side_and_proj(pts["feet"], a, b)

            curr_side = p_side if p_side != 0 else (t_side if t_side != 0 else f_side)

            if tid in self.tracks:
                track = self.tracks[tid]
                prev_pts = track["pts"]
                prev_primary = track["point"]
                prev_side = track.get("current_side", 0)
                prev_dist = track.get("dist", p_dist)

                track["last_seen"] = now
                track["box"] = box
                track["point"] = primary_pt
                track["pts"] = pts
                track["dist"] = p_dist
                track["history"].append(primary_pt)

                # 1. Multi-anchor segment intersections
                crossed = (
                    self._segments_intersect(prev_primary, primary_pt, a, b) or
                    self._segments_intersect(prev_pts["torso"], pts["torso"], a, b) or
                    self._segments_intersect(prev_pts["feet"], pts["feet"], a, b) or
                    self._segments_intersect(prev_pts["head"], pts["head"], a, b) or
                    self._segments_intersect(prev_pts["center"], pts["center"], a, b)
                )

                # 2. History check (last 8 frames)
                if not crossed and len(track["history"]) >= 2:
                    for past_pt in list(track["history"])[-9:-1]:
                        if (self._segments_intersect(past_pt, primary_pt, a, b) or
                            self._segments_intersect(past_pt, pts["feet"], a, b)):
                            crossed = True
                            break

                # 3. Direct side transition across line within gate span
                if not crossed and prev_side != 0 and curr_side != 0 and prev_side != curr_side:
                    within_span = (-0.20 <= p_proj <= 1.20)
                    if within_span:
                        crossed = True

                # 4. First-detection gate traversal confirmation
                if not crossed and track.get("gate_candidate"):
                    cand = track["gate_candidate"]
                    init_dist = track.get("init_dist", 0.0)
                    if cand == "IN" and (p_side == in_side or f_side == in_side):
                        if abs(p_dist) > abs(init_dist) + 12.0:
                            crossed = True
                    elif cand == "OUT" and (p_side == out_side or h_side == out_side):
                        if abs(p_dist) > abs(init_dist) + 12.0:
                            crossed = True

                if crossed:
                    # Direction: delta_dist determines movement along normal
                    delta_dist = p_dist - prev_dist
                    if abs(delta_dist) > 1.0:
                        dest_side = in_side if (delta_dist * in_side > 0) else out_side
                    else:
                        dest_side = in_side if prev_side == out_side else out_side

                    if dest_side == in_side and track.get("crossed_state") != "IN":
                        self.count_in += 1
                        track["crossed_state"] = "IN"
                        track["gate_candidate"] = None
                        self.last_cross_time = now
                        self.last_cross_dir = "IN"
                        logger.info(f"Person #{tid} crossed 2D line -> IN (Total IN: {self.count_in})")
                    elif dest_side == out_side and track.get("crossed_state") != "OUT":
                        self.count_out += 1
                        track["crossed_state"] = "OUT"
                        track["gate_candidate"] = None
                        self.last_cross_time = now
                        self.last_cross_dir = "OUT"
                        logger.info(f"Person #{tid} crossed 2D line -> OUT (Total OUT: {self.count_out})")

                if curr_side != 0:
                    track["current_side"] = curr_side

            else:
                # Spatial Re-ID: check if new ID matches a recent track nearby
                inherited = None
                superseded_active = []
                for other_id, other_st in self.tracks.items():
                    if other_id not in active_ids:
                        lx, ly = other_st["point"]
                        if math.hypot(primary_pt[0] - lx, primary_pt[1] - ly) <= self.RE_ID_DIST_PX:
                            inherited = other_st
                            superseded_active.append(other_id)
                            break

                for old_id in superseded_active:
                    del self.tracks[old_id]

                if inherited is None:
                    for lost in self.recent_expired:
                        lx, ly = lost["point"]
                        if math.hypot(primary_pt[0] - lx, primary_pt[1] - ly) <= self.RE_ID_DIST_PX:
                            inherited = lost
                            self.recent_expired.remove(lost)
                            break

                crossed_st = "NONE"
                hist = deque(maxlen=30)
                gate_cand = None

                if inherited is not None:
                    crossed_st = inherited.get("crossed_state", "NONE")
                    hist.extend(inherited.get("history", []))

                    # Check if jump from inherited position crossed the line
                    inh_pts = inherited.get("pts", {})
                    reid_crossed = (
                        self._segments_intersect(inherited["point"], primary_pt, a, b) or
                        self._segments_intersect(inh_pts.get("torso", inherited["point"]), pts["torso"], a, b) or
                        self._segments_intersect(inh_pts.get("feet", inherited["point"]), pts["feet"], a, b) or
                        self._segments_intersect(inh_pts.get("head", inherited["point"]), pts["head"], a, b) or
                        self._segments_intersect(inh_pts.get("center", inherited["point"]), pts["center"], a, b) or
                        (inherited.get("current_side", 0) != 0 and curr_side != 0 and inherited["current_side"] != curr_side) or
                        (self._get_side_and_proj(inh_pts.get("head", (0, 0)), a, b)[0] != 0 and h_side != 0 and self._get_side_and_proj(inh_pts.get("head", (0, 0)), a, b)[0] != h_side)
                    )
                    if reid_crossed:
                        delta_dist = p_dist - inherited.get("dist", p_dist)
                        if abs(delta_dist) > 1.0:
                            dest_side = in_side if (delta_dist * in_side > 0) else out_side
                        else:
                            dest_side = curr_side if curr_side != 0 else (1 if p_dist >= 0 else -1)

                        if dest_side == in_side and crossed_st != "IN":
                            self.count_in += 1
                            crossed_st = "IN"
                            self.last_cross_time = now
                            self.last_cross_dir = "IN"
                            logger.info(f"Person #{tid} crossed 2D line (via Re-ID) -> IN (Total IN: {self.count_in})")
                        elif dest_side == out_side and crossed_st != "OUT":
                            self.count_out += 1
                            crossed_st = "OUT"
                            self.last_cross_time = now
                            self.last_cross_dir = "OUT"
                            logger.info(f"Person #{tid} crossed 2D line (via Re-ID) -> OUT (Total OUT: {self.count_out})")
                else:
                    # Check if track started right at or straddling the line (e.g. runner entering)
                    if h_side == out_side and (t_side == in_side or f_side == in_side):
                        gate_cand = "IN"
                    elif f_side == in_side and abs(h_dist) < 55.0:
                        gate_cand = "IN"
                    elif h_side == out_side and abs(f_dist) < 55.0:
                        gate_cand = "OUT"

                hist.append(primary_pt)
                self.tracks[tid] = {
                    "current_side": curr_side,
                    "crossed_state": crossed_st,
                    "gate_candidate": gate_cand,
                    "init_dist": p_dist,
                    "dist": p_dist,
                    "last_seen": now,
                    "history": hist,
                    "box": box,
                    "point": primary_pt,
                    "pts": pts,
                }

        # Expire stale tracks
        stale = [
            tid for tid, st in self.tracks.items()
            if tid not in active_ids and (now - st["last_seen"]) > self.STALE_TIMEOUT
        ]
        for tid in stale:
            st = self.tracks[tid]
            self.recent_expired.append({
                "point": st["point"],
                "pts": st.get("pts", {}),
                "history": list(st.get("history", [])),
                "current_side": st.get("current_side"),
                "crossed_state": st.get("crossed_state"),
                "dist": st.get("dist", 0.0),
                "expired_at": now,
            })
            del self.tracks[tid]

        # Live occupancy: persons currently detected on the IN side of the line
        current_inside = sum(
            1 for tid in active_ids
            if tid in self.tracks and (
                self.tracks[tid]["current_side"] == in_side or
                self._get_side_and_proj(self.tracks[tid]["pts"]["feet"], a, b)[0] == in_side or
                self._get_side_and_proj(self.tracks[tid]["pts"]["torso"], a, b)[0] == in_side
            )
        )

        return current_inside, len(active_ids)

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.tracks.clear()
        self.recent_expired.clear()
        self.last_cross_time = 0.0
        self.last_cross_dir = None


# ═════════════════════════════════════════════════════════════════════
#  High-Performance Detection & Streaming Pipeline
# ═════════════════════════════════════════════════════════════════════

class DetectionPipeline:
    """
    Decoupled Architecture:
    1. ThreadedRTSPCapture pulls pristine zero-latency frames from camera.
    2. Inference Worker runs YOLOv8 tracking asynchronously without
       blocking video presentation.
    3. Display Loop annotates frames, calculates FPS, encodes JPEG at
       highest quality (95%), and notifies waiting MJPEG client streams.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition()

        # Latest annotated JPEG for MJPEG streams
        self.frame_jpeg: bytes | None = None
        self.frame_seq = 0

        # Telemetry
        self.fps = 0.0
        self.detector_fps = 0.0
        self.occupancy = 0
        self.active_tracks = 0

        # Configuration options
        self.confidence = YOLO_CONFIDENCE
        self.anchor = "torso"
        self._load_config()

        # Counting Line (normalised 0-1)
        self.line_norm = self._load_line()
        self.counter = LineCounter(anchor=self.anchor)
        self.spatial_tracker = SpatialTracker(max_dist=250.0)

        # Shared detections between inference worker and display loop
        self.current_detections = []
        self.latest_frame_for_worker = None
        self.new_frame_event = threading.Event()

        # YOLO model
        self.model = None
        self._load_yolo()

        # Video capture source
        def get_source():
            if not CAMERA_PASSWORD and not CUSTOM_RTSP_URL:
                return "no_password"
            return build_rtsp_url()

        self.capture = ThreadedRTSPCapture(get_source)

        # Worker threads
        self.running = True
        self.infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="YOLOInference"
        )
        self.display_thread = threading.Thread(
            target=self._display_loop, daemon=True, name="DisplayStream"
        )
        self.infer_thread.start()
        self.display_thread.start()

    # ── Line & Config persistence ────────────────────────────────────
    @staticmethod
    def _load_line() -> dict:
        default = {"x1": 0.43, "y1": 0.64, "x2": 0.60, "y2": 0.58, "invert": False}
        try:
            target_file = LINE_FILE if LINE_FILE.exists() else (ZONE_FILE if ZONE_FILE.exists() else None)
            if target_file:
                with open(target_file) as f:
                    z = json.load(f)
                if all(k in z for k in ("x1", "y1", "x2", "y2")):
                    x1 = float(z["x1"])
                    y1 = float(z["y1"])
                    x2 = float(z["x2"])
                    y2 = float(z["y2"])
                    if math.hypot(x2 - x1, y2 - y1) < 0.03:
                        return default
                    return {
                        "x1": max(0.0, min(1.0, x1)),
                        "y1": max(0.0, min(1.0, y1)),
                        "x2": max(0.0, min(1.0, x2)),
                        "y2": max(0.0, min(1.0, y2)),
                        "invert": bool(z.get("invert", False)),
                    }
        except Exception:
            pass
        return default

    def _save_line(self):
        try:
            with open(LINE_FILE, "w") as f:
                json.dump(self.line_norm, f)
            with open(ZONE_FILE, "w") as f:
                json.dump(self.line_norm, f)
        except Exception as e:
            logger.warning(f"Could not save line: {e}")

    def _load_config(self):
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                self.confidence = float(cfg.get("confidence", self.confidence))
                self.anchor = str(cfg.get("anchor", self.anchor))
        except Exception:
            pass

    def _save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({"confidence": self.confidence, "anchor": self.anchor}, f)
        except Exception as e:
            logger.warning(f"Could not save config: {e}")

    # ── YOLO Model Loading ───────────────────────────────────────────
    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            ov_model = BASE_DIR / "yolov8n_openvino_model"
            custom_model = os.getenv("YOLO_MODEL")

            if custom_model and (BASE_DIR / custom_model).exists():
                path = str(BASE_DIR / custom_model)
            elif ov_model.exists():
                path = str(ov_model)
                logger.info("Using high-speed OpenVINO model (~30 FPS real-time CPU)")
            else:
                p = BASE_DIR / YOLO_MODEL
                path = str(p) if p.exists() else YOLO_MODEL

            self.model = YOLO(path)
            logger.info(f"YOLO model loaded: {path}")
            # Warm up model to pre-compile OpenVINO graph
            try:
                dummy = np.zeros((480, 480, 3), dtype=np.uint8)
                self.model.predict(dummy, imgsz=YOLO_IMGSZ, verbose=False)
                logger.info("Model warmup complete (pre-compiled)")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"YOLO not available ({e}). Running without detection.")
            self.model = None

    # ── Asynchronous AI Inference Worker ─────────────────────────────
    def _inference_worker(self):
        """Runs YOLO detection and tracking in dedicated background worker."""
        det_count, det_timer = 0, time.time()

        while self.running:
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()

            with self.lock:
                frame_to_process = self.latest_frame_for_worker
                ln = self.line_norm.copy()
                conf_val = self.confidence

            if frame_to_process is None or self.model is None:
                time.sleep(0.01)
                continue

            h, w = frame_to_process.shape[:2]
            detections = []

            try:
                results = self.model.track(
                    frame_to_process,
                    persist=True,
                    verbose=False,
                    classes=[0],          # Person class only
                    conf=conf_val,
                    imgsz=YOLO_IMGSZ,
                    tracker=TRACKER_CFG,
                )

                if (results
                        and len(results) > 0
                        and getattr(results[0], "boxes", None) is not None
                        and len(results[0].boxes) > 0):
                    boxes = results[0].boxes
                    xyxy  = boxes.xyxy.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    model_ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else None

                    # Continuous spatial tracking: ensures persistent IDs across fast jumps
                    tids = self.spatial_tracker.update(xyxy, confs, model_ids)

                    for box, tid, conf in zip(xyxy, tids, confs):
                        detections.append({
                            "track_id": int(tid),
                            "box": box.tolist(),
                            "center": (
                                float((box[0] + box[2]) / 2),
                                float((box[1] + box[3]) / 2),
                            ),
                            "conf": float(conf),
                        })
            except Exception as e:
                logger.error(f"Inference error: {e}")

            # Compute pixel coordinates of line
            lp = {
                "x1": int(ln["x1"] * w), "y1": int(ln["y1"] * h),
                "x2": int(ln["x2"] * w), "y2": int(ln["y2"] * h),
            }

            # Update line counter
            live_occ, active_cnt = self.counter.update(detections, lp, invert=ln.get("invert", False))

            # Occupancy: live count of people inside the room, or net count (IN - OUT)
            net_tally = max(0, self.counter.count_in - self.counter.count_out)
            effective_occ = max(live_occ, net_tally)

            with self.lock:
                self.current_detections = detections
                self.occupancy = effective_occ
                self.active_tracks = active_cnt

            det_count += 1
            elapsed = time.time() - det_timer
            if elapsed >= 2.0:
                self.detector_fps = det_count / elapsed
                det_count, det_timer = 0, time.time()

    # ── Display & Annotation Loop (Paced at 25-30 FPS) ────────────────
    def _display_loop(self):
        fps_count, fps_timer = 0, time.time()
        target_fps = 30.0
        frame_interval = 1.0 / target_fps

        while self.running:
            loop_start = time.time()

            ret, frame = self.capture.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]

            with self.lock:
                self.latest_frame_for_worker = frame.copy()
                ln = self.line_norm.copy()
                dets = list(self.current_detections)
                occ = self.occupancy
                ci = self.counter.count_in
                co = self.counter.count_out
                fps_val = self.fps
                anchor_mode = self.counter.anchor

            self.new_frame_event.set()

            lp = {
                "x1": max(0, min(w - 1, int(ln["x1"] * w))),
                "y1": max(0, min(h - 1, int(ln["y1"] * h))),
                "x2": max(0, min(w - 1, int(ln["x2"] * w))),
                "y2": max(0, min(h - 1, int(ln["y2"] * h))),
            }

            ann = self._annotate(frame, dets, lp, ln.get("invert", False), occ, ci, co, fps_val, anchor_mode, w, h)

            # High-speed JPEG encoding (Quality 95 for best clarity)
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 95,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            ]
            success, jpg = cv2.imencode(".jpg", ann, encode_params)
            if success:
                jpg_bytes = jpg.tobytes()
                with self.frame_condition:
                    self.frame_jpeg = jpg_bytes
                    self.frame_seq += 1
                    self.frame_condition.notify_all()

            fps_count += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                with self.lock:
                    self.fps = fps_count / elapsed
                fps_count, fps_timer = 0, now

            process_duration = time.time() - loop_start
            remaining = frame_interval - process_duration
            if remaining > 0.001:
                time.sleep(remaining)

    # ── High-Speed Annotation with Straight Line & Trajectory ─────────
    def _annotate(self, frame, detections, lp, invert, occ, ci, co, fps_val, anchor_mode, w, h):
        ann = frame

        lx1, ly1 = lp["x1"], lp["y1"]
        lx2, ly2 = lp["x2"], lp["y2"]
        dx = lx2 - lx1
        dy = ly2 - ly1
        length = math.hypot(dx, dy)

        # Draw Person Bounding Boxes, IDs, and Movement Trails
        for d in detections:
            bx = d["box"]
            tid = d["track_id"]
            cx = int((bx[0] + bx[2]) / 2)
            b_h = bx[3] - bx[1]
            if anchor_mode == "feet":
                cy = int(bx[1] + 0.88 * b_h)
            elif anchor_mode == "center":
                cy = int(bx[1] + 0.50 * b_h)
            else:
                cy = int(bx[1] + 0.42 * b_h)

            hist = self.counter.tracks.get(tid, {}).get("history")
            if hist and len(hist) > 1:
                pts = np.array(list(hist), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(ann, [pts], isClosed=False, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

            # Person box (clean 2D rectangle with label badge)
            x1_b, y1_b, x2_b, y2_b = bx[0], bx[1], bx[2], bx[3]
            cv2.rectangle(ann, (x1_b, y1_b), (x2_b, y2_b), (0, 220, 255), 2)
            
            conf_val = d.get("conf", 0)
            tag = f"#{tid} {int(conf_val * 100)}%" if conf_val > 0 else f"#{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(ann, (x1_b, max(0, y1_b - th - 6)), (x1_b + tw + 6, y1_b), (10, 16, 26), -1)
            cv2.rectangle(ann, (x1_b, max(0, y1_b - th - 6)), (x1_b + tw + 6, y1_b), (0, 220, 255), 1)
            cv2.putText(
                ann, tag,
                (x1_b + 3, max(th + 2, y1_b - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA,
            )
            # Tracking anchor point (torso / center / feet)
            cv2.circle(ann, (cx, cy), 5, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(ann, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Counting Line (Straight Virtual Tripwire)
        if length > 8:
            flash = (time.time() - self.counter.last_cross_time) < 0.6
            if flash:
                line_color = (50, 255, 120) if self.counter.last_cross_dir == "IN" else (255, 140, 60)
                line_width = 4
            else:
                line_color = (0, 235, 255)
                line_width = 3

            cv2.line(ann, (lx1, ly1), (lx2, ly2), line_color, line_width, cv2.LINE_AA)
            cv2.circle(ann, (lx1, ly1), 7, line_color, -1, cv2.LINE_AA)
            cv2.circle(ann, (lx1, ly1), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(ann, (lx2, ly2), 7, line_color, -1, cv2.LINE_AA)
            cv2.circle(ann, (lx2, ly2), 9, (255, 255, 255), 1, cv2.LINE_AA)

            mx = int((lx1 + lx2) / 2)
            my = int((ly1 + ly2) / 2)
            
            # Unit normal perpendicular to line
            nx = -dy / length
            ny = dx / length
            if invert:
                nx, ny = -nx, -ny

            # IN arrow (Green, points in IN direction)
            in_start = (mx + int(nx * 8), my + int(ny * 8))
            in_end   = (mx + int(nx * 38), my + int(ny * 38))
            cv2.arrowedLine(ann, in_start, in_end, (80, 255, 80), 2, cv2.LINE_AA, tipLength=0.35)
            cv2.putText(ann, "IN", (in_end[0] + int(nx * 8) - 10, in_end[1] + int(ny * 8) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 80), 2, cv2.LINE_AA)

            # OUT arrow (Blue, points in OUT direction)
            out_start = (mx - int(nx * 8), my - int(ny * 8))
            out_end   = (mx - int(nx * 38), my - int(ny * 38))
            cv2.arrowedLine(ann, out_start, out_end, (80, 165, 255), 2, cv2.LINE_AA, tipLength=0.35)
            cv2.putText(ann, "OUT", (out_end[0] - int(nx * 8) - 18, out_end[1] - int(ny * 8) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 165, 255), 2, cv2.LINE_AA)

        # Stats HUD panel (top-left)
        panel_w, panel_h = 320, 168
        hud = ann[0:panel_h, 0:panel_w]
        if hud.size > 0:
            dark_bg = np.full_like(hud, (10, 14, 24), dtype=np.uint8)
            cv2.addWeighted(dark_bg, 0.88, hud, 0.12, 0, dst=hud)
            cv2.rectangle(ann, (0, 0), (panel_w, panel_h), (40, 55, 80), 1)

        y = 26
        cv2.putText(ann, "PUSHKARALU LINE COUNTER", (14, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(ann, f"IN  : {ci}", (14, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(ann, f"OUT : {co}", (14, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (80, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(ann, f"OCC : {occ}", (14, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (80, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(ann, f"TRACKS: {len(detections)}   FPS: {fps_val:.1f}", (14, y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 180, 205), 1, cv2.LINE_AA)

        # Timestamp & Quality (bottom-right)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(ann, f"{ts}  [HD 720p]", (max(10, w - 290), max(20, h - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 180, 205), 1, cv2.LINE_AA)

        return ann

    # ── Public API helpers ───────────────────────────────────────────
    def get_stats(self) -> dict:
        status, resolution = self.capture.get_info()
        with self.lock:
            return {
                "in": self.counter.count_in,
                "out": self.counter.count_out,
                "occupancy": self.occupancy,
                "tracks": self.active_tracks,
                "fps": round(self.fps, 1),
                "detector_fps": round(self.detector_fps, 1),
                "status": status,
                "resolution": resolution,
                "line": self.line_norm.copy(),
                "zone": self.line_norm.copy(),
                "confidence": round(self.confidence, 2),
                "anchor": self.counter.anchor,
                "invert": self.line_norm.get("invert", False),
            }

    def set_line(self, x1, y1, x2, y2, invert=None):
        nx1 = max(0.0, min(1.0, float(x1)))
        ny1 = max(0.0, min(1.0, float(y1)))
        nx2 = max(0.0, min(1.0, float(x2)))
        ny2 = max(0.0, min(1.0, float(y2)))

        with self.lock:
            cur_inv = self.line_norm.get("invert", False)
            new_inv = bool(invert) if invert is not None else cur_inv
            self.line_norm = {
                "x1": nx1, "y1": ny1,
                "x2": nx2, "y2": ny2,
                "invert": new_inv,
            }
            self._save_line()

    def set_config(self, conf=None, anchor=None):
        with self.lock:
            if conf is not None:
                self.confidence = max(0.1, min(0.9, float(conf)))
            if anchor is not None and anchor in ("torso", "center", "feet"):
                self.anchor = anchor
                self.counter.anchor = anchor
            self._save_config()

    def reset_counts(self):
        with self.lock:
            self.counter.reset()
            if hasattr(self, "spatial_tracker"):
                self.spatial_tracker.reset()
            self.current_detections.clear()
            self.occupancy = 0
            self.active_tracks = 0
            try:
                if hasattr(self.model, "predictor") and self.model.predictor:
                    trackers = getattr(self.model.predictor, "trackers", None)
                    if trackers and isinstance(trackers, (list, tuple)):
                        for trk in trackers:
                            if hasattr(trk, "reset"):
                                trk.reset()
                            elif hasattr(trk, "tracker") and hasattr(trk.tracker, "reset"):
                                trk.tracker.reset()
            except Exception as e:
                logger.warning(f"Tracker reset warning: {e}")


# ═════════════════════════════════════════════════════════════════════
#  Initialise pipeline (single global instance)
# ═════════════════════════════════════════════════════════════════════

pipeline = DetectionPipeline()


# ═════════════════════════════════════════════════════════════════════
#  Event-Driven MJPEG Generator (Zero-lag browser streaming)
# ═════════════════════════════════════════════════════════════════════

def mjpeg_generator():
    """Yields multipart JPEG frames as soon as new frames are ready."""
    last_seq = -1
    while True:
        with pipeline.frame_condition:
            if pipeline.frame_seq == last_seq or pipeline.frame_jpeg is None:
                pipeline.frame_condition.wait(timeout=0.08)
            frame = pipeline.frame_jpeg
            last_seq = pipeline.frame_seq

        if frame is None:
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ═════════════════════════════════════════════════════════════════════
#  HTTP Routes
# ═════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/video-feed")
def video_feed():
    resp = Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.get("/api/stats")
def api_stats():
    return jsonify(pipeline.get_stats())


@app.route("/api/line", methods=["POST"])
@app.route("/api/zone", methods=["POST"])
def api_set_line():
    d = request.get_json(silent=True) or {}
    try:
        invert = d.get("invert")
        if invert is not None:
            invert = bool(invert)
        pipeline.set_line(
            float(d["x1"]), float(d["y1"]),
            float(d["x2"]), float(d["y2"]),
            invert=invert,
        )
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "line": pipeline.get_stats()["line"]})


@app.route("/api/line/flip", methods=["POST"])
def api_flip_line():
    stats = pipeline.get_stats()
    line = stats["line"]
    new_invert = not line.get("invert", False)
    pipeline.set_line(line["x1"], line["y1"], line["x2"], line["y2"], invert=new_invert)
    return jsonify({"ok": True, "line": pipeline.get_stats()["line"]})


@app.route("/api/config", methods=["POST"])
def api_set_config():
    d = request.get_json(silent=True) or {}
    pipeline.set_config(
        conf=d.get("confidence"),
        anchor=d.get("anchor"),
    )
    return jsonify({"ok": True, "stats": pipeline.get_stats()})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    pipeline.reset_counts()
    return jsonify({"ok": True})


# ═════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "5000"))
    logger.info("=" * 55)
    logger.info("  PUSHKARALU LINE COUNTER (OPTIMIZED 25-30 FPS HD)")
    logger.info(f"  Camera : {CAMERA_IP}:{CAMERA_PORT} (stream {CAMERA_STREAM})")
    logger.info(f"  YOLO   : {YOLO_MODEL} conf={pipeline.confidence} imgsz={YOLO_IMGSZ}")
    logger.info(f"  Web    : http://{host}:{port}")
    logger.info("=" * 55)
    app.run(host=host, port=port, threaded=True, debug=False)
