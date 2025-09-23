# app.py
# Import the web app FIRST so eventlet is monkey-patched before anything else.
from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os
import cv2
import time
import math
import sqlite3
import threading
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from ultralytics import YOLO

# Keep OpenCV predictable/light on Pi
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# --------------------- FLAGS ---------------------
SHOW_WINDOWS = False  # keep False on headless
PRINT_DEBUG = True

"""
Cameras:
  Pass either numeric indices (0,1,2) or stable device paths:
  /dev/videoN  OR  /dev/v4l/by-id/usb-...-video-index0

  Set via env:
    SC_CAM_PED, SC_CAM_VEH, SC_CAM_TL
"""

# --------------------- CONFIGURATION ---------------------
# YOLO
YOLO_MODEL = os.getenv("SC_YOLO_MODEL", "yolov8n.pt")
YOLO_CONF = float(os.getenv("SC_YOLO_CONF", "0.35"))
YOLO_IMG_SZ = int(os.getenv("SC_YOLO_IMG", "416"))

# Classes
PEDESTRIAN_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

# Frames (lighter for 3 cams)
FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS_TARGET = int(os.getenv("SC_FPS", "12"))
FRAME_TIME = 1.0 / max(1, FPS_TARGET)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "2"))  # infer every Nth frame (default 2)

# LED matrix (guarded init below)
LED_CASCADE = int(os.getenv("SC_LED_CASCADE", "4"))
LED_ORIENTATION = int(os.getenv("SC_LED_ORIENTATION", "-90"))

# Vehicle distance & speed calibration
PIXELS_PER_METER_VEH = float(os.getenv("SC_VEH_PPM", "40.0"))
MIN_TRACK_HITS = 3
SPEED_AVG_WINDOW = 5

# Pedestrian lane reference (in the vehicle camera frame)
PEDESTRIAN_LANE_Y = int(os.getenv("SC_LANE_Y", "250"))

# Safety threshold (meters)
VEHICLE_CLOSE_THRESH_M = float(os.getenv("SC_VEH_CLOSE_M", "6.0"))

# Traffic light ROI (x, y, w, h)
TRAFFIC_LIGHT_ROI = tuple(map(int, os.getenv("SC_TL_ROI", "100,60,120,160").split(",")))

# HSV thresholds
HSV_RED_1 = ((0, 120, 120), (10, 255, 255))
HSV_RED_2 = ((170, 120, 120), (180, 255, 255))
HSV_YELLOW = ((15, 120, 120), (35, 255, 255))
HSV_GREEN  = ((40, 70, 70), (90, 255, 255))

# Logging / DB
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
# Time-based logging cadence (seconds) to avoid spamming DB
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))

# Drawing
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLUE   = (255, 0, 0)

# --------------------- LED (guarded) ---------------------
def _init_led():
    try:
        from luma.core.interface.serial import spi, noop
        from luma.led_matrix.device import max7219
        from luma.core.render import canvas
        from PIL import ImageFont

        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, cascaded=LED_CASCADE, block_orientation=LED_ORIENTATION, rotate=0)
        font = ImageFont.load_default()

        def show_led(msg: str):
            with canvas(device) as draw:
                draw.text((1, -2), msg, fill="white", font=font)
        if PRINT_DEBUG:
            print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        if PRINT_DEBUG:
            print("[LED] Not available, falling back to console:", repr(e))
        def show_led(msg: str):
            print("[LED]", msg)
        return show_led

show_led = _init_led()

# --------------------- HELPERS ---------------------
def _normalize_cam(value):
    """
    Accepts int, numeric string, /dev/videoN, or /dev/v4l/by-id/*.
    Returns an int index (preferred) if it can resolve to /dev/videoN, else the original string/path.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    try:
        if s.startswith("/dev/"):
            real = os.path.realpath(s)  # follows by-id symlinks
            m = re.match(r"^/dev/video(\d+)$", real)
            if m:
                return int(m.group(1))
            # fallback to the resolved path if not /dev/videoN but still exists
            if os.path.exists(real):
                return real
    except Exception:
        pass
    m = re.match(r"^/dev/video(\d+)$", s)
    if m:
        return int(m.group(1))
    return s

# --------------------- CAMERA THREAD ---------------------
class CameraStream:
    def __init__(self, index, width: int, height: int, fps: int):
        self.index = index  # may be int or path string
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self._open_camera()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_camera(self):
        """
        Open the camera using numeric index (preferred) or a device path.
        If a /dev/v4l/by-id symlink is given, resolve it to /dev/videoN and
        pass the numeric index to OpenCV (more reliable on some builds).
        """
        idx = _normalize_cam(self.index)

        # Prefer numeric
        if isinstance(idx, int):
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera index {idx}")
            pretty_name = f"/dev/video{idx}"
        else:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera path {idx}")
            pretty_name = str(idx)

        # lower latency: avoid deep buffering
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          min(self.fps, 20))

        def set_fourcc(code):
            fourcc = cv2.VideoWriter_fourcc(*code)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            return int(self.cap.get(cv2.CAP_PROP_FOURCC)) == fourcc

        if not set_fourcc('MJPG'):
            set_fourcc('YUYV') or set_fourcc('YUY2')

        # warm-up
        self.cap.read()

        # Log negotiated mode
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_val = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_val >> 8*i) & 0xFF) for i in range(4)])
        print(f"[OPEN] {pretty_name} -> {w}x{h}@{f:.1f} FOURCC={fourcc_str}")

    def _reopen_once(self):
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(0.2)
        self._open_camera()

    def _update(self):
        no_frame = 0
        frame_interval = 1.0 / max(1, self.fps)
        last_t = 0.0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                no_frame += 1
                if no_frame == 20:
                    print(f"[INFO] Reopening camera due to stalled frames…")
                    self._reopen_once()
                    no_frame = 0
                time.sleep(0.02)
                continue
            else:
                no_frame = 0

            # Resize here so main loop stays light
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            with self.lock:
                self.ret = True
                self.frame = frame

            # throttle to target FPS (avoid overspinning)
            now = time.time()
            sleep_left = frame_interval - (now - last_t)
            if sleep_left > 0:
                time.sleep(sleep_left)
            last_t = now

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass

# --------------------- SIMPLE TRACKER ---------------------
@dataclass
class Track:
    id: int
    cls: str
    history: deque  # of (t, cx, cy)
    hits: int = 0

class CentroidTracker:
    def __init__(self, max_dist_px: float = 100.0, max_age_s: float = 1.0):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_dist = max_dist_px
        self.max_age = max_age_s

    def update(self, detections: List[Tuple[str, Tuple[int,int,int,int]]], now: float):
        det_centroids = [(lab, ((x1+x2)//2, (y1+y2))) for lab,(x1,y1,x2,y2) in detections]

        unmatched = list(range(len(det_centroids)))
        for tid, tr in list(self.tracks.items()):
            best_j, best_d = None, 1e9
            _, (last_t, last_cx, last_cy) = tr.cls, tr.history[-1]
            for j in unmatched:
                lab, (cx, cy) = det_centroids[j]
                if lab != tr.cls:
                    continue
                d = math.hypot(cx - last_cx, cy - last_cy)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= self.max_dist:
                lab, (cx, cy) = det_centroids[best_j]
                tr.history.append((now, cx, cy))
                tr.hits += 1
                if len(tr.history) > 30:
                    tr.history.popleft()
                unmatched.remove(best_j)

        for j in unmatched:
            lab, (cx, cy) = det_centroids[j]
            tr = Track(id=self.next_id, cls=lab, history=deque(maxlen=30))
            tr.history.append((now, cx, cy))
            tr.hits = 1
            self.tracks[tr.id] = tr
            self.next_id += 1

        # prune stale
        to_del = []
        for tid, tr in self.tracks.items():
            last_t, _, _ = tr.history[-1]
            if now - last_t > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            self.tracks.pop(tid, None)

    def speeds_mps(self, ppm: float) -> Dict[int, float]:
        out = {}
        for tid, tr in self.tracks.items():
            if tr.hits < MIN_TRACK_HITS or len(tr.history) < 2:
                continue
            pts = list(tr.history)[-SPEED_AVG_WINDOW:]
            ds, dt = 0.0, 0.0
            for (t1, x1, y1), (t2, x2, y2) in zip(pts, pts[1:]):
                ds += math.hypot(x2 - x1, y2 - y1)
                dt += (t2 - t1)
            if dt <= 0:
                continue
            m = (ds / ppm) / dt
            out[tid] = m
        return out

# --------------------- YOLO ---------------------
model = YOLO(YOLO_MODEL)

def yolo_detect(frame: np.ndarray, conf: float, img_size: int):
    res = model.predict(frame, conf=conf, imgsz=img_size, verbose=False)
    boxes = []  # (label, (x1,y1,x2,y2), conf)
    r0 = res[0]
    names = r0.names
    for b in r0.boxes:
        cls_id = int(b.cls[0])
        label = names[cls_id].lower()
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        boxes.append((label, (x1, y1, x2, y2), float(b.conf[0])))
    return boxes

def filter_classes(boxes, keep: set):
    return [(label, xyxy, cf) for label, xyxy, cf in boxes if label in keep]

# --------------------- DB ---------------------
def init_db(path: str):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ped_count INTEGER,
            veh_count INTEGER,
            tl_color TEXT,
            nearest_vehicle_distance_m REAL,
            avg_vehicle_speed_mps REAL,
            action TEXT
        );
        """
    )
    con.commit()
    con.close()

def log_event(path: str, ts: float, ped_count: int, veh_count: int, tl_color: str,
              nearest_vehicle_distance_m: float, avg_vehicle_speed_mps: float, action: str):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts, ped_count, veh_count, tl_color, nearest_vehicle_distance_m, avg_vehicle_speed_mps, action) VALUES (?,?,?,?,?,?,?)",
        (ts, ped_count, veh_count, tl_color, nearest_vehicle_distance_m, avg_vehicle_speed_mps, action)
    )
    con.commit()
    con.close()

# --------------------- TRAFFIC LIGHT DETECTION ---------------------
def detect_traffic_light_color(frame: np.ndarray) -> str:
    x, y, w, h = TRAFFIC_LIGHT_ROI
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def mask_range(hsv_img, lo, hi):
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        return cv2.inRange(hsv_img, lo, hi)

    mask_r1 = mask_range(hsv, *HSV_RED_1)
    mask_r2 = mask_range(hsv, *HSV_RED_2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_y = mask_range(hsv, *HSV_YELLOW)
    mask_g = mask_range(hsv, *HSV_GREEN)

    r = int(np.sum(mask_red > 0))
    yv = int(np.sum(mask_y > 0))
    g = int(np.sum(mask_g > 0))

    vals = {"red": r, "yellow": yv, "green": g}
    best = max(vals, key=vals.get)
    if vals[best] < 50:
        return "unknown"
    return best

def safe_camera(index) -> Optional[int | str]:
    """
    Returns normalized index/path if it opens & reads 1 frame, else None.
    Forces MJPG @ modest size/FPS to avoid blocking.
    """
    idx = _normalize_cam(index)
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 6)
        ok, _ = cap.read()
        cap.release()
        return idx if ok else None
    except Exception:
        return None

def pick_cameras():
    """
    Picks cameras based on env vars if provided, else auto-scans /dev/video0..9.
    Supports both numeric indices and device paths.
    """
    ped_env = os.getenv("SC_CAM_PED")
    veh_env = os.getenv("SC_CAM_VEH")
    tl_env  = os.getenv("SC_CAM_TL")

    if ped_env and veh_env and tl_env:
        ped = _normalize_cam(ped_env)
        veh = _normalize_cam(veh_env)
        tl  = _normalize_cam(tl_env)

        ok_all = [safe_camera(ped), safe_camera(veh), safe_camera(tl)]
        if all(x is not None for x in ok_all):
            return ped, veh, tl
        raise RuntimeError(f"One or more SC_CAM_* env cameras failed to open: {ok_all}")

    # else auto-scan /dev/video0..9
    candidates = []
    for idx in range(10):
        ok = safe_camera(idx)
        if ok is not None:
            candidates.append(ok)
        if len(candidates) >= 3:
            break
    if len(candidates) < 3:
        raise RuntimeError(f"Found only {len(candidates)} working cameras: {candidates}.")
    return candidates[0], candidates[1], candidates[2]

# --------------------- MAIN PIPELINE (runs in a thread) ---------------------
def run_pipeline():
    init_db(DB_PATH)

    i_ped, i_veh, i_tl = pick_cameras()

    # Smooth presets
    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS_TARGET)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS_TARGET)
    cam_tl  = CameraStream(i_tl,  480,     270,     max(8, FPS_TARGET // 2))

    print(f"[Pedestrian Cam] {i_ped}")
    print(f"[Vehicle Cam]    {i_veh}")
    print(f"[Traffic Light]  {i_tl}")

    def wait_for_first_frame(stream: CameraStream, name: str, timeout_s: float = 3.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ok, fr = stream.read()
            if ok and fr is not None:
                return True
            time.sleep(0.05)
        print(f"[WARN] {name} did not deliver first frame within {timeout_s:.1f}s")
        return False

    _ = wait_for_first_frame(cam_ped, "Pedestrian")
    _ = wait_for_first_frame(cam_veh, "Vehicle")
    _ = wait_for_first_frame(cam_tl,  "Traffic Light")

    veh_tracker = CentroidTracker(max_dist_px=120.0, max_age_s=1.0)
    frame_idx = 0
    last_log_ts = 0.0

    try:
        while True:
            loop_start = time.time()
            frame_idx += 1

            # Read cameras
            rp, fp = cam_ped.read()
            rv, fv = cam_veh.read()
            rt, ft = cam_tl.read()

            ok_ped = bool(rp and fp is not None)
            ok_veh = bool(rv and fv is not None)
            ok_tl  = bool(rt and ft is not None)
            online = ok_ped or ok_veh or ok_tl  # any cam online

            # Prepare blank frames for any missing cams (keeps MJPEG alive)
            blank_640x360 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            blank_480x270 = np.zeros((270, 480, 3), dtype=np.uint8)

            fp_s = fp if ok_ped else blank_640x360
            fv_s = fv if ok_veh else blank_640x360
            ft_s = ft if ok_tl  else blank_480x270

            do_infer = (frame_idx % max(1, SKIP_FRAMES) == 0)

            # --- Pedestrians ---
            ped_count = 0
            if do_infer and ok_ped:
                det_p = yolo_detect(fp_s, YOLO_CONF, YOLO_IMG_SZ)
                det_p = filter_classes(det_p, PEDESTRIAN_CLASSES)
                ped_count = len(det_p)
                for lab, (x1, y1, x2, y2), cf in det_p:
                    cv2.rectangle(fp_s, (x1, y1), (x2, y2), COLOR_GREEN, 2)
                    cv2.putText(fp_s, f"{lab} {cf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)
            cv2.putText(fp_s, f"Pedestrians: {ped_count}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            # --- Vehicles ---
            veh_count = 0
            nearest_vehicle_distance_m = float('inf')
            avg_speed_mps = 0.0

            if do_infer and ok_veh:
                det_v = yolo_detect(fv_s, YOLO_CONF, YOLO_IMG_SZ)
                det_v = filter_classes(det_v, VEHICLE_CLASSES)
                veh_count = len(det_v)
                now = time.time()
                det_v_in = [(lab, bbox) for lab, bbox, _ in det_v]
                veh_tracker.update(det_v_in, now)

                speed_by_tid = veh_tracker.speeds_mps(PIXELS_PER_METER_VEH)

                for lab, (x1, y1, x2, y2), cf in det_v:
                    # Draw box
                    cv2.rectangle(fv_s, (x1, y1), (x2, y2), COLOR_RED, 2)

                    # Match nearest active track (same class)
                    bx = (x1 + x2) // 2
                    by = (y1 + y2) // 2
                    best_tid, best_d = None, 1e9
                    for tid, tr in veh_tracker.tracks.items():
                        if tr.cls != lab or not tr.history:
                            continue
                        _, cx_tr, cy_tr = tr.history[-1]
                        d = math.hypot(bx - cx_tr, by - cy_tr)
                        if d < best_d:
                            best_d, best_tid = d, tid
                    speed_mps = speed_by_tid.get(best_tid, 0.0) if best_tid is not None else 0.0

                    # Distance to pedestrian lane
                    cx = (x1 + x2) // 2
                    cy = y2  # vehicle "front" proxy
                    dy_px = max(0, PEDESTRIAN_LANE_Y - cy)
                    dist_m = abs(dy_px) / PIXELS_PER_METER_VEH
                    nearest_vehicle_distance_m = min(nearest_vehicle_distance_m, dist_m)
                    cv2.line(fv_s, (cx, cy), (cx, PEDESTRIAN_LANE_Y), COLOR_YELLOW, 1)

                    # Overlay label once (class, conf, speed, distance)
                    label_txt = f"{lab} {cf:.2f} | {speed_mps*3.6:.0f} km/h | {dist_m:.1f} m"
                    cv2.putText(fv_s, label_txt, (x1, max(12, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RED, 1)

                if speed_by_tid:
                    avg_speed_mps = float(np.mean(list(speed_by_tid.values())))

                # Draw track IDs / speeds at the track point
                for tid, tr in veh_tracker.tracks.items():
                    if not tr.history:
                        continue
                    _, cx_t, cy_t = tr.history[-1]
                    s = speed_by_tid.get(tid, 0.0)
                    cv2.circle(fv_s, (cx_t, cy_t), 3, COLOR_BLUE, -1)
                    cv2.putText(fv_s, f"ID {tid} {s:.1f} m/s",
                                (cx_t + 6, cy_t - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLUE, 1)

            cv2.putText(fv_s, f"Vehicles: {veh_count}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            cv2.line(fv_s, (0, PEDESTRIAN_LANE_Y),
                     (FRAME_W, PEDESTRIAN_LANE_Y), COLOR_YELLOW, 2)

            # --- Traffic light color ---
            tl_color = detect_traffic_light_color(ft_s) if ok_tl else "unknown"
            cv2.putText(ft_s, f"TL: {tl_color.upper()}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            x, y, w, h = TRAFFIC_LIGHT_ROI
            cv2.rectangle(ft_s, (x, y), (x + w, y + h), COLOR_WHITE, 2)

            # Push frames to web UI (even if some are blanks)
            publish_frame("ped", fp_s)
            publish_frame("veh", fv_s)
            publish_frame("tl",  ft_s)

            # Prepare status → website
            now = time.time()
            local_hour = time.localtime(now).tm_hour
            flags = {"night": local_hour >= 21, "rush": local_hour == 7}
            extra = {"ambulance": False}

            nearest_m = 0.0 if nearest_vehicle_distance_m == float('inf') else nearest_vehicle_distance_m
            publish_status_from_loop(
                now_ts=now,
                ped_count=ped_count,
                veh_count=veh_count,
                tl_color=tl_color,
                nearest_m=nearest_m,
                avg_mps=avg_speed_mps,
                flags=flags,
                extra=extra,
            )

            # --- LED & Scenario Logic (simple local LED feedback) ---
            action = "OFF"
            veh_close = (nearest_vehicle_distance_m < VEHICLE_CLOSE_THRESH_M) if ok_veh else False
            if tl_color == "red":
                action = "STOP"
            elif tl_color == "green":
                if ped_count > 0 and veh_close:
                    action = "STOP"
                elif ped_count > 0:
                    action = "GO"
                else:
                    action = "OFF"
            elif tl_color == "yellow":
                action = "STOP" if ped_count > 0 else "OFF"
            else:  # unknown
                if ped_count > 0 and (veh_close or veh_count > 0):
                    action = "STOP"
                elif ped_count > 0:
                    action = "GO"
                else:
                    action = "OFF"
            show_led(action)

            # Overlays on vehicle frame for nearest/speed
            cv2.putText(
                fv_s,
                f"Nearest dist: {nearest_m:.1f} m",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2
            )
            cv2.putText(fv_s, f"Avg speed: {avg_speed_mps:.1f} m/s",
                        (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            # --- Logging (time-based, not per N frames) ---
            if now - last_log_ts >= LOG_EVERY_SEC:
                log_event(DB_PATH, now, int(ped_count), int(veh_count),
                          tl_color, float(nearest_m), float(avg_speed_mps), action)
                last_log_ts = now

            if SHOW_WINDOWS:
                cv2.imshow("Pedestrians", fp_s)
                cv2.imshow("Vehicles", fv_s)
                cv2.imshow("Traffic Light", ft_s)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            # pacing the outer loop
            elapsed = time.time() - loop_start
            if elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed)

    finally:
        try: cam_ped.stop()
        except Exception: pass
        try: cam_veh.stop()
        except Exception: pass
        try: cam_tl.stop()
        except Exception: pass
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()

# --------------------- ENTRYPOINT ---------------------
if __name__ == "__main__":
    # Run processing on a background thread
    threading.Thread(target=run_pipeline, daemon=True).start()
    # Run the web server in the MAIN thread (prevents starvation)
    start_http_server(host="0.0.0.0", port=5000)
