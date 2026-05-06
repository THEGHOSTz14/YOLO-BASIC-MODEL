import os
import time
import zipfile
import urllib.request
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Color map ────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "person":     ( 60,  60, 255),
    "cell phone": (255, 180,   0),
    "mouse":      (  0, 200, 255),
    "keyboard":   (255, 100, 200),
    "laptop":     (  0, 255, 200),
    "monitor":    (200, 255,   0),
    "tv":         (200, 255,   0),
    "chair":      (255, 140,   0),
    "bottle":     (100, 255, 100),
    "cup":        (  0, 140, 255),
    "book":       (200, 100, 255),
    "remote":     (255, 255,   0),
    "car":        (  0, 100, 255),
    "cat":        (255,  80, 180),
    "dog":        (180,  80, 255),
}
DEFAULT_COLOR = (0, 255, 100)


def get_color(label: str):
    return CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)


def id_color(track_id) -> tuple:
    """Unique hue per track ID. Accepts int or string."""
    hue = (int(track_id) * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ── Finger counting ───────────────────────────────────────────────────────────
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def count_fingers(landmarks) -> int:
    count = 0
    # thumb: compare x-coordinates
    if landmarks[4][0] < landmarks[3][0]:
        count += 1
    # other four fingers: tip y < pip y  →  finger is up
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip][1] < landmarks[pip][1]:
            count += 1
    return count


def draw_finger_badge(frame, count, bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = max(y1 - 40, 40)
    cv2.circle(frame, (cx, cy), 36, (0, 255, 180), -1)
    cv2.circle(frame, (cx, cy), 36, (0, 0, 0), 2)
    text = str(count)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    cv2.putText(frame, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)


# ── MediaPipe model download ──────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _is_valid_task_file(path: str) -> bool:
    """Returns True only if the file exists and is a valid zip/task archive."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 100_000:   # real model is ~9 MB
        return False
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.namelist()
        return True
    except zipfile.BadZipFile:
        return False


def ensure_hand_model():
    """Download hand_landmarker.task if missing or corrupt."""
    if not _is_valid_task_file(MODEL_PATH):
        if os.path.exists(MODEL_PATH):
            print("[INFO] Removing corrupt hand_landmarker.task ...")
            os.remove(MODEL_PATH)
        print("[INFO] Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
        print(f"[INFO] Download complete ({size_mb:.1f} MB)")


# ── Worker thread ─────────────────────────────────────────────────────────────
class YOLOThread(QThread):
    # signals: annotated frame | detections list | total fingers
    frame_ready  = pyqtSignal(np.ndarray, list, int)
    error_signal = pyqtSignal(str)

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.5):
        super().__init__()
        self.model_path = model_path
        self.conf       = conf
        self.running    = False

        # recording state
        self._writer      = None
        self._recording   = False
        self._record_path = ""

    # ── Recording control (called from the GUI thread) ────────────────────────
    def start_recording(self, path: str):
        self._record_path = path
        self._recording   = True     # VideoWriter created lazily on first frame

    def stop_recording(self):
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[REC] Saved -> {self._record_path}")

    def _write_frame(self, frame: np.ndarray):
        if not self._recording:
            return
        if self._writer is None:
            h, w   = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._record_path, fourcc, 20, (w, h))
        self._writer.write(frame)

    # ── Main detection loop ───────────────────────────────────────────────────
    def run(self):

        # ── Load YOLO + DeepSORT ──────────────────────────────────────────────
        try:
            model   = YOLO(self.model_path)
            tracker = DeepSort(max_age=30, n_init=2, nn_budget=100)
        except Exception as exc:
            self.error_signal.emit(f"Model load error: {exc}")
            return

        # ── Load MediaPipe hand landmarker ────────────────────────────────────
        try:
            ensure_hand_model()

            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            # HandLandmarkerOptions is a STANDALONE class, NOT HandLandmarker.HandLandmarkerOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )

            options = HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            hand_detector = HandLandmarker.create_from_options(options)

        except Exception as exc:
            self.error_signal.emit(f"MediaPipe init error: {exc}")
            return

        # ── Open camera ───────────────────────────────────────────────────────
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error_signal.emit("Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # ── YOLO detection ────────────────────────────────────────────────
            results  = model(frame, conf=self.conf, verbose=False)
            raw_dets = []   # DeepSORT format: ([x,y,w,h], conf, label)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    label = model.names[int(box.cls[0])]
                    raw_dets.append(([x1, y1, x2 - x1, y2 - y1], score, label))

            # ── DeepSORT tracking ─────────────────────────────────────────────
            tracks     = tracker.update_tracks(raw_dets, frame=frame)
            detections = []

            for track in tracks:
                if not track.is_confirmed():
                    continue

                tid             = int(track.track_id)
                label           = track.get_det_class() or "object"
                score           = track.get_det_conf() or 0.0
                score           = score if score else 0.0
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                color           = get_color(label)
                tcolor          = id_color(tid)

                detections.append((x1, y1, x2, y2, label, score, color, tid))

                # bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), tcolor, 2)

                # label background + text
                header = f"{label} #{tid}  {score:.2f}"
                (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame,
                              (x1, max(y1 - th - 10, 0)),
                              (x1 + tw + 6, y1), tcolor, -1)
                cv2.putText(frame, header,
                            (x1 + 3, max(y1 - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # centre trail dot
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, tcolor, -1)

            # ── MediaPipe hand landmarks ──────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_res = hand_detector.detect(mp_img)

            total_fingers = 0

            if hand_res.hand_landmarks:
                fh, fw = frame.shape[:2]
                for hand_lm in hand_res.hand_landmarks:
                    coords  = [(lm.x, lm.y, lm.z) for lm in hand_lm]
                    fingers = count_fingers(coords)
                    total_fingers += fingers

                    px = [int(c[0] * fw) for c in coords]
                    py = [int(c[1] * fh) for c in coords]

                    # draw skeleton
                    for a, b in [
                        (0,1),(1,2),(2,3),(3,4),
                        (0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),
                        (0,13),(13,14),(14,15),(15,16),
                        (0,17),(17,18),(18,19),(19,20),
                        (5,9),(9,13),(13,17),
                    ]:
                        cv2.line(frame, (px[a], py[a]), (px[b], py[b]),
                                 (0, 220, 160), 2)
                    for x, y in zip(px, py):
                        cv2.circle(frame, (x, y), 4, (0, 255, 180), -1)

                    x1h = max(min(px) - 20, 0)
                    y1h = max(min(py) - 20, 0)
                    x2h = min(max(px) + 20, fw)
                    y2h = min(max(py) + 20, fh)
                    draw_finger_badge(frame, fingers, (x1h, y1h, x2h, y2h))

            # ── Recording overlay ─────────────────────────────────────────────
            if self._recording:
                ts = time.strftime("%H:%M:%S")
                cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(frame, f"REC  {ts}",
                            (34, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                self._write_frame(frame)

            self.frame_ready.emit(frame.copy(), detections, total_fingers)

        # ── Cleanup ───────────────────────────────────────────────────────────
        cap.release()
        hand_detector.close()
        if self._writer is not None:
            self._writer.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def set_conf(self, value: int):
        self.conf = value / 100.0