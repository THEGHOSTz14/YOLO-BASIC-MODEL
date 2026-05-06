import os
import time
import cv2
import numpy as np
import turtle
from collections import defaultdict
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QListWidget, QListWidgetItem,
    QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui  import QImage, QPixmap, QColor
from yolo_thread  import YOLOThread

os.makedirs("recordings",  exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ── Gesture map ───────────────────────────────────────────────────────────────
#  finger count  →  (display label, action key)
GESTURES = {
   
    5: ("5 fingers - STOP DETECTION", "stop"),
}
GESTURE_HOLD_SEC = 1.2   # how long to hold before triggering

STYLE = """
QMainWindow, QWidget {
    background: #0d0d0d;
    color: #00ff88;
    font-family: Consolas, monospace;
}
QPushButton {
    background: #111;
    border: 1px solid #00ff88;
    color: #00ff88;
    padding: 10px;
    border-radius: 4px;
    font-size: 13px;
}
QPushButton:hover    { background: #00ff88; color: #0d0d0d; }
QPushButton:disabled { border-color: #2a2a2a; color: #2a2a2a; }
QPushButton#rec_btn  { border-color: #ff4444; color: #ff4444; }
QPushButton#rec_btn:hover { background: #ff4444; color: #fff; }
QPushButton#rec_btn_active {
    background: #ff4444;
    color: #fff;
    border-color: #ff4444;
}
QSlider::groove:horizontal {
    background: #222; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00ff88;
    width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QListWidget {
    background: #0a0a0a; border: 1px solid #1a1a1a; font-size: 12px;
}
QFrame#card { background: #111; border: 1px solid #1c1c1c; border-radius: 6px; }
QLabel#stat_number { font-size: 24px; font-weight: bold; color: #00ff88; }
QLabel#stat_label  { font-size: 10px; color: #446644; }
"""


def make_card(lbl: QLabel, title: str) -> QFrame:
    card = QFrame()
    card.setObjectName("card")
    lay  = QVBoxLayout(card)
    lay.setContentsMargins(8, 6, 8, 6)
    lay.setSpacing(2)
    lbl.setObjectName("stat_number")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    sub = QLabel(title)
    sub.setObjectName("stat_label")
    sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(lbl)
    lay.addWidget(sub)
    return card


def point_in_poly(point, poly):
    return cv2.pointPolygonTest(np.array(poly, dtype=np.float32), point, False) >= 0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        

        self.zones = {
        "restricted": [],
        "safe": [],
        "tracking": []
     }
        self.current_zone = None
        self.drawing_points = []



        self.setWindowTitle("YOLO · DeepSORT · Gesture Control")
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(STYLE)

        self._thread        = None
        self._fps_time      = time.time()
        self._session_start = None
        self._paused        = False
        self._recording     = False
        self._last_frame    = None

        # stats
        self._total = 0
        self._peak  = 0
        self._class_totals: dict[str, int] = defaultdict(int)

        # gesture hold state
        self._gesture_finger   = -1
        self._gesture_since    = 0.0
        self._gesture_fired    = False
        self._gesture_cooldown = 0.0

        self._build_ui()

        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick)
        self._clock.start(500)

        self.video_lbl.setMouseTracking(True)
        self.video_lbl.mousePressEvent = self._mouse_press
        self.video_lbl.mouseMoveEvent = self._mouse_move
        self.video_lbl.mouseReleaseEvent = self._mouse_release


    def _mouse_press(self, event):
       if self.current_zone:
          self.drawing_points = [(int(event.position().x()), int(event.position().y()))]

    def _mouse_move(self, event):
       if self.current_zone and self.drawing_points:
          self.drawing_points.append((int(event.position().x()), int(event.position().y())))

    def _mouse_release(self, event):
       if self.current_zone and self.drawing_points:
          self.zones[self.current_zone].append(self.drawing_points.copy())
          self.drawing_points = []  





    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        def _set_zone(self, z):
            self.current_zone = z

        # ── Left: video feed ──────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(4)

        self.video_lbl = QLabel("[ NO FEED ]")
        self.video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_lbl.setStyleSheet("border:1px solid #00ff88; color:#00ff88;")
        self.video_lbl.setMinimumSize(800, 600)

        self.fps_lbl = QLabel("FPS: --")
        self.fps_lbl.setStyleSheet("color:#336633; font-size:11px;")

        self.gesture_lbl = QLabel("")
        self.gesture_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_lbl.setStyleSheet(
            "color:#00ffcc; font-size:14px; font-weight:bold; padding:4px;")

        left.addWidget(self.video_lbl)
        left.addWidget(self.fps_lbl)
        left.addWidget(self.gesture_lbl)

        # ── Right: controls ───────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        self.start_btn = QPushButton("START")
        self.stop_btn  = QPushButton("STOP")
        self.stop_btn.setEnabled(False)
        self.pause_btn = QPushButton("PAUSE")
        self.pause_btn.setEnabled(False)
        self.rec_btn   = QPushButton("RECORD")
        self.rec_btn.setObjectName("rec_btn")
        self.rec_btn.setEnabled(False)
        self.shot_btn  = QPushButton("SCREENSHOT")
        self.shot_btn.setEnabled(False)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.btn_r = QPushButton("🟥 Restricted Zone")
        self.btn_s = QPushButton("🟩 Safe Zone")
        self.btn_t = QPushButton("🟦 Tracking Zone")

        self.btn_r.clicked.connect(lambda: self._set_zone("restricted"))
        self.btn_s.clicked.connect(lambda: self._set_zone("safe"))
        self.btn_t.clicked.connect(lambda: self._set_zone("tracking"))

        right.addWidget(self.btn_r)
        right.addWidget(self.btn_s)
        right.addWidget(self.btn_t)

        self.pause_btn.clicked.connect(self._toggle_pause)
        self.rec_btn.clicked.connect(self._toggle_record)
        self.shot_btn.clicked.connect(self._screenshot)
    

        # confidence slider
        conf_lbl         = QLabel("Confidence Threshold")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_val = QLabel("0.50")
        self.conf_slider.valueChanged.connect(self._on_conf)
        
         
        # stat cards row 1
        self.lbl_now  = QLabel("0")
        self.lbl_tot  = QLabel("0")
        self.lbl_peak = QLabel("0")
        self.lbl_cls  = QLabel("0")
        cards = QHBoxLayout()
        cards.setSpacing(4)
        for lbl, title in [
            (self.lbl_now,  "NOW"),
            (self.lbl_tot,  "TOTAL"),
            (self.lbl_peak, "PEAK"),
            (self.lbl_cls,  "CLASSES"),
        ]:
            cards.addWidget(make_card(lbl, title))

        # stat cards row 2
        self.lbl_session = QLabel("00:00")
        self.lbl_fingers = QLabel("-")
        self.lbl_fingers.setStyleSheet(
            "font-size:38px; font-weight:bold; color:#00ffcc;")
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(make_card(self.lbl_session, "SESSION"))
        row2.addWidget(make_card(self.lbl_fingers, "FINGERS"))

        # gesture legend
        gesture_frame = QFrame()
        gesture_frame.setObjectName("card")
        glay = QVBoxLayout(gesture_frame)
        glay.setContentsMargins(8, 6, 8, 6)
        glay.setSpacing(2)
        title_lbl = QLabel("GESTURE CONTROLS")
        title_lbl.setStyleSheet("color:#00ff88; font-size:11px; font-weight:bold;")
        glay.addWidget(title_lbl)
        for fc, (desc, _) in GESTURES.items():
            row = QLabel(f"  {desc}")
            row.setStyleSheet("color:#446644; font-size:10px;")
            glay.addWidget(row)

        # hold-progress label
        self.hold_lbl = QLabel("")
        self.hold_lbl.setStyleSheet("color:#00ffcc; font-size:10px; padding:2px;")

        # class breakdown
        bk_lbl = QLabel("CLASS BREAKDOWN")
        bk_lbl.setStyleSheet("color:#446644; font-size:10px;")
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(120)

        # live tracks
        det_lbl = QLabel("LIVE TRACKS")
        det_lbl.setStyleSheet("color:#446644; font-size:10px;")
        self.det_list = QListWidget()

        # reset button
        reset_btn = QPushButton("RESET STATS")
        reset_btn.setStyleSheet(
            "border-color:#334433; color:#334433; font-size:11px; padding:6px;")
        reset_btn.clicked.connect(self._reset_stats)

        # assemble right panel
        for w in [self.start_btn, self.stop_btn,
                  self.pause_btn, self.rec_btn, self.shot_btn]:
            right.addWidget(w)
        right.addSpacing(4)
        right.addWidget(conf_lbl)
        right.addWidget(self.conf_slider)
        right.addWidget(self.conf_val)
        right.addSpacing(4)
        right.addLayout(cards)
        right.addLayout(row2)
        right.addWidget(gesture_frame)
        right.addWidget(self.hold_lbl)
        right.addWidget(bk_lbl)
        right.addWidget(self.class_list)
        right.addWidget(det_lbl)
        right.addWidget(self.det_list)
        right.addStretch()
        right.addWidget(reset_btn)

        root.addLayout(left,  4)
        root.addLayout(right, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_conf(self, v: int):
        self.conf_val.setText(f"{v / 100:.2f}")
        if self._thread:
            self._thread.set_conf(v)

    def _start(self):
        self._session_start = time.time()
        self._thread = YOLOThread(conf=self.conf_slider.value() / 100.0)
        self._thread.frame_ready.connect(self._update_frame)
        self._thread.error_signal.connect(self._show_error)
        self._thread.start()
        for btn in [self.stop_btn, self.pause_btn, self.rec_btn, self.shot_btn]:
            btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.video_lbl.setText("[ LOADING... ]")

    def _stop(self):
        if self._recording:
            self._toggle_record()
        if self._thread:
            self._thread.stop()
            self._thread = None
        self._session_start = None
        self._paused        = False
        self.start_btn.setEnabled(True)
        for btn in [self.stop_btn, self.pause_btn, self.rec_btn, self.shot_btn]:
            btn.setEnabled(False)
        self.video_lbl.setText("[ STOPPED ]")
        self.det_list.clear()
        self.fps_lbl.setText("FPS: --")
        self.lbl_now.setText("0")
        self.lbl_session.setText("00:00")
        self.lbl_fingers.setText("-")
        self.gesture_lbl.setText("")

    def _toggle_pause(self):
        if not self._thread:
            return
        self._paused = not self._paused
        self.pause_btn.setText("RESUME" if self._paused else "PAUSE")
        self.gesture_lbl.setText("[ PAUSED ]" if self._paused else "")

    def _toggle_record(self):
        if not self._thread:
            return
        if not self._recording:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("recordings", f"rec_{ts}.mp4")
            self._thread.start_recording(path)
            self._recording = True
            self.rec_btn.setText("STOP REC")
            self.rec_btn.setObjectName("rec_btn_active")
            self.rec_btn.setStyleSheet(
                "background:#ff4444; color:#fff; border-color:#ff4444;"
                "padding:10px; border-radius:4px; font-size:13px;")
        else:
            self._thread.stop_recording()
            self._recording = False
            self.rec_btn.setText("RECORD")
            self.rec_btn.setStyleSheet("")
            self.rec_btn.setObjectName("rec_btn")

    def _screenshot(self):
        if self._last_frame is None:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("screenshots", f"shot_{ts}.jpg")
        cv2.imwrite(path, self._last_frame)
        self.gesture_lbl.setText(f"Saved -> {path}")
        QTimer.singleShot(2000, lambda: self.gesture_lbl.setText(""))

    def _reset_stats(self):
        self._total = 0
        self._peak  = 0
        self._class_totals.clear()
        for lbl in [self.lbl_tot, self.lbl_peak, self.lbl_cls]:
            lbl.setText("0")
        self.class_list.clear()

    def _tick(self):
        if self._session_start:
            elapsed = int(time.time() - self._session_start)
            m, s    = divmod(elapsed, 60)
            self.lbl_session.setText(f"{m:02d}:{s:02d}")

    # ── Gesture engine ────────────────────────────────────────────────────────
    def _process_gesture(self, finger_count: int):
        now = time.time()

        if now < self._gesture_cooldown:
            self.hold_lbl.setText("")
            return

        if finger_count not in GESTURES:
            self._gesture_finger = -1
            self._gesture_fired  = False
            self.hold_lbl.setText("")
            return

        if finger_count != self._gesture_finger:
            self._gesture_finger = finger_count
            self._gesture_since  = now
            self._gesture_fired  = False

        if not self._gesture_fired:
            held = now - self._gesture_since
            pct  = min(held / GESTURE_HOLD_SEC, 1.0)
            bar  = "#" * int(pct * 12) + "." * (12 - int(pct * 12))
            desc = GESTURES[finger_count][0]
            self.hold_lbl.setText(f"Hold [{bar}]  {desc}")

            if held >= GESTURE_HOLD_SEC:
                self._gesture_fired    = True
                self._gesture_cooldown = now + 2.0
                action = GESTURES[finger_count][1]
                self.hold_lbl.setText(f"OK  {desc}")
                QTimer.singleShot(1500, lambda: self.hold_lbl.setText(""))
                self._fire_action(action)

    def _fire_action(self, action: str):
        if   action == "toggle_pause":   self._toggle_pause()
        elif action == "toggle_record":  self._toggle_record()
        elif action == "screenshot":     self._screenshot()
        elif action == "stop":           self._stop()

    # ── Frame update ──────────────────────────────────────────────────────────
    def _update_frame(self, frame, detections, finger_count):
        if self._paused:
            return

        self._last_frame = frame.copy()

        # FPS
        now            = time.time()
        fps            = 1.0 / max(now - self._fps_time, 1e-6)
        self._fps_time = now
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

        # fingers
        self.lbl_fingers.setText(str(finger_count) if finger_count > 0 else "-")
        self._process_gesture(finger_count)

        # stats
        cur = len(detections)
        self._total += cur
        if cur > self._peak:
            self._peak = cur
        for *_, label, score, color, tid in detections:
            self._class_totals[label] += 1

        self.lbl_now.setText(str(cur))
        self.lbl_tot.setText(str(self._total))
        self.lbl_peak.setText(str(self._peak))
        self.lbl_cls.setText(str(len(self._class_totals)))

        # class breakdown list
        self.class_list.clear()
        for label, count in sorted(
            self._class_totals.items(), key=lambda x: x[1], reverse=True
        ):
            c = next(
                (col for *_, l, sc, col, tid in detections if l == label),
                (0, 255, 100),
            )
            b, g, r = c
            item = QListWidgetItem(f"  {label:<18} x{count}")
            item.setForeground(QColor(r, g, b))
            self.class_list.addItem(item)

        # live tracks list
        self.det_list.clear()
        for x1, y1, x2, y2, label, score, color, tid in detections:
            b, g, r = color
            item = QListWidgetItem(f"  #{tid:<4} {label:<18} {score:.2f}")
            item.setForeground(QColor(r, g, b))
            self.det_list.addItem(item)

        # frame → QPixmap



        def draw_zones(frame, zones):
            for ztype, polys in zones.items():
                for poly in polys:
                    pts = np.array(poly, np.int32)
        
                    if ztype == "restricted":        
                        color = (0, 0, 255)
                    elif ztype == "safe":
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
        
                    cv2.polylines(frame, [pts], True, color, 2)
        
        draw_zones(frame, self.zones)        


        for x1, y1, x2, y2, label, score, color, tid in detections:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            for ztype, polys in self.zones.items():
                for poly in polys:
                    if point_in_poly((cx, cy), poly):
                        if ztype == "restricted":
                            cv2.putText(frame, "RESTRICTED!", (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        elif ztype == "safe":
                            cv2.putText(frame, "SAFE", (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        elif ztype == "tracking":
                            cv2.putText(frame, "TRACKING", (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
          






        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg     = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix      = QPixmap.fromImage(qimg).scaled(
            self.video_lbl.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_lbl.setPixmap(pix)

    def _show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self._stop()

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)