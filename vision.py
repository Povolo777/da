"""
vision.py — webcam-based object + gesture recognition for Alex.

Uses MediaPipe's Tasks API (HandLandmarker) rather than the older
mp.solutions.hands API, since recent mediapipe builds on Windows no
longer ship the legacy solutions module.

Runs in its own thread. Pushes a rolling snapshot of what's currently
seen (objects + gesture) into a thread-safe VisionState object that
main.py / brain.py can read at any time.
"""

import os
import sys
import time
import threading
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from face_id import FaceID


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("[vision] downloading hand landmark model (one-time, ~8MB)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("[vision] model downloaded.")


class VisionState:
    """Thread-safe holder for the latest vision snapshot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._objects = []
        self._gesture = None
        self._face = None
        self._last_update = 0

    def update(self, objects=None, gesture=None, face=None):
        with self._lock:
            if objects is not None:
                self._objects = objects
            self._gesture = gesture
            self._face = face
            self._last_update = time.time()

    def snapshot(self):
        with self._lock:
            return {
                "objects": list(self._objects),
                "gesture": self._gesture,
                "face": self._face,
                "age_seconds": round(time.time() - self._last_update, 1),
            }


def classify_gesture(landmarks):
    """Rule-based gesture classifier from a list of 21 landmark points
    (each with .x, .y in normalized [0,1] image coordinates).

    Returns one of: 'open_palm', 'fist', 'thumbs_up', 'peace', 'point', None
    """

    def finger_up(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    thumb_up = (
        landmarks[4].x < landmarks[3].x
        if landmarks[17].x < landmarks[0].x
        else landmarks[4].x > landmarks[3].x
    )
    index_up = finger_up(8, 6)
    middle_up = finger_up(12, 10)
    ring_up = finger_up(16, 14)
    pinky_up = finger_up(20, 18)

    count_up = sum([index_up, middle_up, ring_up, pinky_up])

    if count_up == 0 and not thumb_up:
        return "fist"
    if count_up == 4:
        return "open_palm"
    if thumb_up and count_up == 0:
        return "thumbs_up"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "point"
    return None


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)


def _open_camera(camera_index: int):
    """Windows' default OpenCV backend (MSMF) has known issues failing
    to open webcams that DirectShow handles fine. Try DSHOW first on
    Windows, fall back to the OS default elsewhere / if DSHOW fails."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def run_vision(state: VisionState, config: dict, stop_event: threading.Event):
    """Main vision loop. Call this in a background thread."""
    vcfg = config["vision"]
    cap = _open_camera(vcfg["camera_index"])
    if not cap.isOpened():
        print(f"[vision] ERROR: could not open camera index {vcfg['camera_index']}. "
              f"Check Windows camera privacy settings (Settings > Privacy > Camera), "
              f"make sure no other app is using the webcam, and confirm the index is "
              f"correct — try camera_index 1 in config.json if you have more than one camera.")
        return

    hand_landmarker = None
    if vcfg["enable_gesture_detection"]:
        ensure_hand_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    yolo_model = None
    if vcfg["enable_object_detection"] and YOLO_AVAILABLE:
        print("[vision] loading YOLOv8n...")
        yolo_model = YOLO("yolov8n.pt")

    face_id = FaceID()
    if vcfg.get("enable_face_recognition", True):
        if face_id.trained:
            print(f"[vision] face recognition ready, enrolled: {list(face_id.labels.values())}")
        else:
            print("[vision] no enrolled face found — run 'python enroll_face.py' first "
                  "to teach Alex your face. Continuing without face recognition.")

    last_detect_time = 0
    interval = vcfg["detection_interval_seconds"]

    print("[vision] started. Press 'q' in the video window to quit vision only.")

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        now = time.time()
        do_detect = (now - last_detect_time) >= interval

        gesture = None
        if hand_landmarker is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hand_landmarker.detect(mp_image)
                if result.hand_landmarks:
                    gestures = []
                    for landmarks in result.hand_landmarks:
                        g = classify_gesture(landmarks)
                        if g:
                            gestures.append(g)
                        draw_landmarks(frame, landmarks)
                    if gestures:
                        # e.g. "peace" for one hand, "peace, fist" for two
                        gesture = ", ".join(gestures)
            except Exception as e:
                print(f"[vision] hand detection failed on this frame: {e}")

        objects = None
        face = None
        if do_detect:
            if vcfg.get("enable_face_recognition", True):
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face = face_id.identify(gray)
                except Exception as e:
                    print(f"[vision] face recognition failed on this frame: {e}")

            if yolo_model is not None:
                try:
                    results = yolo_model(frame, verbose=False)
                    names = yolo_model.names
                    objects = sorted({names[int(c)] for c in results[0].boxes.cls}) if len(results) else []
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = names[int(box.cls[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"[vision] object detection failed on this frame: {e}")

            last_detect_time = now

        if gesture is not None or objects is not None or face is not None:
            state.update(objects=objects, gesture=gesture, face=face)

        if face:
            label = "You" if face != "unknown_face" else "Unknown face"
            cv2.putText(frame, label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("Alex Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[vision] stopped.")


if __name__ == "__main__":
    import json
    with open("config.json") as f:
        cfg = json.load(f)
    st = VisionState()
    stop = threading.Event()
    try:
        run_vision(st, cfg, stop)
    except KeyboardInterrupt:
        stop.set()