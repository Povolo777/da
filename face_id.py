"""
face_id.py — lightweight face recognition so Alex can tell "the user"
apart from "some other face" in front of the camera.

Deliberately uses OpenCV's built-in Haar cascade + LBPH recognizer
instead of dlib/face_recognition: those need a C++ compiler toolchain
on Windows and are a common install headache. This trades a bit of
accuracy for something that works out of the box with just
opencv-contrib-python.

Flow:
  1. Run `python enroll_face.py` once — captures ~20 frames of your
     face from the webcam and trains a small local model on it.
  2. vision.py loads the trained model and, on each detection tick,
     checks whether the primary face in frame matches the enrolled
     user closely enough (confidence threshold) or is "unknown".
"""

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "face_model"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.txt"

# Lower = stricter match required. LBPH confidence is a distance, not
# a probability — smaller means more similar. 70-90 is a reasonable
# starting threshold for "same person" under normal webcam lighting.
MATCH_THRESHOLD = 80

CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
# Fallback local copy — some opencv-contrib-python builds (notably on
# very new Python versions) ship without the bundled data/ files, so
# cv2.data.haarcascades points at a path that doesn't actually exist.
_LOCAL_CASCADE_PATH = Path(__file__).parent / "haarcascade_frontalface_default.xml"


def _resolve_cascade_path() -> str:
    bundled = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(bundled):
        return bundled

    if not _LOCAL_CASCADE_PATH.exists():
        print("[face_id] OpenCV's bundled cascade file is missing on this install — "
              "downloading a copy (one-time, ~1MB)...")
        urllib.request.urlretrieve(CASCADE_URL, _LOCAL_CASCADE_PATH)
        print("[face_id] cascade downloaded.")
    return str(_LOCAL_CASCADE_PATH)


_CASCADE_PATH = _resolve_cascade_path()


class FaceID:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(_CASCADE_PATH)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}  # int label -> name
        self.trained = False
        self._load()

    def _load(self):
        if MODEL_PATH.exists() and LABELS_PATH.exists():
            self.recognizer.read(str(MODEL_PATH))
            with open(LABELS_PATH) as f:
                self.labels = {int(k): v for k, v in (line.strip().split(",", 1) for line in f if line.strip())}
            self.trained = True

    def detect_faces(self, gray_frame):
        """Returns list of (x, y, w, h) boxes."""
        return self.detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    def identify(self, gray_frame) -> str | None:
        """Returns a name if a known face is recognized with enough
        confidence, 'unknown_face' if a face is seen but not recognized,
        or None if no face is in frame at all."""
        faces = self.detect_faces(gray_frame)
        if len(faces) == 0:
            return None
        if not self.trained:
            return "unknown_face"

        # Use the largest face (closest to camera) as the primary subject.
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        label, confidence = self.recognizer.predict(roi)

        if confidence <= MATCH_THRESHOLD:
            return self.labels.get(label, "unknown_face")
        return "unknown_face"

    @staticmethod
    def train(samples: list[np.ndarray], name: str = "user"):
        """samples: list of grayscale face-ROI images (all same size).
        Trains and persists a fresh model, overwriting any previous one."""
        MODEL_DIR.mkdir(exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        labels_arr = np.array([0] * len(samples))
        recognizer.train(samples, labels_arr)
        recognizer.write(str(MODEL_PATH))
        with open(LABELS_PATH, "w") as f:
            f.write(f"0,{name}\n")