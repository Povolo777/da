"""
enroll_face.py — run this ONCE to teach Alex your face.

Opens your webcam, captures ~20 clear samples of your face as you
move/turn slightly, and trains a local recognition model from them.
Nothing is uploaded anywhere — it's a small file in ./face_model/.

Usage:
    python enroll_face.py
    python enroll_face.py --name Pavel     (optional, defaults to "user")
"""

import argparse
import sys
import time

import cv2

from face_id import FaceID

SAMPLES_NEEDED = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="user")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    face_id = FaceID()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera_index}")
        sys.exit(1)

    samples = []
    print(f"Enrolling face for '{args.name}'. Look at the camera and slowly turn "
          f"your head left/right a little. Capturing {SAMPLES_NEEDED} samples...")

    last_capture = 0
    while len(samples) < SAMPLES_NEEDED:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_id.detect_faces(gray)

        now = time.time()
        if len(faces) > 0 and (now - last_capture) > 0.3:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            samples.append(roi)
            last_capture = now
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"  captured {len(samples)}/{SAMPLES_NEEDED}")

        cv2.putText(frame, f"Samples: {len(samples)}/{SAMPLES_NEEDED}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Enroll Face - press q to abort", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

    cap.release()
    cv2.destroyAllWindows()

    FaceID.train(samples, name=args.name)
    print(f"Done. Trained and saved model for '{args.name}' in ./face_model/. "
          f"Alex will now recognize you on startup.")


if __name__ == "__main__":
    main()
    