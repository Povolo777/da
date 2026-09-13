"""
camera_test.py — diagnostic script to figure out why OpenCV can't open
the webcam. Tries every camera index (0-4) against every relevant
backend on this OS and reports which combinations actually work.

Run this standalone: python camera_test.py
"""

import sys
import cv2

if sys.platform == "win32":
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("CAP_ANY (default)", cv2.CAP_ANY),
    ]
else:
    backends = [("CAP_ANY (default)", cv2.CAP_ANY)]

print(f"OpenCV version: {cv2.__version__}")
print(f"Platform: {sys.platform}")
print("-" * 50)

found_any = False
for index in range(5):
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        opened = cap.isOpened()
        frame_ok = False
        if opened:
            ok, frame = cap.read()
            frame_ok = ok and frame is not None
        cap.release()

        status = "OK, frame read" if frame_ok else ("opened but no frame" if opened else "failed to open")
        print(f"index={index:<2} backend={name:<20} -> {status}")
        if frame_ok:
            found_any = True

print("-" * 50)
if found_any:
    print("At least one combination worked — use that index/backend in vision.py.")
else:
    print("Nothing worked. This points to the camera being blocked at the OS/driver")
    print("level rather than an OpenCV problem. Check:")
    print("  1. Settings > Privacy & security > Camera > both toggles ON")
    print("     ('Camera access' AND 'Let desktop apps access your camera')")
    print("  2. Device Manager > Cameras > is the webcam listed with no warning icon?")
    print("  3. Is any other app (Teams/Zoom/browser tab/another Python process)")
    print("     currently holding the camera open?")
    print("  4. Antivirus/security software with a 'webcam protection' feature")
    print("     that silently blocks unrecognized apps (common in Windows Defender,")
    print("     Kaspersky, Norton, Bitdefender — check their camera privacy settings).")
