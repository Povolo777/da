import sounddevice as sd
import numpy as np

print("Recording 3 seconds, speak now...")
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32")
sd.wait()
print("Recording finished.")
print("Max amplitude:", np.abs(audio).max())
print("Mean amplitude:", np.abs(audio).mean())
if np.abs(audio).max() < 0.001:
    print("WARNING: essentially silence captured. Wrong mic or muted input.")
else:
    print("Audio captured successfully - mic is working.")