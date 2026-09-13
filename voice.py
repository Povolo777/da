"""
voice.py — speech in (wake word + command capture via faster-whisper)
and speech out for Alex.

TTS engine is selectable via config: "piper" (offline neural TTS,
sounds far more natural, needs the piper-tts package + a voice model)
or "pyttsx3" (the original robotic-but-zero-setup fallback). If Piper
isn't installed or fails to load, Alex automatically falls back to
pyttsx3 so voice output never just breaks.
"""

import queue
import subprocess
import threading
import numpy as np
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel


class Speaker:
    """Thread-safe TTS wrapper. Tries Piper first if configured, falls
    back to pyttsx3 if Piper isn't available."""

    def __init__(self, config: dict):
        self._lock = threading.Lock()
        vcfg = config["voice"]
        self._rate = vcfg["rate"]
        self._volume = vcfg["volume"]
        self._engine_name = vcfg.get("tts_engine", "pyttsx3")
        self._piper_voice = vcfg.get("piper_voice_path")  # e.g. "en_US-lessac-medium.onnx"
        self._piper = None

        if self._engine_name == "piper":
            self._piper = self._try_load_piper()
            if self._piper is None:
                print("[voice] Piper unavailable, falling back to pyttsx3.")
                self._engine_name = "pyttsx3"

    def _try_load_piper(self):
        if not self._piper_voice:
            print("[voice] tts_engine is 'piper' but voice.piper_voice_path isn't set in config.")
            return None
        try:
            from piper.voice import PiperVoice
            import sounddevice as _sd
            voice = PiperVoice.load(self._piper_voice)
            return voice
        except Exception as e:
            print(f"[voice] couldn't load Piper: {e}")
            return None

    def _speak_piper(self, text: str):
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._piper.config.sample_rate)
            self._piper.synthesize(text, wav_file)
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
            sd.play(audio, samplerate=wav_file.getframerate())
            sd.wait()

    def _speak_pyttsx3(self, text: str):
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def speak(self, text: str):
        with self._lock:
            print(f"[alex says] {text}")
            try:
                if self._engine_name == "piper" and self._piper is not None:
                    self._speak_piper(text)
                else:
                    self._speak_pyttsx3(text)
            except Exception as e:
                print(f"[voice] TTS failed ({e}), retrying with pyttsx3.")
                try:
                    self._speak_pyttsx3(text)
                except Exception as e2:
                    print(f"[voice] pyttsx3 fallback also failed: {e2}")


def record_chunk(duration_seconds: float, sample_rate: int) -> np.ndarray:
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def listen_loop(
    on_command,
    config: dict,
    stop_event: threading.Event,
    speaker=None,
):
    """Continuously listens in chunk_seconds windows, transcribes with
    faster-whisper, and calls on_command(text) whenever the wake word
    is heard, passing along whatever follows it in the same chunk.

    If the wake word is said alone, gives a quick spoken cue ("Da?")
    so the user knows to speak now, then opens a follow-up listening
    window for the actual command.
    """
    acfg = config["audio"]
    sample_rate = acfg["sample_rate"]
    chunk_seconds = acfg["chunk_seconds"]
    wake_word = config["wake_word"].lower()
    # None means "auto-detect" — faster-whisper will figure out per chunk
    # whether the user spoke Romanian, English, or switched mid-conversation.
    forced_language = acfg.get("language")

    print(f"[voice] loading whisper model '{acfg['whisper_model_size']}'...")
    model = WhisperModel(acfg["whisper_model_size"], device="cpu", compute_type="int8")

    print(f"[voice] listening for wake word '{wake_word}'...")

    while not stop_event.is_set():
        audio = record_chunk(chunk_seconds, sample_rate)
        segments, info = model.transcribe(
            audio, language=forced_language, beam_size=5, vad_filter=True
        )
        segments = list(segments)
        text = " ".join(seg.text for seg in segments).strip()

        if not text:
            print("[voice] (heard silence / nothing transcribable)")
            continue

        print(f"[voice] transcribed ({info.language}, p={info.language_probability:.2f}): {text!r}")
        lower = text.lower()
        if wake_word in lower:
            print(f"[voice] heard: {text}")
            idx = lower.find(wake_word)
            remainder = text[idx + len(wake_word):].strip(" ,.:!?")

            if not remainder:
                # Wake word only — cue the user, then capture the next
                # chunk as the command.
                print("[voice] wake word alone — cueing user and listening for the command...")
                if speaker:
                    speaker.speak("Da?")
                follow_up = record_chunk(chunk_seconds, sample_rate)
                segs, follow_info = model.transcribe(
                    follow_up, language=forced_language, beam_size=5, vad_filter=True
                )
                remainder = " ".join(s.text for s in segs).strip()
                if remainder:
                    print(f"[voice] follow-up transcribed ({follow_info.language}): {remainder!r}")
                else:
                    print("[voice] follow-up: heard silence / nothing transcribable — no command captured.")

            if remainder:
                on_command(remainder)


if __name__ == "__main__":
    import json
    with open("config.json") as f:
        cfg = json.load(f)
    sp = Speaker(cfg)
    sp.speak("Voice module online.")
    stop = threading.Event()
    try:
        listen_loop(lambda text: print("COMMAND:", text), cfg, stop)
    except KeyboardInterrupt:
        stop.set()