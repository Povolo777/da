"""
memory.py — Alex's persistent memory: notes and known facts about the
user, stored outside config.json so they survive restarts and aren't
mixed up with settings.

Two kinds of entries:
  - notes: free-form things the user asked Alex to remember
           ("remind me to renew my ID", "my landlord's number is ...")
  - facts: short key/value style things Alex has learned about the
           user (name, preferences) — used to personalize replies

Everything is plain JSON on disk, one file, thread-safe writes.
"""

import json
import threading
import time
import uuid
from pathlib import Path

MEMORY_PATH = Path(__file__).parent / "memory.json"

_EMPTY = {"notes": [], "facts": {}}


class Memory:
    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write(_EMPTY)

    def _read(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return dict(_EMPTY)

    def _write(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    # ---- notes ----------------------------------------------------

    def add_note(self, text: str, tag: str = None) -> str:
        with self._lock:
            data = self._read()
            note = {
                "id": uuid.uuid4().hex[:8],
                "text": text,
                "tag": tag,
                "created": time.time(),
            }
            data["notes"].append(note)
            self._write(data)
            return note["id"]

    def list_notes(self, tag: str = None) -> list:
        data = self._read()
        notes = data["notes"]
        if tag:
            notes = [n for n in notes if n.get("tag") == tag]
        return notes

    def search_notes(self, query: str) -> list:
        q = query.lower()
        return [n for n in self._read()["notes"] if q in n["text"].lower()]

    def delete_note(self, note_id: str) -> bool:
        with self._lock:
            data = self._read()
            before = len(data["notes"])
            data["notes"] = [n for n in data["notes"] if n["id"] != note_id]
            self._write(data)
            return len(data["notes"]) < before

    # ---- facts (name, preferences, etc.) ---------------------------

    def set_fact(self, key: str, value):
        with self._lock:
            data = self._read()
            data["facts"][key] = value
            self._write(data)

    def get_fact(self, key: str, default=None):
        return self._read()["facts"].get(key, default)

    def all_facts(self) -> dict:
        return self._read()["facts"]

    # ---- context helper for brain.py --------------------------------

    def context_summary(self, max_notes: int = 5) -> str:
        """A short string injected into the system prompt so Alex
        always has a bit of persistent context about the user, without
        dumping the whole memory file into every request."""
        data = self._read()
        parts = []
        facts = data["facts"]
        if facts:
            fact_str = ", ".join(f"{k}: {v}" for k, v in facts.items())
            parts.append(f"Known facts about the user: {fact_str}.")
        recent = data["notes"][-max_notes:]
        if recent:
            notes_str = "; ".join(n["text"] for n in recent)
            parts.append(f"Recent notes: {notes_str}.")
        return " ".join(parts) if parts else ""