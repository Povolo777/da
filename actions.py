"""
actions.py — turns structured actions from the brain into real effects.

Anything that modifies Alex's own config or code goes through a
confirmation step printed to console (and spoken) before it's applied.
This is the "self-modification" layer, done safely: propose -> confirm -> apply.
"""

import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

from memory import Memory

CONFIG_PATH = Path(__file__).parent / "config.json"
memory = Memory()

# Common Windows app name -> executable mappings, since a spoken name
# like "notepad" or "spotify" rarely matches os.startfile() directly.
APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "chrome": "chrome",
    "google chrome": "chrome",
    "spotify": "spotify",
    "word": "winword",
    "excel": "excel",
}


def _resolve_app(target: str) -> str:
    """Resolve a spoken app name to something the OS can actually launch."""
    key = target.strip().lower()
    if key in APP_ALIASES:
        return APP_ALIASES[key]
    found = shutil.which(target)
    if found:
        return found
    return target  # last resort, let the OS try


def open_app(target: str):
    resolved = _resolve_app(target)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-a", resolved])
        elif sys.platform == "win32":
            os.startfile(resolved)  # noqa
        else:
            subprocess.Popen([resolved])
        return f"Opening {target}."
    except Exception as e:
        return f"Couldn't open {target}: {e}"


def close_app(target: str):
    """Best-effort close by process name. Windows/Linux only for now."""
    try:
        if sys.platform == "win32":
            name = target if target.lower().endswith(".exe") else f"{target}.exe"
            result = subprocess.run(
                ["taskkill", "/IM", name, "/F"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"Closed {target}."
            return f"Couldn't find {target} running."
        else:
            result = subprocess.run(["pkill", "-i", target], capture_output=True, text=True)
            return f"Closed {target}." if result.returncode == 0 else f"Couldn't find {target} running."
    except Exception as e:
        return f"Couldn't close {target}: {e}"


def type_text(text: str):
    """Types text into whatever window currently has focus."""
    try:
        import pyautogui
        pyautogui.write(text, interval=0.02)
        return f"Typed: {text}"
    except Exception as e:
        return f"Couldn't type text: {e}"


def search_web(query: str):
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    webbrowser.open(url)
    return f"Searching the web for {query}."


def take_screenshot(save_path: str = "screenshot.png"):
    try:
        import pyautogui
        img = pyautogui.screenshot()
        img.save(save_path)
        return f"Screenshot saved to {save_path}."
    except Exception as e:
        return f"Couldn't take screenshot: {e}"


def update_config(key_path: str, value, speaker=None):
    """Propose a config change, ask for confirmation in the terminal,
    only apply if the user confirms. key_path uses dot notation,
    e.g. 'voice.rate'.
    """
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    keys = key_path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    old_value = node.get(keys[-1])

    prompt = f"Alex wants to change {key_path} from {old_value!r} to {value!r}. Apply? [y/N] "
    if speaker:
        speaker.speak(f"I'd like to change {key_path} to {value}. Say or type yes to confirm.")
    answer = input(prompt).strip().lower()

    if answer == "y" or answer == "yes":
        node[keys[-1]] = value
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        return f"Updated {key_path} to {value}."
    return "Change discarded."


def propose_code_edit(file_path: str, description: str, diff: str, speaker=None):
    """Never auto-applies. Prints the diff and requires manual application.
    This keeps a human in the loop for anything touching source code.
    """
    print("\n" + "=" * 60)
    print(f"PROPOSED CODE CHANGE to {file_path}")
    print(f"Reason: {description}")
    print("-" * 60)
    print(diff)
    print("=" * 60)
    if speaker:
        speaker.speak(f"I have a proposed change to {file_path}. Check the terminal to review it.")
    return "Proposed change printed to terminal. Not applied automatically — review and apply it yourself."


def save_note(text: str, tag: str = None):
    note_id = memory.add_note(text, tag)
    return f"Noted. ({note_id})"


def recall_notes(query: str = None, tag: str = None):
    if query:
        notes = memory.search_notes(query)
    else:
        notes = memory.list_notes(tag)
    if not notes:
        return "I don't have any notes matching that."
    return " | ".join(f"[{n['id']}] {n['text']}" for n in notes[-10:])


def forget_note(note_id: str):
    ok = memory.delete_note(note_id)
    return f"Deleted note {note_id}." if ok else f"No note found with id {note_id}."


ACTION_TABLE = {
    "open_app": lambda args, speaker: open_app(args["target"]),
    "close_app": lambda args, speaker: close_app(args["target"]),
    "type_text": lambda args, speaker: type_text(args["text"]),
    "search_web": lambda args, speaker: search_web(args["query"]),
    "take_screenshot": lambda args, speaker: take_screenshot(args.get("save_path", "screenshot.png")),
    "update_config": lambda args, speaker: update_config(args["key_path"], args["value"], speaker),
    "propose_code_edit": lambda args, speaker: propose_code_edit(
        args["file_path"], args["description"], args["diff"], speaker
    ),
    "save_note": lambda args, speaker: save_note(args["text"], args.get("tag")),
    "recall_notes": lambda args, speaker: recall_notes(args.get("query"), args.get("tag")),
    "forget_note": lambda args, speaker: forget_note(args["note_id"]),
}


def dispatch(action_name: str, args: dict, speaker=None) -> str:
    handler = ACTION_TABLE.get(action_name)
    if not handler:
        return f"Unknown action: {action_name}"
    try:
        return handler(args, speaker)
    except Exception as e:
        return f"Action '{action_name}' failed: {e}"
    