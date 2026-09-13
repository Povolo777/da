"""
main.py — run this. Starts vision in a background thread and voice
listening in the main thread, routing recognized commands through
the brain and dispatching any resulting actions.
"""

import json
import threading
from pathlib import Path

from vision import VisionState, run_vision
from voice import Speaker, listen_loop
from brain import Brain
from actions import dispatch

CONFIG_PATH = Path(__file__).parent / "config.json"


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    vision_state = VisionState()
    speaker = Speaker(config)
    brain = Brain(config)
    stop_event = threading.Event()

    vision_thread = threading.Thread(
        target=run_vision, args=(vision_state, config, stop_event), daemon=True
    )
    vision_thread.start()

    speaker.speak(f"{config['assistant_name']} online.")

    def handle_command(text: str):
        snapshot = vision_state.snapshot()
        reply, action = brain.respond(text, snapshot)

        if reply:
            speaker.speak(reply)

        if action:
            result = dispatch(action["name"], action["input"], speaker)
            print(f"[action result] {result}")

    try:
        listen_loop(handle_command, config, stop_event, speaker=speaker)
    except KeyboardInterrupt:
        print("\n[main] shutting down...")
        stop_event.set()
        vision_thread.join(timeout=2)


if __name__ == "__main__":
    main()
    