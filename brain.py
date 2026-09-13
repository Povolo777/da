
from engine import build_engine
from memory import Memory

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application on the user's computer.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "App name or path"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Open a web search in the browser for the given query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the user's screen and save it.",
            "parameters": {
                "type": "object",
                "properties": {"save_path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_config",
            "description": "Change one of Alex's own config values (e.g. voice.rate, wake_word). Always requires user confirmation before applying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key_path": {"type": "string", "description": "Dot-notation path, e.g. 'voice.rate'"},
                    "value": {"description": "New value"},
                },
                "required": ["key_path", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_code_edit",
            "description": "Propose (but do not apply) a code change to one of Alex's own source files. Only use this for actual source code edits, never for simple settings (use update_config for those).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "description": {"type": "string"},
                    "diff": {"type": "string", "description": "Unified diff or before/after snippet"},
                },
                "required": ["file_path", "description", "diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_app",
            "description": "Close/terminate a running application by name.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "App/process name"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into whatever window currently has focus on the user's screen.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a short note or reminder the user asked you to remember. Use this any time the user says things like 'remember that...', 'note that...', or asks you to keep track of something for later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note content"},
                    "tag": {"type": "string", "description": "Optional short category, e.g. 'reminder', 'idea'"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_notes",
            "description": "Look up notes the user previously asked you to remember, by keyword search or tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search notes for"},
                    "tag": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_note",
            "description": "Delete a previously saved note by its id.",
            "parameters": {
                "type": "object",
                "properties": {"note_id": {"type": "string"}},
                "required": ["note_id"],
            },
        },
    },
]


class Brain:
    def __init__(self, config: dict):
        self.bcfg = config["brain"]
        self.engine = build_engine(self.bcfg)
        self.memory = Memory()
        base_prompt = self.bcfg["system_prompt"]
        context = self.memory.context_summary()
        full_prompt = f"{base_prompt}\n\n{context}" if context else base_prompt
        self.history = [{"role": "system", "content": full_prompt}]

        self.engine.check_available()

    def _build_context_message(self, user_text: str, vision_snapshot: dict) -> str:
        objects = vision_snapshot.get("objects") or []
        gesture = vision_snapshot.get("gesture")
        face = vision_snapshot.get("face")
        vision_desc = []
        if face:
            if face == "unknown_face":
                vision_desc.append("camera sees an unrecognized face")
            else:
                vision_desc.append(f"camera recognizes {face} (the user)")
        if objects:
            vision_desc.append(f"visible objects: {', '.join(objects)}")
        if gesture:
            vision_desc.append(f"detected hand gesture: {gesture}")
        vision_str = "; ".join(vision_desc) if vision_desc else "no notable objects or gestures detected"

        return f"[Camera context: {vision_str}]\nUser said: {user_text}"

    def respond(self, user_text: str, vision_snapshot: dict):
        """Returns (spoken_text, action_or_None).
        action_or_None is a dict {'name': str, 'input': dict} if the
        model decided to call a tool.
        """
        content = self._build_context_message(user_text, vision_snapshot)
        self.history.append({"role": "user", "content": content})

        try:
            result = self.engine.chat(
                messages=self.history,
                tools=TOOLS,
                max_tokens=self.bcfg.get("max_tokens", 500),
            )
        except Exception as e:
            # Engine died mid-conversation (crashed, model unloaded, etc).
            # Don't crash the whole assistant over it — drop the failed
            # turn from history and surface a spoken apology instead.
            self.history.pop()
            return f"Sorry, I couldn't reach my brain just now: {e}", None

        spoken_text = result["content"]
        action = None
        if result["tool_calls"]:
            action = result["tool_calls"][0]

        # Safety net: some local models occasionally write a fake tool call
        # as plain JSON text instead of using the real tool-calling
        # mechanism. Catch that so we never speak raw JSON out loud.
        stripped = spoken_text.strip()
        if not action and stripped.startswith("{") and '"name"' in stripped:
            spoken_text = "Sorry, I got a bit confused there — could you rephrase that?"

        self.history.append({"role": "assistant", "content": spoken_text})

        # keep history bounded (system prompt + last ~20 turns)
        if len(self.history) > 21:
            self.history = [self.history[0]] + self.history[-20:]

        return spoken_text.strip(), action