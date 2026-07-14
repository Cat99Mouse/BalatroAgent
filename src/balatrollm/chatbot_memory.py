"""Chatbot-mode memory rendering."""

import json
from typing import Any


class ChatbotMemory:
    """Render a minimal memory block for chatbot mode."""

    def render(self, history: list[dict[str, Any]]) -> str:
        """Render only the last 10 successful actions."""
        recent = history[-10:]
        lines = [
            "# Recent Successful Actions",
            "",
            "Only successful gameplay actions are listed here.",
        ]
        if not recent:
            lines.extend(["", "No successful actions yet."])
            return "\n".join(lines)

        lines.extend(["", "From oldest to newest:"])
        for index, entry in enumerate(recent, 1):
            method = str(entry.get("method", "unknown"))
            params = entry.get("params", {})
            params_text = json.dumps(params, separators=(",", ":"), sort_keys=True)
            lines.append(f"{index}. `{method}({params_text})`")
        return "\n".join(lines)
