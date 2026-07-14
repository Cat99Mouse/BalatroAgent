"""Unit tests for chatbot-mode memory rendering."""

from balatrollm.chatbot_memory import ChatbotMemory


class TestChatbotMemory:
    """Tests for chatbot memory output."""

    def test_empty_history(self) -> None:
        """Empty history should render without errors."""
        result = ChatbotMemory().render([])

        assert "Recent Successful Actions" in result
        assert "No successful actions yet." in result

    def test_renders_only_last_10_actions(self) -> None:
        """Only the last 10 successful actions should be rendered."""
        history = [
            {
                "method": "play",
                "params": {"cards": [i], "reasoning": f"move-{i:02d}"},
            }
            for i in range(12)
        ]

        result = ChatbotMemory().render(history)

        assert "move-00" not in result
        assert "move-01" not in result
        for i in range(2, 12):
            assert f"move-{i:02d}" in result

    def test_does_not_render_error_context(self) -> None:
        """Chatbot memory should not include agent error or failure sections."""
        history = [{"method": "discard", "params": {"cards": [0]}}]

        result = ChatbotMemory().render(history)

        assert "last response was invalid" not in result
        assert "last tool call failed" not in result
        assert "Strategic Context" not in result
