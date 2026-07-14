"""Unit tests for Bot observation tools."""

import json
from types import SimpleNamespace
from typing import Any

from balatrollm.bot import MAX_GLOBAL_MEMORY_CHARS, MEMORY_UPDATE_FIELD, Bot
from balatrollm.client import BalatroError
from balatrollm.config import Config, Task


def _make_bot(mode: str = "agent") -> Bot:
    task = Task(
        model="openai/gpt-4",
        seed="AAAAAAA",
        deck="RED",
        stake="WHITE",
        strategy="default",
    )
    return Bot(task=task, config=Config(model=[task.model], mode=mode))


class FakeBalatro:
    """Minimal fake Balatro client for observation tests."""

    def __init__(self, gamestate: dict[str, Any]) -> None:
        self.gamestate = gamestate
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, params))
        if method != "gamestate":
            raise AssertionError(f"Unexpected method: {method}")
        return self.gamestate


class FakeActionBalatro:
    """Fake Balatro client for action execution tests."""

    def __init__(
        self, gamestate: dict[str, Any] | None = None, fail_actions: bool = False
    ) -> None:
        self.gamestate = gamestate or {
            "state": "SELECTING_HAND",
            "won": False,
            "ante_num": 1,
            "round_num": 1,
        }
        self.fail_actions = fail_actions
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "gamestate":
            return self.gamestate
        if self.fail_actions:
            raise BalatroError(400, "action failed", {"name": "FakeBalatro"})
        return self.gamestate


class FakeCollector:
    """Minimal collector stub for action execution tests."""

    def __init__(self) -> None:
        self.call_outcomes: list[str] = []
        self.failures = 0
        self.gamestates: list[dict[str, Any]] = []
        self.memory_updates: list[dict[str, Any]] = []
        self.reset_count = 0

    def reset_failures(self) -> None:
        self.reset_count += 1

    def record_call(self, outcome: str) -> None:
        self.call_outcomes.append(outcome)

    def record_failure(self) -> None:
        self.failures += 1

    def write_gamestate(self, gamestate: dict[str, Any]) -> None:
        self.gamestates.append(gamestate)

    def write_global_memory_update(
        self,
        *,
        index: int,
        request: str | None,
        method: str,
        memory: str,
        truncated: bool,
    ) -> None:
        self.memory_updates.append(
            {
                "index": index,
                "request": request,
                "method": method,
                "memory": memory,
                "truncated": truncated,
            }
        )


def _response(function_name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name=function_name,
                                arguments=json.dumps(arguments),
                            )
                        )
                    ]
                )
            )
        ]
    )


def _tool_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(tool for tool in tools if tool["function"]["name"] == name)


class TestBotObservationTools:
    """Tests for read-only observation tools added to LLM calls."""

    def test_get_tools_adds_remaining_deck_observation(self) -> None:
        """Agent observations are available for LLM-driven states."""
        bot = _make_bot()

        selecting_tools = bot._get_tools_for_state(
            "SELECTING_HAND", include_observation=True
        )
        selecting_names = [tool["function"]["name"] for tool in selecting_tools]
        assert "observe_remaining_deck" in selecting_names
        assert "observe_run_info" in selecting_names
        assert "score_candidates" not in selecting_names

        shop_tools = bot._get_tools_for_state("SHOP", include_observation=True)
        shop_names = [tool["function"]["name"] for tool in shop_tools]
        assert "observe_remaining_deck" in shop_names
        assert "observe_run_info" in shop_names
        assert "score_candidates" not in shop_names

        action_only_tools = bot._get_tools_for_state(
            "SELECTING_HAND", include_observation=False
        )
        action_only_names = [tool["function"]["name"] for tool in action_only_tools]
        assert "observe_remaining_deck" not in action_only_names
        assert "observe_run_info" not in action_only_names
        assert "score_candidates" not in action_only_names

        play_tool = _tool_by_name(selecting_tools, "play")
        play_params = play_tool["function"]["parameters"]
        assert MEMORY_UPDATE_FIELD in play_params["properties"]
        assert MEMORY_UPDATE_FIELD in play_params["required"]

    def test_get_tools_omits_observations_in_chatbot_mode(self) -> None:
        """Chatbot mode never exposes observation tools."""
        bot = _make_bot(mode="chatbot")

        selecting_tools = bot._get_tools_for_state(
            "SELECTING_HAND", include_observation=True
        )
        selecting_names = [tool["function"]["name"] for tool in selecting_tools]

        assert "play" in selecting_names
        assert "discard" in selecting_names
        assert "observe_remaining_deck" not in selecting_names
        assert "observe_run_info" not in selecting_names
        assert "score_candidates" not in selecting_names

        play_tool = _tool_by_name(selecting_tools, "play")
        play_params = play_tool["function"]["parameters"]
        assert MEMORY_UPDATE_FIELD not in play_params["properties"]
        assert MEMORY_UPDATE_FIELD not in play_params["required"]

    def test_agent_memory_update_does_not_mutate_strategy_tools(self) -> None:
        """Runtime memory_update injection should not alter shared TOOLS.json data."""
        bot = _make_bot()

        bot._get_tools_for_state("SELECTING_HAND", include_observation=True)
        raw_play_tool = _tool_by_name(bot.strategy.get_tools("SELECTING_HAND"), "play")
        raw_params = raw_play_tool["function"]["parameters"]

        assert MEMORY_UPDATE_FIELD not in raw_params["properties"]
        assert MEMORY_UPDATE_FIELD not in raw_params["required"]

    async def test_observe_remaining_deck_returns_aggregate_only(self) -> None:
        """Remaining deck observation returns counts without raw card order."""
        bot = _make_bot()
        gamestate = {
            "cards": {
                "count": 3,
                "limit": 52,
                "cards": [
                    {"key": "S_A", "value": {"suit": "S", "rank": "A"}},
                    {"key": "H_A", "value": {"suit": "H", "rank": "A"}},
                    {"key": "D_9", "value": {"suit": "D", "rank": "9"}},
                ],
            }
        }
        fake_balatro = FakeBalatro(gamestate)
        bot._balatro = fake_balatro  # type: ignore[assignment]
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="observe_remaining_deck",
                arguments='{"reasoning": "Need draw odds"}',
            )
        )

        result = await bot._execute_observation_tool_call(tool_call)

        assert fake_balatro.calls == [("gamestate", None)]
        assert result["summary"] == "3/52 cards remaining; draw order is not shown"
        assert result["draw_order_shown"] is False
        remaining_deck = result["remaining_deck"]
        assert remaining_deck["count"] == 3
        assert remaining_deck["limit"] == 52
        assert remaining_deck["suits"] == [
            {"name": "S", "count": 1},
            {"name": "H", "count": 1},
            {"name": "D", "count": 1},
        ]
        assert remaining_deck["ranks"] == [
            {"name": "A", "count": 2},
            {"name": "9", "count": 1},
        ]
        assert "cards" not in remaining_deck

    async def test_observe_run_info_uses_current_gamestate_without_rpc(self) -> None:
        """Run info observation reads the hidden prompt data without bot RPC."""
        bot = _make_bot()
        fake_balatro = FakeBalatro({"state": "SHOULD_NOT_BE_USED"})
        bot._balatro = fake_balatro  # type: ignore[assignment]
        gamestate = {
            "hands": {
                "Pair": {
                    "level": 2,
                    "chips": 25,
                    "mult": 3,
                    "example": [["S_A", True], ["H_A", True]],
                    "played": 4,
                    "played_this_round": 1,
                }
            }
        }
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="observe_run_info",
                arguments='{"reasoning": "Need hand levels"}',
            )
        )

        result = await bot._execute_observation_tool_call(tool_call, gamestate)

        assert fake_balatro.calls == []
        assert result["summary"] == "1 poker hand run info entries returned"
        assert result["poker_hands"] == gamestate["hands"]


class TestBotAgentGlobalMemory:
    """Tests for agent-mode memory_update handling."""

    async def test_successful_action_updates_and_logs_global_memory(self) -> None:
        bot = _make_bot()
        fake_balatro = FakeActionBalatro()
        fake_collector = FakeCollector()
        bot._balatro = fake_balatro  # type: ignore[assignment]
        bot._collector = fake_collector  # type: ignore[assignment]
        bot._last_request_custom_id = "request-00005"

        result = await bot._execute_tool_call(
            _response(
                "play",
                {
                    "cards": [0, 1],
                    "reasoning": "Score the pair.",
                    "memory_update": "Build around pairs; keep economy stable.",
                },
            )
        )

        assert result is fake_balatro.gamestate
        assert fake_balatro.calls == [
            ("play", {"cards": [0, 1], "reasoning": "Score the pair."})
        ]
        assert bot._global_memory == "Build around pairs; keep economy stable."
        assert fake_collector.memory_updates == [
            {
                "index": 1,
                "request": "request-00005",
                "method": "play",
                "memory": "Build around pairs; keep economy stable.",
                "truncated": False,
            }
        ]
        assert bot._history == [
            {
                "method": "play",
                "params": {"cards": [0, 1], "reasoning": "Score the pair."},
                "reasoning": "Score the pair.",
            }
        ]
        assert fake_collector.call_outcomes == ["successful"]

    async def test_missing_memory_update_is_invalid_in_agent_mode(self) -> None:
        bot = _make_bot()
        fake_balatro = FakeActionBalatro()
        fake_collector = FakeCollector()
        bot._balatro = fake_balatro  # type: ignore[assignment]
        bot._collector = fake_collector  # type: ignore[assignment]

        result = await bot._execute_tool_call(
            _response("play", {"cards": [0, 1], "reasoning": "Score the pair."})
        )

        assert result is fake_balatro.gamestate
        assert fake_balatro.calls == [("gamestate", None)]
        assert bot._global_memory == ""
        assert bot._history == []
        assert fake_collector.memory_updates == []
        assert fake_collector.call_outcomes == ["error"]
        assert fake_collector.failures == 1

    async def test_failed_action_does_not_update_global_memory(self) -> None:
        bot = _make_bot()
        fake_balatro = FakeActionBalatro(fail_actions=True)
        fake_collector = FakeCollector()
        bot._balatro = fake_balatro  # type: ignore[assignment]
        bot._collector = fake_collector  # type: ignore[assignment]
        bot._global_memory = "Existing plan."

        result = await bot._execute_tool_call(
            _response(
                "discard",
                {
                    "cards": [2],
                    "reasoning": "Dig for pair support.",
                    "memory_update": "New plan that should not apply.",
                },
            )
        )

        assert result is fake_balatro.gamestate
        assert fake_balatro.calls == [
            ("discard", {"cards": [2], "reasoning": "Dig for pair support."}),
            ("gamestate", None),
        ]
        assert bot._global_memory == "Existing plan."
        assert bot._history == []
        assert fake_collector.memory_updates == []
        assert fake_collector.call_outcomes == ["failed"]

    async def test_memory_update_is_capped_before_logging(self) -> None:
        bot = _make_bot()
        fake_balatro = FakeActionBalatro()
        fake_collector = FakeCollector()
        bot._balatro = fake_balatro  # type: ignore[assignment]
        bot._collector = fake_collector  # type: ignore[assignment]
        long_memory = "x" * (MAX_GLOBAL_MEMORY_CHARS + 50)

        await bot._execute_tool_call(
            _response(
                "discard",
                {
                    "cards": [2],
                    "reasoning": "Cycle one card.",
                    "memory_update": long_memory,
                },
            )
        )

        assert len(bot._global_memory) == MAX_GLOBAL_MEMORY_CHARS
        assert fake_collector.memory_updates[0]["truncated"] is True
        assert fake_collector.memory_updates[0]["memory"] == (
            "x" * MAX_GLOBAL_MEMORY_CHARS
        )
