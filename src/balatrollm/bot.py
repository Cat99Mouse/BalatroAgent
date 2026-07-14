"""Core LLM-powered Balatro bot implementation."""

import asyncio
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
from openai.types.chat import ChatCompletion

from .chatbot_memory import ChatbotMemory
from .client import BalatroClient, BalatroError
from .collector import (
    ChatCompletionError,
    ChatCompletionResponse,
    Collector,
    FinishReason,
    Stats,
)
from .config import Config, Task, get_model_config
from .llm import LLMClient, LLMClientError, LLMTimeoutError
from .strategy import StrategyManager, _deck_summary

logger = logging.getLogger(__name__)

BUY_TOOL_ALIASES: dict[str, str] = {
    "buy_card": "card",
    "buy_voucher": "voucher",
    "buy_pack": "pack",
}
SELL_TOOL_ALIASES: dict[str, str] = {
    "sell_joker": "joker",
    "sell_consumable": "consumable",
}
REARRANGE_TOOL_ALIASES: dict[str, str] = {
    "rearrange_hand": "hand",
    "rearrange_jokers": "jokers",
    "rearrange_consumables": "consumables",
}
MAX_OBSERVATION_CALLS = 2
MAX_GLOBAL_MEMORY_CHARS = 1200
MEMORY_UPDATE_FIELD = "memory_update"
OBSERVATION_TOOL_NAMES: set[str] = {"observe_remaining_deck", "observe_run_info"}
MEMORY_UPDATE_TOOL_PROPERTY: dict[str, Any] = {
    "type": "string",
    "maxLength": MAX_GLOBAL_MEMORY_CHARS,
    "description": (
        "Complete replacement for the agent's run-level global memory after this "
        "successful action. Keep it concise, durable, and under 1200 characters."
    ),
}
OBSERVE_REMAINING_DECK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "observe_remaining_deck",
        "strict": False,
        "description": (
            "Observe the remaining draw deck as aggregate counts only. This is "
            "read-only and never reveals draw order or the next card. Use it "
            "only when deck composition materially affects the current decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why remaining deck composition is useful now.",
                }
            },
            "required": ["reasoning"],
        },
    },
}
OBSERVE_RUN_INFO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "observe_run_info",
        "strict": False,
        "description": (
            "Observe current poker hand run info: hand levels, base chips, base "
            "mult, examples, and played counts. This is read-only and uses the "
            "current game state already available to the bot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why poker hand run info is useful now.",
                }
            },
            "required": ["reasoning"],
        },
    },
}


class BotError(Exception):
    """Base exception for bot errors."""

    pass


class Bot:
    """One-shot LLM-powered Balatro bot. Creates clients, plays a single game, returns stats."""

    def __init__(self, task: Task, config: Config, port: int | None = None) -> None:
        self.task = task
        self.config = config
        self.port = port if port is not None else config.port
        self.model_config = get_model_config(config.model_config)
        self.strategy = StrategyManager(task.strategy, mode=config.mode)
        self._chatbot_memory = ChatbotMemory()

        self._balatro: BalatroClient | None = None
        self._llm: LLMClient | None = None
        self._collector: Collector | None = None

        self._last_error_msg: str | None = None
        self._last_failed_msg: str | None = None
        self._global_memory: str = ""
        self._last_request_custom_id: str | None = None
        self._history: list[dict[str, Any]] = []

        # Finish reason tracking
        self._finish_reason: FinishReason | None = None
        # Separate counters for error calls vs failed calls
        self._consecutive_errors: int = 0
        self._consecutive_faileds: int = 0

    async def __aenter__(self) -> "Bot":
        """Initialize all clients."""
        self._balatro = BalatroClient(
            host=self.config.host,
            port=self.port,
        )
        await self._balatro.__aenter__()

        self._llm = LLMClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "",
        )
        await self._llm.__aenter__()

        return self

    async def __aexit__(self, *_: Any) -> None:
        """Clean up all clients."""
        if self._llm is not None:
            await self._llm.__aexit__(None, None, None)
            self._llm = None
        if self._balatro is not None:
            await self._balatro.__aexit__(None, None, None)
            self._balatro = None

    def _setup_file_logging(self) -> None:
        """Redirect logging to file in run directory."""
        if self._collector is None:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        log_file = self._collector.run_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    async def _wait_for_menu(self, timeout: float = 10.0) -> None:
        """Wait for game to be in MENU state."""
        assert self._balatro is not None

        start = time.time()
        while time.time() - start < timeout:
            try:
                gamestate = await self._balatro.call("gamestate")
                if gamestate.get("state", "") == "MENU":
                    logger.debug("Confirmed MENU state")
                    return
            except Exception as e:
                logger.debug(f"Gamestate check failed: {e}")
            await asyncio.sleep(0.5)

        self._finish_reason = "connection_abort"
        raise BotError(f"Timeout waiting for MENU state after {timeout}s")

    async def play(self, runs_dir: Path = Path.cwd()) -> Stats:
        """Play a single game run. Returns final Stats."""
        if self._balatro is None or self._llm is None:
            raise RuntimeError(
                "Bot not initialized. Use 'async with Bot(config) as bot:'"
            )

        # Health check before initializing collector
        try:
            await self._balatro.call("gamestate")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            self._finish_reason = "connection_abort"
            raise BotError(
                f"Cannot connect to Balatro on {self.config.host}:{self.port}. "
                "Make sure Balatro instance started correctly."
            ) from e
        except Exception as e:
            self._finish_reason = "connection_abort"
            raise BotError(f"Failed to connect to Balatro: {e}") from e

        self._collector = Collector(self.task, runs_dir, mode=self.config.mode)
        self._setup_file_logging()

        logger.info("Starting game")
        logger.info(f"Run data will be saved to: {self._collector.run_dir}")

        try:
            await self._balatro.call("menu")
            await self._wait_for_menu()
            gamestate = await self._balatro.call(
                "start",
                {
                    "deck": self.task.deck,
                    "stake": self.task.stake,
                    "seed": self.task.seed,
                },
            )
            await self._run_game_loop(gamestate)
        except BotError:
            logger.error("Game ended due to bot error")
            raise
        except Exception as e:
            self._finish_reason = "unexpected_error"
            logger.exception("Unexpected error occurred during gameplay")
            raise BotError(f"Unexpected error: {e}") from e
        finally:
            if self._collector:
                try:
                    reason: FinishReason = self._finish_reason or "unexpected_error"
                    self._collector.write_stats(reason)
                    logger.info("Stats written")
                except Exception as e:
                    logger.debug(
                        f"Could not write stats (normal if run failed early): {e}"
                    )

        return self._collector._calculate_stats(
            self._finish_reason or "unexpected_error"
        )

    async def _run_game_loop(self, gamestate: dict[str, Any]) -> None:
        """Main game loop."""
        assert self._balatro is not None
        assert self._llm is not None
        assert self._collector is not None

        while True:
            if gamestate.get("won", False):
                self._finish_reason = "won"
                logger.info("Game won! Waiting for GAME_OVER state...")
                break

            current_state = gamestate.get("state", "")
            logger.info(f"State: {current_state}")

            await asyncio.sleep(0.5)
            await self._balatro.call("gamestate")

            match current_state:
                case "SELECTING_HAND" | "SHOP" | "SMODS_BOOSTER_OPENED":
                    response = await self._get_llm_response(gamestate)
                    gamestate = await self._execute_tool_call(response)
                case "ROUND_EVAL":
                    gamestate = await self._balatro.call("cash_out")
                case "BLIND_SELECT":
                    # NOTE: This bot always selects and never skips blinds
                    gamestate = await self._balatro.call("select")
                case "GAME_OVER":
                    self._finish_reason = "lost"
                    logger.info("Game over!")
                    break
                case _:
                    await asyncio.sleep(1)
                    gamestate = await self._balatro.call("gamestate")

    async def _get_llm_response(self, gamestate: dict[str, Any]) -> ChatCompletion:
        """Get LLM response for current game state."""
        assert self._balatro is not None
        assert self._llm is not None
        assert self._collector is not None

        strategy_content = self.strategy.render_strategy(gamestate)
        gamestate_content = self.strategy.render_gamestate(gamestate)
        if self.config.mode == "chatbot":
            memory_content = self._chatbot_memory.render(self._history)
        else:
            memory_content = self.strategy.render_memory(
                history=self._history[-10:],
                global_memory=self._global_memory,
                last_error=self._last_error_msg,
                last_failure=self._last_failed_msg,
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": strategy_content,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": gamestate_content},
                    {"type": "text", "text": memory_content},
                ],
            }
        ]

        state = gamestate["state"]
        observation_enabled = self.config.mode == "agent"
        tools = self._get_tools_for_state(
            state, include_observation=observation_enabled
        )

        observation_calls = 0
        while True:
            response = await self._call_llm(messages, tools)
            tool_call = self._first_tool_call(response)
            fn_name = self._tool_call_name(tool_call)

            if fn_name not in OBSERVATION_TOOL_NAMES or not observation_enabled:
                return response

            if observation_calls >= MAX_OBSERVATION_CALLS:
                return response

            observation_calls += 1
            tool_result = await self._execute_observation_tool_call(
                tool_call, gamestate
            )
            logger.info(f"Observation: {fn_name}({tool_result.get('summary', '')})")

            messages = [
                *messages,
                self._assistant_tool_call_message(response),
                {
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", ""),
                    "content": json.dumps(tool_result),
                },
            ]
            tools = self._get_tools_for_state(
                state,
                include_observation=(
                    observation_enabled and observation_calls < MAX_OBSERVATION_CALLS
                ),
            )

    def _get_tools_for_state(
        self, state: str, include_observation: bool = False
    ) -> list[dict[str, Any]]:
        """Get strategy tools, optionally adding read-only observation tools."""
        tools = deepcopy(self.strategy.get_tools(state))
        if self.config.mode == "agent":
            self._add_agent_memory_update_to_tools(tools)
        if include_observation and self.config.mode == "agent":
            tools.append(OBSERVE_REMAINING_DECK_TOOL)
            tools.append(OBSERVE_RUN_INFO_TOOL)
        return tools

    def _add_agent_memory_update_to_tools(self, tools: list[dict[str, Any]]) -> None:
        """Require memory_update on agent action tools without mutating strategy data."""
        for tool in tools:
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                continue
            properties = parameters.setdefault("properties", {})
            if not isinstance(properties, dict):
                continue
            properties[MEMORY_UPDATE_FIELD] = deepcopy(MEMORY_UPDATE_TOOL_PROPERTY)
            required = parameters.setdefault("required", [])
            if isinstance(required, list) and MEMORY_UPDATE_FIELD not in required:
                required.append(MEMORY_UPDATE_FIELD)

    def _first_tool_call(self, response: ChatCompletion) -> Any | None:
        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return None
        return message.tool_calls[0]

    def _tool_call_name(self, tool_call: Any | None) -> str | None:
        if tool_call is None:
            return None
        function_obj = getattr(tool_call, "function", tool_call)
        return getattr(function_obj, "name", None)

    def _assistant_tool_call_message(self, response: ChatCompletion) -> dict[str, Any]:
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        return {
            "role": "assistant",
            "content": getattr(message, "content", "") or "",
            "tool_calls": [
                tool_call.model_dump(exclude_none=True) for tool_call in tool_calls
            ],
        }

    async def _execute_observation_tool_call(
        self, tool_call: Any, gamestate: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a read-only observation tool without updating gameplay history."""
        function_obj = getattr(tool_call, "function", tool_call)
        fn_name = getattr(function_obj, "name", None)
        fn_args_str = getattr(function_obj, "arguments", None)

        if fn_name not in OBSERVATION_TOOL_NAMES:
            return {"error": f"Unknown observation tool: {fn_name}"}

        try:
            if fn_args_str:
                json.loads(fn_args_str)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON in observation tool arguments: {e}"}

        if fn_name == "observe_remaining_deck":
            return await self._observe_remaining_deck()
        if fn_name == "observe_run_info":
            return self._observe_run_info(gamestate)

        return {"error": f"Unknown observation tool: {fn_name}"}

    def _observe_run_info(self, gamestate: dict[str, Any] | None) -> dict[str, Any]:
        """Observe poker hand run info from the current gamestate without RPC."""
        if gamestate is None:
            return {"error": "Current game state is unavailable"}

        hands = gamestate.get("hands")
        if not isinstance(hands, dict):
            return {
                "summary": "Poker hand run info is unavailable",
                "poker_hands": None,
            }

        poker_hands: dict[str, dict[str, Any]] = {}
        for name, hand in hands.items():
            if not isinstance(hand, dict):
                continue
            poker_hands[str(name)] = {
                "level": hand.get("level"),
                "chips": hand.get("chips"),
                "mult": hand.get("mult"),
                "example": hand.get("example"),
                "played": hand.get("played"),
                "played_this_round": hand.get("played_this_round"),
            }

        return {
            "summary": f"{len(poker_hands)} poker hand run info entries returned",
            "poker_hands": poker_hands,
        }

    async def _observe_remaining_deck(self) -> dict[str, Any]:
        """Observe aggregate remaining draw deck composition without card order."""
        assert self._balatro is not None

        try:
            gamestate = await self._balatro.call("gamestate")
            remaining_deck = _deck_summary(gamestate)
        except Exception as e:
            logger.warning(f"Remaining deck observation failed: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

        if remaining_deck is None:
            return {
                "summary": "Remaining draw deck data is unavailable",
                "remaining_deck": None,
                "draw_order_shown": False,
            }

        count = remaining_deck.get("count", 0)
        limit = remaining_deck.get("limit")
        total = f"{count}/{limit}" if limit is not None else str(count)
        return {
            "summary": f"{total} cards remaining; draw order is not shown",
            "remaining_deck": remaining_deck,
            "draw_order_shown": False,
        }

    async def _call_llm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatCompletion:
        """Call the LLM and record request/response artifacts."""
        assert self._balatro is not None
        assert self._llm is not None
        assert self._collector is not None

        request_data = {
            "model": self.task.model,
            "messages": messages,
            "tools": tools,
            **self.model_config,
        }

        custom_id = self._collector.write_request(request_data)
        self._last_request_custom_id = custom_id
        request_id = str(time.time_ns() // 1_000_000)

        try:
            response = await self._llm.call(
                model=self.task.model,
                messages=messages,
                tools=tools,
                model_config=self.model_config,
            )

            try:
                await self._balatro.call(
                    "screenshot",
                    {"path": str(self._collector.screenshot_dir / f"{custom_id}.png")},
                )
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")

            self._collector.write_response(
                id=str(time.time_ns() // 1_000_000),
                custom_id=custom_id,
                response=ChatCompletionResponse(
                    request_id=request_id,
                    status_code=200,
                    body=response.model_dump(),
                ),
            )

            return response

        except LLMTimeoutError as e:
            self._collector.write_response(
                id=str(time.time_ns() // 1_000_000),
                custom_id=custom_id,
                error=ChatCompletionError(code="timeout", message=str(e)),
            )
            self._finish_reason = "llm_abort"
            raise BotError("3 consecutive LLM timeouts") from e

        except LLMClientError as e:
            self._collector.write_response(
                id=str(time.time_ns() // 1_000_000),
                custom_id=custom_id,
                error=ChatCompletionError(code="error", message=str(e)),
            )
            self._finish_reason = "llm_abort"
            raise BotError(f"LLM error: {e}") from e

    async def _execute_tool_call(self, response: ChatCompletion) -> dict[str, Any]:
        """Execute tool call from LLM response."""
        assert self._balatro is not None
        assert self._collector is not None

        ################################################################################
        # Parse tool call from LLM response
        ################################################################################

        message = response.choices[0].message

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return await self._handle_error_call(
                f"No tool calls in LLM response: {message.content}"
            )

        tool_call = message.tool_calls[0]
        function_obj = getattr(tool_call, "function", tool_call)

        fn_name = getattr(function_obj, "name", None)
        if not fn_name:
            return await self._handle_error_call("Invalid tool call: missing name")
        if fn_name in OBSERVATION_TOOL_NAMES:
            if self.config.mode == "chatbot":
                return await self._handle_error_call(
                    f"Observation tools are disabled in chatbot mode: {fn_name}"
                )
            return await self._handle_error_call(
                f"Observation tool {fn_name} was returned as a gameplay action"
            )

        fn_args_str = getattr(function_obj, "arguments", None)
        if not fn_args_str:
            return await self._handle_error_call("Invalid tool call: missing arguments")

        try:
            fn_args = json.loads(fn_args_str)
        except json.JSONDecodeError as e:
            return await self._handle_error_call(
                f"Invalid JSON in tool call arguments: {e}"
            )
        if not isinstance(fn_args, dict):
            return await self._handle_error_call(
                "Invalid tool call: arguments must be a JSON object"
            )

        memory_update_arg = fn_args.pop(MEMORY_UPDATE_FIELD, None)
        memory_update: str | None = None
        memory_truncated = False
        if self.config.mode == "agent":
            if not isinstance(memory_update_arg, str):
                return await self._handle_error_call(
                    f"Invalid tool call: missing {MEMORY_UPDATE_FIELD}"
                )
            memory_update, memory_truncated = self._normalize_global_memory(
                memory_update_arg
            )

        history_fn_name = fn_name
        if fn_name in BUY_TOOL_ALIASES:
            purchase_key = BUY_TOOL_ALIASES[fn_name]
            if purchase_key not in fn_args:
                return await self._handle_error_call(
                    f"Invalid tool call: {fn_name} missing {purchase_key}"
                )
            fn_name = "buy"
            fn_args = {
                purchase_key: fn_args[purchase_key],
                "reasoning": fn_args.get("reasoning", ""),
            }

        if fn_name in SELL_TOOL_ALIASES:
            sell_key = SELL_TOOL_ALIASES[fn_name]
            if sell_key not in fn_args:
                return await self._handle_error_call(
                    f"Invalid tool call: {fn_name} missing {sell_key}"
                )
            fn_name = "sell"
            fn_args = {
                sell_key: fn_args[sell_key],
                "reasoning": fn_args.get("reasoning", ""),
            }

        if fn_name in REARRANGE_TOOL_ALIASES:
            rearrange_key = REARRANGE_TOOL_ALIASES[fn_name]
            if rearrange_key not in fn_args:
                return await self._handle_error_call(
                    f"Invalid tool call: {fn_name} missing {rearrange_key}"
                )
            fn_name = "rearrange"
            fn_args = {
                rearrange_key: fn_args[rearrange_key],
                "reasoning": fn_args.get("reasoning", ""),
            }

        ################################################################################
        # Execute tool call
        ################################################################################

        try:
            logger.info(f"Executing: {history_fn_name}({fn_args})")
            gamestate = await self._balatro.call(fn_name, fn_args)

            self._collector.reset_failures()
            # Reset both consecutive counters on success
            self._consecutive_errors = 0
            self._consecutive_faileds = 0
            self._last_error_msg = None
            self._last_failed_msg = None
            self._collector.record_call("successful")
            self._collector.write_gamestate(gamestate)
            if self.config.mode == "agent" and memory_update is not None:
                self._record_global_memory_update(
                    method=history_fn_name,
                    memory=memory_update,
                    truncated=memory_truncated,
                )
            self._history.append(
                {
                    "method": history_fn_name,
                    "params": fn_args,
                    "reasoning": fn_args.get("reasoning", ""),
                }
            )

            return gamestate

        except BalatroError as e:
            return await self._handle_failed_call(f"BalatroError: {e}")

        except httpx.TransportError as e:
            logger.warning(f"Game transport error during tool call: {e}")
            self._collector.record_call("failed")
            try:
                return await self._balatro.call("gamestate")
            except Exception:
                self._finish_reason = "connection_abort"
                raise BotError(f"Game unresponsive after transport error: {e}") from e

    def _normalize_global_memory(self, memory_update: str) -> tuple[str, bool]:
        """Normalize and cap model-maintained global memory."""
        memory = memory_update.strip()
        if len(memory) <= MAX_GLOBAL_MEMORY_CHARS:
            return memory, False
        return memory[:MAX_GLOBAL_MEMORY_CHARS], True

    def _record_global_memory_update(
        self, method: str, memory: str, truncated: bool
    ) -> None:
        """Apply and persist a successful agent global memory update."""
        assert self._collector is not None

        self._global_memory = memory
        self._collector.write_global_memory_update(
            index=len(self._history) + 1,
            request=self._last_request_custom_id,
            method=method,
            memory=memory,
            truncated=truncated,
        )

    async def _handle_error_call(self, msg: str) -> dict[str, Any]:
        """Handle invalid LLM response (no valid tool call)."""
        assert self._balatro is not None
        assert self._collector is not None

        logger.warning(f"Error call: {msg}")
        self._last_error_msg = msg
        self._collector.record_failure()
        self._collector.record_call("error")

        # Track consecutive error calls separately
        self._consecutive_errors += 1
        self._consecutive_faileds = 0

        if self._consecutive_errors >= Collector.MAX_CONSECUTIVE_FAILURES:
            self._finish_reason = "consecutive_error_calls"
            raise BotError("Too many consecutive error calls")

        return await self._balatro.call("gamestate")

    async def _handle_failed_call(self, msg: str) -> dict[str, Any]:
        """Handle valid tool call that resulted in BalatroError."""
        assert self._balatro is not None
        assert self._collector is not None

        logger.warning(f"Failed call: {msg}")
        self._last_failed_msg = msg
        self._collector.record_failure()
        self._collector.record_call("failed")

        # Track consecutive failed calls separately
        self._consecutive_faileds += 1
        self._consecutive_errors = 0

        if self._consecutive_faileds >= Collector.MAX_CONSECUTIVE_FAILURES:
            self._finish_reason = "consecutive_failed_calls"
            raise BotError("Too many consecutive failed calls")

        return await self._balatro.call("gamestate")
