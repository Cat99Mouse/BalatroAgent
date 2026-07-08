"""Strategy template management for BalatroLLM."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

STRATEGIES_DIR = Path(__file__).parent / "strategies"
SUIT_ORDER: tuple[str, ...] = ("S", "H", "C", "D")
RANK_ORDER: tuple[str, ...] = (
    "A",
    "K",
    "Q",
    "J",
    "T",
    "9",
    "8",
    "7",
    "6",
    "5",
    "4",
    "3",
    "2",
)


def _render_counts(
    counts: dict[str, int], order: tuple[str, ...]
) -> list[dict[str, int | str]]:
    """Render ordered non-zero count items."""
    return [
        {"name": key, "count": counts[key]}
        for key in order
        if counts.get(key, 0) > 0
    ]


def _render_rank_suit_counts(counts: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Render rank-by-suit counts without exposing deck order."""
    rows: list[dict[str, Any]] = []
    for rank in RANK_ORDER:
        suit_counts = counts[rank]
        total = sum(suit_counts.values())
        if total == 0:
            continue
        rows.append(
            {
                "rank": rank,
                "total": total,
                "suits": [
                    {"name": suit, "count": suit_counts[suit]}
                    for suit in SUIT_ORDER
                ],
            }
        )
    return rows


def _deck_summary(gamestate: dict[str, Any]) -> dict[str, Any] | None:
    """Summarize remaining draw deck composition without exposing deck order."""
    cards_info = gamestate.get("cards")
    if not isinstance(cards_info, dict):
        return None

    cards = cards_info.get("cards")
    if not isinstance(cards, list):
        return None

    suit_counts: dict[str, int] = {suit: 0 for suit in SUIT_ORDER}
    rank_counts: dict[str, int] = {rank: 0 for rank in RANK_ORDER}
    rank_suit_counts: dict[str, dict[str, int]] = {
        rank: {suit: 0 for suit in SUIT_ORDER} for rank in RANK_ORDER
    }

    for card in cards:
        if not isinstance(card, dict):
            continue
        value = card.get("value")
        if not isinstance(value, dict):
            continue
        suit = value.get("suit")
        rank = value.get("rank")
        suit_key = suit if isinstance(suit, str) and suit in suit_counts else None
        rank_key = rank if isinstance(rank, str) and rank in rank_counts else None
        if suit_key is not None:
            suit_counts[suit_key] += 1
        if rank_key is not None:
            rank_counts[rank_key] += 1
        if suit_key is not None and rank_key is not None:
            rank_suit_counts[rank_key][suit_key] += 1

    return {
        "count": cards_info.get("count", len(cards)),
        "limit": cards_info.get("limit"),
        "suits": _render_counts(suit_counts, SUIT_ORDER),
        "ranks": _render_counts(rank_counts, RANK_ORDER),
        "composition": _render_rank_suit_counts(rank_suit_counts),
    }


@dataclass(frozen=True)
class StrategyManifest:
    """Strategy metadata from manifest.json."""

    name: str
    description: str
    author: str
    version: str
    tags: list[str]

    @classmethod
    def from_file(
        cls, strategy: str, strategies_dir: Path = STRATEGIES_DIR
    ) -> "StrategyManifest":
        """Load strategy metadata from manifest.json."""
        manifest_path = strategies_dir / strategy / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found for strategy '{strategy}': {manifest_path}"
            )

        with manifest_path.open() as f:
            data = json.load(f)

        required_fields = {"name", "description", "author", "version", "tags"}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(
                f"Manifest for strategy '{strategy}' missing fields: {missing_fields}"
            )

        return cls(
            name=data["name"],
            description=data["description"],
            author=data["author"],
            version=data["version"],
            tags=data["tags"],
        )


class StrategyManager:
    """Manages Jinja2 strategy templates for LLM prompts.

    A strategy consists of four files:
    - STRATEGY.md.jinja: High-level strategic guidance
    - GAMESTATE.md.jinja: Current game state rendering
    - MEMORY.md.jinja: Action history and error context
    - TOOLS.json: Tool definitions for each game state

    Usage:
        sm = StrategyManager("default")
        strategy_text = sm.render_strategy(gamestate)
        gamestate_text = sm.render_gamestate(gamestate)
        memory_text = sm.render_memory(history)
        tools = sm.get_tools("SELECTING_HAND")
    """

    def __init__(self, name: str, strategies_dir: Path = STRATEGIES_DIR):
        """Initialize the strategy manager.

        Args:
            name: Name of the strategy (subdirectory in strategies_dir)
            strategies_dir: Base directory containing strategy folders

        Raises:
            FileNotFoundError: If strategy directory or required files don't exist
        """
        self.path = strategies_dir / name

        if not self.path.exists():
            raise FileNotFoundError(f"Strategy not found: {name}")

        # Verify required files exist
        required = [
            "STRATEGY.md.jinja",
            "GAMESTATE.md.jinja",
            "MEMORY.md.jinja",
            "TOOLS.json",
        ]
        missing = [f for f in required if not (self.path / f).exists()]
        if missing:
            raise FileNotFoundError(f"Strategy '{name}' missing files: {missing}")

        self.env = Environment(loader=FileSystemLoader(self.path))
        self.env.filters["from_json"] = json.loads

        # Load tools
        with open(self.path / "TOOLS.json") as f:
            self._tools = json.load(f)

    def render_strategy(self, gamestate: dict[str, Any]) -> str:
        """Render the strategy guidance template.

        Args:
            gamestate: Current game state from BalatroBot

        Returns:
            Rendered strategy guidance text
        """
        template = self.env.get_template("STRATEGY.md.jinja")
        return template.render(G=gamestate)

    def render_gamestate(self, gamestate: dict[str, Any]) -> str:
        """Render the game state template.

        Args:
            gamestate: Current game state from BalatroBot

        Returns:
            Rendered game state text for LLM context
        """
        template = self.env.get_template("GAMESTATE.md.jinja")
        return template.render(
            G=gamestate,
            D=_deck_summary(gamestate),
        )

    def render_memory(
        self,
        history: list[dict[str, Any]],
        last_error: str | None = None,
        last_failure: str | None = None,
    ) -> str:
        """Render the memory/history template.

        Args:
            history: List of previous actions with method, params, reasoning
            last_error: Error message from last invalid LLM response
            last_failure: Error message from last failed API call

        Returns:
            Rendered memory context text
        """
        template = self.env.get_template("MEMORY.md.jinja")
        return template.render(
            history=history,
            last_error_call_msg=last_error,
            last_failed_call_msg=last_failure,
        )

    def get_tools(self, state: str) -> list[dict[str, Any]]:
        """Get tool definitions for a game state.

        Args:
            state: Game state string (e.g., "SELECTING_HAND", "SHOP")

        Returns:
            List of OpenAI function calling tool definitions
        """
        return self._tools.get(state, [])
