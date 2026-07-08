"""Probe provider support for chat completions tool_choice modes.

Usage:
    python scripts/test_tool_choice.py config/local.yaml
    python scripts/test_tool_choice.py config/local.yaml --modes auto required require
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from balatrollm.config import Config, get_model_config

TEST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "choose_action",
        "description": "Choose a simple test action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["continue"],
                    "description": "The action to choose.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "A brief reason for the action.",
                },
            },
            "required": ["action", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _load_config(path: Path) -> Config:
    config = Config.load(yaml_path=path)
    config.validate()
    if not config.api_key or config.api_key == "replace-with-your-api-key":
        raise SystemExit(f"api_key is not set in {path}")
    return config


def _clip(text: str, limit: int = 700) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


async def probe_tool_choice(config: Config, mode: str) -> None:
    model = config.model[0]
    model_config = get_model_config(config.model_config)
    model_config["tool_choice"] = mode
    model_config["parallel_tool_calls"] = False

    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "This is a tool-calling capability test. "
                    "Call choose_action with action='continue'."
                ),
            }
        ],
        "tools": [TEST_TOOL],
        **model_config,
    }

    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=60.0,
    )
    try:
        response = await client.chat.completions.create(**request)
    except APIStatusError as e:
        body = getattr(e.response, "text", "")
        print(f"[FAIL] tool_choice={mode!r} status={e.status_code}")
        print(_clip(body))
        return
    except Exception as e:
        print(f"[FAIL] tool_choice={mode!r} error={type(e).__name__}: {e}")
        return
    finally:
        await client.close()

    choice = response.choices[0]
    message = choice.message
    tool_calls = message.tool_calls or []

    print(f"[OK] tool_choice={mode!r}")
    print(f"  model={model}")
    print(f"  finish_reason={choice.finish_reason}")
    print(f"  message_content_present={bool(message.content)}")
    print(f"  tool_calls={len(tool_calls)}")
    for i, tool_call in enumerate(tool_calls):
        print(f"  tool_call[{i}].name={tool_call.function.name}")
        print(f"  tool_call[{i}].arguments={tool_call.function.arguments}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        help="Path to BalatroLLM YAML config, usually config/local.yaml.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["auto", "required", "require"],
        help="tool_choice values to test.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    for mode in args.modes:
        await probe_tool_choice(config, mode)


if __name__ == "__main__":
    asyncio.run(main())
