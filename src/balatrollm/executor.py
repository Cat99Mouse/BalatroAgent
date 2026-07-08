"""Task execution for BalatroLLM runs."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from balatrobot import BalatroInstance as _BalatroInstance
from balatrobot import Config as BalatrobotConfig

from .bot import Bot
from .config import Config, Task

logger = logging.getLogger(__name__)
HEALTH_TIMEOUT = 30.0


class BalatroInstance(_BalatroInstance):
    """Balatro instance with a tolerant startup health check."""

    async def _wait_for_health(self, timeout: float = HEALTH_TIMEOUT) -> None:
        """Wait for health endpoint, retrying through transient invalid responses."""
        url = f"http://{self._config.host}:{self._config.port}"
        payload = {"jsonrpc": "2.0", "method": "health", "params": {}, "id": 1}
        start = asyncio.get_event_loop().time()
        last_error: str | None = None

        while asyncio.get_event_loop().time() - start < timeout:
            try:
                async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                    response = await client.post(url, json=payload)
                    if response.status_code != 200:
                        last_error = (
                            f"HTTP {response.status_code}: {response.text[:300]!r}"
                        )
                        await asyncio.sleep(0.5)
                        continue
                    data: Any = response.json()
                    result = data.get("result") if isinstance(data, dict) else None
                    if isinstance(result, dict) and result.get("status") == "ok":
                        return
                    last_error = f"unexpected health response: {data!r}"
            except (httpx.HTTPError, ValueError, AttributeError) as e:
                last_error = f"{type(e).__name__}: {e}"
            await asyncio.sleep(0.5)

        details = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(
            f"Health check failed after {timeout}s on "
            f"{self._config.host}:{self._config.port}.{details}"
        )


@dataclass
class Executor:
    """Executes tasks with parallelism."""

    config: Config
    tasks: list[Task]
    runs_dir: Path = field(default_factory=Path.cwd)

    _instances: dict[int, BalatroInstance] = field(
        default_factory=dict, init=False, repr=False
    )
    _port_pool: asyncio.Queue[int] = field(
        default_factory=asyncio.Queue, init=False, repr=False
    )
    _shutdown: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )

    async def run(self) -> None:
        """Execute all tasks."""
        ports = range(self.config.port, self.config.port + self.config.parallel)
        try:
            await self._start_instances(ports)
            await self._execute_tasks()
        except asyncio.CancelledError:
            print("\nInterrupted! Cleaning up...")
            raise
        finally:
            await self._stop_instances()
        print("Done.")

    async def _start_instances(self, ports: range) -> None:
        """Start Balatro instances."""
        cfg = BalatrobotConfig.from_env()

        # Create all instances with shared session_id
        instances = []
        for port in ports:
            instance = BalatroInstance(cfg, port=port)
            instances.append((port, instance))

        # Start all instances in parallel
        await asyncio.gather(*(instance.start() for _, instance in instances))

        # Register instances in pool
        for port, instance in instances:
            self._instances[port] = instance
            await self._port_pool.put(port)

    async def _stop_instances(self) -> None:
        """Stop all instances."""
        await asyncio.gather(
            *(i.stop() for i in self._instances.values()),
            return_exceptions=True,
        )
        self._instances.clear()

    async def _execute_tasks(self) -> None:
        """Execute tasks with port pool."""
        total = len(self.tasks)
        count = 0

        async def run_task(task: Task) -> None:
            nonlocal count
            if self._shutdown.is_set():
                return
            port = await self._port_pool.get()
            try:
                count += 1
                instance = self._instances[port]
                log_path = instance.log_path
                print(
                    f"[{count:0{len(str(total))}d}/{total}] STARTED   | {log_path} | {task}"
                )
                bot = Bot(task=task, config=self.config, port=port)
                async with bot:
                    await bot.play(self.runs_dir)
                print(
                    f"[{count:0{len(str(total))}d}/{total}] COMPLETED | {log_path} | {task}"
                )
            except Exception:
                logger.exception(f"Run failed: {task}")
                print(
                    f"[{count:0{len(str(total))}d}/{total}] ERROR     | {log_path} | {task}"
                )
            finally:
                await asyncio.sleep(1)
                await self._port_pool.put(port)

        pending = [asyncio.create_task(run_task(t)) for t in self.tasks]
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            self._shutdown.set()
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
