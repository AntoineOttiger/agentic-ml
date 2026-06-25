"""Client MCP synchrone pour les outils ML de l'agent.

Wrapping synchrone du client async MCP : un coroutine unique gère le cycle de vie
de la connexion (connect → wait → disconnect) dans un thread daemon. Les appels
`call_tool()` soumet les requêtes depuis n'importe quel thread via `run_coroutine_threadsafe`.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# src/ résolu depuis ce fichier : src/agentic_ml/mcp_server/client.py → src/
_SRC_DIR = str(Path(__file__).resolve().parents[2])


class MCPToolClient:
    """Context manager synchrone autour d'une session MCP stdio.

    Usage:
        with MCPToolClient() as client:
            result = client.call_tool("get_model_schema_tool", {"model_type": "xgboost"})
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._session: ClientSession | None = None
        # threading.Event pour la synchronisation cross-thread
        self._ready = threading.Event()
        self._stopped = threading.Event()
        # asyncio.Event créé dans le loop dédié (via _run)
        self._stop_signal: asyncio.Event | None = None

    def __enter__(self) -> MCPToolClient:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(timeout=30):
            raise TimeoutError("Le serveur MCP n'a pas répondu dans les 30 secondes.")
        return self

    def __exit__(self, *_: object) -> None:
        if self._stop_signal is not None:
            self._loop.call_soon_threadsafe(self._stop_signal.set)
        if not self._stopped.wait(timeout=15):
            raise TimeoutError("Le serveur MCP ne s'est pas arrêté dans les 15 secondes.")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _run(self) -> None:
        """Coroutine principal : ouvre la connexion, signale 'ready', attend 'stop'."""
        env = {**os.environ, "PYTHONPATH": _SRC_DIR}
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agentic_ml.mcp_server"],
            env=env,
        )
        self._stop_signal = asyncio.Event()

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop_signal.wait()
        finally:
            self._session = None
            self._stopped.set()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Appelle un outil MCP et retourne le résultat désérialisé.

        Timeout de 300 s pour accommoder les pipelines ML longs (Optuna).
        Lève une RuntimeError si l'outil signale une erreur côté serveur.
        """
        assert self._session is not None, (
            "MCPToolClient non initialisé (utiliser comme context manager)"
        )
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments),
            self._loop,
        )
        result = future.result(timeout=300)

        if result.isError:
            error_text = result.content[0].text if result.content else "(no detail)"
            raise RuntimeError(f"Erreur MCP outil '{name}': {error_text}")

        raw = result.content[0].text if result.content else ""
        if not raw:
            raise RuntimeError(f"L'outil MCP '{name}' a retourné un contenu vide.")
        return json.loads(raw)
