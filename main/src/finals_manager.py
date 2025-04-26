import json
from abc import ABC, abstractmethod
from typing import Any

import httpx
import websockets


class FinalsManager(ABC):
    def __init__(self):
        print("initializing participant finals server manager")
        self.client = httpx.AsyncClient()

    async def exit(self):
        await self.client.aclose()

    async def async_post(self, endpoint: str, json: dict | None = None):
        return await self.client.post(endpoint, json=json, timeout=None)

    async def send_result(
        self, websocket: websockets.WebSocketClientProtocol, data: dict[str, Any]
    ):
        return await websocket.send(json.dumps(data))

    @abstractmethod
    async def run_asr(self, audio_bytes: bytes) -> str:
        raise NotImplemented

    @abstractmethod
    async def run_nlp(self, transcript: str) -> dict[str, str]:
        raise NotImplemented

    @abstractmethod
    async def send_heading(self, heading: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    async def reset_cannon(self) -> None:
        raise NotImplemented

    @abstractmethod
    async def run_vlm(self, image_bytes: bytes, caption: str) -> list[int]:
        raise NotImplemented
