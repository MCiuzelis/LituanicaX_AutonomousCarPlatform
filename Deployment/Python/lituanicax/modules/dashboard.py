"""Browser dashboard: MJPEG camera streams, telemetry, and the SPACE gate.

Replaces `dashboard_node.py`.  Runs an aiohttp server on its own thread and
reads robot state directly instead of subscribing to topics.
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from aiohttp import web

from ..config import DASHBOARD_HZ, TELEMETRY_HZ
from ..module import Module

if TYPE_CHECKING:
    from ..robot import Robot

HTML_PATH = Path(__file__).resolve().parents[1] / "dashboard.html"

# EMA smoothing factors (lower = smoother but more lag)
ALPHA_BATTERY = 0.3
ALPHA_CPU = 0.2
ALPHA_TEMP = 0.025


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


class Dashboard(Module):
    """Overrides run(): the web server owns its own asyncio event loop."""

    def __init__(
        self,
        robot: "Robot",
        host: str = "0.0.0.0",
        port: int = 8080,
        stream_fps: float = DASHBOARD_HZ,
        sample_hz: float = TELEMETRY_HZ,
    ) -> None:
        super().__init__("dashboard", stream_fps)

        self.robot = robot
        self.host = host
        self.port = int(port)
        self.stream_fps = float(stream_fps)
        self.sample_hz = float(sample_hz)

        self.html = HTML_PATH.read_text()
        self.ws_clients: set[web.WebSocketResponse] = set()
        self.telemetry = {
            "throttle": 0.0,
            "steer": 0.0,
            "battery": 0.0,
            "current": 0.0,
            "velocity": 0.0,
            "distance": 0.0,
            "active": False,
            "cpu_usage": 0.0,
            "cpu_temp": 0.0,
            "undervoltage": False,
            "overcurrent": False,
        }
        self._filtered_battery = 0.0
        self._filtered_cpu_usage = 0.0
        self._filtered_cpu_temp = 0.0

    # --- server lifecycle ---

    def run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_websocket)
        app.router.add_get("/stream/{side}", self._handle_stream)

        runner = web.AppRunner(app, shutdown_timeout=0.25)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.log.info(f"Dashboard open at http://{get_local_ip()}:{self.port}")

        sampler = asyncio.create_task(self._sample_telemetry())
        broadcaster = asyncio.create_task(self._broadcast_telemetry())
        try:
            while not self.stopping:
                await asyncio.sleep(0.1)
        finally:
            self.robot.set_autopilot_enabled(False)
            for task in (sampler, broadcaster):
                task.cancel()
            await asyncio.gather(sampler, broadcaster, return_exceptions=True)
            await self._close_clients()
            await runner.cleanup()

    async def _close_clients(self) -> None:
        for ws in list(self.ws_clients):
            try:
                await ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY, message=b"dashboard shutting down"
                )
            except Exception:
                pass
        self.ws_clients.clear()

    # --- telemetry ---

    async def _sample_telemetry(self) -> None:
        """Poll the robot at the telemetry rate and low-pass the noisy channels."""
        interval = 1.0 / self.sample_hz
        while True:
            await asyncio.sleep(interval)
            drive = self.robot.drive
            monitoring = self.robot.monitoring

            if monitoring is not None:
                self._filtered_battery = (
                    ALPHA_BATTERY * monitoring.battery_voltage
                    + (1.0 - ALPHA_BATTERY) * self._filtered_battery
                )
                self._filtered_cpu_usage = (
                    ALPHA_CPU * monitoring.cpu_usage
                    + (1.0 - ALPHA_CPU) * self._filtered_cpu_usage
                )
                self._filtered_cpu_temp = (
                    ALPHA_TEMP * monitoring.cpu_temperature
                    + (1.0 - ALPHA_TEMP) * self._filtered_cpu_temp
                )
                self.telemetry.update(
                    {
                        "battery": self._filtered_battery,
                        "current": monitoring.motor_current,
                        "velocity": monitoring.linear_velocity,
                        "distance": monitoring.distance_travelled,
                        "cpu_usage": self._filtered_cpu_usage,
                        "cpu_temp": self._filtered_cpu_temp,
                        "undervoltage": monitoring.undervoltage,
                        "overcurrent": monitoring.overcurrent,
                    }
                )

            if drive is not None:
                self.telemetry.update(
                    {"throttle": drive.last_throttle, "steer": drive.last_steer}
                )
            self.telemetry["active"] = self.robot.autopilot_enabled

    async def _broadcast_telemetry(self) -> None:
        interval = 1.0 / self.stream_fps
        while True:
            await asyncio.sleep(interval)
            payload = json.dumps({"type": "telemetry", **self.telemetry})

            dead: set[web.WebSocketResponse] = set()
            for ws in list(self.ws_clients):
                try:
                    await ws.send_str(payload)
                except Exception:
                    dead.add(ws)
            self.ws_clients -= dead

    # --- routes ---

    async def _handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=self.html, content_type="text/html")

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)
        self.log.info(f"Dashboard client connected: {request.remote}")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "space":
                        self.robot.set_autopilot_enabled(bool(data.get("pressed")))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self.ws_clients.discard(ws)
            self.robot.set_autopilot_enabled(False)
            self.log.info(f"Dashboard client disconnected: {request.remote}")

        return ws

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        camera = self.robot.cameras.get(request.match_info["side"])
        response = web.StreamResponse()
        response.content_type = "multipart/x-mixed-replace; boundary=frame"
        await response.prepare(request)

        interval = 1.0 / self.stream_fps
        try:
            while not self.stopping:
                jpeg = camera.preview_jpeg if camera is not None else None
                if jpeg:
                    await response.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                await asyncio.sleep(interval)
        except (ConnectionResetError, asyncio.CancelledError, Exception):
            pass
        return response
