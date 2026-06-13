from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import threading
from array import array

import aiohttp
import rclpy
from aiohttp import web
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32MultiArray

from .common import FPS_TARGET


_HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")
with open(_HTML_PATH) as _f:
    HTML = _f.read()


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class DashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("dashboard_node")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("stream_fps", FPS_TARGET)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.stream_fps = float(self.get_parameter("stream_fps").value)

        self.ws_clients: set[web.WebSocketResponse] = set()
        self.latest_jpeg = {"left": None, "right": None}
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
        # EMA low-pass filter state
        self._filtered_battery = 0.0
        self._filtered_cpu_usage = 0.0
        self._filtered_cpu_temp = 0.0
        # Smoothing factors (lower = smoother but more lag)
        self._alpha_battery = 0.3
        self._alpha_cpu = 0.2
        self._alpha_temp = 0.025

        self.lock = threading.Lock()

        self.enable_pub = self.create_publisher(Bool, "/autopilot/enabled", 1)
        self.create_subscription(CompressedImage, "/camera/left/preview/compressed", self._left_preview_cb, sensor_qos())
        self.create_subscription(CompressedImage, "/camera/right/preview/compressed", self._right_preview_cb, sensor_qos())
        self.create_subscription(Float32MultiArray, "/monitoring/telemetry", self._telemetry_cb, 1)
        self.create_subscription(Float32MultiArray, "/drive/telemetry", self._drive_cb, 1)
        self.create_subscription(Bool, "/autopilot/enabled", self._enabled_cb, 1)

    def _left_preview_cb(self, msg: CompressedImage) -> None:
        self._store_preview("left", msg)

    def _right_preview_cb(self, msg: CompressedImage) -> None:
        self._store_preview("right", msg)

    def _store_preview(self, side: str, msg: CompressedImage) -> None:
        with self.lock:
            self.latest_jpeg[side] = bytes(msg.data)

    def _telemetry_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 6:
            return
        raw_battery = float(msg.data[0])
        raw_cpu_usage = float(msg.data[2])
        raw_cpu_temp = float(msg.data[3])
        with self.lock:
            self._filtered_battery = self._alpha_battery * raw_battery + (1.0 - self._alpha_battery) * self._filtered_battery
            self._filtered_cpu_usage = self._alpha_cpu * raw_cpu_usage + (1.0 - self._alpha_cpu) * self._filtered_cpu_usage
            self._filtered_cpu_temp = self._alpha_temp * raw_cpu_temp + (1.0 - self._alpha_temp) * self._filtered_cpu_temp
            self.telemetry.update(
                {
                    "battery": self._filtered_battery,
                    "current": float(msg.data[1]),
                    "velocity": float(msg.data[6]) if len(msg.data) >= 7 else 0.0,
                    "distance": float(msg.data[7]) if len(msg.data) >= 8 else 0.0,
                    "cpu_usage": self._filtered_cpu_usage,
                    "cpu_temp": self._filtered_cpu_temp,
                    "undervoltage": bool(msg.data[4]),
                    "overcurrent": bool(msg.data[5]),
                }
            )

    def _drive_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            return
        with self.lock:
            self.telemetry.update(
                {
                    "throttle": float(msg.data[0]),
                    "steer": float(msg.data[1]),
                    "active": bool(msg.data[2]),
                }
            )

    def _enabled_cb(self, msg: Bool) -> None:
        with self.lock:
            self.telemetry["active"] = bool(msg.data)

    def publish_enabled(self, enabled: bool) -> None:
        try:
            msg = Bool()
            msg.data = enabled
            self.enable_pub.publish(msg)
        except Exception:
            pass


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML, content_type="text/html")


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    node: DashboardNode = request.app["node"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    node.ws_clients.add(ws)
    node.get_logger().info(f"Dashboard client connected: {request.remote}")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "space":
                    node.publish_enabled(bool(data.get("pressed")))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        node.ws_clients.discard(ws)
        node.publish_enabled(False)
        node.get_logger().info(f"Dashboard client disconnected: {request.remote}")

    return ws


async def mjpeg_stream(request: web.Request) -> web.StreamResponse:
    node: DashboardNode = request.app["node"]
    side = request.match_info["side"]
    resp = web.StreamResponse()
    resp.content_type = "multipart/x-mixed-replace; boundary=frame"
    await resp.prepare(request)

    interval = 1.0 / node.stream_fps
    try:
        while True:
            with node.lock:
                jpeg = node.latest_jpeg.get(side)
            if jpeg:
                await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(interval)
    except (ConnectionResetError, asyncio.CancelledError, Exception):
        pass
    return resp


async def broadcast_telemetry(app: web.Application) -> None:
    node: DashboardNode = app["node"]
    interval = 1.0 / node.stream_fps
    while True:
        await asyncio.sleep(interval)
        with node.lock:
            payload = json.dumps({"type": "telemetry", **node.telemetry})

        dead: set[web.WebSocketResponse] = set()
        for ws in list(node.ws_clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.add(ws)
        node.ws_clients -= dead


async def on_startup(app: web.Application) -> None:
    app["telemetry_task"] = asyncio.create_task(broadcast_telemetry(app))


async def on_cleanup(app: web.Application) -> None:
    node: DashboardNode = app["node"]
    node.publish_enabled(False)
    for ws in list(node.ws_clients):
        try:
            await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"dashboard shutting down")
        except Exception:
            pass
    node.ws_clients.clear()
    app["telemetry_task"].cancel()
    try:
        await app["telemetry_task"]
    except asyncio.CancelledError:
        pass


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


async def _async_main(node: DashboardNode, app: web.Application) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    runner = web.AppRunner(app, shutdown_timeout=0.25)
    await runner.setup()
    site = web.TCPSite(runner, node.host, node.port)
    await site.start()

    local_ip = get_local_ip()
    node.get_logger().info(f"Dashboard open at http://{local_ip}:{node.port}")

    try:
        while rclpy.ok() and not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
    finally:
        await runner.cleanup()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashboardNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def spin_executor() -> None:
        try:
            executor.spin()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass

    spin_thread = threading.Thread(target=spin_executor, daemon=True)
    spin_thread.start()

    app = web.Application()
    app["node"] = node
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/stream/{side}", mjpeg_stream)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    try:
        asyncio.run(_async_main(node, app))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
