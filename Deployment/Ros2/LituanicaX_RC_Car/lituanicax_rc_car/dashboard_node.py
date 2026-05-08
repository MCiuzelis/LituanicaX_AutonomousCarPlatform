from __future__ import annotations

import asyncio
import json
import threading
from array import array

import aiohttp
import rclpy
from aiohttp import web
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32MultiArray

from .common import FPS_TARGET


HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RC Car Autopilot</title>
<style>
  body {
    background: #111; color: #eee; font-family: monospace;
    margin: 0; display: flex; flex-direction: column;
    align-items: center; padding: 20px; user-select: none;
  }
  h1 { margin: 0 0 14px; }
  #cameras { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
  .cam-wrap img { width: 480px; border: 2px solid #444; display: block; background: #1a1a1a; min-height: 180px; }
  .label { text-align: center; margin-top: 5px; font-size: 13px; color: #aaa; }
  #status { margin-top: 12px; font-size: 14px; }
  #status.active { color: #4f4; }
  #status.inactive { color: #f84; }
  #hint { margin-top: 4px; font-size: 12px; color: #666; }
  #telemetry {
    margin-top: 14px; display: grid;
    grid-template-columns: repeat(4, minmax(150px, 190px)); gap: 12px;
  }
  .gauge-box {
    background: #1e1e1e; border: 1px solid #444;
    border-radius: 6px; padding: 12px;
  }
  .gauge-label { font-size: 11px; color: #888; margin-bottom: 6px; letter-spacing: 1px; }
  .gauge-value { font-size: 22px; font-weight: bold; }
  .bar-bg { background: #2a2a2a; height: 8px; border-radius: 4px; margin-top: 8px; position: relative; }
  .bar-center { position: absolute; top: 0; left: 50%; width: 1px; height: 100%; background: #555; }
  .bar-fg { height: 8px; border-radius: 4px; transition: width 0.06s, margin 0.06s, background 0.1s; }
  #space-btn {
    margin-top: 16px; padding: 12px 40px;
    border: 2px solid #555; border-radius: 8px;
    font-size: 16px; font-family: monospace; color: #888;
    background: #1a1a1a; transition: all 0.05s;
  }
  #space-btn.active { border-color: #4f4; color: #4f4; background: #0a1f0a; }
  @media (max-width: 820px) {
    .cam-wrap img { width: min(92vw, 480px); }
    #telemetry { grid-template-columns: repeat(2, minmax(140px, 1fr)); width: 92vw; }
  }
</style>
</head>
<body>
<h1>RC Car Autopilot</h1>

<div id="cameras">
  <div class="cam-wrap">
    <img src="/stream/left" alt="Left ToF camera">
    <div class="label">Camera 0 (left)</div>
  </div>
  <div class="cam-wrap">
    <img src="/stream/right" alt="Right ToF camera">
    <div class="label">Camera 8 (right)</div>
  </div>
</div>

<div id="status" class="inactive">Connecting...</div>
<div id="hint">Hold SPACE to activate autopilot - click this tab first to capture keys</div>

<div id="telemetry">
  <div class="gauge-box">
    <div class="gauge-label">THROTTLE</div>
    <div class="gauge-value" id="tval">0.000</div>
    <div class="bar-bg">
      <div class="bar-fg" id="tbar" style="width:0%;background:#555;"></div>
    </div>
  </div>
  <div class="gauge-box">
    <div class="gauge-label">STEER</div>
    <div class="gauge-value" id="sval">0.000</div>
    <div class="bar-bg">
      <div class="bar-center"></div>
      <div class="bar-fg" id="sbar" style="width:50%;background:#555;margin-left:50%;"></div>
    </div>
  </div>
  <div class="gauge-box">
    <div class="gauge-label">BATTERY</div>
    <div class="gauge-value" id="bval">0.00 V</div>
    <div class="bar-bg">
      <div class="bar-fg" id="bbar" style="width:0%;background:#4f4;"></div>
    </div>
  </div>
  <div class="gauge-box">
    <div class="gauge-label">MOTOR CURRENT</div>
    <div class="gauge-value" id="cval">0.000 A</div>
    <div class="bar-bg">
      <div class="bar-fg" id="cbar" style="width:0%;background:#fa4;"></div>
    </div>
  </div>
</div>

<div id="space-btn">SPACE - hold to activate</div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const statusEl = document.getElementById('status');
const tval = document.getElementById('tval');
const tbar = document.getElementById('tbar');
const sval = document.getElementById('sval');
const sbar = document.getElementById('sbar');
const bval = document.getElementById('bval');
const bbar = document.getElementById('bbar');
const cval = document.getElementById('cval');
const cbar = document.getElementById('cbar');
const spaceBtn = document.getElementById('space-btn');
let held = false;

ws.onopen = () => { statusEl.textContent = 'Connected - hold SPACE to drive'; statusEl.className = 'inactive'; };
ws.onclose = () => { statusEl.textContent = 'Disconnected'; statusEl.className = 'inactive'; };
ws.onerror = () => { statusEl.textContent = 'Connection error'; statusEl.className = 'inactive'; };

ws.onmessage = ({ data }) => {
  const msg = JSON.parse(data);
  if (msg.type !== 'telemetry') return;

  const tp = Math.max(0, Math.min(100, msg.throttle * 100)).toFixed(1);
  tval.textContent = msg.throttle.toFixed(3);
  tbar.style.width = tp + '%';
  tbar.style.marginLeft = '0';
  tbar.style.background = msg.throttle > 0.05 ? '#4af' : '#555';

  const sv = msg.steer;
  sval.textContent = (sv >= 0 ? '+' : '') + sv.toFixed(3);
  if (sv >= 0) {
    sbar.style.marginLeft = '50%';
    sbar.style.width = (sv * 50).toFixed(1) + '%';
  } else {
    sbar.style.marginLeft = ((0.5 + sv / 2) * 100).toFixed(1) + '%';
    sbar.style.width = (-sv * 50).toFixed(1) + '%';
  }
  sbar.style.background = Math.abs(sv) > 0.05 ? '#fa4' : '#555';

  const bv = msg.battery;
  bval.textContent = bv.toFixed(2) + ' V';
  bbar.style.width = Math.min(100, (bv / 10.0) * 100).toFixed(1) + '%';
  bbar.style.background = bv < 6.0 ? '#f64' : bv < 7.0 ? '#fa4' : '#4f4';

  const cv = msg.current;
  cval.textContent = cv.toFixed(3) + ' A';
  cbar.style.width = Math.min(100, (cv / 5.0) * 100).toFixed(1) + '%';

  statusEl.textContent = msg.active ? 'AUTOPILOT ACTIVE' : 'Policy inactive - hold SPACE';
  statusEl.className = msg.active ? 'active' : 'inactive';
};

function sendSpace(pressed) {
  if (ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: 'space', pressed }));
  spaceBtn.classList.toggle('active', pressed);
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !held) {
    e.preventDefault();
    held = true;
    sendSpace(true);
  }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space') {
    e.preventDefault();
    held = false;
    sendSpace(false);
  }
});
window.addEventListener('blur', () => {
  if (held) { held = false; sendSpace(false); }
});
</script>
</body>
</html>
"""


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
            "active": False,
        }
        self.lock = threading.Lock()

        self.enable_pub = self.create_publisher(Bool, "/autopilot/enabled", 1)
        self.create_subscription(CompressedImage, "/camera/left/preview/compressed", self._left_preview_cb, sensor_qos())
        self.create_subscription(CompressedImage, "/camera/right/preview/compressed", self._right_preview_cb, sensor_qos())
        self.create_subscription(Float32MultiArray, "/hardware/telemetry", self._telemetry_cb, 1)
        self.create_subscription(Bool, "/autopilot/enabled", self._enabled_cb, 1)

    def _left_preview_cb(self, msg: CompressedImage) -> None:
        self._store_preview("left", msg)

    def _right_preview_cb(self, msg: CompressedImage) -> None:
        self._store_preview("right", msg)

    def _store_preview(self, side: str, msg: CompressedImage) -> None:
        with self.lock:
            self.latest_jpeg[side] = bytes(msg.data)

    def _telemetry_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 5:
            return
        with self.lock:
            self.telemetry.update(
                {
                    "throttle": float(msg.data[0]),
                    "steer": float(msg.data[1]),
                    "battery": float(msg.data[2]),
                    "current": float(msg.data[3]),
                    "active": bool(msg.data[4]),
                }
            )

    def _enabled_cb(self, msg: Bool) -> None:
        with self.lock:
            self.telemetry["active"] = bool(msg.data)

    def publish_enabled(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self.enable_pub.publish(msg)


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
    app["telemetry_task"].cancel()
    try:
        await app["telemetry_task"]
    except asyncio.CancelledError:
        pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashboardNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = web.Application()
    app["node"] = node
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/stream/{side}", mjpeg_stream)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    try:
        node.get_logger().info(f"Dashboard available at http://<pi-ip>:{node.port}")
        web.run_app(app, host=node.host, port=node.port)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
