# rc_car — Raspberry Pi 5 RC car

Everything that runs on the car's Pi 5 (`ssh rp5`, `~/rc_car`): hardware drivers, the web
dashboard / remote-control server, calibration tools and hardware tests.

```
rc_car/
  dashboard.py        web dashboard + control server  (http://rp5:8080, tcp :8081)
  web/index.html      the dashboard page
  car_client.py       PC-side keyboard client for the tcp link (stdlib only, copy to the PC)
  hardware/           Python package: one class per subsystem (see "Hardware package")
  tools/              interactive tools (need a TTY: ssh -t rp5 python3 ~/rc_car/tools/...)
  tests/              hardware self-test, PWM sweep, CPU/Hailo stress + soak tests
  scripts/            boot / setup / diagnostics shell scripts
  config/             steering.json (written by tools/steering_cal.py)
  vision/             Hailo-8 object detection (see "Vision")
  results/            test outputs      recordings/  dashboard mp4s      logs/  dashboard.log
  models/             *.hef detectors (built on the PC, not in git)
```

Git: this folder is a repo (`main`). The PC keeps a clone in `RcCarYoloTraining/car/` and pushes
straight into this checkout (`receive.denyCurrentBranch updateInstead`). `recordings/`,
`results/`, `logs/`, `models/` and `*.bak` are not tracked.

## Safety

* The motor can drive the car away. Nothing in this tree applies throttle without an explicit
  request: the dashboard caps it at 20 % and stops the motor after 400 ms without a command,
  `tools/drive_test.py` asks for a typed `yes`, and the tests only touch the steering servo.
* **GPIO13 is the ESC throttle.** `esc-neutral.service` puts it at 1500 µs at boot and
  `hardware/esc.py` never removes that signal. Do not open GPIO13 as a plain `Servo`
  (closing it would drop the signal); `tests/pwm_sweep.py` refuses it without `--allow-esc-pin`.
* The throttle calibration is reversed on this car: **1000 µs = full forward, 2000 µs = full
  brake** (`hardware/esc.py`). `HobbywingESC.throttle(+1)` is still forward.
* After writing files on the Pi run `sync` — a hard power cut has already zeroed files once.

## Dashboard

```
scripts/dashboard_restart.sh          # (re)start detached, log in logs/dashboard.log
python3 dashboard.py                  # or in the foreground
```
Open `http://rp5:8080`: MJPEG camera with resolution switching (640x360 .. 1536x864 at up to
120 fps, 1920x1080 / 2304x1296 at 56 fps), mp4 recording (H.264 to
`recordings/`, playable/deletable from the page), telemetry (drive, battery, board power, CPU,
WiFi, Hailo, links) and keyboard driving (W/S/A/D/space) over a WebSocket.

`car_client.py --host <pi-ip>` on a PC drives over the tcp link (:8081) and shows the health
line. Both links use one JSON object per line: `{"t": -1..1, "s": -1..1}` in (throttle, negative
= brake; steering), health `{"h":1,"v":batt,"temp":..,"rps":..,"us":..,"st":..,"act":..,"lk":..,"stop":..}`
out at 5 Hz. The freshest command from any link wins.

HTTP API: `GET /api/telemetry`, `POST /api/mode {"w","h","fps"}` (409 while recording),
`POST /api/record {"on":bool}`, `GET /api/recordings`, `DELETE /api/recordings/<name>`,
`GET /recordings/<name>` (range requests), `GET /stream.mjpg`, `GET /ws`,
`POST /api/vision {"on","hef","conf","filter"}`, `GET /ws/det` (detection boxes, see "Vision").

## Vision

Apple detector for the track (trained in `RcCarYoloTraining` on the PC). Needs `python3-hailort`
(apt, 4.23.0) + picamera2; no OpenCV.

* `vision/apple_detector.py`: camera lores stream (640 wide, ISP-scaled from the main stream) ->
  letterbox 640x640 -> HailoRT async API (several frames in flight) -> on-chip NMS -> boxes
  normalised 0..1. Standalone benchmark (needs the camera, so stop the dashboard first):
  `python3 vision/apple_detector.py --save 3`.
* Dashboard: **▶ detection** under the video; model = any `models/*.hef` or the stock COCO
  yolov8s (`/usr/share/hailo-models/yolov8s_h8.hef`, "apple only" = COCO class 47). Boxes are
  drawn over the MJPEG in the browser (not burned into recordings). The Vision panel shows
  detection vs camera fps, dropped frames, delay from sensor timestamp to boxes (readout+ISP /
  Hailo / decode) and Hailo chip temperature. Use 1280x720 modes: that is the sensor mode
  (1536x864) the model was trained on.
* `vision/eval_hef.py`: run a HEF over saved JPEGs and dump predictions (accuracy checks).

Measured 2026-09-23 with `models/apple_yolov8s.hef` at 1280x720@120: 120 det/s, delay 24.5 ms p50,
CPU ~27 %. Only one process can own the Hailo: turn detection off before running other Hailo tools.

WiFi only (the Pi 5 antenna sits under the Hailo HAT, so Bluetooth was dropped). Ethernet is
for updates. NetworkManager keyfiles: wlan0 = 192.168.88.126, eth0 = 192.168.88.129.

## Hardware package

```python
import sys; sys.path.insert(0, "/home/matasciuzelis/rc_car")
from hardware import RP5Board, Drive

with RP5Board() as b:
    b.cpu.temperature_c()                 # 45.8
    b.power.sample().total_w              # 2.26  (sum of PMIC rail I*V)
    b.battery.voltage()                   # 6.99  (MCP3202 CH1 * 19.7k/4.7k)
    b.hailo.benchmark(seconds=5)          # BenchmarkResult(hw_only_fps=308.7, latency_ms=6.66, ...)
    b.camera.set_mode(1280, 720, fps=60)  # picks raw 1536x864 mode
    frame = b.camera.capture_frame()      # numpy (720, 1280, 3) RGB888
    b.encoder.read()                      # EncoderReading(angle_deg, turns, velocity_dps, magnet, ...)

with Drive() as d:                        # ESC neutral, steering centred, encoder sampling thread
    d.steer(-0.5)                         # -1 left .. +1 right (calibrated)
    d.forward(0.1); d.brake(); d.stop()   # forward/brake fractions 0..1
    d.rps, d.rpm, d.turns
```

Every class derives from `hardware.base.Subsystem`: `available()` cheap presence check,
`status()` JSON-able dict, `close()` / `with`. Failures raise `hardware.HardwareError`.

| class | file | backend / wiring |
|---|---|---|
| `CpuMonitor` | cpu.py | `/sys/class/thermal`, cpufreq, `vcgencmd get_throttled` |
| `BoardPower`, `PowerLogger` | power.py | `vcgencmd pmic_read_adc`; `logger(interval, csv)` = background sampler with `.stats()` |
| `HailoBoard` | hailo.py | `hailortcli` identify / scan / benchmark. `measure_power()` raises: the AI-HAT Hailo-8 has no current monitor |
| `Camera` | camera.py | picamera2, IMX708 on CAM0. `set_mode(w, h, fps)` picks the smallest raw mode that fits; `set_fps()` live |
| `MCP3202` | adc.py | kernel IIO (`dtoverlay=mcp3202,spi0-0-present`), SPI0 CS0 = GPIO8, VREF 3.3 V |
| `BatteryMonitor` | adc.py | MCP3202 CH1 through 15k/4.7k divider (ratio 4.1915, full scale 13.8 V) |
| `AS5600` | encoder.py | I2C bus 1 (GPIO2/3) addr 0x36 on the motor shaft; `read()` full, `read_fast()` ~1.5 kHz |
| `HardwarePWM`, `Servo` | pwm.py | sysfs `/sys/class/pwm` on RP1 **PWM0**: GPIO12 → chan0, GPIO13 → chan1 (18/19 → 2/3) |
| `Steering` | steering.py | Servo on GPIO12; center/amplitude/invert from `config/steering.json` (defaults 1500/300 µs) |
| `HobbywingESC` | esc.py | QuicRun 1080 G2: throttle pulse on GPIO13 only; the ESC is switched on by hand |
| `Drive` | drive.py | ESC + encoder + steering; median-filtered rps/rpm, odometer in turns |
| `RP5Board` | board.py | lazy facade: `cpu power hailo camera adc battery encoder`, `servo(12)`, `summary()` |

### Wiring summary

| function | Pi GPIO (pin) |
|---|---|
| steering servo PWM | GPIO12 (32) |
| ESC throttle PWM | GPIO13 (33) — the ESC's power button and SET button are operated by hand |
| AS5600 encoder | I2C1: SDA GPIO2 (3), SCL GPIO3 (5) |
| MCP3202 ADC | SPI0: CS GPIO8 (24), MISO GPIO9 (21), MOSI GPIO10 (19), SCLK GPIO11 (23); VDD/VREF 3.3 V |
| battery sense | 2S LiPo (6.4–8.4 V) → MCP3202 CH1 via 15k (top) / 4.7k (bottom) |
| camera | Arducam 12MP IMX708 (no EEPROM) on CAM0, `dtoverlay=imx708,cam0` |
| Hailo-8 26 TOPS | AI HAT on PCIe |

## Tools (`ssh -t rp5 python3 ~/rc_car/tools/<tool>`)

| tool | purpose |
|---|---|
| `steering_cal.py` | a/d centre ±5 µs (A/D ±25), [ ] amplitude, l/r/c end stops, i invert, q quit. Saves `config/steering.json` on every change |
| `esc_throttle.py` | raw throttle pulses (n/f/b, +/-) for the ESC's throttle-range calibration and bench checks |
| `drive_test.py` | ramp forward throttle until the encoder sees motion, hold, stop (asks for `yes`; ESC switched on by hand) |
| `encoder_watch.py` | print AS5600 angle / turns / velocity while turning the shaft |

## Tests

```
python3 tests/selftest.py [--skip camera,hailo] [--batt-min 6.4 --batt-max 8.4]
                                        # every subsystem; results/selftest_<ts>.json
python3 tests/pwm_sweep.py              # steering servo sweep with duty verification via pinctrl
python3 tests/camera_sweep.py           # capture + MJPEG encoder fps per mode -> results/camera/
tests/stress_test.sh                    # idle / CPU / Hailo / both with PMIC logging -> results/
tests/soak_test.sh [seconds]            # CPU + Hailo soak (default 300 s) -> results/soak/
```
Stop the dashboard first (`pkill -f dashboard.py`): the camera and PWM pins cannot be shared.

## Setup scripts (`scripts/`)

| script | purpose |
|---|---|
| `esc_neutral.sh` | run at boot by `/etc/systemd/system/esc-neutral.service`: GPIO13 at 50 Hz / 1500 µs |
| `dashboard_restart.sh` | restart the dashboard detached |
| `setup_pwm_overlay.sh` | adds `dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4` (PWM0 is off in the stock DT; only PWM1, the fan, exists) |
| `setup_camera_overlay.sh [cam0\|cam1]` | `camera_auto_detect=0` + `dtoverlay=imx708,<port>` (the Arducam has no EEPROM) |
| `setup_mcp3202_overlay.sh` | replaces `spi0-2cs` with `mcp3202,spi0-0-present` |
| `diag_camera.sh`, `diag_iio.sh`, `diag_i2c.sh` | detection diagnostics for the camera, the ADC and the I2C buses |

`/boot/firmware/config.txt` needs: `dtparam=i2c_arm=on`, `dtoverlay=mcp3202,spi0-0-present`,
`camera_auto_detect=0`, `dtoverlay=imx708,cam0`, `dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4`.
Access to `/sys/class/pwm` comes from the `gpio` group (`/lib/udev/rules.d/99-com.rules`);
passwordless sudo is configured so `hardware/pwm.py` can `dtoverlay` at runtime if needed.

## Verified figures (selftest, 2026-09-19 .. 21)

* CPU 45.8 °C @ 2400 MHz idle, throttled 0x0; board power idle 2.1–2.3 W (core 0.57 W), EXT5V 5.06 V
* Hailo-8 fw 4.23.0, yolov8s_h8: 308.7 fps hw-only, 6.66 ms latency; `measure-power` unsupported
* MCP3202 CH1 stable within ±3 counts; 2069 counts → 6.99 V with the battery at 7.0 V
* PWM 50 Hz on GPIO12/13: period 20 000 000 ns, 1.5 ms pulse measured ~7.5 % high
* Camera native modes 1536x864@120, 2304x1296@56, 4608x2592@14; 92/92 fps×resolution combos
  hit target within 0.15 fps, mode switch 0.3–0.4 s (`results/camera/`)
* Encoder `read_fast()` ~1.5 kHz
* Camera pipeline (`tests/camera_sweep.py`, 2026-09-21): the software JPEG encoder keeps up with
  every mode up to 1536x864@120 (16 % CPU on YUV420, ~47 Mbit/s) and 2304x1296@56 (~52 Mbit/s);
  the dashboard delivered the full frame rate of every mode over WiFi at the desk (-34 dBm).
  Larger modes will be limited by WiFi throughput at range, not by the Pi.

## Troubleshooting notes

* **Camera not detected** (`imx708 10-001a: failed to read chip id 708, error -5`): the sensor is
  not answering on I2C. It was a ribbon problem — re-seat with the gold contacts facing away
  from the PCB, 22-pin end in the Pi; try the other cable from the kit; `scripts/diag_camera.sh`.
* **No pwmchip for PWM0**: the overlay is missing → `scripts/setup_pwm_overlay.sh` (PWM0 shows
  up as `pwmchip1`; `pwmchip0` is the fan).
* **`hailortcli` strings look padded**: the firmware NUL-pads them; `HailoBoard.identify()` strips
  them. The `hailo_pci` WARNING in dmesg is benign. `hailort.log` is written to the cwd by
  `hailortcli`; it can be deleted.
* **Dashboard fails to start**: another instance (or a test) still owns the camera / PWM —
  `scripts/dashboard_restart.sh` waits for it; check `logs/dashboard.log`.
