# Pi5 — Raspberry Pi 5 RC car

Everything that runs on the car's Pi 5 (`ssh rp5`, `~/LituanicaX_InteligentRobotCompetition/Pi5`): hardware drivers, the web
dashboard / remote-control server, calibration tools and hardware tests.

```
Pi5/
  dashboard.py        web dashboard + control server  (http://rp5:8080, tcp :8081)
  recorder.py         training-data recordings: video + per-frame IMU / odometry (see "Recordings")
  web/index.html      the dashboard page
  car_client.py       PC-side keyboard client for the tcp link (stdlib only, copy to the PC)
  hardware/           Python package: one class per subsystem (see "Hardware package")
  tools/              interactive tools (need a TTY: ssh -t rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/...)
  tests/              hardware self-test, PWM sweep, CPU/Hailo stress + soak tests
  scripts/            boot / setup / diagnostics shell scripts
  config/             steering.json (tools/steering_cal.py), imu.json, camera.json (tools/camera_cal.py)
  vision/             Hailo-8 object detection (see "Vision")
  results/            test outputs      recordings/  one folder per take      logs/  dashboard.log
  models/             *.hef detectors (built on the PC, not in git)
```

Git: this folder is `Pi5/` of the LituanicaX_InteligentRobotCompetition repo; the Pi runs it from its
clone at `~/LituanicaX_InteligentRobotCompetition` (the only copy of the car code on the Pi). The Pi's
clone accepts pushes into its checkout (`receive.denyCurrentBranch updateInstead`), so from the PC:
`git push origin main && git push rp5 main` (remote `rp5` = `rp5:LituanicaX_InteligentRobotCompetition`).
Changes made on the Pi: commit there, then `git pull rp5 main` on the PC and push to GitHub.
`recordings/`, `results/`, `logs/`, `models/` and `*.bak` are not tracked.

## Safety

* The motor can drive the car away. Nothing in this tree applies throttle without an explicit
  request: the dashboard caps it at 40 % and stops the motor after 400 ms without a command,
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
120 fps, 1920x1080 / 2304x1296 at 56 fps; the preview to the page is capped at 30 fps), training-data
recording (see "Recordings": playable / downloadable / deletable from the page), telemetry (drive,
IMU, battery, board power, CPU, WiFi, Hailo, links) and keyboard driving (W/S/A/D/space) over a WebSocket.

`car_client.py --host <pi-ip>` on a PC drives over the tcp link (:8081) and shows the health
line. Both links use one JSON object per line: `{"t": -1..1, "s": -1..1}` in (throttle, negative
= brake; steering), health `{"h":1,"v":batt,"temp":..,"rps":..,"us":..,"st":..,"act":..,"lk":..,"stop":..}`
out at 5 Hz. The freshest command from any link wins.

HTTP API: `GET /api/telemetry`, `POST /api/mode {"w","h","fps"}` (409 while recording),
`POST /api/record {"on":bool}`, `GET /api/recordings`, `DELETE /api/recordings/<id>` (the whole take),
`GET /api/recordings/<id>/download` (take as .tar), `GET /recordings/<id>/<id>.mp4` (range requests),
`GET /stream.mjpg`, `GET /ws`,
`POST /api/vision {"on","hef","conf","filter"}`, `GET /ws/det` (detection boxes, see "Vision"),
`GET /ws/imu` (car-frame attitude / accel / rates at 25 Hz), `POST /api/imu/level` (409 if the car
moves), `POST /api/imu/zero_yaw`. Telemetry `imu`: values, per-sensor Hz / lost, accuracy, resets.

**IMU panel**: attitude indicator + g-g plot (forward vs lateral, rings at 0.25/0.5/0.75 g), roll
(right side down +), pitch (nose up +), relative yaw (left +), yaw rate, fwd/lat accel, tilt. Attitude
is game_rv (no magnetometer: the motor/battery disturb it), so yaw
drifts slowly and is zeroed at start-up / by **zero yaw**. **set level** = car still on level ground
in its nominal pose -> roll = pitch = 0, saved to `config/imu.json` (same as `tools/imu_level.py`).
The dashboard enables only game_rv, accel, gyro, linear_accel, gravity at 200 Hz (`Imu.SENSORS`,
`Imu.RATE_HZ`; measured ~200 Hz game_rv/gyro, ~175 Hz the rest, 0 lost). The magnetometer and rv
(which fuses it) are never enabled: they are useless here and cost hub bandwidth (with them all 7
outputs ran at 100 Hz). Dynamic calibration: accel + gyro (the hub boots with gyro off; the
driver re-applies it after hub resets / worker restarts). Reliability is shown, not assumed: each
`/ws/imu` message carries `age_ms` (age of the attitude sample; the page says "imu data stale" above
150 ms, "imu link lost" if no message for 300 ms), and "dropouts" counts gaps > 50 ms in the
game_rv stream from 2 s after start (fusion runs at ~12 Hz for its first ~0.75 s).
Push-only WebSockets (`/ws/imu`, `/ws/det`) must still read: aiohttp handles heartbeat pongs in
`receive()`, and without a reader it dropped them every 7.5 s (fixed 2026-10-04).

## Recordings (training data)

**● record** on the page writes one take into `recordings/<id>/`, `<id>` =
`YYYYmmdd_HHMMSS_cam0_<w>x<h>_<fps>fps`. Every file carries the id, so a copied-out file still says
which camera / mode / take it is from; **delete** removes the folder (video + IMU + metadata together),
**download** streams it as `<id>.tar`.

| file | content |
|---|---|
| `<id>.mp4` | H.264 (x264 ultrafast, ~0.2 bit/pixel), fragmented mp4 (playable up to the last second after a power cut); tags title = id, comment = camera / mode. Pixels are clean (nothing drawn in) |
| `<id>_frames.csv` | **one row per video frame** (row i = frame i, `pts_us` = video pts): sensor timestamp, exposure, gains, lux, lens position, and the IMU / odometry interpolated to the middle of that frame's exposure: car roll / pitch / yaw + quaternion, body rates, linear accel, **camera -> world quaternion** and camera pitch / roll / heading (CameraModel conventions, ready for `cam.ground(u, v, pitch_deg=, roll_deg=)`), encoder turns / rps, steering, ESC pulse, dropped frames before it |
| `<id>_imu.csv` | every IMU sample (game_rv, gyro, linear_accel, accel, gravity ~150-200 Hz; no magnetometer), raw board frame, same clock |
| `<id>_drive.csv` | encoder turns / rps, steering, ESC pulse, throttle command at 100 Hz |
| `<id>_meta.json` | camera (id `cam0` = CSI CAM0, front IMX708; mount height / pitch / roll; intrinsics; full `config/camera.json`), this stream's ScalerCrop, K and distortion, IMU mount (`config/imu.json`), column descriptions, git commit, summary (frames, measured fps, dropped, frames without IMU, IMU rates) |

World frame of a take: X = car heading at its first frame, Y left, Z up (yaw is relative, game_rv has
no magnetometer). All timestamps are CLOCK_MONOTONIC ns; frame time = SensorTimestamp - exposure/2 +
readout/2 (rolling shutter, middle row).

Frame rate: the camera runs at the selected mode (e.g. 120 fps for detection); the recording takes
every Nth frame, at most 60 fps up to 1536x864 and 30 fps above (`REC_FPS_SMALL/LARGE` in
dashboard.py; the Pi 5 has no H.264 hardware). Measured 2026-10-10, 60 s takes: 1280x720@60,
1536x864 (120 -> 60), 1280x720 (120 -> 60) with detection on, 2304x1296 (56 -> 28): 0 dropped frames,
0 frames without IMU, CPU 35-58 %, video ~1.3-2.1 MB/s. The mp4 is muxed on its own thread: the SD
card stalls writes for up to ~8 s (ext4 commits) and writing from the camera thread dropped frames.
Recording stops by itself below 500 MB free.

```python
import csv, cv2                                   # e.g. frames + camera pose for 3D apple positions
rows = list(csv.DictReader(open(f"{id}/{id}_frames.csv")))
cap = cv2.VideoCapture(f"{id}/{id}.mp4")
for r in rows:
    ok, img = cap.read()                          # frame int(r["frame"])
    pitch, roll = float(r["cam_pitch_deg"]), float(r["cam_roll_deg"])
```

## Vision

Apple detector for the track (trained in `YoloTraining/` on the PC). Needs `python3-hailort`
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

## Camera calibration (3D from detections)

The lens is a wide one (~96 deg pinhole-equivalent HFOV at the calibrated focal length, ~122 deg
diagonal) with strong barrel distortion (k1 -0.215). `config/camera.json` holds intrinsics +
distortion in **full-sensor pixels** (4608x2592) and the mount (height, pitch, roll); one
calibration serves every stream through its ScalerCrop: 1280x720 / 1536x864 modes and the
detector's 640x360 lores = crop (768, 432, 3072, 1728), 2304x1296 / 4608x2592 = whole sensor.
Lens focus is the libcamera default, manual LensPosition 1.0 dioptre: if autofocus or another lens
position is ever used, the focal length changes and the calibration must be redone.

```python
from vision.camera_model import CameraModel, CROP_1536      # numpy only
cam = CameraModel.load()
X, Y = cam.box_ground((x1, y1, x2, y2), CROP_1536)  # detector box (0..1) -> floor point under its
                                                    # bottom-centre, mm: X forward, Y left of the camera
cam.ground(u, v, z=35)                              # sensor px -> point on the plane Z = 35 mm
cam.project([[X, Y, Z]])                            # world mm -> sensor px
mx, my = cam.undistort_maps(CROP_1536, (1280, 720)) # cv2.remap maps for an undistorted image
cam.K(CROP_1536, (1280, 720)), cam.dist             # OpenCV-style K / dist for that stream
```

World frame: origin on the floor under the camera, X forward, Y left, Z up; `ground()` takes
`pitch_deg=` / `roll_deg=` to follow body pitch (static pitch_deg - IMU pitch change, since IMU
pitch is nose-up + and camera pitch is down +).

Calibration 2026-10-04 (`results/camera_cal/20261004_cal/`): one full-res view of 4 A4 boards
(8x6 inner corners, 30 mm squares) flat on the floor + 5 straight room edges (plumb lines) for the
upper half of the frame. f = 2093 px, k1 -0.2151, k2 0.0378, principal point fixed at the sensor
centre (a single view cannot separate it from pitch), camera **186.1 mm** above the floor, pitch
**2.1 deg down**, roll 0.0 deg. Residuals 1.5 px boards (paper curl) / 0.8 px lines at full res;
independent check on the 1280x720 and 2304x1296 captures: 0.70 px. Floor points from the
1280x720 detections land 1-2 mm (max 6 mm) from the fitted board corners at 0.6-0.9 m. One
1280x720 pixel on the floor is 1.7 mm at 0.5 m, 6 mm at 1 m, 25 mm at 2 m, 55 mm at 3 m (the
horizon is row ~328, the nearest visible floor 0.39 m straight ahead, 0.34 m in the bottom
corners). The distortion is measured out to 1780 px from the centre, which covers the whole 1536x864-crop FOV; the corners of the full-FOV
modes are extrapolated. The board scale assumes the printout really has 30 mm squares: height
and all distances scale with it (check with a ruler).

Full context, limitations and ideas for further calibration: `results/camera_cal/CALIBRATION_NOTES.md`.
Redo it (e.g. after moving the camera): boards flat on the floor, then
`pkill -f "[p]ython3 dashboard.py"; python3 tools/camera_cal.py capture; scripts/dashboard_restart.sh`
and `python3 tools/camera_cal.py fit results/camera_cal/<run> --write` (optionally a `lines.json`
of straight edges in the run dir; a few minutes). Needs python3-opencv +
python3-scipy (apt).

## Hardware package

```python
import sys; sys.path.insert(0, "/home/matasciuzelis/LituanicaX_InteligentRobotCompetition/Pi5")
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
| `BNO085` | bno085.py | 9-DOF IMU, SPI0 CE1 (spidev0.1) + INT GPIO6 + RST GPIO5; own worker process, see "IMU (BNO085)" |
| `HardwarePWM`, `Servo` | pwm.py | sysfs `/sys/class/pwm` on RP1 **PWM0**: GPIO12 → chan0, GPIO13 → chan1 (18/19 → 2/3) |
| `Steering` | steering.py | Servo on GPIO12; center/amplitude/invert from `config/steering.json` (defaults 1500/300 µs) |
| `HobbywingESC` | esc.py | QuicRun 1080 G2: throttle pulse on GPIO13 only; the ESC is switched on by hand |
| `Drive` | drive.py | ESC + encoder + steering; median-filtered rps/rpm, odometer in turns |
| `CarFrame` | imu_frame.py | IMU board frame -> car frame (x fwd, y left, z up): mount axes + level reference from `config/imu.json` |
| `RP5Board` | board.py | lazy facade: `cpu power hailo camera adc battery encoder imu`, `servo(12)`, `summary()` |

### Wiring summary

| function | Pi GPIO (pin) |
|---|---|
| steering servo PWM | GPIO12 (32) |
| ESC throttle PWM | GPIO13 (33) — the ESC's power button and SET button are operated by hand |
| AS5600 encoder | I2C1: SDA GPIO2 (3), SCL GPIO3 (5) |
| MCP3202 ADC | SPI0: CS GPIO8 (24), MISO GPIO9 (21), MOSI GPIO10 (19), SCLK GPIO11 (23); VDD/VREF 3.3 V |
| BNO085 IMU (Adafruit 4754, P0+P1 jumpered = SPI) | SPI0: CS GPIO7 (26), SDA->MISO GPIO9, DI->MOSI GPIO10, SCL->SCLK GPIO11; INT GPIO6 (31), RST GPIO5 (29) |
| battery sense | 2S LiPo (6.4–8.4 V) → MCP3202 CH1 via 15k (top) / 4.7k (bottom) |
| camera | Arducam 12MP IMX708 (no EEPROM) on CAM0, `dtoverlay=imx708,cam0` |
| Hailo-8 26 TOPS | AI HAT on PCIe |

## Tools (`ssh -t rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/<tool>`)

| tool | purpose |
|---|---|
| `steering_cal.py` | a/d centre ±5 µs (A/D ±25), [ ] amplitude, l/r/c end stops, i invert, q quit. Saves `config/steering.json` on every change |
| `esc_throttle.py` | raw throttle pulses (n/f/b, +/-) for the ESC's throttle-range calibration and bench checks |
| `drive_test.py` | ramp forward throttle until the encoder sees motion, hold, stop (asks for `yes`; ESC switched on by hand) |
| `encoder_watch.py` | print AS5600 angle / turns / velocity while turning the shaft |
| `camera_cal.py capture / fit <run> [--write]` | camera calibration from checkerboards on the floor -> `config/camera.json` (see "Camera calibration") |
| `imu_level.py [--show]` | capture the IMU level reference (car still, level, nominal pose) into `config/imu.json`; dashboard stopped |

## Tests

```
python3 tests/selftest.py [--skip camera,hailo] [--batt-min 6.4] [--batt-max 8.4]
                                        # supply check: stable and >= batt-min (bench DC supply ok);
                                        # --batt-max only when a 2S LiPo is fitted
                                        # every subsystem incl. encoder + IMU (car at rest, levelled);
                                        # results/selftest_<ts>.json
python3 tests/pwm_sweep.py              # steering servo sweep with duty verification via pinctrl
python3 tests/camera_sweep.py           # capture + MJPEG encoder fps per mode -> results/camera/
tests/stress_test.sh                    # idle / CPU / Hailo / both with PMIC logging -> results/
tests/soak_test.sh [seconds]            # CPU + Hailo soak (default 300 s) -> results/soak/
python3 tests/bno085_test.py           # IMU functional test (~1 min, board still) -> results/bno085_test_<ts>.json
python3 tests/bno085_endurance.py      # IMU under load / faults / soak (~45 min) -> results/bno085_endurance_<ts>.json
python3 tests/imu_dashboard_soak.py [--minutes 15] [--vision]   # dashboard RUNNING: IMU as served
                                        # (/ws/imu gaps, sample age, 100 Hz dropouts, resets) -> results/
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
| `setup_spi0_mode3.sh` | `dtoverlay=mcp3202-mode3` (MCP3202 in SPI mode 3 like the BNO085, see below) + realtime priority for the `gpio` group |
| `diag_camera.sh`, `diag_iio.sh`, `diag_i2c.sh` | detection diagnostics for the camera, the ADC and the I2C buses |

`/boot/firmware/config.txt` needs: `dtparam=i2c_arm=on`, `dtoverlay=mcp3202,spi0-0-present`,
`dtoverlay=mcp3202-mode3`, `camera_auto_detect=0`, `dtoverlay=imx708,cam0`,
`dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4`.
Access to `/sys/class/pwm` comes from the `gpio` group (`/lib/udev/rules.d/99-com.rules`);
passwordless sudo is configured so `hardware/pwm.py` can `dtoverlay` at runtime if needed.

## IMU (BNO085)

```python
from hardware import BNO085
with BNO085() as imu:
    imu.enable_default()                  # rv, game_rv, accel, gyro, mag, linear_accel, gravity @ 100 Hz
    imu.quaternion, imu.euler, imu.acceleration, imu.gyro, imu.magnetic   # latest values
    r = imu.wait("rv")                    # next sample: Reading(sensor, values, accuracy, timestamp, seq)
    with imu.subscribe("accel", "gyro") as sub:
        for r in sub: ...                 # every sample, buffered (nothing dropped while you're busy)
    imu.enable("girv")                    # any sensor, rate_hz=None = its max; batch_ms=20 = more throughput
    imu.stats()                           # per sensor: delivered Hz, lost (seq gaps); link counters
    imu.tare(); imu.clear_tare(); imu.calibration(); imu.save_calibration(); imu.reset()
```
No vendor library (Adafruit's CircuitPython SPI driver is broken: its example cannot run, it
reads each packet twice and drops what the hub sends during writes). `hardware/bno085.py`
speaks SHTP/SH-2 itself. A worker process (`python -m hardware.bno085 --worker`, started and
restarted by the class, SCHED_FIFO) owns spidev0.1 + GPIO5/6 and hands samples over through a
shared-memory ring, so the caller's threads / GIL cannot make it miss the hub. Hub resets
(brown-out, glitch) are detected and the configuration is re-applied (~0.7 s gap).

Measured on this board (firmware 3.2.13):
* max rate per sensor alone: accel 500 Hz, gyro / rv / game_rv / linear_accel / gravity /
  gyro_uncal 400 Hz, arvr_* ~385 Hz, girv 1000 Hz, mag / mag_uncal / geo_rv 100 Hz.
* The hub CPU caps the total at ~800-1200 reports/s: with several sensors it decimates; raw
  sensors at max and girv at 1 kHz starve everything else. gyro/arvr fall back to 100 Hz if
  asked < 2500 us (the driver never asks). Batching >= 20 ms raises the total ~50 % and makes
  it load-proof, but overflows above ~1100 reports/s.
* `enable_default()` (~730 reports/s) is the recommended setup: in the endurance test it
  delivered every sample with CPU (stress-ng x4), Python GIL hogs, SPI ADC hammering and all
  of them together; only extreme memory-bandwidth stress made the hub deliver ~10 % fewer.
* SPI0 must run both devices in mode 3: with the MCP3202 in mode 0 every switch between the
  two added a clock edge and garbled the next IMU packet (`scripts/setup_spi0_mode3.sh`).
* Mounting on the car: the breakout stands on its edge, chip face (+Z) forward, -X up, so
  car (x fwd, y left, z up) = board (+Z, +Y, -X), plus the level reference (mount is ~2.4 deg
  off vertical, measured 2026-10-04). `CarFrame` applies both; raw `imu.euler` is in the board
  frame and near gimbal lock (pitch ~88 deg) on this mount, so use `CarFrame.attitude()`.
* |accel| at rest reads ~9.28 m/s^2 (5 % low): the accelerometer's dynamic calibration has not
  converged (it needs the board held still in ~6 orientations; awkward once mounted).
* Raw sensors (raw_accel/gyro/mag) must reach the hub after their calibrated base sensor:
  in the other order the hub stops asserting INT (enable() orders them, fixed 2026-10-04).
* WAKE (P0) is tied high on the breakout, so the hub only accepts writes during transfers it
  starts; the driver keeps a request in flight at all times (details in the module docstring).
  Wiring P0 to a free GPIO would remove that constraint.

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
