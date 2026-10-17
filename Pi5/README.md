# Pi5 — Raspberry Pi 5 RC car

Everything that runs on the car's Pi 5 (`ssh rp5`, `~/LituanicaX_InteligentRobotCompetition/Pi5`): hardware drivers, the web
dashboard / remote-control server, calibration tools and hardware tests.

```
Pi5/
  dashboard.py        web dashboard (http://rp5:8080): camera preview, detection, recording, map, telemetry
  drive_server.py     throttle / steering / motor encoder + the command links (ws :8082, tcp :8081); its own
                      process, started by dashboard.py (see "Dashboard")
  recorder.py         training-data recordings: video + per-frame IMU / odometry (see "Recordings")
  web/index.html      the dashboard page
  car_client.py       PC-side keyboard client for the tcp link (stdlib only, copy to the PC)
  hardware/           Python package: one class per subsystem (see "Hardware package")
  tools/              interactive tools (need a TTY: ssh -t rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/tools/...)
  tests/              hardware self-test, PWM sweep, CPU/Hailo stress + soak tests
  scripts/            boot / setup / diagnostics shell scripts
  config/             steering.json (tools/steering_cal.py), imu.json, camera.json (tools/camera_cal.py)
  vision/             Hailo-8 object detection (see "Vision"), track map + planner (see "Track map", "Autonomy")
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
  request: the drive server caps it at 40 % and stops the motor after 400 ms without a command
  (and when the dashboard that started it dies),
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
Open `http://rp5:8080`: camera preview (camera starts at 1536x864 @ 120 fps; resolution switching 640x360 .. 1536x864 at up to
120 fps, 1920x1080 / 2304x1296 at 56 fps), training-data recording (see "Recordings": playable /
downloadable / deletable from the page), telemetry (drive, IMU, battery, board power, CPU, WiFi, Hailo,
links) and keyboard driving (W/S/A/D/space).

**Three processes** (2026-10-11; before, everything shared one Python process and one GIL: with the
track map in it the detector fell to 20-40 fps and keyboard commands stalled for up to ~0.5 s):

| process | does | why separate |
|---|---|---|
| `dashboard.py` | camera, Hailo detection, preview, recording, telemetry, web page | |
| `drive_server.py` (started by the dashboard) | ESC + steering + motor encoder (500 Hz), the command links, dead-man; network thread SCHED_FIFO | commands must never wait for anything else |
| `vision/map_worker.py` (started by the dashboard) | track map + planner (20 Hz; plans go to the drive server, driven only when armed) | ~2-3 ms of Python per detection frame + ~6 ms per plan; skips frames > 80 ms behind |

**Driving**: the page's commands go from a Web Worker (unaffected by drawing on the page) straight to
`ws://<pi>:8082/ws/ctl`: `{"t": -1..1, "s": -1..1, "n": seq}` on every key change and 20x/s; the server
acks each one at once (`{"a": seq}`), and the Links panel shows the round trip (p50 / p95 / max) and,
from the server's side, the longest gap between commands and how many arrived > 150 ms late.
In autonomous mode the page sends `{"arm": true|false, "max": level, "n": seq}` instead (see "Autonomy").
`car_client.py --host <pi-ip>` on a PC uses the tcp link (:8081, unchanged): `{"t","s"}` lines in, health
`{"h":1,"v":batt,"temp":..,"rps":..,"us":..,"st":..,"act":..,"lk":..,"stop":..}` out at 5 Hz (also sent
on the WebSocket). The freshest command wins; an `"auto"` command (autonomy, unix socket
`/tmp/rc_drive.sock` from the dashboard) is ignored while a human link sent one in the last 0.5 s.
State for the dashboard / recorder: shared memory `rc_drive_state` (`drive_server.DriveState`).
If the drive server dies the dashboard restarts it; if the dashboard dies the drive server stops the car and exits.

**Preview**: `/ws/video?w=480|640|960` sends the detector's own frame (960x540 lores, scaled) as JPEG
together with the boxes found in that frame, so boxes and video are always in sync. The page acks every
frame and the server keeps at most 2 unacked, always the newest: on a weak link the frame rate drops,
the delay does not grow. 640x360 ~11-25 kB/frame at 30 fps; frame age on the car ~15 ms (detection
off) / ~40 ms (on: includes the Hailo). `/stream.mjpg?w=..` serves the same frames without boxes.

Measured 2026-10-11 (desk, -51 dBm): commands round trip 3-4 ms p50, max 6-21 ms, 0 lost, 0 late,
including 60 s with detection 120 fps + recording + 960-wide preview + CPU at 93 %.

HTTP API: `GET /api/telemetry`, `POST /api/mode {"w","h","fps"}` (409 while recording),
`POST /api/record {"on":bool}`, `GET /api/recordings`, `DELETE /api/recordings/<id>` (the whole take),
`GET /api/recordings/<id>/download` (take as .tar), `GET /recordings/<id>/<id>.mp4` (range requests),
`GET /stream.mjpg`, `GET /ws/video`, `GET /ws` (telemetry),
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
| `<id>_drive.csv` | encoder turns / rps, steering, ESC pulse, throttle command at 100 Hz; since 2026-10-11 afternoon also the active link, `cmd_age_ms` (age of the newest command: a WiFi stall shows as it growing; dead-man at 400), `wifi_dbm`, `auto_state` (0 off / 1 driving / 2 braking) and the autonomy target speed |
| `<id>_meta.json` | camera (id `cam0` = CSI CAM0, front IMX708; mount height / pitch / roll; intrinsics; full `config/camera.json`), this stream's ScalerCrop, K and distortion, IMU mount (`config/imu.json`), column descriptions, git commit, summary (frames, measured fps, dropped, frames without IMU, IMU rates) |

World frame of a take: X = car heading at its first frame, Y left, Z up (yaw is relative, game_rv has
no magnetometer). All timestamps are CLOCK_MONOTONIC ns; frame time = SensorTimestamp - exposure/2 +
readout/2 (rolling shutter, middle row).

Frame rate: the camera runs at the selected mode (e.g. 120 fps for detection); the recording takes
every Nth frame, at most 30 fps (`REC_FPS_SMALL/LARGE` in dashboard.py; the Pi 5 has no H.264
hardware: x264 at 1536x864@60 took ~1.9 cores and pulled detection down to ~70 fps; at 30 fps
detection keeps 120 fps while recording, CPU ~60 %, measured 2026-10-11). Takes before 2026-10-11 are
60 fps. Measured 2026-10-10 (then at 60 fps), 60 s takes: 1280x720@60,
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
  drawn by the browser on the preview frame they belong to (not burned into recordings). The Vision panel shows
  detection vs camera fps, dropped frames, delay from sensor timestamp to boxes (readout+ISP /
  Hailo / decode) and Hailo chip temperature. Use 1280x720 modes: that is the sensor mode
  (1536x864) the model was trained on.
* `vision/eval_hef.py`: run a HEF over saved JPEGs and dump predictions (accuracy checks).

Measured 2026-09-23 with `models/apple_yolov8s.hef` at 1280x720@120: 120 det/s, delay 24.5 ms p50,
CPU ~27 %. Only one process can own the Hailo: turn detection off before running other Hailo tools.

WiFi only (the Pi 5 antenna sits under the Hailo HAT, so Bluetooth was dropped). Ethernet is
for updates. NetworkManager keyfiles: wlan0 = 192.168.88.126, eth0 = 192.168.88.129.

## Track map (apples from many frames -> track boundaries)

`vision/track_map.py` (numpy; `TrackMapper`) keeps a **rolling** map of the apples around the car and
chains them into a left and a right track boundary. Nothing is remembered from earlier laps: an apple
not seen for `window_s` (4 s) or more than `window_behind_mm` (1 m) behind the camera is forgotten, and
the path shown is the last `trail_s` (10 s). So the car works on any track from the first metre, and
the map cannot fill up with drifted copies of earlier laps (lap-to-lap drift was 0.3-0.5 m). Live it
runs in its own process (`vision/map_worker.py`, `dashboard.MapProcess`: fed every game_rv sample, the
encoder at 100 Hz and every detection frame in 10 ms batches; ~25-45 apples in the window);
`tools/replay_map.py` runs the identical code over a recorded take.

* **Odometry**: rear-axle dead reckoning, encoder distance (`config/odometry.json` `mm_per_turn`;
  |turns|, the ESC cannot reverse) along the IMU heading, pose history of the last 10 s. The encoder
  is on the motor shaft (gearbox + open differential to the wheels): an unloaded inner wheel spinning
  in a corner and gear lash on throttle <-> brake both reach it, so the filter trusts it less than a
  wheel encoder (`odo_scale_sigma` 10 %, `odo_lat_sigma` 5 %) and leans on the apples.
* **Late detections**: boxes arrive ~30 ms after the exposure (`Vision` passes each frame's exposure
  middle `t_mid` + ScalerCrop). Each frame is placed with the pose **and camera attitude at t_mid**
  from the history, then associated (Mahalanobis gate, range-dependent noise: radial >> sideways),
  and the matches with confirmed apples correct the pose (small Kalman update of x, y, heading). The
  correction is applied to the odometry frame, so the path driven since the exposure moves with it.
* **Camera motion**: the IMU is on the camera bracket. Car roll / pitch are **high-passed** (2 s):
  suspension motion moves the camera, a sloped driveway does not (the apples lie on the same slope;
  `Params.attitude`: `highpass` | `absolute` vs the level reference | `static` = ignore the IMU).
  Height vs pitch / roll comes from `config/suspension.json` (`tools/suspension_cal.py`).
* **Apples**: box bottom-centre on the floor (for a ~80 mm apple this is within 1 cm of its contact
  point, independent of its size); boxes cut by the frame edge are not used. New apples start
  tentative, are confirmed after 5 hits, dropped when they are in view (0.45-2.2 m) for 40
  frames without a detection, and forgotten outside the rolling window (above).
* **Boundaries** (10 Hz): apples the car passes get their side; confirmed apples 0.6 m behind to 4 m
  ahead are chained (max gap 1.4 m, max bend 55 deg, never across the other chain) into left / right
  polylines + centre line + track width, car frame and world.
* Tunables: `Params` in vision/track_map.py; overrides in an optional `config/track_map.json`
  (e.g. `{"attitude": "static"}`). Pitch-bias estimation (`estimate_pitch`) exists but is off: on the
  2026-10-10 takes it ran to its limit, soaking up the unmeasured odometry scale.

Dashboard: **Track map** panel under the video (top-down: follow car heading-up / world-up / whole
map, wheel or +/- to zoom; apples blue = left, orange = right, grey = side unknown, hollow =
tentative; yellow + = this frame's detections; path; boundaries + centre line; violet = the
autopilot's path with the apples it took as its left / right gates, see "Autonomy")
and the mapped apples + boundaries projected back onto the camera image ("on camera"). Telemetry
`trackmap`; `/ws/map` (10 Hz), `POST /api/map {"reset": true}` (refused while recording).

**Recording**: every take gets the map (the map is reset when the take starts, so its world frame is
the take's start: camera floor point, X = car heading, Y left, mm):

| file | content |
|---|---|
| `<id>_det.csv` | every detection box of every camera frame (120 fps, not just the video's frames; join `_frames.csv` on `sensor_ts_ns`): raw box 0..1 + score, then status (used / edge / range), floor point (camera frame), its noise, world point, the apple it was matched to / started, camera pitch / roll / height used, pose, correction |
| `<id>_pose.csv` | 50 Hz fused pose, odometry-only pose, car roll / pitch, the high-passed part, camera pitch / roll / height, pose sigma |
| `<id>_landmarks.csv` | 10 Hz: every active apple (estimated position, sigma, state, side, hits, misses, position in the car frame) |
| `<id>_track.jsonl` | 10 Hz: left / right boundary + centre line in the car frame, apple ids per side, width |
| `<id>_map.json` | final map: the apples of the last window, the last trail_s of path, parameters, odometry / suspension config, statistics |

Review / re-tune offline (`results/trackmap/<id>/`; only the raw box columns are read, so any take
can be re-mapped with other parameters):
```
python3 tools/replay_map.py run recordings/<id> --png [--video] [--attitude static] [--mm-per-turn 30]
python3 tools/detect_take.py recordings/<id>     # older takes without _det.csv: run the HEF on the mp4 (dashboard stopped)
```
`--video` writes `<id>_review.mp4`: detections, mapped apples / boundaries projected back into the
image, inset map. If the projected dots slide off the apples as the car drives, the odometry scale or
the camera pitch is off.

**Calibration the map needs** (in this order):
1. `tools/odometry_cal.py`: push the car exactly 4 m straight, it counts the encoder turns (repeat a
   few times, q saves the median; works with the dashboard running). **Not measured yet**: `mm_per_turn` 23.25 is the Sim's
   guess. The 2026-10-10 takes cannot settle it (estimates 13-57 mm/turn depending on the method:
   the IMU was not calibrated then, so camera pitch and scale cannot be separated, and the loop
   has almost no straights); `replay_map.py fit-scale` exists for takes with a good IMU.
2. `tools/suspension_cal.py capture` / `fit --write`: one 8x6 board flat ~0.5 m in front, press the
   car down in every direction for ~60 s -> camera height vs IMU pitch / roll, and checks that the
   IMU really turns with the camera (gain ~1). Pose math verified on the 2026-10-10 calibration
   image (`check`): single boards give h 191-197 mm, pitch 11.3-11.8 deg vs 193.4 / 11.53 fitted.

Measured 2026-10-11 (live, car still, 1280x720@120 + detection): map loop 0.35 ms p50, CPU 36 %,
detection 120 fps / 30.6 ms. Replay of the 2026-10-10 takes (static attitude, unmeasured scale):
~130 detection frames/s on the Pi, median match residual ~20-27 mm; laps drift against each other.

## Autonomy

**On the page**: enable keyboard driving, set the throttle % (the same value is the autonomous throttle
ceiling, so autonomy never drives faster than holding W would), press **autonomous mode**, then
**hold SPACE**: the car follows the violet path. **Release SPACE -> it brakes to a stop.** W/A/S/D take
over at once while held (manual override); releasing them with SPACE still held hands back to autonomy.
A banner over the video says what is happening (hold SPACE / waiting for a path / AUTO DRIVING / AUTO
BRAKING / MANUAL OVERRIDE / no link), the Track map table has "autopilot plan" (steering, target
speed, path mode, confidence, which limit sets the speed) and "autonomy" (state, speed, ceiling,
plan age).

How it is wired: `vision/map_worker.py` runs map tick + planner at 20 Hz and sends `{"plan": {"s", "v",
"m"}}` straight to the drive server (unix socket; not through the dashboard). The page's control
worker sends `{"arm": true, "max": level}` at 20 Hz while SPACE is held. `drive_server.Control.auto_loop`
(50 Hz, realtime thread) drives only while some page's arm message is < 0.25 s old and the plan is
< 0.3 s old and wants > 0 m/s. Speed control (since 2026-10-11 evening; it was bang-bang ceiling /
coast / brake, 10-24 switches a second, jerky): PI on the throttle as a fraction of the page's level
(`KP_PER_MM`, `KI_PER_MM`), target speed rate-limited (+1.5 / -3 m/s^2), motor speed low-passed;
more than 250 mm/s too fast -> brake 25-80 %. Otherwise (released, WiFi gap, planner silent, path lost)
-> brake 60 % until the motor says < 120 mm/s, then neutral.
Arming is kept **per page**: a second dashboard tab cannot cancel it (2026-10-11: two tabs made
autonomy drop 10-20 times a second); the page shows a red banner when more than one control page is
connected. A manual command only overrides autonomy when a key is pressed (t or s not 0); in
autonomous mode, releasing W/A/S/D without SPACE stops the manual command at once (before, the last W
throttle stayed on until S: fixed 2026-10-11). Simulated in
`tests/autonomy_sim.py` (no hardware; 8 scenarios: drive, slow down, release, link loss, planner
quiet, path lost, manual override, plan without arming; + a second page, smooth throttle, W released
in autonomous mode, W released with SPACE held): stop in 0.5-0.8 s from ~1 m/s.
Every take now also has `<id>_plan.csv` (the live planner at 20 Hz: mode, why there is no path, steering,
target speed, apples per side, input age) and `_drive.csv` has the autonomy state and target; the
page's autopilot row shows why there is no path. `scripts/dashboard_restart.sh` keeps the previous log
as `logs/dashboard.log.1`. `tools/analysis/auto_throttle.py` summarises throttle / brake behaviour.

* **Path = gates**: Delaunay triangulation of the confirmed apples around the car; every edge about a
  track width long (0.45-1.5 m) is a gate (one apple left, one right). A beam search walks gate
  midpoints through neighbouring triangles from the gate the car is in, scoring constant width,
  smooth turns and length, never using an apple on both sides.
* **Side memory** (2026-10-11 pm): every plan votes its left / right apples back into the map
  (`Landmark.votes`, +-8; an apple the car passed counts 6 for its side). The gate search never uses an
  apple with >= 4 votes on the other side and pays a cost for weaker disagreement, so sides stop
  flipping between frames. The map draws apples in their remembered side colour, and the blue / orange
  lines are now the sides the planner actually uses (before, they were the map's own straight-on
  chaining, which in corners joined the inner row to the outer one: the "blue line to an orange apple").
* **Rows** (2026-10-11 late): apples closer than 0.6 x track width (<= 520 mm) are one row (union-find;
  along a row p50 ~350 mm, across the track >= ~550 mm). Two apples of one row never make a gate (two
  apples apart they look like a perfect 0.8 m gate: the path crossed the outer row in corners), and a
  row whose net side memory is >= 6 (one passed apple) lends its side to all its apples, so the outer
  row seen first in a corner keeps its side. A row with strong memory both ways (bridged by a stray
  apple) lends nothing. Only gate paths vote (a wrong one-row guess used to become the whole row's
  memory: the "everything orange" screenshots). The start gate no longer has to straddle the car's
  axis (deep in a hairpin the whole track is to one side).
* **One row in view** ("row" mode, deep in sharp corners): the visible row is chained from the apple
  nearest the car (within its row group; across a gap <= 0.7 m only nearly straight on - a 1.3 m jump
  reached the other row in hairpins) and followed half a track width inside it. Its side comes from the
  rows / votes, else from geometry at the row's apple nearest the car. A gate path from < 0.6 s ago beats
  a one-row guess.
* **Path memory**: the last gate path is kept in map coordinates; with nothing usable in view it is
  followed for up to 1.5 s (mode "memory", confidence 0.5) instead of stopping, because a stopped car
  sees the same nothing forever (the sim's hairpin stall, the "refuses to start" in corners).
* **Racing line**: instead of the gate midpoints, the smoothest line through the gates that keeps 220 mm
  from every gate apple (bounded least squares on the squared curvature, `racing_line()`), starting at
  the car's position and heading. On a 90 deg / 0.75 m corner the peak curvature drops 1.40 -> 1.11 1/m.
* "How far the path is known" is the farthest path point from the camera (it was the forward x, which
  in a sharp corner stays small: the path was rejected as too short and the car stopped).
* **Smoothing**: the gate midpoints (one per apple pair) form a kinked polyline; it is resampled every
  40 mm and smoothed with a Gaussian along its length (sigma 250 mm, `smooth_sigma_mm`), so the car
  follows and the page draws a curve. Steering metrics on the replays are unchanged at 250 mm; 400 mm
  starts cutting corners.
* **Steering** (`control` "ff", 2026-10-11 late): the path's curvature where the car will be (feed-forward,
  50 ms further on for the servo) + feedback on the sideways and heading error, critically damped over
  `ff_dist_mm` 600 mm of travel (k = k_path - e / D^2 - 2 psi / D, speed independent), times `steer_gain`
  (default **1.0** now; dashboard slider, config/planner.json), plus `steer_trim` -0.05 (at 0 the car
  curved right); curvature -> servo with the measured full lock (1.45 1/m = 0.69 m radius, ~67 ms servo +
  chassis lag). It is computed from the **predicted** pose: along the arc the car is driving (speed, yaw
  rate) for the input's age + `actuation_lag_s` 0.08 s - heading included (at 135 deg/s the car turns
  ~18 deg in 0.14 s). Pure pursuit ("pp", lookahead 0.65 m + 0.25 s) asked for 0.24 1/m half a second
  before a 0.75 m corner and only ramped up when its target reached the bend - with ~0.2 s of loop delay
  the car ran wide into the outer row (needing gain 1.5-1.6); "ff" asks 0.98-1.04 of the driven corner
  curvature at gain 1.0 on the manual takes.
* **Speed** (brakes whenever the path is not straight or not certain): lateral acceleration <= 0.15 g
  for the commanded curvature, and able to brake (1.5 m/s^2) to every path point's curve speed before
  reaching it (it was the 75th percentile curvature of the whole path, so a corner that was a small part
  of it barely slowed the car), never faster than the known path allows in 1 s, and top speed x
  confidence (one-sided path / memory 0.5, < 4 gates 0.7, halved per 0.5 steering jump between plans);
  at least 0.4 m/s while any path exists; no path -> 0 (the drive server brakes), steering held 0.3 s.

**Closed-loop simulator** `tools/track_sim.py [--track loop|hairpin|bean] [--laps 2] [--vmax 1500]
[--plot p.png] [--set planner_field=v] [--map map_field=v] [--seed n]`: the real TrackMapper + Planner
drive a simulated car (measured steering + lag, camera projection with pitch wobble and misses, 120 fps,
delivery delays, 20 Hz plans) on synthetic two-row tracks with 0.75 m corners. Replays of manual takes
cannot show what the planner's own steering leads to; this can. 4 seeds x 3 tracks x 2 laps: the old
control (pure pursuit on the midline, gain 1.55) touched 70 apples, the current defaults 17 (24/24 laps,
0 off track; at 2 m/s: 16).

**Map worker timing** (the "stops and refuses to start" of 2026-10-11): with 30-60 apples in view the
worker needed 4.8 ms per detection frame (x 120 fps) + 9 ms per plan; competing with camera / Hailo /
x264 it fell behind, its plans were up to 2 s late (input age p90 120-200 ms even while driving, interval
p90 110-200 ms) and plans older than 0.15 s are not sent -> "planner quiet -> braking", and no restart
while it stayed behind. Now: undistort once per frame and stop iterating when converged, vectorised pose
correction, chained boundaries at 5 Hz -> 2.5 ms per frame, ~6 ms per plan; the worker plans first and
maps frames only until the next plan is due, skipping frames > 80 ms behind (`skipped` in the map stats,
"skipped" rows in `_det.csv`). `tools/worker_rt_replay.py recordings/<id> --burn 4` feeds a take to the
real worker process at real speed with all 4 cores busy: old 7.1 / 15.5 plans/s with up to 3.6 s gaps,
new 19.9 / 20.1 plans/s, input age p50 35 ms, no gap > 0.35 s (144122 stationary with ~35 apples /
144507 laps). `_plan.csv` also logs yaw rate, commanded curvature, e / psi / feed-forward and skipped
frames; plan messages carry their input age ("a").
Since 2026-10-11 evening the worker plans at **50 Hz** (`PLAN_HZ`; side votes and the steering-jump
confidence are per second, not per plan) and the drive server applies a new plan's steering at once while
autonomy drives (it waited for its next 50 Hz step). `tools/analysis/map_bench.py recordings/<id>
--plan-hz 50`: map + plans ~0.33 core at 2.8 GHz (0.49 at 2.4); real worker with 4 cores busy: 50.1
plans/s, interval p99 32 ms / max 42 ms, input age p50 34 ms. The closed-loop sim shows no difference
between 20 / 33 / 50 Hz on the loop / hairpin tracks (the latency prediction already covers it); the
larger delays are detection (~30 ms), the IMU (below) and the servo (~67 ms, 50 Hz PWM frames).

**CPU**: overclocked to 2.8 GHz (`arm_freq=2800` in /boot/firmware/config.txt, backup
config.txt.bak_20261011_pre_oc; the firmware sets 0.94 V) + `performance` governor
(/etc/systemd/system/cpu-performance.service). 10 min stress-ng --verify on all cores: 0 errors, max
67.5 C, never throttled (active cooler). 3.0 GHz + 50 mV did not boot (tested with `reboot "0 tryboot"`,
which falls back to config.txt after a failed boot - use that for any further trial).

**IMU lag** (`tools/analysis/imu_lag.py recordings/<id>`: camera yaw rate from apple bearings at 120 fps
vs the IMU streams): the gyro reports describe the car ~17 ms before their timestamps (r 0.96-0.98, 3
takes), game_rv another ~4.5 ms. 200 Hz is how often a sample comes, not how old it is: the delay is in
the BNO085 (sensor low-pass + hub processing; the driver's timestamps follow the SH-2 base / delay
fields of the INT edge correctly), possibly a few ms of it in the camera timestamp convention (libcamera
SensorTimestamp = frame start; t_mid assumes start of readout). `tools/imu_latency_test.py` (dashboard
stopped, turn the car by hand) measures raw_gyro / gyro / game_rv / girv against each other: if girv or
raw_gyro is much earlier, the map should use it instead of game_rv.

**2026-10-16 (first full autonomous lap, 2304x1296 / 56 fps / 96 deg mode)**: (1) *Turned in early, hit apexes*
(143705: 13 of 40 inside apples touched): the path went straight from the camera to the first racing-line
point, often 20-30 deg to the side, so the heading-error term alone asked for full lock (`_plan.csv`: k_ff ~0,
psi -20..-33 deg) - lowering the gain only scaled it. Every path now starts along the car's heading
(`_from_car`): track_sim inside touches 19 -> 12. (2) *Coasting in corners*: not the throttle ceiling - the
planner's target fell to 0.45-0.75 m/s (one-row / memory confidence 0.5, steering-jump halving) and the speed
loop rose at only 1.5 m/s^2. Now confidence 0.8 / 0.8 / 0.85, jump penalty halved, corners 2.0 m/s^2, min
0.6 m/s; drive server accel 3 m/s^2, stronger PI, brakes from 150 mm/s over target
(`tools/analysis/speed_limits.py` shows which limit set the target). Sim mean speed 1.21 -> 1.30 m/s, same
touches. (3) *Started next to the left row, drove into it*: the first "gate" was crossed lengthwise / the gap
to the next straight's row looked like the track. Now a path must cross every gate at >= 35 deg, the car must
be on the track side of the first gate's two rows, and the one-row fallback prefers rows near the car.
track_sim starts 30-60 deg turned towards a row (`--start-offset/--start-yaw/--start-at`) now succeed (60 deg
on the hairpin track is physically too tight). **Autonomy tuning** panel on the page: planner
`TUNABLES` (gain, steer calmness, preview, apex margin, top speed, corner accel, bend braking, min speed,
one-row / memory speed) -> config/planner.json, drive server `DRIVE_TUNABLES` (accel, decel, PI, start
throttle, brake threshold / softness) -> config/drive.json, applied at once. Logs: `_plan.csv` + limits,
virtual apples, target; `_planpath.jsonl` (10 Hz path / sides / target); `_drive.csv` + throttle fraction,
filtered speed, ceiling, last plan; meta has the planner params + drive tuning.
`tools/analysis/apex_clearance.py` (body clearance of passed apples by inside / outside of turns),
`edge_bearing.py` (lens model at the image edges: the 96 deg mode is ~1 deg off at the outer 10 %).

**Rows as chains, one-side fallback, boundary memory** (2026-10-15 evening, `tools/side_eval.py`: every plan
scored against the side each apple was really passed on). Rows were distance groups (union-find, 0.48 m):
where a corner wraps round, the inner row's end came within 0.34-0.43 m of the outer row and both became one
"mixed" row without protection - the planner built a gate inside the outer row of a left corner and steered
right (150022). Now rows are chains: shortest links first, <= 2 per apple, the second continuing the first
(bend <= 60 deg), links up to 0.68 m, near-duplicates (< 0.18 m) one node; no gate inside a row even with
mixed memory; a mixed row takes the side holding 3x more memory. The one-row fallback tries every row (side
from memory, else geometry: the car is on the track side of a boundary), offsets it half a width towards
the track and takes the one reaching furthest (a firm side counts 1 m): before, it took the row nearest the
car, often the inner row's 0.4 m stub, and stopped. `BoundaryMemory`: apples a gate path put on a side (and
apples passed) are kept in map coordinates for 8 s; map apples within 0.25 m inherit the side, remembered
apples the map has lost are planned with again. Plans with a wrong-side apple within 1.5 m ahead, 8 takes
(4572 moving plans): 10.1 % -> 3.4 % (150022 23 -> 0.3 %, 145819 14 -> 1.8 %); one-row fallback in 150022
26 -> 13 % of the time; track_sim 4 seeds x 4 tracks: 26 -> 13 apples touched, all laps.
The 2026-10-15 15:32 take stalled for darkness (5 lux, gain 16, exposure capped at 7.9 ms by 120 fps):
half the apples were not detected.

**Map fixes from the same takes**: (1) the BNO085 game_rv lags ~17.5 ms (`imu_lag_s`): replays with the
IMU shifted 15-20 ms earlier create half as many duplicate apples in 110-145 deg/s corners (144507: 2951
-> 1458), the smeared double rows the planner then paired wrongly. (2) New apples only start within
2.5 m (`max_init_range_mm`, was 3 m): 1 deg of pitch error moves an apple ~0.8 m at 3 m but ~0.36 m at
2 m, and after hard acceleration far rows appeared as clouds 1-2 m too close (high-pass pitch
undershoot); ignoring the IMU attitude, `absolute` attitude or a longer high-pass were all worse on the
planner replays. `tools/analysis/plan_views.py` shows the video frame next to what the planner saw.

`tools/autonomy_eval.py recordings/<id> ... [--plot p.png] [--csv c.csv] [--no-gates]` replays takes
through the map + planner and compares, every 0.1 s, its curvature with what the car really drove
(yaw rate / speed after the lag), and checks the left / right apples it used against the path the
driver took over the next 1.5 m. 2026-10-11 takes (keyboard driving: bang-bang steering, ~1.2 m/s,
some apples hit), with gates: a path 98-100 % of the time, known p50 1.4-1.8 m ahead, turn direction
agrees with the driven one 79-92 %, right-side apples on the wrong side of the driven path 0.4-6 %,
left (inside of these mostly-left loops) 8-22 %; the video shows the car cutting / hitting inside
apples at several of those moments, so part of that is the driver's line, not the map.
Chained boundaries only (`--no-gates`): turn direction 73-93 %, left wrong side 13-40 %.
Needs scipy (python3-scipy, installed on the Pi).
First autonomous laps (2026-10-11 13:31-13:36, gain 1.6 set by hand): servo response is linear
(0.87-1.03 of 1.45 1/m x command above 0.2), so the understeer came from the planner: its pure-pursuit
curvature was ~0.6-0.9 of what the manually driven corners needed; gain 1.5-1.6 brings it to
0.87-1.01 (`autonomy_eval.py ... --set steer_gain=1.5` prints "corners asked/needed"). With side
memory + row fallback on the afternoon manual takes: path lost <= 2 %, turn direction agrees 83-96 %.

**Faster speed policy** (2026-10-16 evening, autonomous takes 1016_152200-152748: laps at 1.2-1.3 m/s, "slow,
coasts"). The car followed its target closely (throttle never at the 40 % ceiling, u ~0.6); the *target* was
low: top speed 1500 set it 41-75 % of the time, "known path / 1 s" another 25-43 % - the path's end stays put
while the car approaches it, so the target fell to 0.6-0.8 m/s before every corner (brake -0.4..-0.8) although
the curve allowed 1.2-3.5 m/s, then rose only as new path came into view. Corners: 1.1-1.6 m/s^2 lateral at
0.86-1.08 m/s; the manual laps 2026-10-11: 2.8-3.7 m/s^2 at 1.4-1.6 m/s. Now: the path beyond the known one may
be a full-lock corner, so the car must be able to brake to that corner's speed (x `known_end_frac`) by the end
of the known path; curvatures above full lock no longer lower the speed (the error feedback asked for more than
the car can steer); top speed 2500, corner grip 2800 (config/planner.json had 2000: set to 2800). Drive
server: `brake_max` tunable (was 0.8 fixed, now 1.0; brake 0.4-0.8 slowed only ~1.2 m/s^2 median), I term
not integrated while P alone saturates. track_sim (`--acc/--dec/--grip` added; 4 tracks x 3 seeds x 2 laps,
--dec 1800, --vmax 2500): old 1.30 m/s, 23.2 s per 2 laps, 4 apples touched; new 1.94 m/s, 15.5 s, 3 touched,
none off; brakes 1.2 m/s^2: same; grip 3000 instead of 4000: 7 touched, none off; "race" preset 2.15 m/s.
The tuning panel is grouped (speed / corners + steering / throttle + brakes), shows changed sliders with a
reset arrow, and has presets calm / default / race (speed + throttle settings only) and reset all.
Pi backup: results/backup_20261016_pre_aggressive/. At 2304x1296 @ 56 fps the detector ran at only 37-41 fps
(1536x864: ~100 fps).

**"Turns to the outside at some corner entries"** (2026-10-17, 19 takes 1017_145132-152035, 1.5-1.8 m/s). Two
causes. (1) *Weave*: steering +-0.6..1.0 and yaw rate +-70..110 deg/s at 2.2-2.5 Hz on the straights while
e / psi stay small; a corner entered in the outward phase looks like a turn to the outside. Steering 1.2-4 Hz RMS
0.07-0.15 at ~1.2 m/s (10-16) -> 0.2-0.37 at 1.5-1.7 m/s. The pose prediction (along the arc for input age +
actuation lag) is a yaw-rate damping term ~ v x latency / ff_dist on a yaw rate ~75 ms old; with ff_dist fixed
in mm it grows with speed and resonates. track_sim reproduces it once its delays match the car (`--odo-lag 0.05
--det-lag 0.06 --steer-delay 0.09`: input age ~45-60 ms as live, steering -> gyro 80-100 ms measured); more
prediction (actuation_lag_s 0.12-0.15) made it worse (0.43-0.50). Fix: errors corrected over
max(ff_dist_mm, v x `ff_time_s`) (0.5 s, slider "steer calmness s"): weave 0.27 -> 0.11 at 2.3 m/s, unchanged at
1.3 m/s. (2) *Path flips*: gate paths resting on one apple a side alternated between two gates (lookahead target
+-300-400 mm sideways every 2-4 plans, full lock each way): live logs 0.66 % of such plans flip vs 0.04 % with 2+
gates (35 % of gate plans are weak, mostly at corner entries). Fix: a weak path that disagrees with the fresh
remembered path by > 250 mm must hold for 0.15 s first, the remembered path is followed meanwhile (`weak_*`
params). `tools/analysis/flip_eval.py` (open-loop replay of the takes): gate flips 29 -> 12. track_sim, both
fixes, realistic delays, 3 seeds x 4 tracks x 2 laps @ 2.3 m/s: weave 0.288 -> 0.122, 1.71 -> 1.83 m/s, apples
touched 4 -> 4, none off. `tools/analysis/overlay_video.py --plan`: video with boxes + scores, the live planned
path (violet), the boundary apples used (blue / orange) and a HUD. Pi backup: results/backup_20261017_pre_weave/.

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
| `odometry_cal.py` | push the car exactly 4 m straight -> `config/odometry.json` mm per motor turn (track map) |
| `suspension_cal.py capture / fit <run> [--write]` | camera height / attitude vs IMU from a board on the floor -> `config/suspension.json` |
| `replay_map.py run / fit-scale <take>` | re-run the track map over a take (map png, review video) / estimate the odometry scale |
| `detect_take.py <take>` | run the detector over a take's video (takes recorded without detection) |
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
