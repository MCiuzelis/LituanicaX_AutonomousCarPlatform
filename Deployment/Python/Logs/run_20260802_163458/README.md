# LituanicaX run log (schema 1)

Written by `lituanicax.modules.recorder`. Every file is append-only and every
row carries the same two clocks:

* `t` — `CLOCK_MONOTONIC` seconds, the join key across all files. The ToF
  frame timestamps come from the camera processes on this same system-wide
  clock, so they are directly comparable.
* `t_wall` — unix time. `t_wall = t + meta.json["clock_offset"]`.

## Files

| File | One row per | Key fields |
|------|-------------|------------|
| `meta.json` | run | clocks, enabled modules, full `config.py` snapshot, obs layout |
| `policy.jsonl` | policy tick (30 Hz) | `step`, `t`, `left_image_index`, `obs_index`, `action_*`, `drive_*` |
| `telemetry.jsonl` | telemetry tick (30 Hz) | `t`, battery/current/CPU/encoder, `drive_*`, `dash_*` |
| `frames.jsonl` | stored ToF frame | `index`, `side`, `frame_id`, `t` (capture time) |
| `obs.bin` | policy tick | float32 `[n, 4331]`, row `n` = `obs_index` |
| `tof_<side>_depth.bin` | stored frame | float32 `[n, 180, 240]`, mm |
| `tof_<side>_conf.bin` | stored frame | uint8 `[n, 180, 240]` |
| `console.log` | log line | `[LEVEL] [t_wall] [t] [module]: message` |
| `summary.json` | run | counts, bytes, dropped records |

Cameras: left, right.

## Which image produced which action

`policy.jsonl` names the record it used — no timestamp matching required:

```python
import json, numpy as np

meta  = json.load(open("meta.json"))
steps = [json.loads(line) for line in open("policy.jsonl")]
depth = np.memmap("tof_left_depth.bin", dtype=np.float32,
                  mode="r").reshape(-1, 180, 240)
obs   = np.memmap("obs.bin", dtype=np.float32, mode="r").reshape(-1, 4331)

s = steps[100]
frame  = depth[s["left_image_index"]]        # exactly what the camera saw
vector = obs[s["obs_index"]]                   # exactly what ONNX was given
print(s["t"], s["action_throttle_raw"], s["action_steer"], s["drive_duty"])
```

`*_image_index` is `null` when that camera had no fresh frame (the policy then
commands zero) or when the size budget was reached. Consecutive steps often
share an index: the policy reads the newest frame, and the camera does not
always have a new one.

## Lining a step up with telemetry

`telemetry.jsonl` is a separate 30 Hz stream, so match on `t`:

```python
import bisect
tel = [json.loads(line) for line in open("telemetry.jsonl")]
ts  = [row["t"] for row in tel]
near = tel[min(bisect.bisect_left(ts, s["t"]), len(tel) - 1)]
```

`dash_*` fields are the exact websocket payload the browser received (EMA
filtered); the unprefixed fields next to them are the unfiltered module state.

## Reproducing an observation

`process_for_policy(depth, conf)` in `modules/tof_camera.py` turns a raw frame
into the `depth_left` / `depth_right` slices of `obs.bin`, using the crop rows
and constants recorded in `meta.json`.
