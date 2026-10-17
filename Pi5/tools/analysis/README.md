# tools/analysis: one-off analyses of recorded takes / the running car (2026-10-11)

| script | runs on | what |
|---|---|---|
| `take_timing.py` | PC / Pi | detection rate, drive sampler, map size over each take (5 s bins) |
| `frames_grid.py` | PC | video frames with the live detections drawn in |
| `overlay_video.py` | PC (ffmpeg) | shareable videos of a take: `<id>_clean.mp4` (remuxed) + `<id>_boxes.mp4` (live detections drawn in, `--hud` adds speed / steering); `--plan` -> `<id>_plan.mp4` with scores, the live planned path, boundary apples and a HUD |
| `flip_eval.py` | PC (`../pc_run.py`) | planner path flips (target jumps sideways + steering jumps) on takes replayed open loop, by plan mode; `--set` to compare settings |
| `plot_maps.py` | PC | top-down plot of the maps recorded live |
| `steer_model.py` | PC / Pi | steering command -> curvature + lag |
| `car_frame_views.py` | PC (`../pc_run.py`) | car-frame map / planner / driven path at chosen moments |
| `imu_lag.py` | PC (`../pc_run.py`) | delay of the gyro / game_rv streams vs the camera (apple bearings at 120 fps) and vs each other |
| `map_bench.py` | **Pi** | CPU cost of the map worker (frames + plans) per second of a take, `--plan-hz` |
| `plan_views.py` | PC (`../pc_run.py`) | video frame + large car-frame view of the planner (side memory, sides used, path, target, driven path) at chosen moments |
| `profile_map.py` | **Pi** | CPU cost of the track map per frame |
| `cpu_by_proc.py` | **Pi** | CPU per process / thread |
| `link_test.py` | PC | command round trips + preview video against the running dashboard (neutral commands) |

The main replay tools are one level up: `replay_map.py` (map), `autonomy_eval.py` (map + planner vs. the drive).
On the PC: `cd Pi5 && uv run --project ../YoloTraining --with scipy python tools/pc_run.py tools/autonomy_eval.py recordings/<id> ...`
(`Pi5/recordings/<id>/` on the PC holds copies of the takes' CSV / JSON files; the videos are in
`YoloTraining/data/videos_<date>/`, flat).
