# RC car apple-track detector

Single-class (`apple`) YOLOv8s detector for the Pi 5 + Hailo-8 car: trained on this PC (RTX 3070),
compiled to a HEF in the Hailo AI Software Suite docker image. Trained at 960 and run at 960x544
(the 1536x864 camera frame scales to it with no letterbox padding).

## Layout
```
YoloTraining/
  README.md  pyproject.toml  uv.lock   uv project (all Python here runs via `uv run`)
  pipeline.sh              labeling + two-round training end to end (steps 2-4b)
  scripts/                 01..08 pipeline steps (table below)
  data/
    videos/                car recordings (1536x864 @ 60 fps): <id>.mp4 + _frames/_imu/_drive.csv + _meta.json
    frames/                frames while driving, deduplicated
    labels/ labels_r2/     Grounding DINO labels / + round-2 additions (YOLO txt)
    preview/ preview_r2/   label previews (boxes drawn; red = added in round 2)
    yolo/                  train/val dataset + data.yaml (split by video)
  weights/                 pretrained yolov8s.pt (training start point)
  runs/
    apple_yolov8s/         round-1 model
    apple_yolov8s_r2/      final model (weights/best.pt)
    old_20260923_r2/       previous model (1280x720 recordings, 640x640)
    pred_*.mp4             annotated held-out video
  export/                  apple_yolov8s.onnx, calib/, compile_hailo.sh -> apple_yolov8s.hef (old_20260923_*.hef = previous model)
  logs/                    autolabel / pipeline / Hailo compile logs
  third_party/hailo/       Hailo AI SW Suite 2026-07 docker bundle (zip) + its run script
```

## Pipeline
| step | command | output |
|---|---|---|
| 0. get videos | `scp 'rp5:LituanicaX_InteligentRobotCompetition/Pi5/recordings/*/*' data/videos/` | 9 sessions (2026-10-10), 1536x864 @ 60 fps + sensor CSVs |
| 1. frames | `uv run scripts/01_extract_frames.py` | `data/frames/`: every 3rd frame while driving (motor >= 5 rev/s from `_frames.csv`), blurry + near-duplicate frames dropped |
| 2. auto-label | `uv run scripts/02_autolabel.py --preview` | `data/labels/`, `data/preview/` |
| 3. dataset | `uv run scripts/03_build_dataset.py` | `data/yolo/` split by video (val = sessions `131416`, `131458`, `131840`) |
| 4. train | `uv run scripts/04_train.py` | `runs/apple_yolov8s/weights/best.pt` |
| 4b. round-2 labels | `uv run scripts/07_refine_labels.py --preview`, then 3 + 4 with `--labels data/labels_r2` / `--name apple_yolov8s_r2` | `data/labels_r2/`, `runs/apple_yolov8s_r2/` |
| 5. export | `uv run scripts/05_export_hailo.py` | `export/apple_yolov8s.onnx` + `export/calib/` |
| 6. compile | see "Compiling for Hailo" | `export/apple_yolov8s.hef` |
| 7. deploy | `scp export/apple_yolov8s.hef rp5:LituanicaX_InteligentRobotCompetition/Pi5/models/ && ssh rp5 sync` | model appears in the dashboard |
| check | `uv run scripts/06_predict_video.py data/videos/<clip>.mp4` | annotated mp4 in `runs/` |
| check | `scripts/08_compare_hef.py` (instructions in its docstring) | HEF-on-Hailo vs PyTorch accuracy |

## Labeling notes
- Grounding DINO base (`IDEA-Research/grounding-dino-base`), prompt
  `apple. basket. box. leaf. stone. shoe. car.`; only `apple` boxes are kept, the other phrases
  absorb look-alikes.
- Boxes whose bottom edge is above 25 % of the frame height are dropped: track apples lie on the
  ground and the camera is fixed low (horizon of the 1536x864 mode is at ~26 %).
- Boxes that contain the centres of >= 2 other boxes are dropped (DINO sometimes boxes a whole row).
- The motion gate in step 1 drops static clip starts/ends and the tail after the car hits apples
  (car stopped, apples still rolling).
- Each frame is labeled whole plus 3 zoomed tiles of the ground band, merged with NMS.
- Grounding DINO misses apples cut off by the frame edge; round 2 (`07_refine_labels.py`) adds
  the first model's confident detections that DINO missed, and the model is retrained.
- More clips (other lighting, track layouts, apple-free frames) will help more than any
  hyper-parameter.

## Results (2026-10-10)
Held-out sessions `131416` + `131458` + `131840` (441 frames, 7826 apples, never trained on), 960x544:

| | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| round 1 (`runs/apple_yolov8s`, DINO labels) | 0.994 | 0.918 | 0.983 | 0.986 |
| final (`runs/apple_yolov8s_r2`, round-2 labels) | 0.994 | 0.920 | 0.991 | 0.986 |

| HEF int8 on the car's Hailo-8 (AP50 / P / R at 0.35, `08_compare_hef.py`) | 0.9885 | | 0.980 | 0.991 |
| PyTorch fp32, same matcher | 0.9893 | | 0.982 | 0.991 |

Visual check: `runs/pred_20261010_131840_cam0_1536x864_60fps.mp4`.
On the car (`models/apple_yolov8s.hef`; the previous 640x640 model is kept as `apple_yolov8s_640_20260923.hef`):
`hailortcli run` 145.7 FPS (640x640 model: 185.6); dashboard at 1280x720@120 with 960x540 lores:
120 det/s, delay sensor->boxes 29.7 ms p50 (640x640 model: 24.5 ms). The detector reads the input size
from the HEF and the dashboard sizes the lores stream to fit it, so both models stay selectable.

## Compiling for Hailo
Docker Desktop + the suite image `hailo8_ai_sw_suite_2026-07:latest` (DFC 3.34 / Model Zoo 2.19 /
HailoRT 4.24, ~35 GB once loaded). Its HEFs **load and run on the car's HailoRT 4.23.0**.
Load once (if the image is gone from Docker):
```
unzip -p third_party/hailo/hailo8_ai_sw_suite_2026-07_docker.zip hailo8_ai_sw_suite_2026-07.tar.gz | docker load
```
Compile (Git Bash, from the repo root; ~20 min with the GPU):
```
MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$(cygpath -wa export):/work" -w /work --entrypoint bash \
  hailo8_ai_sw_suite_2026-07:latest -lc \
  "source /local/workspace/hailo_virtualenv/bin/activate && bash compile_hailo.sh" > logs/compile_hailo.log 2>&1
```
Gotchas: `MSYS_NO_PATHCONV=1` + an absolute mount path, or Git Bash mangles `/work`;
`--end-node-names` at the six raw head convs (in `compile_hailo.sh`), or the parser pulls in
Ultralytics' DFL/decode tail and NMS fails with "expected conv but found activation layer".
`PERFORMANCE=1` in the container env searches for max FPS (stock yolov8s gets 307 FPS that way);
it can take hours and is not needed for the 100 FPS target.

## Car code (`../Pi5/` = `rp5:~/LituanicaX_InteligentRobotCompetition/Pi5`)
The car code lives in `Pi5/` of this repo; the Pi runs it from its own clone of the repo. Edit on the PC,
then `git push rp5 main` - the Pi accepts pushes into its checkout (`receive.denyCurrentBranch updateInstead`),
then restart what changed, e.g. `ssh rp5 'cd LituanicaX_InteligentRobotCompetition/Pi5 && bash scripts/dashboard_restart.sh'`.
Changes made on the Pi: commit there, `git pull rp5 main`. Not tracked (runtime/data): `recordings/`,
`results/`, `logs/`, `models/` (HEFs come from `export/` here), `*.bak`.

- `vision/apple_detector.py`: camera lores (640 wide, ISP-scaled) -> letterbox 640x640 -> HailoRT
  async API (several frames in flight, results collected on a separate thread) -> on-chip NMS.
  Standalone benchmark: `ssh rp5 python3 ~/LituanicaX_InteligentRobotCompetition/Pi5/vision/apple_detector.py --save 3`
  (dashboard must not hold the camera).
- `vision/eval_hef.py`: run a HEF over saved frames (used by `scripts/08_compare_hef.py`).
- Dashboard http://rp5:8080: **▶ detection**, model dropdown (`~/LituanicaX_InteligentRobotCompetition/Pi5/models/*.hef` + stock COCO
  yolov8s with an "apple only" filter), confidence slider, box/label toggles; boxes drawn over the
  MJPEG (colour = score) via `/ws/det` at up to 30 Hz; Vision panel with detection vs camera fps,
  dropped frames, delay p50/p95/max split into readout+ISP / Hailo / decode + WiFi hop, box stats,
  Hailo chip temperature, detector input and sensor mode (warns on modes with a wider FoV than the
  training recordings - use 1280x720 modes).
