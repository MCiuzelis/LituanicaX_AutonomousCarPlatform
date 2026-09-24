#!/usr/bin/env bash
# Compile export/apple_yolov8s.onnx -> apple_yolov8s.hef for the car's Hailo-8.
#
# Runs inside the Hailo AI Software Suite docker image (hailo8_ai_sw_suite_2026-07:
# DFC 3.34 / Model Zoo 2.19 / HailoRT 4.24). The car runs HailoRT 4.23.0 - see README for
# whether the HEF loads there.
#
#   bash compile_hailo.sh                  # default compile
#   PERFORMANCE=1 bash compile_hailo.sh    # search for max FPS (can take hours)
#
# From Windows (Git Bash), with the image loaded via `docker load`:
#   MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$(cygpath -wa export):/work" -w /work --entrypoint bash \
#     hailo8_ai_sw_suite_2026-07:latest -lc \
#     "source /local/workspace/hailo_virtualenv/bin/activate && bash compile_hailo.sh"
set -euo pipefail
cd "$(dirname "$0")"

# Cut the graph at the six raw head convs (box regression cv2.*, class scores cv3.* per stride).
# Without this the parser swallows Ultralytics' DFL/decode tail and the on-chip NMS config
# (conv41/42, conv52/53, conv62/63 in yolov8s.alls) no longer matches.
hailomz compile yolov8s \
  --ckpt apple_yolov8s.onnx \
  --hw-arch hailo8 \
  --calib-path calib/ \
  --classes 1 \
  --end-node-names \
    /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv \
    /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv \
    /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv \
  ${PERFORMANCE:+--performance}

mv yolov8s.hef apple_yolov8s.hef
echo "done -> $(pwd)/apple_yolov8s.hef"
echo "deploy: scp export/apple_yolov8s.hef rp5:rc_car/models/ && ssh rp5 sync"
echo "test:   ssh rp5 python3 rc_car/vision/apple_detector.py --save 3"
