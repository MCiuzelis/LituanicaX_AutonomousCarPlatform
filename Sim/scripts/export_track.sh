#!/usr/bin/env bash
# Re-export assets/blender/Track.blend -> assets/export/{track_objects.usdc, walls.usdc, track.json}.
# Run after every change to the .blend.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
blender -b assets/blender/Track.blend --python assets/blender/export_track.py -- --out assets/export "$@"
