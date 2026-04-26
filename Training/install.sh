#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
isaaclab_dir="${script_dir}/IsaacLab"
isaaclab_ref="v2.3.0"

git -C "${repo_root}" submodule sync --recursive
git -C "${repo_root}" submodule update --init --recursive Training/IsaacLab

# Isaac Sim 5.1 resolves cleanly with Isaac Lab v2.3.0; newer Isaac Lab refs
# pull in a Starlette pin that conflicts with Isaac Sim's FastAPI stack.
git -C "${isaaclab_dir}" fetch --tags origin release/2.3.0
git -C "${isaaclab_dir}" checkout --detach "${isaaclab_ref}"

cd "${script_dir}"
uv sync
