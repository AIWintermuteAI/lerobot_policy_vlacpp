#!/usr/bin/env bash
# Validate the vlacpp lerobot plugin against the LIBERO simulator (no real robot
# required yet). Split topology: LIBERO physics + lerobot-eval run HERE (host);
# vla-server runs on the Ventuno Q and is reached over the LAN. Requires
# `pip install -e ".[libero]"` in the lerobot checkout, and this plugin installed
# (`pip install -e integrations/lerobot/lerobot_policy_vlacpp`).
set -euo pipefail

# ---- CONFIG (edit these) ----
# BOARD_ADDR changes between sessions/networks (DHCP) -- confirm the board's
# current IP before running, e.g. `ping <board>` or check its DHCP lease.
BOARD_ADDR="${BOARD_ADDR:-tcp://192.168.0.171:5555}"   # vla-server on the Ventuno Q
PRETRAINED_DIR="${PRETRAINED_DIR:-./vlacpp-libero}"
TASK_SUITE="${TASK_SUITE:-libero_object}"               # matches the v2 model's benchmark suite
N_EPISODES="${N_EPISODES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/libero_vlacpp_smoke}"
# --------------------------------

echo "[1/2] building LIBERO-shaped pretrained dir at $PRETRAINED_DIR..."
vlacpp-build-pretrained "$PRETRAINED_DIR" \
  --server-address "$BOARD_ADDR" \
  --image-size 256 \
  --real-action-dim 7 \
  --image-keys observation.images.image observation.images.image2

echo "[2/2] running lerobot-eval against LIBERO ($TASK_SUITE)..."
lerobot-eval \
  --policy.path="$PRETRAINED_DIR" \
  --env.type=libero \
  --env.task="$TASK_SUITE" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=vlacpp_libero_smoke
