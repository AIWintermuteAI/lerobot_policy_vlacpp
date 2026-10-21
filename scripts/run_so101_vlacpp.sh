#!/usr/bin/env bash
# Launch the SmolVLA-on-SO-101 stack on the Ventuno Q:
#   1) start vla-server on the GGUF (localhost ZMQ),
#   2) build the vlacpp pretrained dir,
#   3) run lerobot-rollout driving the SO-101 with the vlacpp policy.
#
# Edit the CONFIG block for your paths / camera / serial port before running.
set -euo pipefail

# ---- CONFIG (edit these) ----
VLA_SERVER_BIN="${VLA_SERVER_BIN:-./bin/vla-server}"
GGUF="${GGUF:-./models/smolvla-so101.gguf}"
MMPROJ="${MMPROJ:-./models/smolvla-so101.mmproj.gguf}"
CONFIG_JSON="${CONFIG_JSON:-./models/smolvla-so101.config.json}"
# vla-server binds this address; the client connects to CLIENT_ADDR (localhost).
BIND_ADDR="${BIND_ADDR:-tcp://*:5555}"
CLIENT_ADDR="${CLIENT_ADDR:-tcp://localhost:5555}"
# mmproj is optional (required for SmolVLA); leave empty to omit the positional.
# --------------------------------

PRETRAINED_DIR="${PRETRAINED_DIR:-./vlacpp-so101}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
CAMERA_NAME="${CAMERA_NAME:-front}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
TASK="${TASK:-pick up the cube}"
FPS="${FPS:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-25}"
# Inference backend: "sync" (default, blocks/stalls once the queue empties) or
# "rtc" (async prefetch once the queue drops below RTC_QUEUE_THRESHOLD, so the
# control loop doesn't freeze - see VlaCppPolicy.supports_rtc for what RTC does
# and doesn't give us: async prefetch + delay-cropping, not true chunk-overlap
# guidance, since vla-server exposes no denoising-time hook).
INFERENCE_TYPE="${INFERENCE_TYPE:-sync}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-15}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"

echo "[1/3] starting vla-server..."
# vla-server CLI: [--bind ADDR] [--config PATH] [<mmproj.gguf>] <ckpt>  (mmproj + ckpt are positional).
"$VLA_SERVER_BIN" --bind "$BIND_ADDR" --config "$CONFIG_JSON" ${MMPROJ:+"$MMPROJ"} "$GGUF" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 3

echo "[2/3] building pretrained dir at $PRETRAINED_DIR..."
vlacpp-build-pretrained "$PRETRAINED_DIR" \
  --server-address "$CLIENT_ADDR" \
  --real-action-dim 6 \
  --n-action-steps "$N_ACTION_STEPS" \
  --image-keys "observation.images.${CAMERA_NAME}"

INFERENCE_FLAGS=(--inference.type="$INFERENCE_TYPE")
if [[ "$INFERENCE_TYPE" == "rtc" ]]; then
  INFERENCE_FLAGS+=(
    --inference.queue_threshold="$RTC_QUEUE_THRESHOLD"
    --inference.rtc.execution_horizon="$RTC_EXECUTION_HORIZON"
  )
fi

echo "[3/3] launching lerobot-rollout..."
lerobot-rollout \
  --policy.path="$PRETRAINED_DIR" \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.cameras="{ ${CAMERA_NAME}: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: ${FPS}}}" \
  --task="$TASK" \
  --fps="$FPS" \
  --strategy.type=base \
  "${INFERENCE_FLAGS[@]}" \
  --duration=0
