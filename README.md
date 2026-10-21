# lerobot_policy_vlacpp

A [lerobot](https://github.com/huggingface/lerobot) policy plugin that runs a
vla.cpp SmolVLA GGUF as a **remote inference backend**. The policy proxies
`select_action` / `predict_action_chunk` to a local `vla-server` over ZMQ, so the
standard `lerobot-rollout` control loop can drive an SO-101 arm.

Install (editable, in your lerobot env):

    pip install -e integrations/lerobot/lerobot_policy_vlacpp

Once installed, lerobot auto-discovers it (distribution name starts with
`lerobot_policy_`) and the `vlacpp` policy type becomes available.

See `scripts/run_so101_vlacpp.sh` for the on-board launch sequence.

## Run on the Ventuno Q (SO-101)

Prerequisites on the board: the cross-compiled `vla-server` + Hexagon runtime libs
deployed (see `ci/deploy_snapdragon.sh`), the SmolVLA GGUF/mmproj/config, the SO-101
on a serial port, and one camera. Install the plugin into the board's lerobot env:

    pip install -e integrations/lerobot/lerobot_policy_vlacpp

Then edit and run:

    ./integrations/lerobot/lerobot_policy_vlacpp/scripts/run_so101_vlacpp.sh

### Real-time control loop: sync vs RTC inference

By default the policy runs in lerobot's `sync` inference mode: it fetches a full
action chunk only once the internal queue empties, which **blocks the control
loop** for the duration of one `vla-server` round trip (roughly 1.4-1.9s in our
CPU benchmarks) every `n_action_steps` ticks - a visible stutter on a real arm.

Set `INFERENCE_TYPE=rtc` to use lerobot's Real-Time Chunking engine instead,
which prefetches the next chunk asynchronously once the queue drops below
`RTC_QUEUE_THRESHOLD`, so a fresh chunk is usually ready before the old one
runs out:

    INFERENCE_TYPE=rtc RTC_QUEUE_THRESHOLD=15 RTC_EXECUTION_HORIZON=10 \
      ./integrations/lerobot/lerobot_policy_vlacpp/scripts/run_so101_vlacpp.sh

**Caveat:** `VlaCppPolicy.supports_rtc()` returns `True` so lerobot's RTC engine
accepts it, but this only gives *interface-level* compatibility - async prefetch
+ delay-cropping via lerobot's `ActionQueue`. It does **not** give true RTC
*overlap guidance* (blending old/new chunks inside the denoising loop via
`RTCProcessor.denoise_step`), because `vla-server` is a remote black box with no
exposed intermediate denoising state. If `vla-server`'s own inference latency
regularly exceeds `RTC_QUEUE_THRESHOLD * (1/FPS)` seconds, you'll still see
occasional stalls - raise the threshold or lower `FPS`/`N_ACTION_STEPS`.

### Action-key ordering (must verify once)

`lerobot-rollout` maps the returned action vector to the SO-101 joints in this order:

    shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos

This MUST match the joint order the GGUF was trained on. If the arm moves the wrong
joints, the training dataset used a different order — remap with `--rename_map` or
retrain/convert with matching order. Confirm on a slow, low-torque first run.
