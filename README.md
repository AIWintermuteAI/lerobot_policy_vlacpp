# lerobot_policy_vlacpp

A [lerobot](https://github.com/huggingface/lerobot) policy plugin that runs a
vla.cpp SmolVLA GGUF as a **remote inference backend**. The policy proxies
`select_action` / `predict_action_chunk` to a local `vla-server` over ZMQ, so the
standard `lerobot-rollout` control loop can drive an SO-101 arm.

Install (editable, in your lerobot env):

    pip install -e integrations/lerobot/lerobot_policy_vlacpp

(The plugin is also usable as a standalone repository - point `$VLA_CPP_SRC` at
the vla.cpp tree so the bundled-`vla.proto` check can find `src/serving/vla.proto`.)

Once installed, lerobot auto-discovers it (distribution name starts with
`lerobot_policy_`) and the `vlacpp` policy type becomes available.

See `scripts/run_so101_vlacpp.sh` for the on-board launch sequence.

## Run on the Ventuno Q (SO-101)

Prerequisites on the board: the cross-compiled `vla-server` + Hexagon runtime libs
deployed (see `ci/deploy_snapdragon.sh`), the SmolVLA GGUF/mmproj/config, the SO-101
on a serial port, and the cameras. Install the plugin into the board's lerobot env:

    pip install -e integrations/lerobot/lerobot_policy_vlacpp

Build the pretrained dir (the CLI defaults now match the GGUF and the tuned
control loop, so the only thing that has to be stated explicitly is the camera
key order):

    vlacpp-build-pretrained ~/vla-rollout/vlacpp-ducks \
      --server-address tcp://127.0.0.1:5555 --real-action-dim 6 \
      --image-keys observation.images.wrist_cam observation.images.overhead_cam

Then edit and run:

    ./integrations/lerobot/lerobot_policy_vlacpp/scripts/run_so101_vlacpp.sh

## The control loop: prefetch, the latency budget, and the delay-crop

Under `--inference.type=sync` the policy runs its **own** background prefetch:
once the live queue drops to `prefetch_threshold` rows, the next chunk is fetched
on a background thread while the current queue drains. Actions are always
consumed strictly in order (no catch-up/skip logic), so a chunk fetch must finish
before the queue runs dry or the loop blocks.

### The blindness floor: `L`

`vla-server` is a remote black box: one round trip produces the whole 50-action
chunk, and nothing can be observed in between. So while the arm plays the queue,
the policy is **blind for one round trip** and sees the result one round trip
late. `L` is that round trip, measured through the plugin's own client on the
target board (ventunoq, QCS8300 / HTP v75, smolvla-ducks-v4 f16, 2 x 480x640
views + 6-D state + the real 7-token instruction, n=60):

| `L` (client round trip) | ms |
|---|---|
| **p50** | **495.6** |
| p90 | 528.5 |
| p99 | 935.0 |
| mean | 507.7 |

The 455 ms vla-bench figure is **not** `L`: it is server-side and was taken with
16 lang tokens instead of the real instruction's 7.

### The budget rule: `lookahead + n_action_steps <= chunk_size`

    k         = L * fps                 # steps that go stale during one round trip
    lookahead = ceil(k * 1.4) + 1       # 1.4x safety factor, +1 for the boundary tick
    budget    = lookahead + n_action_steps <= chunk_size

At the shipped 10 fps (250 ms per step, the policy's training rate):

    k(L_p50)  = 0.4956 * 10 = 4.96 steps
    lookahead = ceil(4.96 * 1.4) + 1 = 7 + 1 = 8     -> budget 33/50, runway 0.80 s
    p99 tail  = ceil(9.35 * 1.4) + 1 = 15            -> budget 40/50, runway 1.50 s

The shipped default is **12**, not the strict p50 value of 8: the measured p99
round trip (935 ms) is *longer* than 8 steps of runway, so a tail-latency fetch
would drain the queue and stall the loop. 12 gives a 1.20 s runway = 2.42x p50
and 1.28x p99 at 37/50 of the budget, and measured **0 stalls / 0 backward steps
over 450 ticks**. At 30 fps the budget cannot cover the tail at all (lookahead 41
-> 66 > 50), which is why 10 fps is the shipped profile.

The rule is validated at startup (`VlaCppConfig.validate_latency_budget`, also
called from `VlaCppPolicy.__init__`) and **raises**, because running over budget
does not warn at runtime - it blocks the loop for ~`L` and makes the delay-crop a
silent no-op. If it fires, lower `fps`/`n_action_steps` or raise `chunk_size` and
re-derive from a fresh `L`.

### The delay-crop (why chunk seams do not jump backwards)

The chunk returned by the server was computed from an observation taken one round
trip ago, so its first rows target poses the arm has already passed. `_merge_chunk`
drops exactly the rows that were still queued when that observation was captured:

    correct_offset = L * fps + pending_at_merge
                   == queue depth at observation-capture time

Capture-time measurement is the whole point. Measured on a controlled-trajectory
seam leg (ideal tracker, chunk = `pose_at_capture + j/step`):

| crop | backward steps | worst delta |
|---|---|---|
| **capture-time (shipped)** | **0** | **0.0** |
| none | 21 | -8.0 (= the lookahead width) |
| `L*fps` only | 0 (*masked by the blend+clamp*) | -3.0 raw residual (reappears as 7 backward steps with the clamp relaxed) |

and the queue depth at capture (8) vs at merge (0-3) shows why a merge-time crop
would be a no-op. `blend_horizon` is seam polish on top of that - never the fix,
and never a way to hide a stale chunk.

### Observability

- **Stall counter** - `VlaCppPolicy.stall_count` / `bootstrap_count` / `stats()`.
  Every fetch that starts on an empty queue is counted and logged; the first tick
  of a rollout is tagged as the by-design bootstrap, anything after it is a real
  stall (the arm froze for ~`L`). Totals are logged per rollout on `reset()`.
  A non-zero count after the bootstrap is the direct signal that the latency
  margin is too thin.
- **Per-chunk log line** - fetched-chunk index, fetch latency and its EMA, the
  computed crop, and the queue depth at capture vs at merge:

      vlacpp chunk #7: fetch=498.3ms ema=503.1ms crop=12 (capture_depth=12 merge_depth=0) queued=25 queue_after=25

### RTC inference mode

`INFERENCE_TYPE=rtc` is accepted (`supports_rtc()` returns `True`), but it only
gives *interface-level* compatibility - async prefetch + delay-cropping via
lerobot's `ActionQueue`. It does **not** give true RTC *overlap guidance*
(blending old/new chunks inside the denoising loop via `RTCProcessor.denoise_step`),
because `vla-server` exposes no intermediate denoising state. For an arm on this
board, the sync path above is the tuned one:

    INFERENCE_TYPE=rtc RTC_QUEUE_THRESHOLD=12 RTC_EXECUTION_HORIZON=10 \
      ./integrations/lerobot/lerobot_policy_vlacpp/scripts/run_so101_vlacpp.sh

### Action-key ordering (must verify once)

`lerobot-rollout` maps the returned action vector to the SO-101 joints in this order:

    shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos

This MUST match the joint order the GGUF was trained on. If the arm moves the wrong
joints, the training dataset used a different order — remap with `--rename_map` or
retrain/convert with matching order. Confirm on a slow, low-torque first run.

The **image** pass order is the other half of that contract and is *not* validated
by anything at runtime: pass 0 is `observation.images.wrist_cam`, pass 1 is
`observation.images.overhead_cam` (LeRobot never sorts image keys; it preserves
the dataset's `input_features` order). It is inferred from the same 2-camera
SO-101 lineage, not proven for this GGUF - verify once with a slow run.
