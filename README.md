# lerobot_policy_vlacpp

A [lerobot](https://github.com/huggingface/lerobot) policy plugin that runs a vla.cpp SmolVLA GGUF through a remote `vla-server` over ZMQ, so the standard `lerobot-rollout` loop can drive an SO-101 arm. The policy proxies `select_action` / `predict_action_chunk` to the server; everything else is stock lerobot.

Install it editable into your lerobot env:

    pip install -e integrations/lerobot/lerobot_policy_vlacpp

As a standalone repo, point `$VLA_CPP_SRC` at your vla.cpp tree so the bundled-`vla.proto` check can find `src/serving/vla.proto`. Once installed, lerobot auto-discovers it (distribution name starts with `lerobot_policy_`) and the `vlacpp` policy type just works.

## Running on the Ventuno Q (SO-101)

You need: the cross-compiled `vla-server` + Hexagon runtime libs deployed (see `ci/deploy_snapdragon.sh`), the SmolVLA GGUF/mmproj/config, the arm on a serial port, and both cameras. Then install the plugin into the board's lerobot env and build the pretrained dir. The CLI defaults already match the GGUF and our tuned control loop, so the only thing worth stating explicitly is the camera key order:

    vlacpp-build-pretrained ~/vla-rollout/vlacpp-ducks \
      --server-address tcp://127.0.0.1:5555 --real-action-dim 6 \
      --image-keys observation.images.wrist_cam observation.images.overhead_cam

Then edit and run `scripts/run_so101_vlacpp.sh`.

## The control loop: prefetch, latency budget, delay-crop

Under `--inference.type=sync` the policy runs its own background prefetch: when the live queue drops to `prefetch_threshold` rows, the next chunk is fetched on a background thread while the current one drains. Actions are consumed strictly in order, so a fetch must land before the queue runs dry or the loop blocks.

### The latency floor

The server is a black box: one round trip returns a whole 50-action chunk and nothing can be observed in between. So while the arm plays the queue we are blind for one round trip, and we see the result one round trip late. We call that round trip `L` and measure it on the target board with the plugin's own client (ventunoq, QCS8300 / HTP v75, smolvla-ducks-v4 f16, 2 x 480x640 views + 6-D state + the real instruction, n=60):

| `L` (client round trip) | ms |
|---|---|
| p50 | 495.6 |
| p90 | 528.5 |
| p99 | 935.0 |
| mean | 507.7 |

Note the 455 ms vla-bench figure is not `L`: it's server-side, taken with 16 language tokens instead of the real instruction's 7.

### The budget rule

    k         = L * fps                 # steps that go stale during one round trip
    lookahead = ceil(k * 1.4) + 1       # 1.4x safety factor, +1 for the boundary tick
    budget    = lookahead + n_action_steps <= chunk_size

At our shipped 10 fps (250 ms per step, the policy's training rate): p50 gives lookahead 8 (budget 33/50), but the measured p99 round trip (935 ms) is longer than 8 steps of runway, so a tail-latency fetch would drain the queue and stall. We ship **lookahead 12**: 1.20 s runway, 2.42x p50, 1.28x p99, budget 37/50, and measured 0 stalls / 0 backward steps over 450 ticks. At 30 fps the budget can't cover the tail at all (lookahead 41 -> 66 > 50), which is why 10 fps is the shipped profile.

The rule is validated at startup (`VlaCppConfig.validate_latency_budget`, also called from `VlaCppPolicy.__init__`) and raises, because running over budget doesn't warn at runtime: it blocks the loop for ~`L` and turns the delay-crop into a silent no-op. If it fires, lower `fps` / `n_action_steps` or raise `chunk_size`, then re-derive from a fresh `L`.

### The delay-crop

The chunk was computed from an observation taken one round trip ago, so its first rows target poses the arm has already passed. `_merge_chunk` drops exactly the rows that were still queued when that observation was captured:

    correct_offset = L * fps + pending_at_merge
                   == queue depth at observation-capture time

Measuring the depth at capture time is the whole point; at merge time it's 0-3 and a crop computed there would do nothing. On a controlled-trajectory seam leg (ideal tracker, chunk = `pose_at_capture + j/step`):

| crop | backward steps | worst delta |
|---|---|---|
| capture-time (shipped) | 0 | 0.0 |
| none | 21 | -8.0 (= the lookahead width) |
| `L*fps` only | 0 (masked by blend+clamp) | -3.0 raw residual (reappears as 7 backward steps with the clamp relaxed) |

`blend_horizon` is seam polish on top of that, never the fix and never a way to hide a stale chunk.

### Observability

- `VlaCppPolicy.stall_count` / `bootstrap_count` / `stats()`: every fetch that starts on an empty queue is counted and logged. The first tick of a rollout is tagged as the by-design bootstrap; anything after it means the arm froze for ~`L`. Totals are logged per rollout on `reset()`, and a non-zero post-bootstrap count is your signal that the margin is too thin.
- One log line per chunk: index, fetch latency + EMA, the computed crop, queue depth at capture vs merge:

      vlacpp chunk #7: fetch=498.3ms ema=503.1ms crop=12 (capture_depth=12 merge_depth=0) queued=25 queue_after=25

### RTC mode

`INFERENCE_TYPE=rtc` is accepted (`supports_rtc()` returns `True`), but it only gives interface-level compatibility: async prefetch + delay-cropping through lerobot's `ActionQueue`. No true RTC overlap guidance, because the server exposes no intermediate denoising state. For an arm on this board the sync path above is the tuned one:

    INFERENCE_TYPE=rtc RTC_QUEUE_THRESHOLD=12 RTC_EXECUTION_HORIZON=10 \
      ./integrations/lerobot/lerobot_policy_vlacpp/scripts/run_so101_vlacpp.sh

### Key ordering (verify once)

`lerobot-rollout` maps the returned action vector to SO-101 joints in this order:

    shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos

That must match the joint order the GGUF was trained on. If the arm moves the wrong joints, the dataset used a different order: remap with `--rename_map` or retrain/convert with matching order. Confirm on a slow, low-torque first run.

The image pass order is the other half of that contract and nothing validates it at runtime: pass 0 is `observation.images.wrist_cam`, pass 1 is `observation.images.overhead_cam` (lerobot preserves the dataset's `input_features` order and never sorts keys). We inferred it from the same two-camera SO-101 lineage rather than proving it for this GGUF, so verify once with a slow run.
