# Changelog

All notable changes to `lerobot_policy_vlacpp` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`vlacpp-build-pretrained` wrote a pretrained dir the server rejects.**
  `--image-size` defaulted to `512` while the model (`smolvla-ducks-v4`) and
  `VlaCppConfig` are natively `256`, so a bare build produced a
  mismatched contract that only showed up on a real run. The default is now
  `256`, and `--n-action-steps` went from `50` to `25` (50 violated the new
  lookahead budget). Every CLI default that mirrors a config field is now
  asserted equal to it, so the two cannot drift again.
- `VlaCppPolicy._merge_chunk`'s per-chunk log read `_last_fetch_ms` before the
  first fetch had set it (latent `AttributeError` on a hand-driven merge).
- The bundled-proto test no longer reports a pass while silently skipping when
  the plugin is checked out outside the vla.cpp tree: it now finds a sibling
  `vla.cpp` checkout (or `$VLA_CPP_SRC`) and pins the wire format by md5
  (`a4d21736ae5badad4f091d55b0bb950e`).

### Changed
- **Control loop retuned to the *measured* client round trip, not the 455 ms
  vla-bench figure.** Measured `L` (client round trip through
  `VlaCppSmolvlaClient.predict_chunk`, ventunoq / QCS8300 HTP v75,
  smolvla-ducks-v4, 2 x 480x640 views + 6-D state + the real 7-token
  instruction, n=60): **p50 495.6 / p90 528.5 / p99 935.0 ms** (the bench figure
  used 16 lang tokens and is server-side, not `L`).

  | knob | before | after | why |
  |---|---|---|---|
  | `prefetch_threshold` (lookahead) | 25 | **12** | `ceil(L_p50*fps*1.4)+1` = 8 at 10 fps, raised to 12 so the runway (1.20 s) also covers the measured p99 tail (935 ms > the 0.8 s the strict value gives). Budget 12 + 25 = 37 <= 50. |
  | `blend_horizon` | 16 | **5** | after the delay-crop the seam is already continuous (0 backward steps with the blend off too) - the blend is polish, not the fix, and must never be used to hide a stale chunk. |
  | `max_action_delta` | 1.0 | **10.0** | 1.0 clipped every step of a normal trajectory, rate-limiting the arm and *masking* staleness instead of fixing it (it hides the symptom while costing speed). Now a loose safety clamp only. |

  `n_action_steps` (25), `chunk_size` (50) and `fps` (10) are unchanged; at
  30 fps the budget cannot cover the p99 tail at all (lookahead 41 -> 66 > 50),
  which is why 10 fps - also the training rate - is the shipped profile.

### Added
- **Startup latency-budget assertion**: `prefetch_threshold + n_action_steps <=
  chunk_size` is validated in `VlaCppConfig.__post_init__` (and again in
  `VlaCppPolicy.__init__`). Running over budget does not warn at runtime - it
  drains the queue, blocks the control loop for ~`L`, and makes the delay-crop a
  silent no-op. The error names the derivation (`ceil(L*fps*1.4) + 1`).
- **Stall counter** (`VlaCppPolicy.stall_count` / `bootstrap_count` /
  `stats()`): every fetch that starts on an empty queue is counted and logged,
  with the by-design first-tick bootstrap distinguished from a real drain; the
  per-rollout totals are logged on `reset()`. This is the direct signal that the
  latency margin is too thin.
- **Per-chunk observability log**: fetched-chunk index, fetch latency and its
  EMA, the computed crop, and the queue depth at capture vs at merge - the three
  numbers that were invisible during the last debugging session (the failure
  mode was the queue draining into the blocking path with the crop silently
  becoming 0).
- `derive_lookahead()` plus the measured `L_P50_MS` / `L_P90_MS` / `L_P99_MS` /
  `CONTROL_FPS` constants, so the shipped default can be re-derived and
  regression-tested instead of living as a magic number; the config docstring
  carries the full arithmetic.
- Tests: latency-budget assertion (fires when misconfigured, passes at the
  shipped values), lookahead-vs-measured-`L` arithmetic, CLI/config default
  agreement, a bare build being loadable with the tuned loop, the stall counter
  (increments on drain, bootstrap-only when the lookahead covers the fetch),
  per-rollout reset logging, and the delay-crop using the capture-time depth
  (asserting it is *not* the merge-time depth and *not* `L*fps`).
- A document comment on `_merge_chunk` recording why the crop must stay
  capture-time: measured 0 backward steps for capture-time, 21 for no crop (raw
  yank -8.0 = the lookahead width) and a -3.0 under-crop residual for a bare
  `L*fps` crop that the blend+clamp merely masks.
