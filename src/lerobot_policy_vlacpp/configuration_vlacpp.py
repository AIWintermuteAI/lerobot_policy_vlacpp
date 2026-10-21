"""Config for the vla.cpp-backed lerobot policy.

Registered under the name "vlacpp". The class name (VlaCppConfig) drives
lerobot's naming-convention resolution: modeling_vlacpp.VlaCppPolicy and
processor_vlacpp.make_vlacpp_pre_post_processors.

Latency budget, the lookahead, and the blindness floor (READ BEFORE TUNING)
--------------------------------------------------------------------------
The server is a remote black box: a whole 50-action chunk is produced in one
round trip, so while the arm plays the queue the policy is *blind* for one round
trip and can only observe the result one round trip late. ``L`` is the measured
client round trip through
:class:`~lerobot_policy_vlacpp.client.VlaCppSmolvlaClient.predict_chunk` on the
target board (ventunoq, QCS8300 / HTP v75, smolvla-ducks-v4 f16, 2 x 480x640
views + 6-D state + the real 7-token instruction, n=60):

    L p50 = 495.6 ms    p90 = 528.5 ms    p99 = 935.0 ms    mean = 507.7 ms

The 455 ms vla-bench figure is NOT ``L``: it is server-side, and it was taken
with 16 lang tokens instead of the real instruction's 7.

``prefetch_threshold`` is the lookahead: the number of queued steps that must
still be pending when a fetch starts, i.e. the runway (in steps) that has to
cover ``L``. The formulas the shipped defaults come from:

    k         = L * fps                  # steps that go stale during a round trip
    lookahead = ceil(k * 1.4) + 1        # 1.4x safety factor, +1 for the boundary tick
    budget    = lookahead + n_action_steps <= chunk_size

Worked arithmetic at the shipped 10 fps (250 ms per step - the policy's training
rate, and the rate every record/rollout command on this board uses):

    k(10 fps, L_p50) = 0.4956 * 10                       = 4.96 steps
    lookahead        = ceil(4.96 * 1.4) + 1 = ceil(6.94)+1 = 7 + 1 = 8
    strict p50 budget: 8 + 25 = 33 <= 50    (runway 0.80 s = 1.61x p50)

    8 is the strict p50 value; it is NOT what ships. The measured p99 round trip
    (935 ms) is longer than the 8-step runway (0.80 s), so a p99 round trip would
    drain the queue and stall the loop:

      shipped 12 -> 1.20 s runway = 2.42x p50 and 1.28x p99, budget 37 <= 50
      p99-complete lookahead = ceil(9.35 * 1.4) + 1 = 15 -> budget 40 <= 50

    12 is the smallest shipped value that covers the measured p99 tail, and it
    was measured at 0 stalls / 0 backward steps over 450 ticks in the loop
    harness. At 30 fps the same arithmetic gives 22 from p50 (47/50) but 41 for
    the p99 tail (66/50) - the budget cannot cover the tail at all at 30 fps,
    which is why 10 fps is the shipped profile (stale fraction k/n_action_steps
    is 0.198 at 10 fps vs 0.595 at 30 fps).

Re-derive from a fresh ``L`` if the board, the model, the image count or the fps
change; ``validate_latency_budget`` fails loudly at construction instead of
stalling silently at runtime.

The delay-crop in ``modeling_vlacpp._merge_chunk`` is *not* a separate knob: it
drops the rows that went stale during the round trip, measured from the queue
depth at observation-capture time. See that method's docstring before touching
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig

# --- measured round-trip latency L (ms) and the shipped control rate ---
# Measured on ventunoq against the live vla-server through the plugin's own
# client, n=60 (see the module docstring). These constants exist so the shipped
# defaults can be re-derived and regression-tested instead of being magic
# numbers in the field declarations below.
L_P50_MS: float = 495.6
L_P90_MS: float = 528.5
L_P99_MS: float = 935.0
CONTROL_FPS: int = 10  # the policy's training rate; the shipped rollout profile
LOOKAHEAD_SAFETY: float = 1.4


def derive_lookahead(
    l_ms: float, fps: int = CONTROL_FPS, safety: float = LOOKAHEAD_SAFETY
) -> int:
    """``lookahead = ceil(L * fps * safety) + 1`` - see the module docstring.

    Returns the runway in steps that the safety factor demands for a round trip
    of ``l_ms`` at ``fps``. The shipped ``prefetch_threshold`` default is not
    this value for ``L_P50_MS`` (that is 8); it is raised to cover the measured
    p99 tail - the docstring has the arithmetic.
    """
    return math.ceil((l_ms / 1000.0) * fps * safety) + 1


@PreTrainedConfig.register_subclass("vlacpp")
@dataclass
class VlaCppConfig(PreTrainedConfig):
    # --- vla-server connection ---
    server_address: str = "tcp://localhost:5555"
    recv_timeout_ms: int = 30_000

    # --- SmolVLA preprocessing (must match the GGUF checkpoint) ---
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"
    image_size: int = 256
    max_state_dim: int = 32
    tokenizer_max_length: int = 48

    # --- action chunking (should match the checkpoint's chunk_size) ---
    chunk_size: int = 50
    n_action_steps: int = 25

    # --- control-loop tuning (derivation: module docstring) ---
    # lookahead: start the next fetch when this many actions are still queued.
    # 12 = ceil(L_p50 * fps * 1.4) + 1 (= 8) raised to cover the measured p99 tail
    # (935 ms > the 8-step 0.80 s runway). Budget: 12 + 25 = 37 <= chunk_size 50.
    prefetch_threshold: int = 12
    # Seam polish only. After the capture-time delay-crop the seam is already
    # continuous (measured 0 backward steps with the crop off too); the blend
    # must NOT be used to hide a stale chunk.
    blend_horizon: int = 5
    blend_power: float = 1.0
    # Loose safety clamp on the per-step joint delta, in the action's own units
    # (degrees for the SO-101 GGUF) - NOT a rate limiter. 1.0 clipped every step
    # of a normal trajectory, which masked staleness and cost speed without
    # fixing it; 10.0 is far above any physically plausible per-step target at
    # 10 fps, so it never shapes a healthy trajectory and only catches a gross
    # glitch (sign flip, wrong-chunk jump). None disables it entirely.
    max_action_delta: float | None = 10.0

    # select_action strategy used under `--inference.type=sync`:
    #   "prefetch" - our background-prefetch + blend + delta-clamp path (default)
    #   "basic"    - plain blocking fetch, serve a FIFO of n_action_steps rows,
    #                no prefetch/blend/clamp (reference baseline).
    # Ignored under `--inference.type=rtc`, which drives predict_action_chunk.
    select_action_mode: str = "prefetch"

    # --- action shape ---
    # vla-server pads action_chunk to the checkpoint's max_action_dim; only the
    # first real_action_dim columns carry values (see include/vla.h). This MUST
    # match the checkpoint (e.g. 6 for the SO-101 GGUF, 7 for LIBERO GGUFs).
    real_action_dim: int = 6

    # --- observation mapping ---
    # lerobot exposes camera frames at observation.images.<camera_name>; list the
    # keys in the order the model expects its image views.
    image_keys: list[str] = field(default_factory=lambda: ["observation.images.front"])

    # --- RTC (Real-Time Chunking) inference mode ---
    # Set by build_rollout_context from --inference.rtc.* when --inference.type=rtc.
    # We have no local denoising loop to guide, so RTC gives us async prefetch +
    # delay-cropping only (via the engine's ActionQueue), not true overlap guidance.
    rtc_config: RTCConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )
        self.validate_latency_budget()

    def validate_latency_budget(self) -> None:
        """Fail loudly when the loop cannot cover one server round trip.

        ``prefetch_threshold`` is the lookahead - the runway, in steps, that has
        to cover the measured latency ``L`` (``k = L * fps`` steps go stale
        during one round trip) - and the fetched chunk must still be able to
        supply ``n_action_steps`` rows behind it. Violating the budget does not
        warn at runtime: the queue drains, the control loop blocks for ~``L`` on
        the empty-queue path, and the delay-crop silently becomes a no-op (a
        crop of 0), so the failure is invisible from the outside. See the module
        docstring for the derivation and the measured numbers.

        Called from ``__post_init__`` and from ``VlaCppPolicy.__init__`` so it
        fires at startup, not after an arm has been driven into a wall.
        """
        if self.prefetch_threshold < 0:
            raise ValueError(
                f"prefetch_threshold (lookahead) must be >= 0, got {self.prefetch_threshold}"
            )
        budget = self.prefetch_threshold + self.n_action_steps
        if budget > self.chunk_size:
            raise ValueError(
                f"latency budget violated: prefetch_threshold (lookahead) "
                f"{self.prefetch_threshold} + n_action_steps {self.n_action_steps} = "
                f"{budget} > chunk_size {self.chunk_size}. lookahead = ceil(L*fps*1.4) + 1 "
                f"with L in seconds (measured L p50 = {L_P50_MS} ms at {CONTROL_FPS} fps here); "
                f"raise chunk_size, lower n_action_steps, or lower fps and re-derive. "
                f"Running over budget drains the queue and stalls the control loop."
            )

    # PreTrainedConfig abstract members. This policy performs no local training or
    # temporal stacking, so deltas are None and optimizer/scheduler presets are
    # only nominal (never used at inference time).
    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig()

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        # Inference-only remote backend: features are provided by the robot side.
        return None
