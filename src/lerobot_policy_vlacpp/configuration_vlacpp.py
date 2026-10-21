"""Config for the vla.cpp-backed lerobot policy.

Registered under the name "vlacpp". The class name (VlaCppConfig) drives
lerobot's naming-convention resolution: modeling_vlacpp.VlaCppPolicy and
processor_vlacpp.make_vlacpp_pre_post_processors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig


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

    # configuration_vlacpp.py
    prefetch_threshold: int = 25   # start next fetch when queue has this many actions left
    blend_horizon: int = 16
    blend_power: float = 1.0
    max_action_delta: float | None = 1

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
