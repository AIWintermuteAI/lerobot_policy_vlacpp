import pytest

import lerobot_policy_vlacpp  # noqa: F401  (import triggers registration)
from lerobot.configs import PreTrainedConfig

from lerobot_policy_vlacpp.configuration_vlacpp import (
    CONTROL_FPS,
    L_P50_MS,
    L_P90_MS,
    L_P99_MS,
    VlaCppConfig,
    derive_lookahead,
)


def test_vlacpp_registered():
    assert "vlacpp" in PreTrainedConfig.get_known_choices()


def test_config_defaults_and_type():
    # These are the *config* defaults: the model contract (image_size 256,
    # chunk_size 50) plus the latency-tuned control loop. build_pretrained.py's
    # CLI defaults used to disagree with them (512 / 50) and the CLI was the one
    # that was wrong - see test_build_pretrained.py::test_cli_defaults_match_config_defaults.
    cfg = VlaCppConfig(device="cpu")
    assert cfg.type == "vlacpp"
    assert cfg.server_address == "tcp://localhost:5555"
    assert cfg.image_size == 256
    assert cfg.max_state_dim == 32
    assert cfg.chunk_size == 50
    assert cfg.n_action_steps == 25
    # latency-tuned loop (derivation in the module docstring)
    assert cfg.prefetch_threshold == 12
    assert cfg.blend_horizon == 5
    assert cfg.blend_power == 1.0
    assert cfg.max_action_delta == 10.0
    assert cfg.select_action_mode == "prefetch"
    assert cfg.real_action_dim == 6
    assert cfg.image_keys == ["observation.images.front"]
    # abstract members are satisfied
    assert cfg.observation_delta_indices is None
    assert cfg.action_delta_indices is None
    assert cfg.reward_delta_indices is None
    cfg.validate_features()  # no-op, must not raise


def test_lookahead_follows_the_measured_latency_arithmetic():
    """The shipped lookahead must be the one the measured L derives.

    L is the real client round trip through VlaCppSmolvlaClient.predict_chunk on
    ventunoq (n=60), not the 455 ms bench figure - see the module docstring.
    """
    assert (L_P50_MS, L_P90_MS, L_P99_MS) == (495.6, 528.5, 935.0)
    assert CONTROL_FPS == 10

    # lookahead = ceil(L * fps * 1.4) + 1
    assert derive_lookahead(L_P50_MS) == 8    # ceil(4.956 * 1.4) + 1 = 7 + 1
    assert derive_lookahead(L_P90_MS) == 9    # ceil(5.285 * 1.4) + 1 = 8 + 1
    assert derive_lookahead(L_P99_MS) == 15   # ceil(9.350 * 1.4) + 1 = 14 + 1

    cfg = VlaCppConfig(device="cpu")
    # Shipped 12: the strict p50 value (8) raised so the runway also covers the
    # measured p99 tail, because 8 steps = 0.80 s < the 935 ms p99 round trip.
    assert cfg.prefetch_threshold == 12
    assert cfg.prefetch_threshold >= derive_lookahead(L_P50_MS)
    runway_s = cfg.prefetch_threshold / CONTROL_FPS
    assert runway_s >= L_P99_MS / 1000.0  # 1.20 s >= 0.935 s
    assert runway_s < derive_lookahead(L_P99_MS) / CONTROL_FPS  # and not the 1.5 s tail-complete value
    # the strict p50 value would have been too short for the tail
    assert derive_lookahead(L_P50_MS) / CONTROL_FPS < L_P99_MS / 1000.0
    # budget rule
    assert cfg.prefetch_threshold + cfg.n_action_steps <= cfg.chunk_size


def test_latency_budget_assertion_fires_when_misconfigured():
    """lookahead + n_action_steps <= chunk_size must fail loudly at startup."""
    # shipped tuning is inside budget - must not raise
    VlaCppConfig(device="cpu")
    # exactly at the limit is fine
    VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=25)
    # one step over is not
    with pytest.raises(ValueError, match="latency budget violated"):
        VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=26)
    # the old CLI default (n_action_steps=50) with the shipped lookahead
    with pytest.raises(ValueError, match="latency budget violated"):
        VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=50, prefetch_threshold=12)
    # a negative lookahead is nonsense
    with pytest.raises(ValueError, match="must be >= 0"):
        VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=-1)


def test_budget_error_message_names_the_derivation():
    with pytest.raises(ValueError) as exc:
        VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=40)
    msg = str(exc.value)
    assert "lookahead" in msg
    assert "ceil(L*fps*1.4) + 1" in msg
    assert str(L_P50_MS) in msg


def test_policy_construction_validates_the_budget_too():
    """Startup assertion: a config mutated after construction is still caught."""
    from lerobot_policy_vlacpp.modeling_vlacpp import VlaCppPolicy

    cfg = VlaCppConfig(device="cpu")
    cfg.prefetch_threshold = 40  # someone retuned it without re-deriving the budget
    with pytest.raises(ValueError, match="latency budget violated"):
        VlaCppPolicy(cfg)
