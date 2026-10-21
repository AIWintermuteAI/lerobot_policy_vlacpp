import lerobot_policy_vlacpp  # noqa: F401  (import triggers registration)
from lerobot.configs import PreTrainedConfig


def test_vlacpp_registered():
    assert "vlacpp" in PreTrainedConfig.get_known_choices()


def test_config_defaults_and_type():
    from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig

    cfg = VlaCppConfig(device="cpu")
    assert cfg.type == "vlacpp"
    assert cfg.server_address == "tcp://localhost:5555"
    assert cfg.image_size == 512
    assert cfg.max_state_dim == 32
    assert cfg.chunk_size == 50
    assert cfg.n_action_steps == 50
    assert cfg.real_action_dim == 6
    assert cfg.image_keys == ["observation.images.front"]
    # abstract members are satisfied
    assert cfg.observation_delta_indices is None
    assert cfg.action_delta_indices is None
    assert cfg.reward_delta_indices is None
    cfg.validate_features()  # no-op, must not raise
