from pathlib import Path

import pytest
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.factory import make_pre_post_processors

from lerobot_policy_vlacpp.build_pretrained import build_parser, build_pretrained_dir
from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig


def test_build_pretrained_dir_roundtrip(tmp_path):
    out = tmp_path / "vlacpp-so101"
    build_pretrained_dir(
        str(out),
        server_address="tcp://localhost:5555",
        image_keys=["observation.images.front"],
        chunk_size=50,
        n_action_steps=25,
    )
    # required files exist
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert (out / "policy_preprocessor.json").exists()
    assert (out / "policy_postprocessor.json").exists()

    # config loads and resolves to our policy type
    cfg = PreTrainedConfig.from_pretrained(str(out))
    assert cfg.type == "vlacpp"

    # policy loads weightlessly
    cfg.pretrained_path = out
    policy = get_policy_class("vlacpp").from_pretrained(out, config=cfg)
    assert policy is not None

    # processors load from the saved JSONs
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(out))
    assert pre is not None and post is not None


def test_cli_defaults_match_config_defaults():
    """Regression: the CLI and the config used to disagree.

    `build_pretrained.py` defaulted to image_size=512 (the model is natively
    256) and n_action_steps=50 (which violates the lookahead budget), while
    VlaCppConfig defaults to 256 / 25 - so a bare `vlacpp-build-pretrained`
    silently wrote a pretrained dir the server (and now the config validator)
    rejects. Only a real run showed it. Every CLI default that mirrors a config
    field must equal that field's default.
    """
    cfg = VlaCppConfig(device="cpu")
    args = build_parser().parse_args(["out"])

    assert args.image_size == cfg.image_size == 256
    assert args.n_action_steps == cfg.n_action_steps == 25
    assert args.chunk_size == cfg.chunk_size
    assert args.max_state_dim == cfg.max_state_dim
    assert args.tokenizer_max_length == cfg.tokenizer_max_length
    assert args.real_action_dim == cfg.real_action_dim
    assert args.server_address == cfg.server_address
    assert args.tokenizer_name == cfg.tokenizer_name
    assert args.image_keys == cfg.image_keys


def test_default_build_dir_is_loadable_and_carries_the_tuned_loop(tmp_path):
    """A bare build (no explicit sizes) must produce a dir that loads and whose
    loop config is the tuned one - rebuilding must not silently un-tune it."""
    out = build_pretrained_dir(str(tmp_path / "vlacpp"))

    cfg = PreTrainedConfig.from_pretrained(str(out))
    assert cfg.image_size == 256
    assert cfg.n_action_steps == 25
    assert cfg.chunk_size == 50
    assert cfg.prefetch_threshold == 12
    assert cfg.blend_horizon == 5
    assert cfg.max_action_delta == 10.0
    # the budget rule survives the round trip (this is what __post_init__ checks)
    assert cfg.prefetch_threshold + cfg.n_action_steps <= cfg.chunk_size


def test_build_pretrained_dir_rejects_a_budget_violating_build(tmp_path):
    """The builder must not be able to write a dir the loop cannot run."""
    with pytest.raises(ValueError, match="latency budget violated"):
        build_pretrained_dir(
            str(tmp_path / "bad"),
            chunk_size=50,
            n_action_steps=50,  # the old CLI default: lookahead 12 + 50 > 50
        )
