from pathlib import Path

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.factory import make_pre_post_processors

from lerobot_policy_vlacpp.build_pretrained import build_pretrained_dir


def test_build_pretrained_dir_roundtrip(tmp_path):
    out = tmp_path / "vlacpp-so101"
    build_pretrained_dir(
        str(out),
        server_address="tcp://localhost:5555",
        image_keys=["observation.images.front"],
        chunk_size=50,
        n_action_steps=50,
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
