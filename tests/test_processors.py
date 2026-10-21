import torch

from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig
from lerobot_policy_vlacpp.processor_vlacpp import make_vlacpp_pre_post_processors


def test_processors_build_and_are_passthrough():
    cfg = VlaCppConfig(device="cpu")
    pre, post = make_vlacpp_pre_post_processors(cfg)

    # No normalization steps are present (server-side normalization).
    step_names = [type(s).__name__ for s in pre.steps]
    assert "NormalizerProcessorStep" not in step_names
    post_names = [type(s).__name__ for s in post.steps]
    assert "UnnormalizerProcessorStep" not in post_names

    out = pre({"observation.state": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), "task": "x"})
    assert "observation.state" in out
