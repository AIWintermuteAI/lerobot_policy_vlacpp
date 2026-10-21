import torch

from lerobot.rollout.inference.rtc import supports_rtc_inference

from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig
from lerobot_policy_vlacpp.modeling_vlacpp import VlaCppPolicy


class FakeClient:
    def predict_chunk(self, images_chw, state, task):
        return __import__("numpy").zeros((4, 6), dtype="float32")

    def close(self):
        pass


def _policy():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4)
    return VlaCppPolicy(cfg, client=FakeClient())


def test_supports_rtc_inference_gate_passes():
    # This is lerobot's own gate (rollout/context.py raises if this is False),
    # not a reimplementation - proves our policy is accepted by --inference.type=rtc.
    assert supports_rtc_inference(_policy()) is True


def test_rtc_config_field_is_settable_like_the_rollout_engine_does():
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    policy = _policy()
    # build_rollout_context does exactly this when --inference.type=rtc:
    #   policy.config.rtc_config = cfg.inference.rtc
    policy.config.rtc_config = RTCConfig(execution_horizon=10)
    assert policy.config.rtc_config.execution_horizon == 10


def test_predict_action_chunk_accepts_rtc_kwargs_and_ignores_them():
    policy = _policy()
    batch = {
        "observation.images.front": torch.zeros(1, 3, 240, 320),
        "observation.state": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        "task": "pick up the cube",
    }
    prev_left_over = torch.zeros(10, 6)
    chunk = policy.predict_action_chunk(batch, inference_delay=3, prev_chunk_left_over=prev_left_over)
    assert chunk.shape == (1, 4, 6)


def test_no_init_rtc_processor_needed():
    # We have no local denoising state to set up; rollout/context.py only calls
    # init_rtc_processor if hasattr(policy, "init_rtc_processor") - confirm we
    # correctly don't define one (would be dead code / misleading otherwise).
    assert not hasattr(_policy(), "init_rtc_processor")
