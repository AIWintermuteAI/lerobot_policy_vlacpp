import numpy as np
import torch

from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig
from lerobot_policy_vlacpp.modeling_vlacpp import VlaCppPolicy


class FakeClient:
    def __init__(self, chunk_size=4, action_dim=6):
        self.calls = 0
        self._chunk = np.arange(chunk_size * action_dim, dtype=np.float32).reshape(chunk_size, action_dim)

    def predict_chunk(self, images_chw, state, task):
        self.calls += 1
        return self._chunk

    def close(self):
        pass


def _batch():
    return {
        "observation.images.front": torch.zeros(1, 3, 240, 320),
        "observation.state": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        "task": "pick up the cube",
    }


def test_select_action_queue_reuses_one_chunk():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4,
                       image_keys=["observation.images.front"])
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=4, action_dim=6))

    a0 = policy.select_action(_batch())
    a1 = policy.select_action(_batch())
    assert a0.shape == (1, 6)
    assert policy._client.calls == 1  # second call served from the queue
    assert torch.allclose(a0[0], torch.arange(0, 6, dtype=torch.float32))
    assert torch.allclose(a1[0], torch.arange(6, 12, dtype=torch.float32))


def test_reset_clears_queue():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4)
    policy = VlaCppPolicy(cfg, client=FakeClient())
    policy.select_action(_batch())
    policy.reset()
    policy.select_action(_batch())
    assert policy._client.calls == 2  # fresh chunk after reset


def test_predict_action_chunk_shape():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4)
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=4, action_dim=6))
    chunk = policy.predict_action_chunk(_batch())
    assert chunk.shape == (1, 4, 6)
