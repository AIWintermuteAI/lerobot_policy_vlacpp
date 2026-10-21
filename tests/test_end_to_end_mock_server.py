import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

from lerobot_policy_vlacpp.build_pretrained import build_pretrained_dir


def test_end_to_end_select_action(tmp_path, mock_server):
    out = build_pretrained_dir(
        str(tmp_path / "vlacpp"),
        server_address=mock_server.addr,
        image_keys=["observation.images.front"],
        chunk_size=mock_server.chunk_size,
        # must satisfy prefetch_threshold + n_action_steps <= chunk_size
        n_action_steps=25,
    )
    cfg = PreTrainedConfig.from_pretrained(str(out))
    cfg.pretrained_path = out
    policy = get_policy_class("vlacpp").from_pretrained(out, config=cfg)

    batch = {
        "observation.images.front": torch.zeros(1, 3, 240, 320),
        "observation.state": torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]),
        "task": "grab the block",
    }
    action = policy.select_action(batch)

    assert action.shape == (1, mock_server.action_dim)
    # server returned a ramp; row 0 = [0,1,2,3,4,5]
    assert torch.allclose(action[0], torch.arange(0, mock_server.action_dim, dtype=torch.float32))
    # the request actually reached the mock server
    assert mock_server.last_request is not None
    assert len(mock_server.last_request.images) == 1
