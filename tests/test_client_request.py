import numpy as np

from lerobot_policy_vlacpp.client import VlaCppSmolvlaClient


def test_client_builds_request_and_parses_chunk(mock_server):
    client = VlaCppSmolvlaClient(
        server_address=mock_server.addr,
        tokenizer_name="HuggingFaceTB/SmolVLM2-500M-Instruct",
        image_size=512,
        max_state_dim=32,
        tokenizer_max_length=48,
    )
    img = np.zeros((3, 240, 320), dtype=np.float32)  # CHW in [0,1]
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    chunk = client.predict_chunk(images_chw=[img], state=state, task="pick up the cube")

    req = mock_server.last_request
    assert len(req.images) == 1
    assert req.images[0].encoding == client.pb.Image.F32_RGB_01
    assert req.images[0].height == 512 and req.images[0].width == 512
    assert len(req.state) == 32  # zero-padded to max_state_dim
    assert np.allclose(req.state[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # float32 transport
    assert len(req.lang_tokens) > 0
    assert chunk.shape == (50, 6)
    assert chunk[0, 0] == 0.0 and chunk[0, 1] == 1.0  # ramp ordering preserved


def test_client_slices_padded_action_to_real_action_dim(mock_server_padded):
    # vla-server always returns max_action_dim columns (padded); only the first
    # real_action_dim carry values. The client must trim before returning.
    client = VlaCppSmolvlaClient(
        server_address=mock_server_padded.addr,
        tokenizer_name="HuggingFaceTB/SmolVLM2-500M-Instruct",
        image_size=512,
        max_state_dim=32,
        tokenizer_max_length=48,
        real_action_dim=7,
    )
    img = np.zeros((3, 240, 320), dtype=np.float32)
    state = np.zeros(8, dtype=np.float32)

    chunk = client.predict_chunk(images_chw=[img], state=state, task="grab the block")

    assert chunk.shape == (mock_server_padded.chunk_size, 7)
    # ramp preserved for the first 7 (real) columns, padded 8..31 dropped
    assert chunk[0, 0] == 0.0 and chunk[0, 6] == 6.0

