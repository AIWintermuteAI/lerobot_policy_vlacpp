import threading

import numpy as np
import pytest
import zmq

from lerobot_policy_vlacpp import _proto


class MockVlaServer:
    """A minimal ZMQ REP server that records the last request and returns a
    canned action chunk. Runs on an ephemeral localhost port."""

    def __init__(self, chunk_size=50, action_dim=6):
        self.pb = _proto.load_pb()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.last_request = None
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self.addr = f"tcp://127.0.0.1:{self.port}"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(50)).get(self._sock) != zmq.POLLIN:
                continue
            body = self._sock.recv()
            req = self.pb.PredictRequest()
            req.ParseFromString(body)
            self.last_request = req
            resp = self.pb.PredictResponse()
            resp.request_id = req.request_id
            resp.chunk_size = self.chunk_size
            resp.action_dim = self.action_dim
            # deterministic ramp so tests can assert ordering
            chunk = np.arange(self.chunk_size * self.action_dim, dtype=np.float32)
            resp.action_chunk.extend(chunk.tolist())
            self._sock.send(resp.SerializeToString())

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close(0)


@pytest.fixture
def mock_server():
    srv = MockVlaServer().start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def mock_server_padded():
    # Mirrors vla-server's real wire format: action_chunk columns are always
    # padded to max_action_dim, regardless of the checkpoint's real_action_dim.
    srv = MockVlaServer(chunk_size=10, action_dim=32).start()
    try:
        yield srv
    finally:
        srv.stop()
