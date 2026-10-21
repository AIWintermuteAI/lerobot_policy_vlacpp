import logging

import numpy as np
import pytest
import torch

from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig
from lerobot_policy_vlacpp.modeling_vlacpp import VlaCppPolicy

LOGGER_NAME = "lerobot_policy_vlacpp.modeling_vlacpp"


class FakeClient:
    def __init__(self, chunk_size=4, action_dim=6):
        self.calls = 0
        self._chunk = np.arange(chunk_size * action_dim, dtype=np.float32).reshape(chunk_size, action_dim)

    def predict_chunk(self, images_chw, state, task):
        self.calls += 1
        return self._chunk

    def close(self):
        pass


class RampClient:
    """Chunk = a +1/step ramp on channel 0: chunk[j][0] == j.

    Models the round trip honestly: whatever is queued when the call arrives
    plays while the 'server computes', so the queue is drained by call time and
    is empty by merge time - exactly what the board measured (capture depth 8,
    merge depth 0-3).
    """

    def __init__(self, policy_getter, chunk_size=50, action_dim=6):
        self._get = policy_getter
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.calls = 0
        self.depth_at_call = None

    def predict_chunk(self, images_chw, state, task):
        self.calls += 1
        policy = self._get()
        with policy._lock:
            self.depth_at_call = len(policy._queue)
            for _ in range(self.depth_at_call):
                policy._queue.popleft()
        rows = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        rows[:, 0] = np.arange(self.chunk_size, dtype=np.float32)
        return rows

    def close(self):
        pass


def _batch(keys=("observation.images.front",)):
    batch = {k: torch.zeros(1, 3, 240, 320) for k in keys}
    batch["observation.state"] = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    batch["task"] = "pick up the cube"
    return batch


def test_select_action_queue_reuses_one_chunk():
    # prefetch_threshold=0 so the queue is only refilled when it is empty: the
    # point of this test is that a fetch is NOT issued while rows remain.
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4, prefetch_threshold=0,
                       image_keys=["observation.images.front"])
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=4, action_dim=6))

    a0 = policy.select_action(_batch())
    a1 = policy.select_action(_batch())
    assert a0.shape == (1, 6)
    assert policy._client.calls == 1  # second call served from the queue
    assert torch.allclose(a0[0], torch.arange(0, 6, dtype=torch.float32))
    assert torch.allclose(a1[0], torch.arange(6, 12, dtype=torch.float32))


def test_reset_clears_queue():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4, prefetch_threshold=0)
    policy = VlaCppPolicy(cfg, client=FakeClient())
    policy.select_action(_batch())
    policy.reset()
    policy.select_action(_batch())
    assert policy._client.calls == 2  # fresh chunk after reset


def test_predict_action_chunk_shape():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4, prefetch_threshold=0)
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=4, action_dim=6))
    chunk = policy.predict_action_chunk(_batch())
    assert chunk.shape == (1, 4, 6)


# ---------------------------------------------------------------------------
# instrumentation: the stall counter
# ---------------------------------------------------------------------------
def test_stall_counter_increments_when_the_queue_empties():
    """With no lookahead (prefetch_threshold=0) every tick after the bootstrap
    enters on an empty queue - that is the stall signal the budget exists to
    prevent, so it must be counted (and the first one tagged as by-design)."""
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=1, prefetch_threshold=0)
    client = FakeClient(chunk_size=4, action_dim=6)
    policy = VlaCppPolicy(cfg, client=client)

    policy.select_action(_batch())
    assert policy.stall_count == 1          # bootstrap (by design)
    assert policy.bootstrap_count == 1
    policy.select_action(_batch())
    assert policy.stall_count == 2          # real drain
    policy.select_action(_batch())
    assert policy.stall_count == 3
    assert policy.bootstrap_count == 1      # only ever the first tick

    stats = policy.stats()
    assert stats["stalls"] == 3
    assert stats["bootstraps"] == 1
    assert stats["steps_emitted"] == 3
    assert stats["chunks_fetched"] == 3     # one blocking fetch per drained tick
    assert client.calls == 3


def test_stall_count_is_just_the_bootstrap_when_the_lookahead_covers_the_fetch():
    """The healthy case: the queue never runs dry, so the only empty-queue fetch
    is the by-design first tick."""
    cfg = VlaCppConfig(device="cpu", chunk_size=8, n_action_steps=2, prefetch_threshold=2)
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=8, action_dim=6))

    for _ in range(6):
        policy.select_action(_batch())
        if policy._fetch_thread is not None:      # keep the background fetch deterministic
            policy._fetch_thread.join(timeout=5)

    stats = policy.stats()
    assert stats["steps_emitted"] == 6
    assert stats["stalls"] == 1
    assert stats["bootstraps"] == 1
    assert stats["latency_ema_ms"] is not None   # the fetch latency was measured


def test_reset_logs_and_restarts_the_rollout_counters(caplog):
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=1, prefetch_threshold=0)
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=4, action_dim=6))
    policy.select_action(_batch())
    policy.select_action(_batch())

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        policy.reset()

    totals = [m for m in caplog.messages if "rollout totals" in m]
    assert totals, "reset() must log the per-rollout totals"
    assert "stalls=2" in totals[0]
    assert policy.stall_count == 0                       # new rollout
    assert policy.stats()["steps_emitted"] == 0


# ---------------------------------------------------------------------------
# the delay-crop semantics
# ---------------------------------------------------------------------------
def test_crop_uses_capture_time_depth_not_merge_time_or_latency_only():
    """The delay-crop offset must be the queue depth measured when the
    OBSERVATION WAS CAPTURED (== L*fps + pending_at_merge), not the depth at
    merge time (measured 0-3 on the board, i.e. crops nothing) and not a bare
    L*fps (measured -3.0 residual yank; no crop at all is -8.0).

    Here the observation is captured with 8 rows pending and the 'server' drains
    all 8 while it computes, so:
      capture-time crop (this) -> head is chunk[8]
      merge-time crop          -> head would be chunk[0]
      L*fps=5 crop             -> head would be chunk[5]
    """
    holder = {}
    client = RampClient(lambda: holder["policy"], chunk_size=50, action_dim=6)
    cfg = VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=8,
                       max_action_delta=None, blend_horizon=0)
    policy = VlaCppPolicy(cfg, client=client)
    holder["policy"] = policy

    with policy._lock:
        for k in range(8):  # 8 rows pending at observation-capture time
            policy._queue.append(np.full(6, 900.0 + k, dtype=np.float32))

    policy._start_prefetch(_batch())
    policy._fetch_thread.join(timeout=5)

    assert client.depth_at_call == 8          # captured with 8 pending
    assert len(policy._queue) == 25           # 50 - 8 cropped rows, n_action_steps rows queued
    head = policy._queue[0]
    assert head[0] == 8.0
    assert head[0] != 0.0                     # not a merge-time crop (queue empty at merge)
    assert head[0] != 5.0                     # not a bare L*fps crop (L*fps = 5)
    assert policy.stats()["last_crop"] == 8


def test_merge_logs_crop_and_both_queue_depths(caplog):
    """The per-chunk line must carry the latency EMA, the crop, and the queue
    depth at capture vs at merge - the three things that made the last
    debugging session blind."""
    cfg = VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=8)
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=50, action_dim=6))

    with policy._lock:
        for _ in range(3):  # 3 rows pending -> merge-time depth is 3
            policy._queue.append(np.zeros(6, dtype=np.float32))
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            policy._merge_chunk(np.zeros((50, 6), dtype=np.float32), crop=8)

    lines = [m for m in caplog.messages if "vlacpp chunk #1" in m]
    assert len(lines) == 1
    line = lines[0]
    assert "crop=8" in line
    assert "capture_depth=8" in line
    assert "merge_depth=3" in line
    assert "ema=" in line
    assert "queued=25" in line


def test_bootstrap_merge_is_not_cropped_and_is_logged(caplog):
    """The first fetch of a rollout has nothing queued ahead of it, so it must
    merge at crop=0 (a crop there would drop real rows)."""
    cfg = VlaCppConfig(device="cpu", chunk_size=50, n_action_steps=25, prefetch_threshold=12)
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=50, action_dim=6))

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        policy.select_action(_batch())

    assert policy.stats()["last_crop"] == 0
    assert policy.bootstrap_count == 1
    assert any("bootstrap" in m for m in caplog.messages)


def test_subclass_overriding_merge_chunk_with_the_crop_signature_still_works():
    """API contract: the loop harnesses override ``_merge_chunk(self, chunk, crop=0)``
    to substitute a crop mode. Adding a keyword argument to the base signature
    silently broke them (TypeError on the first merge, zero actions emitted), so
    pin the signature from the outside."""
    seen = []

    class CropObservedPolicy(VlaCppPolicy):
        def __init__(self, config, client=None):
            super().__init__(config, client=client)
            self.records = []

        def _merge_chunk(self, chunk, crop=0):
            capture_depth = int(crop)
            merge_depth = len(self._queue)
            super()._merge_chunk(chunk, crop=crop)
            seen.append("merge")
            self.records.append({
                "crop_used": int(crop),
                "capture_depth": capture_depth,
                "merge_depth": merge_depth,
                "queue_after": len(self._queue),
            })

    cfg = VlaCppConfig(device="cpu", chunk_size=8, n_action_steps=2, prefetch_threshold=2)
    policy = CropObservedPolicy(cfg, client=FakeClient(chunk_size=8, action_dim=6))

    for _ in range(3):
        policy.select_action(_batch())
        if policy._fetch_thread is not None:
            policy._fetch_thread.join(timeout=5)

    assert seen.count("merge") == len(policy.records) >= 3
    assert policy.stats()["steps_emitted"] == 3
    # the crop the plugin computed IS the capture-time depth
    last = policy.records[-1]
    assert last["crop_used"] == last["capture_depth"] == 2
    assert policy.stats()["last_crop"] == 2


def test_basic_mode_serves_rows_in_order_without_prefetch():
    cfg = VlaCppConfig(device="cpu", chunk_size=4, n_action_steps=4, prefetch_threshold=0,
                       select_action_mode="basic")
    policy = VlaCppPolicy(cfg, client=FakeClient(chunk_size=4, action_dim=6))
    rows = [policy.select_action(_batch())[0].tolist() for _ in range(4)]
    assert rows[0][0] == 0.0 and rows[3][0] == 18.0
    assert policy._client.calls == 1
    assert policy.stall_count == 0  # the blocking fetch is the design here, not a stall
