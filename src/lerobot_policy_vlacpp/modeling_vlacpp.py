"""lerobot PreTrainedPolicy that runs inference on a remote vla-server.

Runs in plain sync mode (`--inference.type=sync`) with our own background
prefetch: once the local queue drops below `prefetch_threshold`, a fetch for
the next chunk starts in a background thread while the current queue drains.
Actions are always consumed strictly in order (no index-skipping/catch-up
logic like lerobot's RTC ActionQueue) - we trade "keeps pace under overload"
for "never skips a commanded step", which is the trade we want here since we
can't do true RTC guidance against a remote black-box server anyway.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_vlacpp import VlaCppConfig

logger = logging.getLogger(__name__)


class VlaCppPolicy(PreTrainedPolicy):
    config_class = VlaCppConfig
    name = "vlacpp"

    def __init__(self, config: VlaCppConfig, client=None):
        super().__init__(config)
        self.config = config
        self.register_buffer("_dummy", torch.zeros(1))
        self._client = client
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None  # last action actually emitted

        self._lock = threading.Lock()
        self._fetch_thread: threading.Thread | None = None
        self._fetch_error: Exception | None = None

    # --- lifecycle ---
    def _ensure_client(self):
        if self._client is None:
            from .client import VlaCppSmolvlaClient

            self._client = VlaCppSmolvlaClient(
                server_address=self.config.server_address,
                tokenizer_name=self.config.tokenizer_name,
                image_size=self.config.image_size,
                max_state_dim=self.config.max_state_dim,
                tokenizer_max_length=self.config.tokenizer_max_length,
                real_action_dim=self.config.real_action_dim,
                recv_timeout_ms=self.config.recv_timeout_ms,
            )
        return self._client

    def reset(self):
        with self._lock:
            self._queue.clear()
            self._last_action = None
            self._fetch_error = None
        # Let any in-flight fetch finish naturally; its result will just be
        # discarded on merge since the queue was reset (see _merge_chunk).

    def supports_rtc(self) -> bool:
        # We deliberately don't rely on lerobot's RTC engine (see module
        # docstring) - this policy is meant to run under sync inference with
        # its own prefetch below. Kept False so `--inference.type=rtc` isn't
        # silently accepted and doesn't fight this logic.
        return False

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]):
        raise NotImplementedError("VlaCppPolicy is inference-only; training is done in vla.cpp/PyTorch upstream.")

    # --- batch -> client inputs ---
    def _extract_inputs(self, batch: dict[str, Tensor]):
        images_chw = []
        for key in self.config.image_keys:
            if key not in batch:
                raise KeyError(f"image key '{key}' not in batch keys {list(batch.keys())}")
            img = batch[key]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 4:
                img = img[0]
            if img.ndim != 3:
                raise ValueError(f"{key}: expected CHW (or B,CHW), got {img.shape}")
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            images_chw.append(np.ascontiguousarray(img, dtype=np.float32))

        state = batch["observation.state"]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        task = batch.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        if isinstance(task, bytes):
            task = task.decode()
        return images_chw, state, str(task)

    def _fetch_chunk_sync(self, images_chw, state, task) -> np.ndarray:
        chunk = self._ensure_client().predict_chunk(images_chw=images_chw, state=state, task=task)
        return np.asarray(chunk, dtype=np.float32)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Required by PreTrainedPolicy's abstract interface. Not used by our
        select_action path (see module docstring) — provided so lerobot's
        generic tooling that calls this directly still works."""
        images_chw, state, task = self._extract_inputs(batch)
        chunk = self._fetch_chunk_sync(images_chw, state, task)
        return torch.from_numpy(chunk).unsqueeze(0)

    def _blend_weights(self, horizon: int) -> np.ndarray:
        t = np.linspace(1.0, 0.0, num=horizon, endpoint=False, dtype=np.float32)
        if self.config.blend_power != 1.0:
            t = t**self.config.blend_power
        return t[:, None]

    def _clamp_delta(self, action: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        limit = self.config.max_action_delta
        if limit is None or prev is None:
            return action
        delta = np.clip(action - prev, -limit, limit)
        return prev + delta

    def _merge_chunk(self, chunk: np.ndarray, expected_generation: int) -> None:
        """Blend+clamp the newly fetched chunk against the last emitted action,
        then extend the live queue. Must hold self._lock when calling."""
        n = min(self.config.n_action_steps, chunk.shape[0])
        rows = chunk[:n].copy()

        prev = self._last_action if self._queue == deque() else (self._queue[-1] if self._queue else self._last_action)
        # ^ blend against whatever the arm will actually be doing right before
        # these new rows play: tail of the still-pending queue if non-empty,
        # else the last action we've already emitted.

        horizon = min(self.config.blend_horizon, n)
        if horizon > 0 and prev is not None:
            w_old = self._blend_weights(horizon)
            rows[:horizon] = w_old * prev + (1.0 - w_old) * rows[:horizon]

        last = prev
        for i in range(n):
            rows[i] = self._clamp_delta(rows[i], last)
            last = rows[i]

        for row in rows:
            self._queue.append(np.ascontiguousarray(row, dtype=np.float32))

    def _start_prefetch(self, batch: dict[str, Tensor]) -> None:
        if self._fetch_thread is not None and self._fetch_thread.is_alive():
            return  # a fetch is already in flight
        images_chw, state, task = self._extract_inputs(batch)
        generation = id(self._queue)  # cheap token; reset() clears the queue so stale merges are harmless no-ops

        def _worker():
            try:
                chunk = self._fetch_chunk_sync(images_chw, state, task)
                with self._lock:
                    self._merge_chunk(chunk, generation)
                    self._fetch_error = None
            except Exception as exc:  # noqa: BLE001 - surface on next select_action
                logger.exception("background chunk fetch failed")
                with self._lock:
                    self._fetch_error = exc

        self._fetch_thread = threading.Thread(target=_worker, daemon=True)
        self._fetch_thread.start()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        with self._lock:
            if self._fetch_error is not None:
                err, self._fetch_error = self._fetch_error, None
                raise RuntimeError("vla-server prefetch failed") from err

            queue_len = len(self._queue)
            need_bootstrap = queue_len == 0

        if need_bootstrap:
            # First call, or queue ran dry (fetch too slow / server hiccup):
            # block synchronously so we always have something to emit.
            images_chw, state, task = self._extract_inputs(batch)
            chunk = self._fetch_chunk_sync(images_chw, state, task)
            with self._lock:
                self._merge_chunk(chunk, id(self._queue))
        elif queue_len <= self.config.prefetch_threshold:
            self._start_prefetch(batch)

        with self._lock:
            action = self._queue.popleft()
            self._last_action = action.copy()
        return torch.from_numpy(action).unsqueeze(0)