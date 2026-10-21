"""lerobot PreTrainedPolicy that runs inference on a remote vla-server.

Runs in plain sync mode (`--inference.type=sync`) with our own background
prefetch: once the local queue drops below `prefetch_threshold`, a fetch for
the next chunk starts in a background thread while the current queue drains.
Actions are always consumed strictly in order (no index-skipping/catch-up
logic like lerobot's RTC ActionQueue) - we trade "keeps pace under overload"
for "never skips a commanded step", which is the trade we want here since we
can't do true RTC guidance against a remote black-box server anyway.

The loop depends on one invariant - `prefetch_threshold + n_action_steps <=
chunk_size` - and on one semantic - the delay-crop is measured at
observation-capture time (see `_merge_chunk`). Both are asserted/validated at
startup (`configuration_vlacpp.validate_latency_budget`) and then observed at
runtime, because their failure modes are invisible from the outside:

  * the queue draining into the blocking bootstrap path: the arm freezes for a
    full round trip (~L) and the crop collapses to 0. Counted in
    `stall_count`/`bootstrap_count` and logged per rollout - this is the direct
    signal that the latency margin is too thin (see `configuration_vlacpp` for
    the L / fps / budget arithmetic).
  * a merge-time instead of capture-time crop: the crop silently becomes 0 and
    the chunk-seam jumps backward. Logged per chunk as
    `crop=<n> (capture_depth=<n> merge_depth=<n>)`.

Per-chunk log line: fetched-chunk index, fetch latency and its EMA, computed
crop, queue depth at capture vs at merge, rows queued, queue after.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_vlacpp import VlaCppConfig

logger = logging.getLogger(__name__)

# Exponential moving average weight for the measured fetch latency. 0.2 ~ a
# ~5-chunk window: fast enough to show a server that degraded, slow enough not
# to be dominated by one outlier. Reported per chunk and per rollout.
LATENCY_EMA_ALPHA = 0.2


class VlaCppPolicy(PreTrainedPolicy):
    config_class = VlaCppConfig
    name = "vlacpp"

    def __init__(self, config: VlaCppConfig, client=None):
        super().__init__(config)
        self.config = config
        # Fail at startup, not after the arm has been driven into a wall.
        config.validate_latency_budget()
        self.register_buffer("_dummy", torch.zeros(1))
        self._client = client
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None  # last action actually emitted

        self._lock = threading.Lock()
        self._fetch_thread: threading.Thread | None = None
        self._fetch_error: Exception | None = None

        # --- rollout instrumentation (per-rollout; cleared by reset()) ---
        self._steps_emitted = 0      # select_action calls served
        self._chunks_fetched = 0     # merged chunks
        self._stall_count = 0        # fetches that started with an EMPTY queue
        self._bootstrap_count = 0    # the by-design first fetch of a rollout
        self._latency_ema_ms: float | None = None  # measured fetch round trip
        self._last_fetch_ms: float | None = None   # most recent fetch round trip
        self._last_crop: int | None = None

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
            # Log the rollout the counters describe, then start a fresh one: a
            # non-zero stall count here is the signal that the latency margin
            # was too thin for this rollout (see configuration_vlacpp).
            self._log_rollout_totals()
            self._queue.clear()
            self._last_action = None
            self._fetch_error = None
            self._steps_emitted = 0
            self._chunks_fetched = 0
            self._stall_count = 0
            self._bootstrap_count = 0
            self._latency_ema_ms = None
            self._last_crop = None
        # Let any in-flight fetch finish naturally; its result will just be
        # discarded on merge since the queue was reset (see _merge_chunk).

    @property
    def stall_count(self) -> int:
        """Fetches this rollout that started with an empty queue.

        Includes the by-design bootstrap fetch of the first tick
        (`bootstrap_count`); every other one means the arm froze for ~L because
        the lookahead did not cover the round trip.
        """
        return self._stall_count

    @property
    def bootstrap_count(self) -> int:
        return self._bootstrap_count

    def stats(self) -> dict:
        """Per-rollout counters/log state, for callers that would rather read
        them than parse the log (the loop harness does this)."""
        with self._lock:
            return {
                "steps_emitted": self._steps_emitted,
                "chunks_fetched": self._chunks_fetched,
                "stalls": self._stall_count,
                "bootstraps": self._bootstrap_count,
                "latency_ema_ms": self._latency_ema_ms,
                "last_crop": self._last_crop,
                "queue_depth": len(self._queue),
            }

    def _log_rollout_totals(self) -> None:
        """Must hold self._lock when calling."""
        if not self._steps_emitted and not self._chunks_fetched:
            return  # nothing ran; don't emit an empty line on a bare reset()
        logger.info(
            "vlacpp rollout totals: emitted=%d chunks=%d stalls=%d (bootstraps=%d) "
            "latency_ema=%s crop_last=%s queue=%d",
            self._steps_emitted,
            self._chunks_fetched,
            self._stall_count,
            self._bootstrap_count,
            f"{self._latency_ema_ms:.1f}ms" if self._latency_ema_ms is not None else "n/a",
            self._last_crop if self._last_crop is not None else "n/a",
            len(self._queue),
        )

    def supports_rtc(self) -> bool:
        # Enabled so `--inference.type=rtc` is accepted: lerobot's RTC engine
        # drives predict_action_chunk (which takes **kwargs) and owns the
        # queue/delay-crop. Our own prefetch+blend path is a separate strategy
        # selected via `select_action_mode` and only runs under sync inference.
        return True

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
        t0 = time.perf_counter()
        chunk = self._ensure_client().predict_chunk(images_chw=images_chw, state=state, task=task)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Instrumentation: the measured round trip is L. The EMA is what the
        # per-chunk log reports; a fetch that drifts well above the L the
        # lookahead was derived from is the early warning that the loop is about
        # to stall. Written from the prefetch thread without the lock - a float
        # store is atomic under the GIL and a rare interleaving only skews the
        # average, which is all this number is for.
        self._last_fetch_ms = dt_ms
        self._latency_ema_ms = (
            dt_ms
            if self._latency_ema_ms is None
            else LATENCY_EMA_ALPHA * dt_ms + (1.0 - LATENCY_EMA_ALPHA) * self._latency_ema_ms
        )
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

    def _merge_chunk(self, chunk: np.ndarray, crop: int = 0) -> None:
        """Delay-crop, blend+clamp the newly fetched chunk against the last
        emitted action, then extend the live queue. Must hold self._lock when
        calling.

        ``crop`` is the number of actions still queued ahead of this chunk when
        its observation was captured, and it is the correct offset:

            correct_offset = L * fps + pending_at_merge
                           == queue depth at observation-capture time

        because the rows queued at capture are exactly the rows that play while
        the server works (``L * fps`` of them) plus the rows the loop could not
        have consumed before the fetch started. All of them target poses the arm
        has already passed, so ``chunk[:crop]`` is stale by construction;
        dropping it aligns ``chunk[crop]`` with the arm's actual pose at play
        time and removes the backward hitch at the chunk seam.

        DO NOT "simplify" this to a fixed crop of ``L * fps``, and DO NOT measure
        the depth here instead of at capture. Both alternatives were measured on
        the board's controlled-trajectory seam leg (ideal tracker, chunk =
        pose_at_capture + j/step):

          * capture-time crop (this):  0 backward steps, worst delta 0.0
          * no crop:                   21 backward steps, raw yank -8.0
                                       (= exactly the lookahead width)
          * crop = L*fps only:         0 backward steps *masked by the shipped
                                       blend+clamp*, raw yank -3.0 - the
                                       under-crop residual (capture depth 8 -
                                       L*fps 5); the 3-step reversal reappears
                                       as 7 backward steps the moment the clamp
                                       is relaxed.

        Timings that make a merge-time crop useless: queue depth at capture 8,
        at merge 0-3. A merge-time crop would drop 0-3 rows instead of 8, i.e.
        under-crop by 5-8 - the "looks implemented, does nothing" failure.

        Keep this signature: subclasses (the loop harnesses) override it as
        ``_merge_chunk(self, chunk, crop=0)`` to substitute a crop mode, and
        ``crop`` is by construction the capture-time depth, so extra keyword
        arguments here would break them.
        """
        offset = min(max(0, crop), max(0, chunk.shape[0] - 1))
        chunk = chunk[offset:]
        capture_depth = offset  # measured at observation-capture time
        merge_depth = len(self._queue)

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

        self._last_crop = offset
        self._chunks_fetched += 1
        logger.info(
            "vlacpp chunk #%d: fetch=%s ema=%s crop=%d (capture_depth=%d merge_depth=%d) "
            "queued=%d queue_after=%d",
            self._chunks_fetched,
            f"{self._last_fetch_ms:.1f}ms" if self._last_fetch_ms is not None else "n/a",
            f"{self._latency_ema_ms:.1f}ms" if self._latency_ema_ms is not None else "n/a",
            offset,
            capture_depth,
            merge_depth,
            len(rows),
            len(self._queue),
        )

    def _start_prefetch(self, batch: dict[str, Tensor]) -> None:
        if self._fetch_thread is not None and self._fetch_thread.is_alive():
            return  # a fetch is already in flight
        images_chw, state, task = self._extract_inputs(batch)
        # Queue depth at observation-capture time = actions that will play
        # before this chunk reaches the arm = leading rows to delay-crop.
        # THE OBSERVATION IS CAPTURED HERE, so this is the only correct moment
        # to measure it (see _merge_chunk for why a merge-time depth is wrong).
        with self._lock:
            crop = len(self._queue)
            step_no = self._steps_emitted
        logger.debug(
            "vlacpp prefetch @emit step %d: queue depth %d at capture -> crop=%d",
            step_no,
            crop,
            crop,
        )

        def _worker():
            try:
                chunk = self._fetch_chunk_sync(images_chw, state, task)
                with self._lock:
                    self._merge_chunk(chunk, crop=crop)
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
            self._steps_emitted += 1  # the emit step being served (rollout counter)
        if self.config.select_action_mode == "basic":
            return self._select_action_basic(batch)
        return self._select_action_prefetch(batch)

    def _select_action_basic(self, batch: dict[str, Tensor]) -> Tensor:
        """Reference baseline: blocking fetch, serve a FIFO of n_action_steps
        rows, no prefetch/blend/clamp/delay-crop."""
        with self._lock:
            need_fetch = len(self._queue) == 0

        if need_fetch:
            images_chw, state, task = self._extract_inputs(batch)
            chunk = self._fetch_chunk_sync(images_chw, state, task)
            n = min(self.config.n_action_steps, chunk.shape[0])
            with self._lock:
                # Reference path: the blocking fetch IS the design here, so it is
                # counted as a chunk but deliberately not as a stall.
                self._chunks_fetched += 1
                for row in chunk[:n]:
                    self._queue.append(np.ascontiguousarray(row, dtype=np.float32))

        with self._lock:
            action = self._queue.popleft()
            self._last_action = action.copy()
        return torch.from_numpy(action).unsqueeze(0)

    @torch.no_grad()
    def _select_action_prefetch(self, batch: dict[str, Tensor]) -> Tensor:
        with self._lock:
            if self._fetch_error is not None:
                err, self._fetch_error = self._fetch_error, None
                raise RuntimeError("vla-server prefetch failed") from err

            queue_len = len(self._queue)
            need_bootstrap = queue_len == 0

        if need_bootstrap:
            # First call, or queue ran dry (fetch too slow / server hiccup):
            # block synchronously so we always have something to emit. This is
            # the failure the latency budget exists to prevent: the arm freezes
            # for ~L and the merge that follows crops 0 rows, so nothing in the
            # emitted trajectory shows the freeze.
            with self._lock:
                by_design = self._chunks_fetched == 0 and self._last_action is None
                self._stall_count += 1
                self._bootstrap_count += int(by_design)
                step_no = self._steps_emitted
                fetches = self._chunks_fetched
                ema = self._latency_ema_ms
            if by_design:
                logger.info(
                    "vlacpp bootstrap: queue empty on emit step %d -> blocking fetch "
                    "(by design; the loop is blind for ~L on this tick)",
                    step_no,
                )
            else:
                logger.warning(
                    "vlacpp STALL #%d: queue empty on emit step %d (chunks=%d, latency_ema=%s) "
                    "-> blocking fetch and the delay-crop collapses to 0. The lookahead (%d) does "
                    "not cover one round trip: re-derive it against a re-measured L",
                    self._stall_count,
                    step_no,
                    fetches,
                    f"{ema:.1f}ms" if ema is not None else "n/a",
                    self.config.prefetch_threshold,
                )
            images_chw, state, task = self._extract_inputs(batch)
            chunk = self._fetch_chunk_sync(images_chw, state, task)
            with self._lock:
                self._merge_chunk(chunk, crop=0)
        elif queue_len <= self.config.prefetch_threshold:
            logger.debug(
                "vlacpp prefetch trigger: queue=%d <= lookahead=%d (emit step %d)",
                queue_len,
                self.config.prefetch_threshold,
                self._steps_emitted,
            )
            self._start_prefetch(batch)

        with self._lock:
            action = self._queue.popleft()
            self._last_action = action.copy()
        return torch.from_numpy(action).unsqueeze(0)