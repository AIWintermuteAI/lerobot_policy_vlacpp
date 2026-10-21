# client.py
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
import zmq

from ._proto import load_pb


def _resize_with_pad(img_chw: np.ndarray, target_h: int, target_w: int, pad_value: float = 0.0) -> np.ndarray:
    """Aspect-preserving resize into a (target_h, target_w) canvas, padding top/left."""
    import torch
    import torch.nn.functional as F

    if img_chw.ndim != 3 or img_chw.shape[0] != 3:
        raise ValueError(f"_resize_with_pad expected CHW [3,H,W], got {img_chw.shape}")

    t = torch.from_numpy(np.ascontiguousarray(img_chw)).unsqueeze(0)  # [1,3,H,W]
    cur_h, cur_w = t.shape[2], t.shape[3]
    ratio = max(cur_w / target_w, cur_h / target_h)
    rh = max(1, int(cur_h / ratio))
    rw = max(1, int(cur_w / ratio))
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    pad_h = max(0, target_h - rh)
    pad_w = max(0, target_w - rw)
    t = F.pad(t, (pad_w, 0, pad_h, 0), value=pad_value)

    out = t[0].numpy()  # explicit index, not squeeze(0) — always drops exactly the batch dim
    if out.ndim != 3 or out.shape[0] != 3:
        raise RuntimeError(
            f"_resize_with_pad produced unexpected shape {out.shape} "
            f"from input {img_chw.shape} (target {target_h}x{target_w}, rh={rh}, rw={rw})"
        )
    return out


class VlaCppSmolvlaClient:
    def __init__(
        self,
        server_address: str = "tcp://localhost:5555",
        *,
        tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Instruct",
        image_size: int = 256,
        max_state_dim: int = 32,
        tokenizer_max_length: int = 48,
        real_action_dim: int = 6,
        recv_timeout_ms: int = 30_000,
    ):
        self.pb = load_pb()
        self.image_size = int(image_size)
        self.max_state_dim = int(max_state_dim)
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.real_action_dim = int(real_action_dim)

        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)

        self._ctx = zmq.Context.instance()
        self.sock = self._ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self.sock.connect(server_address)
        self._step = 0

        # REQ sockets are a strict send/recv state machine - calling send()
        # from two threads before the matching recv() completes raises
        # "Operation cannot be accomplished in current state". Our prefetch
        # thread and the bootstrap/blocking path share this client, so all
        # request/response cycles must be fully serialized.
        self._sock_lock = threading.Lock()

        self._cached_task: str | None = None
        self._cached_token_ids: np.ndarray | None = None

    def _tokenize_task(self, task: str) -> np.ndarray:
        task = task or ""
        if not task.endswith("\n"):
            task = task + "\n"
        if task == self._cached_task and self._cached_token_ids is not None:
            return self._cached_token_ids
        toks = self.tok(
            task, padding=False, truncation=True,
            max_length=self.tokenizer_max_length, return_tensors="np",
        )
        ids = toks["input_ids"][0].astype(np.int32)
        self._cached_task = task
        self._cached_token_ids = ids
        return ids

    def predict_chunk(self, images_chw: Sequence[np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1

        for img in images_chw:
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[0] != 3:
                raise ValueError(f"expected CHW float32 [3,H,W], got {arr.shape}")
            arr = _resize_with_pad(arr, self.image_size, self.image_size, pad_value=0.0)
            hwc = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)), dtype=np.float32)
            ip = req.images.add()
            ip.encoding = self.pb.Image.F32_RGB_01
            ip.height = hwc.shape[0]
            ip.width = hwc.shape[1]
            ip.data = hwc.tobytes()

        s = np.asarray(state, dtype=np.float32).reshape(-1)
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[: s.shape[0]] = s
        req.state.extend(state_padded.tolist())

        req.lang_tokens.extend(self._tokenize_task(task).tolist())

        payload = req.SerializeToString()
        with self._sock_lock:
            self.sock.send(payload)
            resp = self.pb.PredictResponse()
            resp.ParseFromString(self.sock.recv())

        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        chunk = np.asarray(resp.action_chunk, dtype=np.float32).reshape(resp.chunk_size, resp.action_dim)
        return chunk[:, : self.real_action_dim]

    def close(self):
        self.sock.close(0)