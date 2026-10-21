import importlib.resources
import os
from pathlib import Path

import pytest

BUNDLED_MD5 = "a4d21736ae5badad4f091d55b0bb950e"


def _candidate_sources() -> list[Path]:
    """Where the authoritative `src/serving/vla.proto` can live.

    The plugin is normally vendored inside the vla.cpp tree
    (`integrations/lerobot/lerobot_policy_vlacpp`), but it is also used as a
    standalone repo (this checkout), where the tree has to be pointed at. Both
    are checked so this stops being a silent skip whenever a tree is around -
    a bundled proto that drifts from the server's wire format is a hard failure,
    not a warning.
    """
    here = Path(__file__).resolve()
    cands = [here.parents[4] / "src" / "serving" / "vla.proto"]  # in-tree layout
    if env := os.environ.get("VLA_CPP_SRC"):
        cands.append(Path(env) / "src" / "serving" / "vla.proto")
    for up in here.parents[:6]:  # sibling checkouts (workspace/vla.cpp, ...)
        cands.append(up / "vla.cpp" / "src" / "serving" / "vla.proto")
    return cands


def _repo_proto() -> Path | None:
    return next((p for p in _candidate_sources() if p.exists()), None)


def test_bundled_proto_matches_source():
    src = _repo_proto()
    if src is None:
        pytest.skip("vla.cpp source tree not present (installed standalone)")
    bundled = importlib.resources.files("lerobot_policy_vlacpp").joinpath("vla.proto").read_text()
    assert bundled == src.read_text(), f"bundled vla.proto drifted from {src}"


def test_bundled_proto_is_the_known_wire_format():
    """Pin the wire format itself: client and server must agree byte-for-byte."""
    import hashlib

    bundled = importlib.resources.files("lerobot_policy_vlacpp").joinpath("vla.proto").read_bytes()
    assert hashlib.md5(bundled).hexdigest() == BUNDLED_MD5


def test_proto_module_compiles_and_exposes_messages():
    from lerobot_policy_vlacpp import _proto

    pb = _proto.load_pb()
    assert hasattr(pb, "PredictRequest")
    assert hasattr(pb, "PredictResponse")
