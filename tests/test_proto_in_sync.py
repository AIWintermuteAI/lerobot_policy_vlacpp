import importlib.resources
from pathlib import Path

import pytest


def _repo_proto() -> Path:
    # tests/ -> lerobot_policy_vlacpp/ -> lerobot/ -> integrations/ -> repo root
    return Path(__file__).resolve().parents[4] / "src" / "serving" / "vla.proto"


def test_bundled_proto_matches_source():
    src = _repo_proto()
    if not src.exists():
        pytest.skip("vla.cpp source tree not present (installed standalone)")
    bundled = importlib.resources.files("lerobot_policy_vlacpp").joinpath("vla.proto").read_text()
    assert bundled == src.read_text(), "bundled vla.proto drifted from src/serving/vla.proto"


def test_proto_module_compiles_and_exposes_messages():
    from lerobot_policy_vlacpp import _proto

    pb = _proto.load_pb()
    assert hasattr(pb, "PredictRequest")
    assert hasattr(pb, "PredictResponse")
