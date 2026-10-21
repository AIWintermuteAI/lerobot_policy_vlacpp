"""Runtime compilation of the bundled vla.proto into a Python module.

Mirrors eval/client/vla_cpp_client.py::_load_pb but sources the proto from the
package data so the plugin is self-contained when installed on the robot.
"""

from __future__ import annotations

import importlib.resources
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_PB = None


def load_pb():
    """Compile the bundled vla.proto once and return the generated module."""
    global _PB
    if _PB is not None:
        return _PB

    proto_text = importlib.resources.files("lerobot_policy_vlacpp").joinpath("vla.proto").read_text()
    tmpdir = tempfile.mkdtemp(prefix="lerobot-vlacpp-pb-")
    proto_path = os.path.join(tmpdir, "vla.proto")
    with open(proto_path, "w") as f:
        f.write(proto_text)

    subprocess.check_call([
        "protoc",
        f"--proto_path={tmpdir}",
        f"--python_out={tmpdir}",
        proto_path,
    ])
    if tmpdir not in sys.path:
        sys.path.insert(0, tmpdir)
    import vla_pb2  # noqa: E402  (generated at runtime)

    _PB = vla_pb2
    return _PB
