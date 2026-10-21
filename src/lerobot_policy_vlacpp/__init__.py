"""lerobot policy plugin backed by vla.cpp's vla-server (SmolVLA GGUF over ZMQ).

Importing this package registers the ``vlacpp`` policy type with lerobot by
importing the configuration module, whose decorator runs at import time.
"""

from . import configuration_vlacpp as configuration_vlacpp  # noqa: F401  (registers "vlacpp")

__all__ = ["configuration_vlacpp"]
