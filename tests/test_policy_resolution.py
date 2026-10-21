import lerobot_policy_vlacpp  # noqa: F401  (registers "vlacpp")
from lerobot.policies.factory import get_policy_class
from lerobot.policies.factory import _make_processors_from_policy_config
from lerobot_policy_vlacpp.configuration_vlacpp import VlaCppConfig
from lerobot_policy_vlacpp.modeling_vlacpp import VlaCppPolicy


def test_get_policy_class_resolves_vlacpp():
    cls = get_policy_class("vlacpp")
    assert cls is VlaCppPolicy


def test_processor_factory_resolves_by_convention():
    cfg = VlaCppConfig(device="cpu")
    pre, post = _make_processors_from_policy_config(cfg, dataset_stats=None)
    assert pre is not None and post is not None
