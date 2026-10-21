import importlib.metadata


def test_distribution_name_has_plugin_prefix():
    # lerobot discovers plugins by distribution Name prefix; it must match exactly.
    dist = importlib.metadata.distribution("lerobot_policy_vlacpp")
    assert dist.metadata["Name"] == "lerobot_policy_vlacpp"


def test_top_level_module_importable():
    mod = importlib.import_module("lerobot_policy_vlacpp")
    assert mod is not None
