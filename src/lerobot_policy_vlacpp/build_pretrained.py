"""Create a lerobot-loadable pretrained directory for the vlacpp policy.

Writes config.json (the VlaCppConfig), model.safetensors (the weightless policy),
and the passthrough processor JSONs so `lerobot-rollout --policy.path=<dir>` works.

Every default below mirrors the corresponding `VlaCppConfig` field default, and
`tests/test_build_pretrained.py::test_cli_defaults_match_config_defaults` asserts
it stays that way. That test exists because they DID diverge: the CLI defaulted
to `image_size=512` while the model (smolvla-ducks-v4) and the config are
natively 256, so a bare `vlacpp-build-pretrained` silently wrote a pretrained dir
that the server rejects - only visible on a real run. The loop defaults
(`prefetch_threshold` 12, `blend_horizon` 5, `max_action_delta` 10) come along
with them, so rebuilding the dir can no longer reset the tuned control loop to
some other value (see configuration_vlacpp for the latency arithmetic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_vlacpp import VlaCppConfig
from .modeling_vlacpp import VlaCppPolicy
from .processor_vlacpp import make_vlacpp_pre_post_processors


def build_pretrained_dir(
    out_dir: str,
    *,
    server_address: str = "tcp://localhost:5555",
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Instruct",
    image_size: int = 256,
    max_state_dim: int = 32,
    tokenizer_max_length: int = 48,
    chunk_size: int = 50,
    n_action_steps: int = 25,
    real_action_dim: int = 6,
    image_keys: list[str] | None = None,
    device: str = "cpu",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = VlaCppConfig(
        device=device,
        server_address=server_address,
        tokenizer_name=tokenizer_name,
        image_size=image_size,
        max_state_dim=max_state_dim,
        tokenizer_max_length=tokenizer_max_length,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        real_action_dim=real_action_dim,
        image_keys=image_keys or ["observation.images.front"],
    )

    # config.json + model.safetensors (weightless policy with dummy buffer)
    policy = VlaCppPolicy(cfg)
    policy.save_pretrained(out)

    # processor JSONs for make_pre_post_processors(pretrained_path=...)
    pre, post = make_vlacpp_pre_post_processors(cfg)
    pre.save_pretrained(out, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    post.save_pretrained(out, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
    return out


def build_parser() -> argparse.ArgumentParser:
    """The `vlacpp-build-pretrained` CLI.

    Every default here mirrors the matching `VlaCppConfig` field default;
    `test_cli_defaults_match_config_defaults` enforces that, so the CLI can no
    longer write a pretrained dir the config itself would reject.
    """
    p = argparse.ArgumentParser(description="Build a vlacpp pretrained dir for lerobot-rollout")
    p.add_argument("out_dir")
    p.add_argument("--server-address", default="tcp://localhost:5555")
    p.add_argument("--tokenizer-name", default="HuggingFaceTB/SmolVLM2-500M-Instruct")
    p.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="must match the GGUF's smolvla.image_size (256 for smolvla-ducks-v4); "
        "the old default of 512 produced a dir the server rejects",
    )
    p.add_argument("--max-state-dim", type=int, default=32)
    p.add_argument("--tokenizer-max-length", type=int, default=48)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=25,
        help="rows served per fetched chunk; must satisfy "
        "prefetch_threshold + n_action_steps <= chunk_size (validated on load)",
    )
    p.add_argument("--real-action-dim", type=int, default=6)
    p.add_argument("--image-keys", nargs="+", default=["observation.images.front"])
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    out = build_pretrained_dir(
        args.out_dir,
        server_address=args.server_address,
        tokenizer_name=args.tokenizer_name,
        image_size=args.image_size,
        max_state_dim=args.max_state_dim,
        tokenizer_max_length=args.tokenizer_max_length,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        real_action_dim=args.real_action_dim,
        image_keys=args.image_keys,
    )
    print(f"wrote pretrained dir: {out}")


if __name__ == "__main__":
    main()
