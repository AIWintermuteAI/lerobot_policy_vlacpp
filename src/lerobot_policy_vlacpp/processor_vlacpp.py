"""Passthrough pre/post processors for the vlacpp policy.

Normalization is baked into the GGUF and applied by vla-server, so we deliberately
omit Normalizer/Unnormalizer and Tokenizer steps (the policy tokenizes itself).
The pipelines only rename observations, add a batch dimension, and place tensors
on the right device (and back to CPU on output).
"""

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    make_policy_processor_pipelines,
)

from .configuration_vlacpp import VlaCppConfig


def make_vlacpp_pre_post_processors(
    config: VlaCppConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    device = config.device or "cpu"
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=device),
    ]
    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
