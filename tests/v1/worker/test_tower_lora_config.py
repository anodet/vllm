# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Settings that split the vision tower's input behind the per-step
tower/connector LoRA mapping must be refused at start-up."""
from types import SimpleNamespace

import pytest

from vllm.config.vllm import VllmConfig
from vllm.model_executor.models import vision


def _config(*, tower=True, cudagraph=False, eager=False, tp_mode="weights", tp=1):
    return SimpleNamespace(
        lora_config=SimpleNamespace(enable_tower_connector_lora=tower),
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=cudagraph),
        model_config=SimpleNamespace(
            enforce_eager=eager,
            multimodal_config=SimpleNamespace(mm_encoder_tp_mode=tp_mode),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


def test_tower_lora_refuses_encoder_cudagraphs():
    with pytest.raises(ValueError, match="cudagraph_mm_encoder"):
        VllmConfig._validate_tower_connector_lora(_config(cudagraph=True))
    # eager mode never captures encoder graphs; without tower LoRA it is fine
    VllmConfig._validate_tower_connector_lora(_config(cudagraph=True, eager=True))
    VllmConfig._validate_tower_connector_lora(_config(cudagraph=True, tower=False))


def test_tower_lora_refuses_data_parallel_vision_encoder():
    with pytest.raises(ValueError, match="mm_encoder_tp_mode"):
        VllmConfig._validate_tower_connector_lora(_config(tp_mode="data", tp=2))
    VllmConfig._validate_tower_connector_lora(_config(tp_mode="data", tp=1))
    VllmConfig._validate_tower_connector_lora(_config(tp_mode="weights", tp=2))
    VllmConfig._validate_tower_connector_lora(_config(tower=False, tp_mode="data", tp=2))


def test_tower_lora_refuses_silent_data_parallel_fallback(monkeypatch):
    """Vision heads not divisible by TP make vLLM fall back to a data-parallel
    vision encoder; with tower LoRA that must be an error, not a warning."""
    monkeypatch.setattr(vision, "get_tensor_model_parallel_world_size", lambda: 2)
    fake = SimpleNamespace(
        lora_config=SimpleNamespace(enable_tower_connector_lora=True), model_config=None
    )
    monkeypatch.setattr(vision, "get_current_vllm_config_or_none", lambda: fake)
    with pytest.raises(ValueError, match="divisible"):
        vision.is_vit_use_data_parallel(num_heads=3)
    assert vision.is_vit_use_data_parallel(num_heads=16) is False

    fake.lora_config = None
    assert vision.is_vit_use_data_parallel(num_heads=3) is True
