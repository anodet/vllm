# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.multimodal.utils import count_mm_batches
from vllm.v1.worker.gpu.mm import encoder_runner as encoder_runner_module
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache, is_encoder_input
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras


class _FakeModel:
    """Records calls to get_mm_lora_token_counts so the test can assert it
    is never invoked for the "prompt_embeds" passthrough modality."""

    def __init__(self):
        self.calls: list[str] = []

    def get_mm_lora_token_counts(self, *, modality, mm_kwargs, num_mm_embeds):
        del mm_kwargs
        self.calls.append(modality)
        return num_mm_embeds, num_mm_embeds

    def get_mm_mapping(self):
        return SimpleNamespace(connector=None)


class _FakeLoRAManager:
    def __init__(self):
        self.set_active_adapters_calls: list[tuple] = []

    def supports_tower_connector_lora(self) -> bool:
        return True

    def set_active_adapters(self, requests, mapping) -> None:
        self.set_active_adapters_calls.append((requests, mapping))


_FIELD = object()


def _kwargs_item():
    """A stand-in for a MultiModalKwargsItem: one key, the same field type for
    every item, so items of one modality batch together."""
    return {"pixel_values": SimpleNamespace(field=_FIELD, data=None)}


def _mm_feature(modality: str, num_embeds: int, identifier: str = "", data=True):
    """A scheduled item. `data=True` gives it a placeholder payload."""
    return SimpleNamespace(
        modality=modality,
        identifier=identifier or f"{modality}-{num_embeds}",
        data=_kwargs_item() if data is True else data,
        mm_position=SimpleNamespace(get_num_embeds=lambda: num_embeds),
    )


class _FakeModelWithConnector(_FakeModel):
    """Qwen3-VL-style counts: 4 tower rows (patches) per embedding, and a
    connector."""

    def get_mm_lora_token_counts(self, *, modality, mm_kwargs, num_mm_embeds):
        del mm_kwargs
        self.calls.append(modality)
        return num_mm_embeds * 4, num_mm_embeds

    def get_mm_mapping(self):
        return SimpleNamespace(connector=["visual.merger"])


def test_set_active_mm_loras_skips_prompt_embeds_passthrough():
    """`prompt_embeds` is a passthrough modality: the tensor is already in
    the model's embedding space, no encoder runs, and there is nothing to
    apply tower/connector LoRA to. set_active_mm_loras must skip it instead
    of forwarding it to the model's get_mm_lora_token_counts (which doesn't
    know about it and would raise "Unsupported modality")."""
    req_id = "req-0"
    model = _FakeModel()
    lora_manager = _FakeLoRAManager()
    encoder_cache = SimpleNamespace(
        mm_features={
            req_id: [
                _mm_feature("prompt_embeds", num_embeds=32),
                _mm_feature("image", num_embeds=4),
            ]
        },
        encoder_outputs={},
    )
    lora_state = SimpleNamespace(
        lora_ids=np.array([7], dtype=np.int32),
        lora_requests={req_id: object()},
    )

    set_active_mm_loras(
        model=model,
        lora_manager=lora_manager,
        encoder_cache=encoder_cache,
        req_id_to_index={req_id: 0},
        lora_state=lora_state,
        scheduled_encoder_inputs={req_id: [0, 1]},
    )

    # Only the real "image" item should ever reach the model.
    assert model.calls == ["image"]
    assert len(lora_manager.set_active_adapters_calls) == 1
    _, mapping = lora_manager.set_active_adapters_calls[0]
    assert mapping.index_mapping == (7, 7, 7, 7)


def _items_for_skip_cases():
    """Two requests' scheduled items covering every reason the encoder skips one.

    req-a (LoRA 7): a new image; an image the worker already holds (the
    scheduler evicted its entry and allocated it again in the same pass, so
    `get_freed_mm_hashes` kept the worker's tensor); an image without a
    payload; a new prompt_embeds passthrough item; another new image.
    req-b (no LoRA): a new image.
    """
    cached = _mm_feature("image", 5, "a-cached")
    features = {
        "req-a": [
            _mm_feature("image", 4, "a-new-1"),
            cached,
            _mm_feature("image", 6, "a-no-data", data=None),
            _mm_feature(
                "prompt_embeds",
                2,
                "a-prompt-embeds",
                data={"embedding": SimpleNamespace(data=torch.zeros(2, 8))},
            ),
            _mm_feature("image", 3, "a-new-2"),
        ],
        "req-b": [_mm_feature("image", 2, "b-new")],
    }
    encoder_outputs = {cached.identifier: torch.zeros(5, 8)}
    scheduled = {"req-a": [0, 1, 2, 3, 4], "req-b": [0]}
    lora_state = SimpleNamespace(
        lora_ids=np.array([7, 0], dtype=np.int32),
        lora_requests={"req-a": object()},
    )
    return features, encoder_outputs, scheduled, lora_state


def test_set_active_mm_loras_maps_only_items_the_encoder_runs():
    """The tower/connector mapping must have rows only for items the encoder
    runs. Rows for a skipped item (already cached, or without a payload) make
    the mapping longer than the encoder input, and the LoRA kernels then index
    past it: a CUDA illegal memory access."""
    features, encoder_outputs, scheduled, lora_state = _items_for_skip_cases()
    encoder_cache = EncoderCache()
    encoder_cache.mm_features = features
    encoder_cache.encoder_outputs = encoder_outputs
    model = _FakeModelWithConnector()
    lora_manager = _FakeLoRAManager()

    set_active_mm_loras(
        model=model,
        lora_manager=lora_manager,
        encoder_cache=encoder_cache,
        req_id_to_index={"req-a": 0, "req-b": 1},
        lora_state=lora_state,
        scheduled_encoder_inputs=scheduled,
    )

    assert encoder_cache.lora_mapped_identifiers == ["a-new-1", "a-new-2", "b-new"]
    assert model.calls == ["image", "image", "image"]
    (_, tower), (_, connector) = lora_manager.set_active_adapters_calls
    # 4 and 3 embeddings for LoRA 7, then 2 for the base model (id 0)
    assert tower.prompt_mapping == (7, 7, 0)
    assert tower.index_mapping == (7,) * 16 + (7,) * 12 + (0,) * 8
    assert connector.index_mapping == (7,) * 4 + (7,) * 3 + (0,) * 2


def test_set_active_mm_loras_sets_nothing_when_the_encoder_runs_nothing():
    """Every scheduled item is already cached: the encoder runs nothing, so no
    mapping is set, and the guard expects an empty encoder input."""
    encoder_cache = EncoderCache()
    item = _mm_feature("image", 4, "cached")
    encoder_cache.mm_features = {"req-a": [item]}
    encoder_cache.encoder_outputs = {"cached": torch.zeros(4, 8)}
    lora_manager = _FakeLoRAManager()

    set_active_mm_loras(
        model=_FakeModelWithConnector(),
        lora_manager=lora_manager,
        encoder_cache=encoder_cache,
        req_id_to_index={"req-a": 0},
        lora_state=SimpleNamespace(
            lora_ids=np.array([7], dtype=np.int32), lora_requests={}
        ),
        scheduled_encoder_inputs={"req-a": [0]},
    )

    assert lora_manager.set_active_adapters_calls == []
    assert encoder_cache.lora_mapped_identifiers == []
    encoder_cache.check_lora_mapping([], [])  # nothing encoded: passes


def test_base_model_only_steps_are_not_checked():
    """A step whose items all use the base model gives a mapping with no LoRA
    rows, which the LoRA kernels never read, so it is not checked: a
    mixed-modality base-model step must keep working."""
    encoder_cache = EncoderCache()
    encoder_cache.mm_features = {
        "req-b": [_mm_feature("image", 4, "img"), _mm_feature("video", 6, "vid")]
    }
    set_active_mm_loras(
        model=_FakeModelWithConnector(),
        lora_manager=_FakeLoRAManager(),
        encoder_cache=encoder_cache,
        req_id_to_index={"req-b": 0},
        lora_state=SimpleNamespace(
            lora_ids=np.array([0], dtype=np.int32), lora_requests={}
        ),
        scheduled_encoder_inputs={"req-b": [0, 1]},
    )
    assert encoder_cache.lora_mapped_identifiers is None
    # two encoder batches (image, video), but no LoRA rows: no error
    encoder_cache.check_lora_mapping(
        ["img", "vid"], [("image", _kwargs_item()), ("video", _kwargs_item())]
    )


def test_execute_mm_encoder_checks_the_mapping_before_encoding():
    """ModelState.execute_mm_encoder must run check_lora_mapping between
    prepare_mm_inputs and the encoder, so a mismatched mapping raises before
    any LoRA kernel runs."""
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    encoded = []
    encoder_runner = SimpleNamespace(
        prepare_mm_inputs=lambda scheduled: (["a"], [("image", _kwargs_item())]),
        execute_mm_encoder=lambda mm_kwargs: encoded.append(mm_kwargs) or [None],
    )
    encoder_cache = EncoderCache()
    encoder_cache.lora_mapped_identifiers = ["a", "b"]  # mapping covers more
    state = SimpleNamespace(encoder_runner=encoder_runner, encoder_cache=encoder_cache)
    with pytest.raises(RuntimeError, match="does not match the encoder input"):
        ModelState.execute_mm_encoder(state, {"req-a": [0]})
    assert encoded == []


def test_mapping_covers_exactly_what_prepare_mm_inputs_encodes(monkeypatch):
    """The mapping (built first) and the encoder's own list (built by
    prepare_mm_inputs right after) must name the same items in the same order,
    which check_lora_mapping then verifies."""
    monkeypatch.setattr(
        encoder_runner_module, "async_tensor_h2d", lambda t, device: t
    )
    features, encoder_outputs, scheduled, lora_state = _items_for_skip_cases()
    encoder_cache = EncoderCache()
    encoder_cache.mm_features = features
    encoder_cache.encoder_outputs = encoder_outputs

    set_active_mm_loras(
        model=_FakeModelWithConnector(),
        lora_manager=_FakeLoRAManager(),
        encoder_cache=encoder_cache,
        req_id_to_index={"req-a": 0, "req-b": 1},
        lora_state=lora_state,
        scheduled_encoder_inputs=scheduled,
    )
    runner = SimpleNamespace(encoder_cache=encoder_cache, device="cpu")
    mm_hashes, mm_kwargs = EncoderRunner.prepare_mm_inputs(runner, scheduled)

    assert mm_hashes == ["a-new-1", "a-new-2", "b-new"]
    # the passthrough item was cached, not encoded
    assert "a-prompt-embeds" in encoder_cache.encoder_outputs
    encoder_cache.check_lora_mapping(mm_hashes, mm_kwargs)
    assert encoder_cache.lora_mapped_identifiers is None


def test_is_encoder_input_matches_prepare_mm_inputs(monkeypatch):
    """Every combination of payload, cached and modality: is_encoder_input
    (used by the LoRA mapping) agrees with what prepare_mm_inputs encodes."""
    monkeypatch.setattr(
        encoder_runner_module, "async_tensor_h2d", lambda t, device: t
    )
    combos = [
        (has_data, cached, modality)
        for has_data in (True, False)
        for cached in (True, False)
        for modality in ("image", "video", "prompt_embeds")
    ]
    for has_data, cached, modality in combos:
        data = (
            {"embedding": SimpleNamespace(data=torch.zeros(1, 8))}
            if modality == "prompt_embeds"
            else _kwargs_item()
        )
        item = _mm_feature(modality, 1, "x", data=data if has_data else None)
        encoder_cache = EncoderCache()
        encoder_cache.mm_features = {"r": [item]}
        if cached:
            encoder_cache.encoder_outputs["x"] = torch.zeros(1, 8)
        expected = is_encoder_input(item, encoder_cache.encoder_outputs)
        runner = SimpleNamespace(encoder_cache=encoder_cache, device="cpu")
        mm_hashes, _ = EncoderRunner.prepare_mm_inputs(runner, {"r": [0]})
        assert (mm_hashes == ["x"]) == expected, (has_data, cached, modality)


def test_check_lora_mapping_raises_on_mismatch():
    encoder_cache = EncoderCache()
    encoder_cache.check_lora_mapping(["a"], [("image", object())])  # unset: no-op

    encoder_cache.lora_mapped_identifiers = ["a", "b"]
    with pytest.raises(RuntimeError, match="does not match the encoder input"):
        encoder_cache.check_lora_mapping(["a"], [("image", object())])
    assert encoder_cache.lora_mapped_identifiers is None  # cleared either way


def test_check_lora_mapping_raises_on_several_encoder_batches():
    """One mapping per step cannot describe two encoder calls."""
    encoder_cache = EncoderCache()
    encoder_cache.lora_mapped_identifiers = ["a", "b"]
    with pytest.raises(RuntimeError, match="2 encoder batches"):
        encoder_cache.check_lora_mapping(
            ["a", "b"], [("video", object()), ("image", object())]
        )


def test_count_mm_batches():
    def item(key):
        return {key: SimpleNamespace(field=object(), data=None)}

    assert count_mm_batches([]) == 0
    assert count_mm_batches([("image", item("pixel_values"))] * 3) == 1
    # different fields cannot share a batch, nor can different modalities
    assert (
        count_mm_batches(
            [
                ("image", item("pixel_values")),
                ("image", item("image_embeds")),
                ("video", item("pixel_values_videos")),
            ]
        )
        == 3
    )
