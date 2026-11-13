# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import replace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
from vllm.multimodal.utils import count_mm_batches


def is_new_encoder_input(
    mm_feature: MultiModalFeatureSpec, encoder_outputs: Mapping[str, torch.Tensor]
) -> bool:
    """Whether a scheduled item has a payload and its output is not cached yet.

    A scheduled item can already be cached on the worker: the scheduler may evict
    an entry and allocate it again within one scheduling pass, and then keeps the
    worker's tensor instead of freeing it (`EncoderCacheManager.get_freed_mm_hashes`).
    """
    return mm_feature.data is not None and mm_feature.identifier not in encoder_outputs


def is_encoder_input(
    mm_feature: MultiModalFeatureSpec, encoder_outputs: Mapping[str, torch.Tensor]
) -> bool:
    """Whether `EncoderRunner.prepare_mm_inputs` passes this scheduled item to the
    encoder: a new input (see `is_new_encoder_input`) that is not the
    `prompt_embeds` passthrough modality.

    The tower/connector LoRA mapping (`gpu/mm/lora.py`) must cover exactly these
    items, in this order. The LoRA kernels take each adapter's rows from the
    mapping, so a mapping with more rows than the encoder input makes them read
    and write past it (a CUDA illegal memory access).
    """
    return (
        is_new_encoder_input(mm_feature, encoder_outputs)
        and mm_feature.modality != "prompt_embeds"
    )


class EncoderCache:
    def __init__(self):
        # req_id -> MM features
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash -> encoder outputs
        self.encoder_outputs: dict[str, torch.Tensor] = {}
        # The identifiers this step's tower/connector LoRA mapping covers, in
        # scheduling order. Set by `set_active_mm_loras`, checked and cleared
        # by `check_lora_mapping` before the encoder runs; None when no such
        # mapping was set.
        self.lora_mapped_identifiers: list[str] | None = None

    def __len__(self) -> int:
        return len(self.encoder_outputs)

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        # The uniprocess executor can share this list with EngineCore.
        self.mm_features[req_id] = list(mm_features)

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def update_encoder_inputs(
        self, req_id: str, updates: dict[int, MultiModalKwargsItem | None]
    ) -> None:
        features = self.mm_features[req_id]
        for input_id, data in updates.items():
            # Replace entries so EngineCore keeps its original payload for replay.
            features[input_id] = replace(features[input_id], data=data)

    def reset_mm_cache(self) -> None:
        """Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # NOTE: v2 encoder cache profiling skips the multi-modal cache
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)

    def check_lora_mapping(
        self,
        mm_hashes: list[str],
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    ) -> None:
        """Raise before the encoder runs if this step's tower/connector LoRA
        mapping does not cover exactly the items about to be encoded
        (`mm_hashes`, from `prepare_mm_inputs`), or if those items need more
        than one `embed_multimodal` call: one mapping is set per step, so each
        call would see rows of the others. Either way the LoRA kernels would
        index past the encoder input, corrupting memory instead of failing here.

        Splits inside one call (encoder CUDA graphs, a data-parallel vision
        encoder) are not visible here; neither is supported with tower LoRA."""
        mapped = self.lora_mapped_identifiers
        if mapped is None:
            return
        self.lora_mapped_identifiers = None
        if mapped != mm_hashes:
            raise RuntimeError(
                "Tower/connector LoRA mapping does not match the encoder input: "
                f"it covers {len(mapped)} items, the encoder runs {len(mm_hashes)}. "
                f"Mapped only: {sorted(set(mapped) - set(mm_hashes))[:4]}; "
                f"encoded only: {sorted(set(mm_hashes) - set(mapped))[:4]}."
            )
        if mm_kwargs:
            num_batches = count_mm_batches(sorted(mm_kwargs, key=lambda x: x[0]))
            if num_batches > 1:
                raise RuntimeError(
                    "Tower/connector LoRA sets one mapping per step, but this "
                    f"step's {len(mm_kwargs)} encoder inputs need {num_batches} "
                    "encoder batches (different modalities or fields)."
                )
