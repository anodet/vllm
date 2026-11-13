# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from typing import Any

import numpy as np

from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache, is_encoder_input

logger = init_logger(__name__)

# Debug aid, not needed for correctness: with VLLM_LOG_MM_LORA_SKIPS=1, log (at most
# once a minute) how many scheduled items the mapping has left out because the
# encoder skips them, and how many tower rows they would have added beyond the
# encoder input without that filter.
_LOG_SKIPS = os.environ.get("VLLM_LOG_MM_LORA_SKIPS") == "1"
_skip_stats = {"cached": 0, "no_data": 0, "rows": 0, "steps": 0, "logged": 0, "t": 0.0}


def _count_skipped(model: Any, mm_features, input_ids, encoder_outputs) -> None:
    stats = _skip_stats
    skipped = 0
    for i in input_ids:
        feature = mm_features[i]
        if feature.modality == "prompt_embeds" or is_encoder_input(
            feature, encoder_outputs
        ):
            continue
        skipped += 1
        stats["no_data" if feature.data is None else "cached"] += 1
        stats["rows"] += model.get_mm_lora_token_counts(
            modality=feature.modality,
            mm_kwargs=feature.data,
            num_mm_embeds=feature.mm_position.get_num_embeds(),
        )[0]
    if not skipped:
        return
    stats["steps"] += 1
    total = stats["cached"] + stats["no_data"]
    now = time.monotonic()
    if total > stats["logged"] and now - stats["t"] >= 60:
        logger.info(
            "Tower/connector LoRA mapping has left out %d already-cached and %d "
            "no-data scheduled items in %d steps (%d tower rows that the "
            "unfiltered mapping would have had beyond the encoder input)",
            stats["cached"],
            stats["no_data"],
            stats["steps"],
            stats["rows"],
        )
        stats["logged"], stats["t"] = total, now


def set_active_mm_loras(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> None:
    if (
        not scheduled_encoder_inputs
        or encoder_cache is None
        or not lora_manager.supports_tower_connector_lora()
    ):
        return

    prompt_lora_mapping: list[int] = []
    token_lora_mapping: list[int] = []
    lora_requests = set()
    encoder_token_counts: list[int] = []
    connector_token_counts: list[int] = []

    mm_items: list[tuple[int, MultiModalFeatureSpec]] = []
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            continue

        lora_id = int(lora_state.lora_ids[req_idx])
        mm_features = encoder_cache.mm_features[req_id]
        # Only the items the encoder runs (is_encoder_input): not items with no
        # payload, not items the worker already holds (an entry the scheduler
        # evicted and allocated again within one pass), and not the
        # "prompt_embeds" passthrough modality. A mapping with rows for items
        # the encoder skips makes the tower/connector LoRA kernels index past
        # the encoder input, an illegal memory access.
        mm_items.extend(
            (lora_id, mm_features[i])
            for i in encoder_input_ids
            if is_encoder_input(mm_features[i], encoder_cache.encoder_outputs)
        )
        if _LOG_SKIPS:
            _count_skipped(
                model, mm_features, encoder_input_ids, encoder_cache.encoder_outputs
            )

        if lora_id > 0:
            lora_request = lora_state.lora_requests.get(req_id)
            if lora_request is not None:
                lora_requests.add(lora_request)

    # Checked against the encoder's own list before it runs (check_lora_mapping).
    # A mapping without LoRA rows cannot overrun: the kernels return before
    # reading it. An empty one is checked too, since it leaves the previous
    # step's mapping in place.
    has_lora_rows = any(lora_id > 0 for lora_id, _ in mm_items)
    encoder_cache.lora_mapped_identifiers = (
        [f.identifier for _, f in mm_items] if has_lora_rows or not mm_items else None
    )

    # Runs items sorted by modality.
    mm_items.sort(key=lambda x: x[1].modality)
    for lora_id, mm_feature in mm_items:
        pos_info = mm_feature.mm_position

        tower_tokens, connector_tokens = model.get_mm_lora_token_counts(
            modality=mm_feature.modality,
            mm_kwargs=mm_feature.data,
            num_mm_embeds=pos_info.get_num_embeds(),
        )

        prompt_lora_mapping.append(lora_id)
        token_lora_mapping.extend([lora_id] * tower_tokens)
        encoder_token_counts.append(tower_tokens)
        connector_token_counts.append(connector_tokens)

    if not prompt_lora_mapping:
        return

    lora_manager.set_active_adapters(
        lora_requests,
        LoRAMapping(
            tuple(token_lora_mapping),
            tuple(prompt_lora_mapping),
            is_prefill=True,
            type=LoRAMappingType.TOWER,
        ),
    )

    mm_mapping = model.get_mm_mapping() if hasattr(model, "get_mm_mapping") else None
    if (
        mm_mapping is None
        or not mm_mapping.connector
        or not all(count is not None for count in connector_token_counts)
    ):
        return

    connector_token_mapping = np.repeat(
        np.array(prompt_lora_mapping, dtype=np.int32),
        np.array(connector_token_counts, dtype=np.int32),
    )
    lora_manager.set_active_adapters(
        lora_requests,
        LoRAMapping(
            index_mapping=tuple(connector_token_mapping.tolist()),
            prompt_mapping=tuple(prompt_lora_mapping),
            is_prefill=True,
            type=LoRAMappingType.CONNECTOR,
        ),
    )
