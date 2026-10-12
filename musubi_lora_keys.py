"""Load LoRA / DoRA files from the SECourses Musubi Trainer under the names the trainer saves.

ComfyUI registers some models' LoRA names only in the diffusers / SimpleTuner layout: Qwen-Image 2.1's MLP halves
(fused into one gate_up weight at load), Z-Image's q / k / v (slices of its fused qkv weight) and every HiDream-O1
layer (its checkpoint nests the model under "model."). The trainer saves "lora_unet_<module path>" for all of them,
so those layers were silently skipped. This registers the trainer's names as aliases of the targets ComfyUI already
knows (slices included); every name ComfyUI maps itself stays as it is. A DoRA "<name>.dora_scale" follows its layer.
"""

from __future__ import annotations

import logging

import comfy.lora
import comfy.model_base

_MARK = "_secourses_musubi_lora_keys"


def add_musubi_names(model, key_map: dict) -> int:
    """Add the trainer's names for the targets in ``key_map``; returns how many were added."""
    added = 0
    for key, target in list(key_map.items()):
        if key.startswith("lycoris_"):
            alias = "lora_unet_" + key[len("lycoris_"):]
            if alias not in key_map:
                key_map[alias] = target
                added += 1
    hidream_o1 = getattr(comfy.model_base, "HiDreamO1", None)
    if hidream_o1 is not None and isinstance(model, hidream_o1):
        for key, target in list(key_map.items()):
            if key.startswith("lora_unet_") and not key.startswith("lora_unet_model_"):
                alias = "lora_unet_model_" + key[len("lora_unet_"):]
                if alias not in key_map:
                    key_map[alias] = target
                    added += 1
    return added


def install() -> None:
    if getattr(comfy.lora, _MARK, False):
        return
    original = comfy.lora.model_lora_keys_unet

    def model_lora_keys_unet(model, key_map=None):
        key_map = original(model) if key_map is None else original(model, key_map)
        try:
            add_musubi_names(model, key_map)
        except Exception as exc:  # never block a LoRA load over the extra names
            logging.warning(f"SECourses: Musubi LoRA names not added: {exc}")
        return key_map

    model_lora_keys_unet.__wrapped__ = original
    comfy.lora.model_lora_keys_unet = model_lora_keys_unet
    setattr(comfy.lora, _MARK, True)
