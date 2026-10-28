"""Use the SECourses gallery with Anima's native encoders and reference adapter."""

import importlib
import random
import re

import torch

import comfy.model_management
import comfy.samplers
import folder_paths
import nodes

from .fibo2_nodes import gallery_images


class SEAnimaGalleryPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references": ("SECOURSES_REF_PACK",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "prepare"
    CATEGORY = "SECourses/Anima"
    DESCRIPTION = "Gallery prompt, added prompts and .txt folder batches. Resolves {a|b} choices with the linked seed; keeps native prompt weights and schedules."

    def prepare(self, references, seed=0):
        text = references.get("prompt", "")
        rng = random.Random(seed)
        choices = re.compile(r"(?<!\\)\{([^{}]*\|[^{}]*)\}")
        while choices.search(text):
            text = choices.sub(lambda match: rng.choice(match.group(1).split("|")), text)
        return (text,)


class SEAnimaGalleryGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "references": ("SECOURSES_REF_PACK",),
            "width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 16}),
            "use_reference": ("BOOLEAN", {"default": True, "tooltip": "On: use the first gallery image for character and style. Off: ordinary Anima text to image. The adapter is loaded only on the reference path."}),
            "cfg": ("FLOAT", {"default": 4.5, "min": 0, "max": 30, "step": 0.1}),
            "adapter_lora": (folder_paths.get_filename_list("loras"), {"default": "anima_ref_adapter_pilot_r32.safetensors"}),
            "adapter_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05}),
            "ref_guidance": ("FLOAT", {"default": 4.5, "min": 0, "max": 20, "step": 0.1}),
            "correction": (["apg", "rescale", "none"],),
            "rescale_phi": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
        }, "optional": {"batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}

    RETURN_TYPES = ("GUIDER", "LATENT")
    RETURN_NAMES = ("guider", "latent")
    FUNCTION = "build"
    CATEGORY = "SECourses/Anima"
    DESCRIPTION = "Text to image or single-image character/style reference. Uses the native Anima Ref Adapter nodes, with white letterboxing and separate reference guidance. Ordinary optional LoRAs belong before this node. Use BF16 compute; fp16 overflows."

    def build(self, model, positive, negative, vae, references, width, height,
              use_reference, cfg, adapter_lora, adapter_strength, ref_guidance, correction, rescale_phi, batch_size=1):
        if not use_reference:
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(cfg)
            latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            return guider, {"samples": latent}
        images = gallery_images(references)
        if not images:
            raise ValueError("Add one reference image to the gallery, or turn Use Reference off for text to image.")
        required = ("AnimaRefAdapterLoader", "AnimaRefLatent", "AnimaRefGuider")
        if any(name not in nodes.NODE_CLASS_MAPPINGS for name in required):
            raise ValueError("Install Anima Ref Adapter with the SECourses custom-node installer, then restart ComfyUI.")
        model, = nodes.NODE_CLASS_MAPPINGS[required[0]]().load(model, adapter_lora, adapter_strength, 1.0)
        reference, latent, _ = nodes.NODE_CLASS_MAPPINGS[required[1]]().encode(images[0], vae, width, height, 255)
        if batch_size > 1:
            reference = {key: value.repeat(batch_size, *([1] * (value.ndim - 1))) for key, value in reference.items()}
            latent["samples"] = latent["samples"].repeat(batch_size, 1, 1, 1)
        guider, = nodes.NODE_CLASS_MAPPINGS[required[2]]().get_guider(
            model, positive, negative, reference, cfg, ref_guidance, correction, rescale_phi)
        return guider, latent


class _AnimaNoise:
    def __init__(self, seed, variation_seed, variation_strength, add_noise):
        self.seed, self.variation_seed, self.variation_strength = seed, variation_seed, variation_strength
        self.add_noise = add_noise

    def generate_noise(self, input_latent):
        latent = input_latent["samples"]
        if not self.add_noise:
            return torch.zeros_like(latent, device="cpu")
        sampler = importlib.import_module(nodes.NODE_CLASS_MAPPINGS["SwarmKSampler"].__module__)
        return sampler.swarm_fixed_noise(self.seed, latent, self.variation_seed, self.variation_strength)


class SEAnimaNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "variation_strength": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
            "add_noise": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "build"
    CATEGORY = "SECourses/Anima"

    def build(self, noise_seed, variation_seed, variation_strength, add_noise):
        return (_AnimaNoise(noise_seed, variation_seed, variation_strength, add_noise),)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SEAnimaGalleryPrompt, SEAnimaGalleryGuider, SEAnimaNoise)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEAnimaGalleryPrompt": "Anima Gallery Prompt",
    "SEAnimaGalleryGuider": "Anima Gallery Reference / Text Guider",
    "SEAnimaNoise": "Anima Seed + Variation Noise",
}
