"""Use the SECourses gallery with Anima's native encoders and the Anima reference adapter.

The reference adapter is adapted from ElectronGirl/ComfyUI-Anima-Ref-Adapter at 0171144 (Apache-2.0;
docs/AnimaRefAdapter-LICENSE.txt). Modified: one gallery guider instead of three nodes, the adapter LoRA is
hooked only for the duration of a reference forward call (no global adapter cache), the reference is padded
to the patch size, and the half-size reference for half-resolution samplers is not encoded.
"""

import importlib
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.conds
import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers
import nodes

from .fibo2_nodes import gallery_images

REF_KEY = "se_anima_ref_latent"


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


def _adapter_weights(lora_name, strength):
    weights = {}
    tensors = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
    for key, down in tensors.items():
        if key.endswith(".lora_down.weight"):
            up = tensors[key.replace("lora_down", "lora_up")]
            alpha = float(tensors.get(key.replace("lora_down.weight", "alpha"), down.shape[0]))
            weights[key[len("diffusion_model."):-len(".lora_down.weight")]] = [down, up, strength * alpha / down.shape[0]]
    if not weights:
        raise ValueError(f"{lora_name} has no lora_down/lora_up weights; choose the Anima reference adapter.")
    return weights


def _hook_adapter(diffusion_model, weights):
    """Add the adapter LoRA to the output of its linear layers; the caller removes the returned hooks."""
    def hook(entry):
        def add_lora(module, inputs, output):
            x = inputs[0]
            if entry[0].device != x.device or entry[0].dtype != x.dtype:
                entry[0] = entry[0].to(device=x.device, dtype=x.dtype)
                entry[1] = entry[1].to(device=x.device, dtype=x.dtype)
            return output + F.linear(F.linear(x, entry[0]), entry[1]) * entry[2]
        return add_lora
    return [diffusion_model.get_submodule(name).register_forward_hook(hook(entry)) for name, entry in weights.items()]


def _ref_rope(pe, hp, wp, device):
    """VideoRopePosition3DEmb.generate_embeddings for the target frame (t 0) and the reference frame (t 1),
    whose rows start one target height lower so the model cannot copy it pixel-aligned."""
    hh, ww = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing="ij")
    t = torch.cat([torch.zeros_like(hh).flatten(), torch.ones_like(hh).flatten()])
    h = torch.cat([hh.flatten(), hh.flatten() + hp])
    w = torch.cat([ww.flatten(), ww.flatten()])
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _ref_forward(dm, x, timesteps, context, ref, transformer_options):
    """MiniTrainDIT._forward with the reference as a second frame at timestep 0."""
    orig_shape = list(x.shape)
    patch = (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, patch)
    if ref.ndim == 4:
        ref = ref.unsqueeze(2)
    ref = comfy.ldm.common_dit.pad_to_patch_size(ref.to(device=x.device, dtype=x.dtype), patch)
    x_emb, _, extra = dm.prepare_embedded_sequence(torch.cat([x[:, :, :1], ref[:, :, :1]], dim=2), fps=None, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape
    rope = _ref_rope(dm.pos_embedder, hp, wp, x.device)

    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    timesteps = torch.cat([timesteps, torch.zeros_like(timesteps)], dim=1)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](timesteps).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    kwargs = {"rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0), "adaln_lora_B_T_3D": adaln,
              "extra_per_block_pos_emb": extra, "transformer_options": transformer_options}
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()  # residual stream in fp32, as MiniTrainDIT._forward does for fp16
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        x_emb = block(x_emb, t_emb, context, **kwargs)
    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    return dm.unpatchify(out)[:, :, :1][:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


def _ref_wrapper(weights):
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        ref = kwargs.pop(REF_KEY, None)
        if ref is None:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        hooks = _hook_adapter(executor.class_obj, weights)
        try:
            return _ref_forward(executor.class_obj, x, timesteps, context, ref, kwargs.get("transformer_options", {}))
        finally:
            for handle in hooks:
                handle.remove()
    return wrapper


def _ref_extra_conds(extra_conds, process_latent_in):
    # Object patches can stay applied on the shared model, so wrap the original function, not an earlier patch.
    extra_conds = getattr(extra_conds, "_se_anima_inner", extra_conds)

    def patched(**kwargs):
        out = extra_conds(**kwargs)
        if kwargs.get(REF_KEY) is not None:
            out[REF_KEY] = comfy.conds.CONDRegular(process_latent_in(kwargs[REF_KEY]))
        return out
    patched._se_anima_inner = extra_conds
    return patched


def _letterbox(image, width, height):
    """The reference fitted inside the output size on white, as the adapter was trained."""
    src = Image.fromarray((image[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
    scale = min(width / src.width, height / src.height)
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(src.resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)


class _AnimaRefGuider(comfy.samplers.CFGGuider):
    """Text CFG and reference guidance as separate terms, with APG or CFG-rescale correction."""

    def set_reference(self, positive, negative, reference, cfg, ref_guidance, correction, rescale_phi):
        self.cfg, self.ref_guidance, self.correction, self.rescale_phi = cfg, ref_guidance, correction, rescale_phi
        negative = node_helpers.conditioning_set_values(negative, {"prompt_type": "negative"})
        self.inner_set_conds({
            "positive": node_helpers.conditioning_set_values(positive, {REF_KEY: reference}),
            "negative": node_helpers.conditioning_set_values(negative, {REF_KEY: reference}),
            "negative_base": negative,
        })

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        conds = [self.conds.get("positive"), self.conds.get("negative")]
        if self.ref_guidance != 1.0:  # at 1 the reference is in both branches: plain CFG, no base branch
            conds.append(self.conds.get("negative_base"))
        out = comfy.samplers.calc_cond_batch(self.inner_model, conds, x, timestep, model_options)
        pos, neg = out[0], out[1]
        base = out[2] if len(out) > 2 else neg
        # x0 space: base + ref_guidance * (reference effect) + cfg * (text effect given the reference)
        guided = base + self.ref_guidance * (neg - base) + self.cfg * (pos - neg)
        dims = tuple(range(1, pos.ndim))
        if self.correction == "apg":
            update = (guided - pos).double()
            unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
            guided = pos + (update - (update * unit).sum(dim=dims, keepdim=True) * unit).to(pos.dtype)
        elif self.correction == "rescale":
            ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True)
            guided = self.rescale_phi * guided * ratio + (1 - self.rescale_phi) * guided
        # Post-CFG model patches (e.g. Anima Safe PAG), called as comfy.samplers.cfg_function does.
        for fn in model_options.get("sampler_post_cfg_function", []):
            guided = fn({"denoised": guided, "cond": conds[0], "uncond": conds[1], "cond_scale": self.cfg,
                         "model": self.inner_model, "uncond_denoised": neg, "cond_denoised": pos, "sigma": timestep,
                         "model_options": model_options, "input": x})
        return guided


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
    DESCRIPTION = "Text to image or single-image character/style reference with the built-in Anima reference adapter: white letterboxing, separate reference guidance and APG. Ordinary optional LoRAs belong before this node. Use BF16 compute; fp16 overflows."

    def build(self, model, positive, negative, vae, references, width, height,
              use_reference, cfg, adapter_lora, adapter_strength, ref_guidance, correction, rescale_phi, batch_size=1):
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if not use_reference:
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(cfg)
            return guider, {"samples": latent}
        images = gallery_images(references)
        if not images:
            raise ValueError("Add one reference image to the gallery, or turn Use Reference off for text to image.")
        model = model.clone()
        model.add_object_patch("extra_conds", _ref_extra_conds(model.get_model_object("extra_conds"), model.get_model_object("process_latent_in")))
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "se_anima_ref_adapter",
                                   _ref_wrapper(_adapter_weights(adapter_lora, adapter_strength)))
        reference = vae.encode(_letterbox(images[0], width, height))  # one image: the Wan VAE reads a batch as video frames
        if batch_size > 1:
            reference = reference.repeat(batch_size, *([1] * (reference.ndim - 1)))
        guider = _AnimaRefGuider(model)
        guider.set_reference(positive, negative, reference, cfg, ref_guidance, correction, rescale_phi)
        return guider, {"samples": latent}


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
