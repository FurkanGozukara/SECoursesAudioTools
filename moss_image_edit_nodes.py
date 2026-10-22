"""MOSS-Image edits with ChenkinLab's Moss-Image-Edit LoRA on ComfyUI's native MOSS-Image model.

The adapter was trained with a reference branch: the latents of up to four reference images pass through every
MOSS-Image block at timestep 0 and attend only to each other, while the text and target tokens keep MOSS-Image's own
attention and also attend to the references' keys and values of the same block. The references never see the text or
the target, so their keys and values are computed once per sampling run and reused for every step and both CFG
branches. The LoRA itself (W + B @ A on q, k, v, o and the FFN) is loaded with the native LoRA loader.
"""

import torch

import comfy.ldm.flux.math
import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.ldm.moss_image.model import gated_residual, modulated_rms_norm

from .fibo2_nodes import gallery_images

KEY = "secourses_moss_image_edit"


def reference_size(width, height, max_edge):
    """The official reference size: scaled down (never up) to max_edge on the long side, each side rounded to 16."""
    limit = max_edge // 16 * 16
    scale = min(1.0, limit / max(width, height))
    return tuple(min(limit, max(16, int(value * scale / 16 + 0.5) * 16)) for value in (width, height))


def qkv(block, y):
    if block.fused_qkv:
        return block.proj["qkv"](y).chunk(3, dim=-1)
    return block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)


def norm_rope(block, q, k, pe):
    """q_norm / k_norm and split-half RoPE with the fused kernel, as comfy.ldm.moss_image.model.attention does."""
    B, N, _ = q.shape
    q, k = q.view(B, N, block.num_heads, -1), k.view(B, N, block.num_heads, -1)
    q_scale, _, q_stream = comfy.ops.cast_bias_weight(block.q_norm, q, offloadable=True)
    k_scale, _, k_stream = comfy.ops.cast_bias_weight(block.k_norm, k, offloadable=True)
    q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, block.q_norm.eps)
    comfy.ops.uncast_bias_weight(block.q_norm, q_scale, None, q_stream)
    comfy.ops.uncast_bias_weight(block.k_norm, k_scale, None, k_stream)
    return q.flatten(2), k.flatten(2)


def finish(block, x, out, mod):
    _, _, gate_a, shift_f, scale_f, gate_f = mod
    x = gated_residual(x, block.proj["o"](out), gate_a)
    y = modulated_rms_norm(x, block.norm2, scale_f, shift_f)
    return gated_residual(x, comfy.ops.linear_input_act(block.proj["ffn_down"], block.proj["ffn_up_gate"](y), "situ"), gate_f)


def attend(block, q, k, v, transformer_options):
    return optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), block.num_heads,
                               transformer_options=transformer_options, preferred_attention=block.comfy_attention)


def reference_rope(diffusion_model, sizes, device):
    """(r, f, h, w) positions of the references: r = 1, 2, ... per image, f = 0, rows and columns centered."""
    ids = []
    for index, (h, w) in enumerate(sizes):
        rows = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
        cols = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
        ids.append(torch.stack((torch.full_like(rows, index + 1), torch.zeros_like(rows), rows, cols), dim=-1))
    ids = torch.cat(ids).unsqueeze(0)
    pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000) for i, d in enumerate(diffusion_model.rope_axes_dim)], dim=-3)
    return pe.unsqueeze(1).transpose(1, 2).contiguous()


def reference_cache(diffusion_model, latents, device, dtype, transformer_options):
    """Keys and values of every block for the references, which run at timestep 0 and attend only to each other."""
    latents = [latent.to(device=device, dtype=dtype) for latent in latents]
    pe = reference_rope(diffusion_model, [latent.shape[-2:] for latent in latents], device)
    vec = diffusion_model.time_embedding(timestep_embedding(torch.zeros(1, device=device), 256).to(dtype))
    mod = diffusion_model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    x = torch.cat([diffusion_model.video_embedder(latent.flatten(2).transpose(1, 2)) for latent in latents], dim=1)
    cache = []
    for block in diffusion_model.layers:
        shift_a, scale_a = mod[0], mod[1]
        q, k, v = qkv(block, modulated_rms_norm(x, block.norm1, scale_a, shift_a))
        q, k = norm_rope(block, q, k, pe)
        cache.append((k, v))
        # the attention backend may consume its inputs; the cached keys and values must stay intact
        x = finish(block, x, attend(block, q, k.clone(), v.clone(), transformer_options), mod)
    return cache


class EditRun:
    """Reference keys and values of one sampling run (OUTER_SAMPLE scope)."""

    def __init__(self, latents):
        self.latents = latents
        self.cache = None


def outer_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(original)
    transformer_options = guider.model_options["transformer_options"]
    transformer_options[KEY] = EditRun(transformer_options[KEY].latents)
    try:
        return executor(*args, **kwargs)
    finally:
        guider.model_options = original


def diffusion_model_wrapper(executor, x, timestep, context, transformer_options, **kwargs):
    run = transformer_options[KEY]
    if run.cache is None:
        run.cache = reference_cache(executor.class_obj, run.latents, x.device, x.dtype, transformer_options)
    return executor(x, timestep, context, transformer_options, **kwargs)


def edit_block(diffusion_model, index):
    block = diffusion_model.layers[index]

    def forward(args, extra_args):
        x, mod, pe, transformer_options = args["img"], args["mod"], args["pe"], args["transformer_options"]
        ref_k, ref_v = transformer_options[KEY].cache[index]
        shift_a, scale_a = mod[0], mod[1]
        q, k, v = qkv(block, modulated_rms_norm(x, block.norm1, scale_a, shift_a))
        q, k = norm_rope(block, q, k, pe)
        batch = x.shape[0]
        k = torch.cat((k, ref_k.expand(batch, -1, -1)), dim=1)
        v = torch.cat((v, ref_v.expand(batch, -1, -1)), dim=1)
        return {"img": finish(block, x, attend(block, q, k, v, transformer_options), mod)}

    return forward


class SEMossImageEditReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MOSS-Image (SFT or Base) with the Moss-Image-Edit LoRA loaded."}),
                "vae": ("VAE",),
                "references": ("SECOURSES_REF_PACK", {"tooltip": "The gallery's images, in order: image 1, image 2, ... (up to 4 recommended)."}),
                "reference_max_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16,
                                               "tooltip": "Long side of each reference in pixels (scaled down only, multiples of 16). The result has the first reference's size: the adapter lines the result up with image 1 token by token. Official: 512."}),
            },
            "optional": {
                "width": ("INT", {"forceInput": True, "tooltip": "Result width without gallery images."}),
                "height": ("INT", {"forceInput": True, "tooltip": "Result height without gallery images."}),
            },
        }

    RETURN_TYPES = ("MODEL", "INT", "INT")
    RETURN_NAMES = ("model", "width", "height")
    FUNCTION = "apply"
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = ("Reference branch of ChenkinLab's Moss-Image-Edit: the gallery images become the images to edit or combine "
                   "(image 1, image 2, ... in the prompt). Load the Moss-Image-Edit LoRA on the model first. The result size is image 1's "
                   "reference size; a different size loses the layout and identity of image 1. Without gallery images the model passes "
                   "through unchanged with the width and height inputs.")

    def apply(self, model, vae, references, reference_max_edge=1024, width=1024, height=1024):
        images = gallery_images(references)
        if not images:
            return (model, width, height)
        latents, sizes = [], []
        for image in images:
            size = reference_size(image.shape[2], image.shape[1], reference_max_edge)
            if size != (image.shape[2], image.shape[1]):
                image = comfy.utils.common_upscale(image.movedim(-1, 1), *size, "lanczos", "disabled").movedim(1, -1)
            latents.append(model.model.process_latent_in(vae.encode(image[..., :3])))
            sizes.append(size)
        diffusion_model = model.get_model_object("diffusion_model")
        model = model.clone()
        model.model_options["transformer_options"][KEY] = EditRun(latents)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, outer_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, diffusion_model_wrapper)
        for index in range(len(diffusion_model.layers)):
            model.set_model_patch_replace(edit_block(diffusion_model, index), "dit", "single_block", index)
        return (model, *sizes[0])


NODE_CLASS_MAPPINGS = {"SEMossImageEditReferences": SEMossImageEditReferences}
NODE_DISPLAY_NAME_MAPPINGS = {"SEMossImageEditReferences": "MOSS-Image Edit References"}
