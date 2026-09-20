"""AvatarForever's AR forward on native ComfyUI LTX-2.3 modules.

Adapted from leeruibin/avatarforever at 4dfc42b, transformer/transformer.py
and transformer_args.py (LTX-2 Community License; docs/AvatarForever-LICENSE.txt).
Modified to use native ComfyUI modules and operations from the loaded model clone.
No global hooks or changes to ComfyUI's model classes.
"""

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.model_prefetch
import comfy.quant_ops
from comfy.ldm.common_dit import rms_norm
from comfy.ldm.lightricks.av_model import CompressedTimestep


def slice_tokens(tensor, selection):
    if isinstance(tensor, CompressedTimestep):
        tensor = tensor.expand()
    return tensor[:, selection] if tensor.shape[1] != 1 else tensor


def slice_rope(pe, selection):
    # Native LTX RoPE: (rotation [B, tokens, heads, half_dim, 2, 2], split).
    return pe[0][:, selection], pe[1]


def history(tensor, selection, dim=1):
    return torch.cat((tensor.narrow(dim, 0, selection.start),
                      tensor.narrow(dim, selection.stop, tensor.shape[dim] - selection.stop)), dim=dim)


def cached_block(block, args, cache, current, reuse, cache_device):
    """Official chunk-local ForeverCache, including the uncached reference path."""
    vx, ax = args["img"]
    vs, aus = current
    to = args["transformer_options"]
    vt, at = args["v_timestep"], args["a_timestep"]
    vpe, ape = args["v_pe"], args["a_pe"]
    vcpe, acpe = args["v_cross_pe"], args["a_cross_pe"]
    if reuse:
        vx, ax = vx[:, vs].contiguous(), ax[:, aus].contiguous()
        vt, at = slice_tokens(vt, vs), slice_tokens(at, aus)
        vpe, ape = slice_rope(vpe, vs), slice_rope(ape, aus)
        vcpe, acpe = slice_rope(vcpe, vs), slice_rope(acpe, aus)

    def context(name, x, pe, selection):
        if reuse:
            old = cache[name]
            full_pe = args[{"vs": "v_pe", "as": "a_pe", "vc": "v_cross_pe", "ac": "a_cross_pe"}[name]]
            return (torch.cat((old.to(x.device), x), dim=1),
                    (torch.cat((history(full_pe[0], selection), pe[0]), dim=1), pe[1]))
        if cache is not None:
            cache[name] = history(x, selection).to(cache_device).clone()
        return x, pe

    def self_and_text(x, table, ts, attn, text_attn, text, prompt_table, prompt_ts, pe, name, selection):
        shift, scale, gate = block.get_ada_values(table, x.shape[0], ts, slice(0, 3))
        norm = comfy.quant_ops.ck.rms_adaln(x, scale, shift)
        ctx, kpe = context(name, norm, pe, selection)
        x = x + attn(norm, context=ctx, pe=pe, k_pe=kpe, transformer_options=to) * gate
        return x + block._apply_text_cross_attention(x, text, text_attn, table, prompt_table, ts,
                                                     prompt_ts, args["attention_mask"], to)

    vx = self_and_text(vx, block.scale_shift_table, vt, block.attn1, block.attn2,
                       args["v_context"], block.prompt_scale_shift_table, args["v_prompt_timestep"], vpe, "vs", vs)
    # Cache the history BEFORE cross-modal attention, as in the released model.
    vctx, vctxpe = context("vc", vx, vcpe, vs)
    ax = self_and_text(ax, block.audio_scale_shift_table, at, block.audio_attn1, block.audio_attn2,
                       args["a_context"], block.audio_prompt_scale_shift_table, args["a_prompt_timestep"], ape, "as", aus)
    actx, actxpe = context("ac", ax, acpe, aus)
    vnorm, anorm = rms_norm(vx), rms_norm(ax)

    def cross_values(table, ts, gates, selection, batch):
        scale, shift = block.get_ada_values(table[:4], batch, ts, selection)
        gate = block.get_ada_values(table[4:], batch, gates)[0]
        return scale, shift, gate

    vscale, vshift, vgate = cross_values(block.scale_shift_table_a2v_ca_video,
        args["v_cross_scale_shift_timestep"], args["v_cross_gate_timestep"], slice(0, 2), vx.shape[0])
    ascale, ashift, _ = cross_values(block.scale_shift_table_a2v_ca_audio,
        args["a_cross_scale_shift_timestep"], args["a_cross_gate_timestep"], slice(0, 2), ax.shape[0])
    vx = vx + block.audio_to_video_attn(vnorm * (1 + vscale) + vshift,
        context=rms_norm(actx) * (1 + ascale) + ashift, pe=vcpe, k_pe=actxpe,
        transformer_options=to) * vgate

    # Upstream reuse attends to current video AFTER a2v; the uncached path uses
    # its pre-a2v norm. This asymmetry is deliberate in the released cache.
    if reuse:
        vctx = torch.cat((cache["vc"].to(vx.device), vx), dim=1)
    vscale, vshift, _ = cross_values(block.scale_shift_table_a2v_ca_video,
        args["v_cross_scale_shift_timestep"], args["v_cross_gate_timestep"], slice(2, 4), vx.shape[0])
    ascale, ashift, agate = cross_values(block.scale_shift_table_a2v_ca_audio,
        args["a_cross_scale_shift_timestep"], args["a_cross_gate_timestep"], slice(2, 4), ax.shape[0])
    ax = ax + block.video_to_audio_attn(anorm * (1 + ascale) + ashift,
        context=rms_norm(vctx) * (1 + vscale) + vshift, pe=acpe, k_pe=vctxpe,
        transformer_options=to) * agate

    for x, table, ts, ff in ((vx, block.scale_shift_table, vt, block.ff),
                              (ax, block.audio_scale_shift_table, at, block.audio_ff)):
        shift, scale, gate = block.get_ada_values(table, x.shape[0], ts, slice(3, 6))
        x.addcmul_(ff(comfy.quant_ops.ck.rms_adaln(x, scale, shift)), gate)
    if reuse:
        # The native output unpatchifier expects the complete selected window.
        full_v, full_a = args["img"]
        full_v[:, vs], full_a[:, aus] = vx, ax
        return full_v, full_a
    return vx, ax


def avatar_forward(dm, x, timestep, context, attention_mask=None, frame_rate=25,
                   transformer_options=None, **kwargs):
    """Run on the sampler's loaded model, including after a non-dynamic rebuild."""
    opts = transformer_options
    state = opts["avatarforever"]
    sigma = state["sigma"]
    vs, aus = state["current"]
    hidden, coords, extra = dm._process_input(x, None, None)
    vx, ax = hidden
    frame_tokens = x[0].shape[3] * x[0].shape[4]
    if state["channel"] is not None:
        condition = dm.patchifier.patchify(state["channel"])[0]
        proj, gate_w, gate_b = state["weights"]
        condition = F.linear(condition, proj.to(vx))
        if state["channel_mode"] == "gated":
            condition = condition * (2 * torch.sigmoid(F.linear(condition, gate_w.to(vx), gate_b.to(vx))))
        vx[:, vs] += condition.repeat(1, (vs.stop - vs.start) // frame_tokens, 1)

    vi = torch.tensor(state["video_positions"], device=vx.device).repeat_interleave(frame_tokens)
    coords[0][:, 0, :, 0] = (vi * 8 - 7).clamp_min(0)
    coords[0][:, 0, :, 1] = vi * 8 + 1
    ai = torch.tensor(state["audio_positions"], device=ax.device)
    ap = dm.a_patchifier
    for side in (0, 1):
        mel = (ai + side) * ap.audio_latent_downsample_factor
        if ap.is_causal:
            mel = (mel + 1 - ap.audio_latent_downsample_factor).clamp_min(0)
        coords[1][:, 0, :, side] = mel * ap.hop_length / ap.sample_rate

    ts, embedded, _ = dm._prepare_timestep(timestep[0], vx.shape[0], vx.dtype,
                                           a_timestep=timestep[1], **extra)
    # Official modality.sigma stays sigma even with a zero audio denoise mask.
    # Both cross-modality scale/shift and gates use the OTHER stream's scalar sigma.
    def embed(module, value):
        result, _ = module(value.reshape(-1), {"resolution": None, "aspect_ratio": None},
                           batch_size=vx.shape[0], hidden_dtype=vx.dtype)
        return result.reshape(vx.shape[0], -1, result.shape[-1])

    scaled = sigma * dm.timestep_scale_multiplier
    cross_scaled = sigma * dm.av_ca_timestep_scale_multiplier
    ts[2] = [embed(dm.av_ca_audio_scale_shift_adaln_single, scaled),
             embed(dm.av_ca_video_scale_shift_adaln_single, scaled),
             embed(dm.av_ca_a2v_gate_adaln_single, cross_scaled),
             embed(dm.av_ca_v2a_gate_adaln_single, cross_scaled)]
    ts[3], ts[4] = embed(dm.prompt_adaln_single, scaled), embed(dm.audio_prompt_adaln_single, scaled)
    contexts, attention_mask = dm._prepare_context(context, vx.shape[0], hidden, attention_mask)
    attention_mask = dm._prepare_attention_mask(attention_mask, vx.dtype)
    pe = dm._prepare_positional_embeddings(coords, frame_rate, vx.dtype)
    args = dict(v_context=contexts[0], a_context=contexts[1], attention_mask=attention_mask,
                v_timestep=ts[0], a_timestep=ts[1], v_pe=pe[0][0], a_pe=pe[1][0],
                v_cross_pe=pe[0][1], a_cross_pe=pe[1][1],
                a_cross_scale_shift_timestep=ts[2][0], v_cross_scale_shift_timestep=ts[2][1],
                v_cross_gate_timestep=ts[2][2], a_cross_gate_timestep=ts[2][3],
                v_prompt_timestep=ts[3], a_prompt_timestep=ts[4], transformer_options=opts)
    caches = state["cache"]
    reuse = caches is not None and bool(caches)
    queue = comfy.model_prefetch.make_prefetch_queue(list(dm.transformer_blocks), vx.device, opts)
    for index, block in enumerate(dm.transformer_blocks):
        mm.throw_exception_if_processing_interrupted()
        comfy.model_prefetch.prefetch_queue_pop(queue, vx.device, block, malloc_scope="block")
        args["img"] = vx, ax
        entry = None if caches is None else caches.setdefault(index, {})
        vx, ax = cached_block(block, args, entry, (vs, aus), reuse, state["cache_device"])
    comfy.model_prefetch.prefetch_queue_pop(queue, vx.device, None, malloc_scope="block")
    return dm._process_output([vx, ax], embedded, None, **extra)
