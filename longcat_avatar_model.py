"""LongCat-Video-Avatar 1.6 diffusion transformer, ported for native ComfyUI loading.

The architecture follows meituan-longcat/LongCat-Video (longcat_video/modules/avatar/
longcat_video_dit_avatar.py, attention.py, blocks.py, rope_3d.py; commit 58eb336) and keeps
its parameter names, so the official BF16 shards load without renaming. Differences that do
not change the arithmetic:

* attention runs through ComfyUI's optimized attention (SageAttention when ComfyUI was
  started with it, PyTorch SDPA otherwise) instead of flash-attn;
* linears are ComfyUI ops, so low-VRAM offload, dynamic VRAM and native INT8 work;
* row-wise operations (norm + modulation, FFN, projections) run in token chunks, which bounds
  peak memory at large sizes and gives the same values;
* condition tokens (reference + previous frames) do not depend on the noisy tokens, so their
  per-block keys and values can be cached across denoising steps (exact; optional).

Context parallel and block-sparse attention (used only by the 720p refinement of LongCat-Video,
not by the avatar model) are not ported.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.modules.attention as comfy_attention
import comfy.ops

# Token rows processed at once by the row-wise parts of a block (65536 rows x 4096 channels x 4 bytes = 1 GiB of
# FP32 temporaries per chunk). Whole segments up to 720p run unchunked; 1920p runs in about six frame chunks.
ROW_CHUNK = 65536


def _attention(q, k, v):
    """q, k, v: [B, H, N, D] -> [B, H, N, D] with ComfyUI's active attention backend."""
    heads = q.shape[1]
    out = comfy_attention.optimized_attention(q, k, v, heads, skip_reshape=True, skip_output_reshape=True)
    return out


def _linear_as(module, x, dtype):
    """Run a ComfyUI Linear in a given compute dtype (the official code uses fp32 autocast)."""
    x = x.to(dtype)
    with comfy.ops.CastBiasWeightContext(module, x, offloadable=True) as (weight, bias):
        return F.linear(x, weight, bias)


def _chunks(n, size=ROW_CHUNK):
    for start in range(0, n, size):
        yield start, min(n, start + size)


class RMSNormFP32(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        # official: (x.float() normalized).type_as(x) * weight; the fused kernel accumulates in fp32 and rounds once
        out = F.rms_norm(x, (x.shape[-1],), eps=self.eps)
        weight = comfy.ops.cast_to_input(self.weight, out, copy=False)
        return out * weight


class LayerNormFP32(nn.Module):
    """LayerNorm computed in fp32 (official LayerNorm_FP32)."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.weight is not None and x.dtype in (torch.bfloat16, torch.float16) and x.is_cuda:
            # The fused kernel accumulates in fp32 and rounds once: the official fp32 LayerNorm + cast,
            # without a full fp32 copy of x (bf16 weights are exact in fp32).
            return F.layer_norm(x, (self.dim,), self.weight.to(device=x.device, dtype=x.dtype),
                                self.bias.to(device=x.device, dtype=x.dtype), self.eps)
        weight = None if self.weight is None else self.weight.to(device=x.device, dtype=torch.float32)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=torch.float32)
        return F.layer_norm(x.float(), (self.dim,), weight, bias, self.eps).to(x.dtype)


def _mod_ln_eager(x, shift, scale, eps: float = 1e-6):
    out = F.layer_norm(x.float(), (x.shape[-1],), None, None, eps)
    return (out * (scale + 1) + shift).to(x.dtype)


def _gate_res_eager(x, gate, y):
    return (x.float() + gate * y.float()).to(x.dtype)


class _Fused:
    """One-kernel versions of the fp32 elementwise chains (torch.compile); same math, eager fallback on any failure."""

    def __init__(self, fn):
        self.eager = fn
        self.compiled = None
        self.failed = False

    def __call__(self, *args):
        if not self.failed and args[0].is_cuda:
            try:
                if self.compiled is None:
                    self.compiled = torch.compile(self.eager, fullgraph=True, dynamic=True)
                return self.compiled(*args)
            except Exception as error:  # no Triton / compiler: keep the identical eager math
                self.failed = True
                import logging
                logging.warning("[LongCat-Avatar] fused kernels unavailable, using eager ops: %s", str(error)[:200])
        return self.eager(*args)


mod_ln = _Fused(_mod_ln_eager)
gate_res = _Fused(_gate_res_eager)


def _modulate(norm, x, shift, scale):
    """Official modulate_fp32: norm in fp32, x * (1 + scale) + shift, back to x.dtype."""
    if getattr(norm, "weight", None) is None and getattr(norm, "eps", None) == 1e-6:
        return mod_ln(x, shift, scale)
    dtype = x.dtype
    out = norm(x.to(torch.float32))
    return torch.addcmul(shift, out, scale + 1).to(dtype)


# ----------------------------------------------------------------------------- RoPE

def _rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


_ROPE_CACHE = {}


def rope_3d_cos_sin(head_dim, grid, device, frame_index=None, num_ref_latents=None):
    """cos/sin [N, head_dim] in fp32 for a (T, H, W) grid (official RotaryPositionalEmbedding)."""
    key = (head_dim, tuple(grid), frame_index, num_ref_latents, device.type, device.index)
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    num_frames, height, width = grid
    base = 10000
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    freqs_t = 1.0 / (base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
    if frame_index is not None and num_ref_latents is not None:
        grid_t = torch.cat([torch.tensor([float(frame_index)]),
                            torch.arange(0, num_frames - num_ref_latents, dtype=torch.float32)])
    else:
        grid_t = torch.arange(0, num_frames, dtype=torch.float32)
    grid_h = torch.arange(0, height, dtype=torch.float32)
    grid_w = torch.arange(0, width, dtype=torch.float32)
    ft = torch.repeat_interleave(grid_t[:, None] * freqs_t[None], 2, dim=-1)
    fh = torch.repeat_interleave(grid_h[:, None] * freqs_h[None], 2, dim=-1)
    fw = torch.repeat_interleave(grid_w[:, None] * freqs_w[None], 2, dim=-1)
    T, H, W = grid_t.shape[0], height, width
    freqs = torch.cat([
        ft[:, None, None, :].expand(T, H, W, ft.shape[-1]),
        fh[None, :, None, :].expand(T, H, W, fh.shape[-1]),
        fw[None, None, :, :].expand(T, H, W, fw.shape[-1]),
    ], dim=-1).reshape(T * H * W, -1)
    freqs = freqs.to(device)
    cos, sin = freqs.cos(), freqs.sin()
    cached = (cos, sin, rope_matrices(cos, sin))
    if len(_ROPE_CACHE) > 16:
        _ROPE_CACHE.clear()
    _ROPE_CACHE[key] = cached
    return cached


def rope_matrices(cos, sin):
    """[N, D] interleaved cos/sin -> [1, 1, N, D/2, 2, 2] fp32 rotation matrices [[c, -s], [s, c]] (flux layout)."""
    c, s = cos[:, 0::2], sin[:, 0::2]
    return torch.stack([c, -s, s, c], dim=-1).reshape(1, 1, c.shape[0], c.shape[1], 2, 2).contiguous()


def apply_rope(x, cos, sin, mats=None):
    """x [B, H, N, D]; official rotate_half RoPE computed in fp32, returned in x.dtype.

    Uses ComfyUI's apply_rope1 (fused comfy_kitchen kernel when available) on the same interleaved pairs:
    out = (x0 c - x1 s, x0 s + x1 c), exactly the official q * cos + rotate_half(q) * sin.
    """
    if mats is None:
        mats = rope_matrices(cos, sin)
    try:
        from comfy.ldm.flux.math import apply_rope1
        return apply_rope1(x, mats)
    except Exception:
        out = torch.empty_like(x)
        for start, end in _chunks(x.shape[2], 8192):
            xs = x[:, :, start:end].float()
            out[:, :, start:end] = (xs * cos[start:end] + _rotate_half(xs) * sin[start:end]).to(x.dtype)
        return out


def rope_1d(x, pos):
    """Official RotaryPositionalEmbedding1D; x [B, H, N, D], pos [N]."""
    head_dim = x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    freqs = freqs.to(pos.device)
    freqs = torch.repeat_interleave(pos.float()[:, None] * freqs[None], 2, dim=-1)
    xs = x.float()
    out = xs * freqs.cos() + _rotate_half(xs) * freqs.sin()
    return out.type_as(x)


# ----------------------------------------------------------------------------- modules

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=True, device=device, dtype=dtype)
        self.q_norm = RMSNormFP32(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = RMSNormFP32(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def qkv_heads(self, x):
        """x [B, N, C] -> q, k, v [B, H, N, D] after RMS norms (chunked over tokens)."""
        B, N, C = x.shape
        q = torch.empty((B, self.num_heads, N, self.head_dim), dtype=x.dtype, device=x.device)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        for start, end in _chunks(N):
            part = self.qkv(x[:, start:end]).view(B, end - start, 3, self.num_heads, self.head_dim)
            part = part.permute(2, 0, 3, 1, 4)
            q[:, :, start:end] = self.q_norm(part[0])
            k[:, :, start:end] = self.k_norm(part[1])
            v[:, :, start:end] = part[2]
        return q, k, v

    def out_proj(self, x):
        """x [B, H, N, D] -> [B, N, C]."""
        B, H, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, N, H * D)
        out = torch.empty((B, N, self.dim), dtype=x.dtype, device=x.device)
        for start, end in _chunks(N):
            out[:, start:end] = self.proj(x[:, start:end])
        return out


class TextCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.kv_linear = operations.Linear(dim, dim * 2, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.q_norm = RMSNormFP32(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = RMSNormFP32(self.head_dim, eps=1e-6, device=device, dtype=dtype)

    def text_kv(self, cond):
        """cond [B, L, C] -> k, v [B, H, L, D]; constant across steps and blocks' calls."""
        B, L, _ = cond.shape
        kv = self.kv_linear(cond).view(B, L, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        return k, v

    def forward(self, x, text_k, text_v):
        """x [B, N, C] (noisy tokens), every batch item attends to its own L text tokens."""
        B, N, C = x.shape
        out = torch.empty_like(x)
        for start, end in _chunks(N):
            q = self.q_linear(x[:, start:end]).view(B, end - start, self.num_heads, self.head_dim)
            q = self.q_norm(q).transpose(1, 2)
            o = _attention(q, text_k, text_v).transpose(1, 2).reshape(B, end - start, C)
            out[:, start:end] = self.proj(o)
        return out


class AudioCrossAttention(nn.Module):
    """Official SingleStreamAttention: each latent frame's tokens attend to that frame's audio tokens."""

    def __init__(self, dim, encoder_hidden_states_dim, num_heads, class_range=24, class_interval=4,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.q_norm = RMSNormFP32(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.kv_linear = operations.Linear(encoder_hidden_states_dim, dim * 2, bias=True, device=device, dtype=dtype)
        self.k_norm = RMSNormFP32(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, class_interval)
        self.rope_h2 = (class_range - class_interval, class_range)
        self.rope_bak = int(class_range // 2)

    def forward(self, x, cond, frames, x_ref_attn_map=None, human_num=None):
        """x [B, frames*S, C] noisy tokens; cond [B*frames, M, Ca] audio tokens."""
        B, N, C = x.shape
        S = N // frames
        xf = x.reshape(B * frames, S, C)
        BF = xf.shape[0]
        q = self.q_linear(xf).view(BF, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        if x_ref_attn_map is not None:
            q = self._multi_rope_q(q, x_ref_attn_map, frames, B)
        M = cond.shape[1]
        kv = self.kv_linear(cond).view(BF, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)
        if x_ref_attn_map is not None:
            k = self._multi_rope_k(k, x_ref_attn_map, frames, B, human_num)
        o = _attention(q, k, v)
        o = o.transpose(1, 2).reshape(BF, S, C)
        o = self.proj(o)
        return o.reshape(B, N, C)

    # Multi-person position encoding (official _process_cross_attn with x_ref_attn_map).
    def _multi_rope_q(self, q, x_ref_attn_map, frames, B):
        max_values = x_ref_attn_map.max(1).values[:, None, None]
        min_values = x_ref_attn_map.min(1).values[:, None, None]
        max_min = torch.cat([max_values, min_values], dim=2)
        h1_max, h1_min = max_min[0, :, 0].max(), max_min[0, :, 1].min()
        h2_max, h2_min = max_min[1, :, 0].max(), max_min[1, :, 1].min()

        def scale(col, lo, hi, new_lo, new_hi):
            return (col - lo) / (hi - lo + 1e-8) * (new_hi - new_lo) + new_lo

        human1 = scale(x_ref_attn_map[0], h1_min, h1_max, *self.rope_h1)
        human2 = scale(x_ref_attn_map[1], h2_min, h2_max, *self.rope_h2)
        background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
        back = torch.full((x_ref_attn_map.size(1),), background_pos, dtype=human1.dtype, device=human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0).clamp(max=2)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        pos = normalized_map[torch.arange(x_ref_attn_map.size(1), device=normalized_map.device), max_indices]
        BF, H, S, D = q.shape
        qq = q.reshape(B, frames, H, S, D).permute(0, 2, 1, 3, 4).reshape(B, H, frames * S, D)
        qq = rope_1d(qq, pos)
        return qq.reshape(B, H, frames, S, D).permute(0, 2, 1, 3, 4).reshape(BF, H, S, D)

    def _multi_rope_k(self, k, x_ref_attn_map, frames, B, human_num):
        BF, H, M, D = k.shape
        per_frame = torch.zeros(M, dtype=k.dtype, device=k.device)
        h1 = (self.rope_h1[0] + self.rope_h1[1]) / 2
        h2 = (self.rope_h2[0] + self.rope_h2[1]) / 2
        if human_num is not None and human_num > 2:
            background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
            per = M // human_num
            per_frame[:per] = h1
            per_frame[per:2 * per] = h2
            per_frame[2 * per:] = background_pos
        else:
            per_frame[:M // 2] = h1
            per_frame[M // 2:] = h2
        pos = per_frame.repeat(frames)
        kk = k.reshape(B, frames, H, M, D).permute(0, 2, 1, 3, 4).reshape(B, H, frames * M, D)
        kk = rope_1d(kk, pos)
        return kk.reshape(B, H, frames, M, D).permute(0, 2, 1, 3, 4).reshape(BF, H, M, D)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AudioProjModel(nn.Module):
    def __init__(self, seq_len=5, seq_len_vf=8, blocks=5, channels=1280, intermediate_dim=512,
                 output_dim=768, context_tokens=32, device=None, dtype=None, operations=None):
        super().__init__()
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.proj1 = operations.Linear(seq_len * blocks * channels, intermediate_dim, device=device, dtype=dtype)
        self.proj1_vf = operations.Linear(seq_len_vf * blocks * channels, intermediate_dim, device=device, dtype=dtype)
        self.proj2 = operations.Linear(intermediate_dim, intermediate_dim, device=device, dtype=dtype)
        self.proj3 = operations.Linear(intermediate_dim, context_tokens * output_dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(output_dim, device=device, dtype=dtype)

    def forward(self, first, latter):
        """first [B, 1, W, S, C]; latter [B, T-1, W', S, C] -> [B, T, 32, 768]."""
        B = first.shape[0]
        frames = first.shape[1] + latter.shape[1]
        a = torch.relu(self.proj1(first.reshape(B * first.shape[1], -1)))
        b = torch.relu(self.proj1_vf(latter.reshape(B * latter.shape[1], -1)))
        c = torch.cat([a.view(B, -1, a.shape[-1]), b.view(B, -1, b.shape[-1])], dim=1)
        c = torch.relu(self.proj2(c.reshape(B * frames, -1)))
        tokens = self.proj3(c).reshape(B * frames, self.context_tokens, self.output_dim)
        tokens = self.norm(tokens)
        return tokens.reshape(B, frames, self.context_tokens, self.output_dim)


class AvatarBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, adaln_tembed_dim, output_dim=768,
                 class_range=24, class_interval=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.hidden_size = hidden_size
        ops = operations
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), ops.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True, device=device, dtype=dtype))
        self.audio_adaLN_modulation = nn.Sequential(nn.SiLU(), ops.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True, device=device, dtype=dtype))
        self.mod_norm_attn = LayerNormFP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNormFP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNormFP32(hidden_size, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        self.pre_video_crs_attn_norm = LayerNormFP32(hidden_size, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        self.attn = SelfAttention(hidden_size, num_heads, device=device, dtype=dtype, operations=ops)
        self.cross_attn = TextCrossAttention(hidden_size, num_heads, device=device, dtype=dtype, operations=ops)
        self.audio_cross_attn = AudioCrossAttention(hidden_size, output_dim, num_heads, class_range, class_interval,
                                                    device=device, dtype=dtype, operations=ops)
        self.ffn = FeedForwardSwiGLU(hidden_size, int(hidden_size * mlp_ratio), device=device, dtype=dtype, operations=ops)

    def _mod(self, linear, t, n):
        out = _linear_as(linear, F.silu(t.float()), torch.float32)  # [B, T, n*C]
        return out.unsqueeze(2).chunk(n, dim=-1)

    def _self_attention(self, q, k, v, grid, num_cond, num_ref, ref_img_index, mask_frame_range, cache):
        """Official Attention.forward split logic. Returns [B, H, N(or noisy), D].

        cache: None (no caching), {"store": True} to record condition keys/values, or
        {"k": k_cond, "v": v_cond} to reuse them (q/k/v hold only noisy tokens then).
        """
        T, Hh, Ww = grid
        per = Hh * Ww
        if cache is not None and "k" in cache:
            k_full = torch.cat([cache["k"], k], dim=2)
            v_full = torch.cat([cache["v"], v], dim=2)
            return self._noise_attention(q, k_full, v_full, per, num_cond, num_ref, ref_img_index,
                                         mask_frame_range, T - num_cond), None
        if num_cond == 0:
            return _attention(q, k, v), None
        nc = num_cond * per
        if num_cond == 1:
            x_cond = _attention(q[:, :, :nc], k[:, :, :nc], v[:, :, :nc])
        else:
            nr = per  # one reference latent
            x_ref = _attention(q[:, :, :nr], k[:, :, :nr], v[:, :, :nr])
            x_prev = _attention(q[:, :, nr:nc], k[:, :, nr:nc], v[:, :, nr:nc])
            x_cond = torch.cat([x_ref, x_prev], dim=2)
        if cache is not None and cache.get("store"):
            cache["k"], cache["v"] = k[:, :, :nc].contiguous(), v[:, :, :nc].contiguous()
            cache.pop("store", None)
        if num_cond == T:
            return None, x_cond
        x_noise = self._noise_attention(q[:, :, nc:], k, v, per, num_cond, num_ref, ref_img_index,
                                        mask_frame_range, T - num_cond)
        return x_noise, x_cond

    @staticmethod
    def _noise_attention(q_noise, k, v, per, num_cond, num_ref, ref_img_index, mask_frame_range, num_noisy):
        if num_cond > 1 and mask_frame_range is not None and mask_frame_range > 0 and ref_img_index is not None:
            start = ref_img_index - mask_frame_range - num_cond + num_ref
            end = ref_img_index + mask_frame_range - num_cond + num_ref + 1
            if start >= 0 and end > start and end <= num_noisy:
                s, e = start * per, end * per
                k_non_ref, v_non_ref = k[:, :, per:], v[:, :, per:]
                front = _attention(q_noise[:, :, :s], k, v) if s > 0 else None
                masked = _attention(q_noise[:, :, s:e], k_non_ref, v_non_ref)
                back = _attention(q_noise[:, :, e:], k, v) if e < q_noise.shape[2] else None
                return torch.cat([p for p in (front, masked, back) if p is not None], dim=2)
        return _attention(q_noise, k, v)

    def forward(self, x, text_kv, t, grid, num_cond, num_ref, audio_tokens, ref_img_index, mask_frame_range,
                rope, cache=None, ref_maps=None, human_num=None):
        """x [B, N, C] (N = all tokens, or only noisy tokens when cache holds the condition)."""
        B, N, C = x.shape
        T, Hh, Ww = grid
        per = Hh * Ww
        cached = cache is not None and "k" in cache
        t_rows = t[:, num_cond:] if cached else t  # [B, T_x, Ct]
        Tx = t_rows.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._mod(self.adaLN_modulation[1], t_rows, 6)

        # self attention
        xm = torch.empty_like(x)
        xv = x.view(B, Tx, per, C)
        for f0, f1 in _chunks(Tx, max(1, ROW_CHUNK // per)):
            xm.view(B, Tx, per, C)[:, f0:f1] = _modulate(self.mod_norm_attn, xv[:, f0:f1], shift_msa[:, f0:f1], scale_msa[:, f0:f1])
        q, k, v = self.attn.qkv_heads(xm)
        del xm
        cos, sin, mats = rope
        if cached:
            off = num_cond * per
            q = apply_rope(q, cos[off:], sin[off:], mats[:, :, off:])
            k = apply_rope(k, cos[off:], sin[off:], mats[:, :, off:])
        else:
            q = apply_rope(q, cos, sin, mats)
            k = apply_rope(k, cos, sin, mats)
        x_noise, x_cond = self._self_attention(q, k, v, grid, num_cond, num_ref, ref_img_index, mask_frame_range, cache)
        ref_attn_map = None
        if ref_maps is not None and x_noise is not None:
            ref_attn_map = attn_map_with_target(q[:, :, -x_noise.shape[2]:], k, grid, ref_maps)
        del q, k, v
        if cached or x_cond is None:
            attn_out = x_noise  # cached condition, or no condition at all (audio + text)
        elif x_noise is None:
            attn_out = x_cond
        else:
            attn_out = torch.cat([x_cond, x_noise], dim=2)
        del x_noise, x_cond
        attn_out = self.attn.out_proj(attn_out)
        # Residual updates run in frame chunks, in place: every op below is per token (text keys/values
        # are shared, each frame's audio tokens belong to that frame), so chunking keeps the exact values
        # while the fp32 temporaries stay small at large sizes.
        xv = x.view(B, Tx, per, C)
        av = attn_out.view(B, Tx, per, C)
        fstep = max(1, ROW_CHUNK // per)
        for f0, f1 in _chunks(Tx, fstep):
            xv[:, f0:f1] = gate_res(xv[:, f0:f1], gate_msa[:, f0:f1], av[:, f0:f1])
        del attn_out, av

        noisy_frames = T - num_cond
        f_off = 0 if cached else num_cond  # first noisy frame inside x
        if noisy_frames > 0:
            a_shift, a_scale, a_gate = self._mod(self.audio_adaLN_modulation[1], t[:, num_cond:], 3)
            tk, tv = text_kv
            M, Ca = audio_tokens.shape[1], audio_tokens.shape[2]
            at = audio_tokens.view(B, noisy_frames, M, Ca)
            for f0, f1 in _chunks(noisy_frames, fstep):
                nf = f1 - f0
                flat = xv[:, f_off + f0:f_off + f1].reshape(B, nf * per, C)
                # text cross attention (condition tokens receive nothing)
                flat = flat + self.cross_attn(self.pre_crs_attn_norm(flat), tk, tv)
                # audio cross attention, modulated by the noisy frames' timesteps
                ao = self.audio_cross_attn(self.pre_video_crs_attn_norm(flat), at[:, f0:f1].reshape(B * nf, M, Ca), nf,
                                           None, human_num)
                ao = _modulate(self.mod_norm_attn, ao.view(B, nf, per, C), a_shift[:, f0:f1], a_scale[:, f0:f1])
                xv[:, f_off + f0:f_off + f1] = gate_res(flat.view(B, nf, per, C), a_gate[:, f0:f1], ao)
                del flat, ao

        # feed forward
        for f0, f1 in _chunks(Tx, fstep):
            xm = _modulate(self.mod_norm_ffn, xv[:, f0:f1], shift_mlp[:, f0:f1], scale_mlp[:, f0:f1])
            h = self.ffn(xm.reshape(B, -1, C)).view(B, f1 - f0, per, C)
            xv[:, f0:f1] = gate_res(xv[:, f0:f1], gate_mlp[:, f0:f1], h)
        return x


def attn_map_with_target(q_noise, k, grid, ref_target_masks):
    """Official get_attn_map_with_target (multi-person); q [B,H,Nq,D], k [B,H,Nk,D]."""
    T, Hh, Ww = grid
    per = Hh * Ww
    ref_k = k[:, :, :per]  # reference image tokens
    scale = 1.0 / math.sqrt(q_noise.shape[-1])
    maps = []
    for mask in ref_target_masks:  # [per]
        idx = mask > 0
        total = None
        for start, end in _chunks(q_noise.shape[2], 8192):
            att = torch.matmul(q_noise[:, :, start:end].float() * scale, ref_k.float().transpose(-1, -2)).softmax(-1)
            val = att[..., idx].sum(-1).mean(1)[0]  # [chunk]
            total = val if total is None else torch.cat([total, val])
        maps.append(total)
    return torch.stack(maps, dim=0)


class LongCatVideoAvatarTransformer(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, hidden_size=4096, depth=48, num_heads=32,
                 caption_channels=4096, mlp_ratio=4, adaln_tembed_dim=512, frequency_embedding_size=256,
                 patch_size=(1, 2, 2), text_tokens_zero_pad=True, audio_window=5, audio_block=5,
                 audio_channel=1280, intermediate_dim=512, output_dim=768, context_tokens=32, vae_scale=4,
                 class_range=24, class_interval=4, image_model=None, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        ops = operations
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vae_scale = vae_scale
        self.audio_window = audio_window
        self.text_tokens_zero_pad = text_tokens_zero_pad
        self.dtype = dtype
        self.frequency_embedding_size = frequency_embedding_size

        self.x_embedder = nn.Module()
        self.x_embedder.proj = ops.Conv3d(in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size,
                                          device=device, dtype=dtype)
        self.t_embedder = nn.Module()
        self.t_embedder.mlp = nn.Sequential(
            ops.Linear(frequency_embedding_size, adaln_tembed_dim, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(adaln_tembed_dim, adaln_tembed_dim, bias=True, device=device, dtype=dtype))
        self.y_embedder = nn.Module()
        self.y_embedder.y_proj = nn.Sequential(
            ops.Linear(caption_channels, hidden_size, bias=True, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype))
        self.blocks = nn.ModuleList([
            AvatarBlock(hidden_size, num_heads, mlp_ratio, adaln_tembed_dim, output_dim, class_range, class_interval,
                        device=device, dtype=dtype, operations=ops)
            for _ in range(depth)])
        self.audio_proj = AudioProjModel(seq_len=audio_window, seq_len_vf=audio_window + vae_scale - 1,
                                         blocks=audio_block, channels=audio_channel, intermediate_dim=intermediate_dim,
                                         output_dim=output_dim, context_tokens=context_tokens,
                                         device=device, dtype=dtype, operations=ops)
        self.final_layer = nn.Module()
        self.final_layer.norm_final = LayerNormFP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.final_layer.linear = ops.Linear(hidden_size, math.prod(self.patch_size) * out_channels, bias=True,
                                             device=device, dtype=dtype)
        self.final_layer.adaLN_modulation = nn.Sequential(
            nn.SiLU(), ops.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True, device=device, dtype=dtype))

    # --------------------------------------------------------------- embeddings
    def compute_dtype(self):
        w = self.x_embedder.proj.weight
        dtype = getattr(w, "dtype", None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.bfloat16
        return dtype

    def timestep_embed(self, timestep):
        """timestep [B, T] (0-1000) -> [B, T, 512] fp32 (official TimestepEmbedder under fp32 autocast)."""
        B, T = timestep.shape
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timestep.device) / half)
        args = timestep.float().flatten()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = _linear_as(self.t_embedder.mlp[0], emb, torch.float32)
        h = F.silu(h)
        h = _linear_as(self.t_embedder.mlp[2], h, torch.float32)
        return h.reshape(B, T, -1)

    def text_embed(self, context, mask=None):
        """context [B, L, 4096] umt5 hidden states; mask [B, L] (1 = real token)."""
        dtype = self.compute_dtype()
        y = self.y_embedder.y_proj(context.to(dtype))
        if self.text_tokens_zero_pad and mask is not None:
            y = y * mask.to(y.dtype)[:, :, None]
        return y

    def audio_tokens(self, audio_embs, num_frames, num_ref_latents):
        """audio_embs [B, F_video, W, S, Ca] -> [B*T, 32, 768] tokens for the T latent frames (official forward)."""
        dtype = self.compute_dtype()
        audio = audio_embs.to(dtype)
        first = audio[:, :1]
        latter = audio[:, 1:]
        B, Fm1, Wn = latter.shape[0], latter.shape[1], latter.shape[2]
        latter = latter.reshape(B, Fm1 // self.vae_scale, self.vae_scale, Wn, *latter.shape[3:])
        mid = self.audio_window // 2
        first_of = latter[:, :, :1, :mid + 1].flatten(2, 3)
        last_of = latter[:, :, -1:, mid:].flatten(2, 3)
        middle = latter[:, :, 1:-1, mid:mid + 1].flatten(2, 3)
        latter_s = torch.cat([first_of, middle, last_of], dim=2)
        tokens = self.audio_proj(first, latter_s)  # [B, T_audio, 32, 768]
        if num_ref_latents:
            tokens = torch.cat([tokens[:, :1], tokens], dim=1)
        tokens = tokens[:, -num_frames:]
        return tokens

    # --------------------------------------------------------------- forward
    def forward(self, x, timestep, text_y, audio_tok, num_cond=0, num_ref=0, ref_img_index=None,
                mask_frame_range=None, text_kv_cache=None, cond_cache=None, ref_target_masks=None, human_num=None):
        """Velocity prediction.

        x [B, 16, T, H, W] latents (all frames, condition first); timestep [B, T] in 0..1000;
        text_y [B, L, C] from text_embed; audio_tok [B, T, 32, 768] or multi-person [1, T, P*32, 768].
        text_kv_cache: optional dict block index -> (k, v), filled on first use.
        cond_cache: optional dict block index -> condition key/value cache (see AvatarBlock).
        Returns [B, 16, T, H, W] float32 (only noisy frames are meaningful).
        """
        B, _, T, H, W = x.shape
        Nt, Nh, Nw = T, H // self.patch_size[1], W // self.patch_size[2]
        per = Nh * Nw
        dtype = self.compute_dtype()
        if ref_target_masks is not None:
            raise NotImplementedError("Multi-person LongCat-Avatar is not supported by this node yet.")
        # official forward casts the timesteps to the model dtype before the fp32 embedding
        t = self.timestep_embed(timestep.to(dtype))  # fp32 [B, T, 512]

        use_cache = cond_cache is not None and num_cond > 0
        cached = use_cache and len(cond_cache) > 0 and all("k" in c for c in cond_cache.values())
        if cached:
            hidden = self.x_embedder.proj(x[:, :, num_cond:].to(dtype)).flatten(2).transpose(1, 2)
        else:
            hidden = self.x_embedder.proj(x.to(dtype)).flatten(2).transpose(1, 2)

        ref_index = ref_img_index if num_ref else None
        rope = rope_3d_cos_sin(self.hidden_size // self.num_heads, (Nt, Nh, Nw), hidden.device,
                               ref_index, num_ref if num_ref else None)

        noisy_frames = Nt - num_cond
        if noisy_frames > 0:
            at = audio_tok.to(dtype)
            if human_num is None:
                at = at[:, num_cond:].reshape(-1, at.shape[2], at.shape[3])
            else:
                at = at[:, num_cond:].reshape(-1, at.shape[2], at.shape[3])
        else:
            at = None

        ref_maps = None
        if ref_target_masks is not None:
            m = F.interpolate(ref_target_masks.unsqueeze(0).float(), size=(Nh, Nw), mode="nearest").squeeze(0)
            ref_maps = (m > 0).reshape(m.shape[0], -1).to(dtype)

        for i, block in enumerate(self.blocks):
            if text_kv_cache is not None and i in text_kv_cache:
                tkv = text_kv_cache[i]
            else:
                tkv = block.cross_attn.text_kv(text_y.to(dtype))
                if text_kv_cache is not None:
                    text_kv_cache[i] = tkv
            cache = None
            if use_cache:
                cache = cond_cache.setdefault(i, {"store": True})
            hidden = block(hidden, tkv, t, (Nt, Nh, Nw), num_cond, num_ref, at, ref_img_index, mask_frame_range,
                           rope, cache=cache, ref_maps=ref_maps, human_num=human_num)

        # final layer (fp32)
        t_rows = t[:, num_cond:] if cached else t
        Tx = t_rows.shape[1]
        mod = _linear_as(self.final_layer.adaLN_modulation[1], F.silu(t_rows), torch.float32).unsqueeze(2)
        shift, scale = mod.chunk(2, dim=-1)
        Cout = math.prod(self.patch_size) * self.out_channels
        out = torch.empty((B, Tx * per, Cout), dtype=torch.float32, device=hidden.device)
        hv = hidden.view(B, Tx, per, -1)
        ov = out.view(B, Tx, per, Cout)
        for f0, f1 in _chunks(Tx, max(1, ROW_CHUNK // per)):
            # official FinalLayer_FP32: the modulated rows return to the model dtype before the fp32 linear
            hm = _modulate(self.final_layer.norm_final, hv[:, f0:f1], shift[:, f0:f1], scale[:, f0:f1])
            ov[:, f0:f1] = _linear_as(self.final_layer.linear, hm, torch.float32)
        del hidden
        # unpatchify: B (T H W) (pt ph pw C) -> B C T H W
        pt, ph, pw = self.patch_size
        out = out.view(B, Tx, Nh, Nw, pt, ph, pw, self.out_channels)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, Tx * pt, Nh * ph, Nw * pw)
        if cached and num_cond:
            pad = torch.zeros((B, self.out_channels, num_cond, out.shape[3], out.shape[4]), dtype=out.dtype, device=out.device)
            out = torch.cat([pad, out], dim=2)
        return out
