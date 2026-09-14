"""MiniMax H3 streaming sampler: continuous audio + image -> long lip-synced video.

Port of the TaoMate-H3 direct streaming method (Alibaba TaoLive AIGC,
https://github.com/TaoLiveAIGC/TaoMate-H3) to ComfyUI's native MiniMax H3
implementation, adapted for a *given* soundtrack:

* the video is generated chunk by chunk (TaoMate phases of 2,2,2,1 native
  17-frame groups per 5 second request) with the 3-step TaoMate LoRA schedule
  (sigma indices 0 / 16 / 33 / 49 of the 50-step shifted grid);
* every finished chunk runs one extra sigma-0 "clean" forward whose per-layer
  keys/values are committed to a persistent KV cache; the next chunk's media
  rows attend to the text/first-frame condition rows, the retained clean
  history (first-chunk video sink + the most recent chunks) and themselves,
  while condition rows attend only to condition rows (TaoMate's rule);
* the soundtrack is not generated: the user's audio is VAE-encoded once and
  its chunk slice enters each step as a teacher-forced latent (noised to the
  step's audio sigma like TaoMate's Base10 milestones, or clean at t=1);
* RoPE positions live on one global timeline (video 5/3 units per frame,
  audio 1 unit per 40 Hz latent), the prompt slides along with each request,
  the first frame stays anchored at the origin;
* clean chunk latents are affine-normalized to the first chunk's statistics
  (TaoMate prefix normalization) so long streams do not drift;
* the video VAE decodes finished 5-latent windows incrementally and frames are
  piped straight into ffmpeg, so a five minute video never has to sit in RAM.

Only the ComfyUI native MiniMax H3 model code is used underneath (its blocks,
projections, RoPE tables and attention backend); no model implementation is
duplicated. The TaoMate adapter is loaded with the stock LoRA loader after the
full-rank conversion shipped next to this file's docs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
import wave
from fractions import Fraction

import torch

import comfy.model_management as mm
import comfy.model_prefetch
import comfy.utils
import folder_paths

try:  # ComfyUI without MiniMax H3 support still imports the package
    from comfy.ldm.minimax import model as h3
    from comfy.ldm.modules.attention import optimized_attention
    import comfy.latent_formats
    import comfy.nested_tensor
    import comfy.quant_ops
except ImportError:  # pragma: no cover - depends on the ComfyUI version
    h3 = None

LOG = logging.getLogger("SECoursesH3Streaming")

FPS = 24
AUDIO_LATENT_RATE = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VIDEO_PREFIX_LATENTS = 2
GROUP_LATENTS = 5
GROUP_FRAMES = 17
TAOMATE_STATE_INDICES = (0, 16, 33, 49)
TAOMATE_GROUPS = "2,2,2,1"
AUDIO_MODES = ["noised to step (TaoMate teacher-like)", "clean pinned (t=1)"]
ANCHOR_MODES = ["all chunks", "first request only", "off"]
TEXT_MODES = ["slide per request (TaoMate)", "fixed at start"]
KV_DTYPES = ["fp8_e4m3", "bf16"]
KV_DEVICES = ["auto", "gpu", "cpu pinned", "cpu"]
WEIGHT_MODES = ["resident (non-dynamic load, fastest when it fits)", "dynamic (ComfyUI default streaming)"]
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_HOP = 800
FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def align_frame_count(frames):
    frames = max(5, int(frames))
    while frames % GROUP_FRAMES != 5:
        frames += 1
    return frames


def latents_for_frames(frames):
    return VIDEO_PREFIX_LATENTS + GROUP_LATENTS * ((frames - 5) // GROUP_FRAMES)


def frames_before(latent_index):
    """Pixel frames represented by video latents [0, latent_index)."""
    full, rest = divmod(int(latent_index), GROUP_LATENTS)
    return full * GROUP_FRAMES + sum(FRAME_PER_TOKEN[:rest])


def round_half_even(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def audio_boundary(frame):
    """40 Hz audio latent index at a 24 fps frame boundary (TaoMate rounding)."""
    return round_half_even(Fraction(int(frame) * AUDIO_LATENT_RATE, FPS))


def parse_groups(text):
    groups = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("chunk_groups entries must be positive 17-frame group counts")
        groups.append(value)
    if not groups:
        raise ValueError("chunk_groups needs at least one group count, eg '2,2,2,1'")
    return tuple(groups)


def build_chunk_plan(total_latents, groups=(2, 2, 2, 1)):
    """Split a 2+5k latent stream into TaoMate-style chunks on the global timeline."""
    if total_latents < VIDEO_PREFIX_LATENTS + GROUP_LATENTS or (total_latents - VIDEO_PREFIX_LATENTS) % GROUP_LATENTS:
        raise ValueError(f"MiniMax H3 video latents must be 2+5k and at least 7, got {total_latents}")
    remaining_groups = (total_latents - VIDEO_PREFIX_LATENTS) // GROUP_LATENTS
    chunks = []
    latent = 0
    request = 0
    phase = 0
    while remaining_groups > 0:
        count = min(groups[phase], remaining_groups)
        stop = latent + count * GROUP_LATENTS + (VIDEO_PREFIX_LATENTS if latent == 0 else 0)
        chunks.append({
            "index": len(chunks), "request": request, "phase": phase, "groups": count,
            "lat_start": latent, "lat_stop": stop,
            "frame_start": frames_before(latent), "frame_stop": frames_before(stop),
            "aud_start": audio_boundary(frames_before(latent)), "aud_stop": audio_boundary(frames_before(stop)),
        })
        remaining_groups -= count
        latent = stop
        phase += 1
        if phase == len(groups):
            phase = 0
            request += 1
    return chunks


def shifted_sigmas(num_steps, shift):
    base = torch.linspace(1.0, 0.0, num_steps, dtype=torch.float64)
    return (shift * base / (1.0 + (shift - 1.0) * base)).tolist()


def state_indices(steps, grid=50):
    steps = int(steps)
    if steps == 3:
        return TAOMATE_STATE_INDICES
    return tuple(int(round(i * (grid - 1) / steps)) for i in range(steps + 1))


def select_sigmas(steps, shift):
    schedule = shifted_sigmas(50, shift)
    return [schedule[i] for i in state_indices(steps)]


# ---------------------------------------------------------------------------
# audio helpers (same normalization as the SECourses init-audio nodes)
# ---------------------------------------------------------------------------

def normalize_waveform(audio, sample_rate):
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.detach().to(dtype=torch.float32, device="cpu")
    if waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    source_rate = int(audio["sample_rate"])
    if source_rate != int(sample_rate):
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, source_rate, int(sample_rate))
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    return waveform.contiguous()


def fit_waveform(waveform, samples):
    length = waveform.shape[-1]
    if length > samples:
        return waveform[..., :samples].contiguous()
    if length < samples:
        return torch.nn.functional.pad(waveform, (0, samples - length))
    return waveform


def write_wav(path, waveform, sample_rate):
    pcm = (waveform.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(path, "wb") as handle:
        handle.setnchannels(waveform.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# persistent clean KV cache
# ---------------------------------------------------------------------------

def _quantize_fp8(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(torch.float16)
    return (x / scale.to(x.dtype)).to(torch.float8_e4m3fn), scale


class KVCache:
    """Committed clean audio/video keys and values, one entry per chunk, per layer."""

    def __init__(self, num_layers, storage_dtype, storage_device, compute_device, pinned=False):
        self.num_layers = num_layers
        self.fp8 = storage_dtype == "fp8_e4m3"
        self.storage_device = torch.device(storage_device)
        self.compute_device = compute_device
        self.pinned = bool(pinned) and self.storage_device.type == "cpu"
        self.entries = []  # dict(audio_rows, video_rows, layers=[per layer tuple])
        self._staged = None
        self.bytes = 0

    def begin_commit(self):
        self._staged = [None] * self.num_layers

    def stage(self, layer, key, value):
        if self._staged is None:
            raise RuntimeError("KV commit is not active")
        key = key.detach().to(torch.bfloat16)
        value = value.detach().to(torch.bfloat16)
        if self.fp8:
            kq, ks = _quantize_fp8(key)
            vq, vs = _quantize_fp8(value)
            item = tuple(self._store(t) for t in (kq, ks, vq, vs))
        else:
            item = tuple(self._store(t) for t in (key, value))
        self._staged[layer] = item

    def _store(self, tensor):
        if tensor.device == self.storage_device:
            return tensor
        tensor = tensor.to(self.storage_device)
        if self.pinned:
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                self.pinned = False
        return tensor

    def commit(self, audio_rows, video_rows):
        if self._staged is None or any(item is None for item in self._staged):
            missing = [i for i, item in enumerate(self._staged or []) if item is None]
            self._staged = None
            raise RuntimeError(f"KV commit is missing layers {missing[:8]}")
        entry = {"audio_rows": int(audio_rows), "video_rows": int(video_rows), "layers": self._staged}
        self._staged = None
        self.entries.append(entry)
        self._recount()

    def rollback(self):
        self._staged = None

    def _recount(self):
        total = 0
        for entry in self.entries:
            for item in entry["layers"]:
                total += sum(t.numel() * t.element_size() for t in item)
        self.bytes = total

    @property
    def rows(self):
        return sum(e["audio_rows"] + e["video_rows"] for e in self.entries)

    @staticmethod
    def _slice(entry, start, stop):
        layers = [tuple(t[start:stop].contiguous() for t in item) for item in entry["layers"]]
        return layers

    def retain(self, sink_rows, recent_chunks):
        """Keep the first chunk's leading video rows as a sink plus the most recent chunks."""
        if len(self.entries) <= max(1, recent_chunks):
            return
        first = self.entries[0]
        kept = []
        if sink_rows > 0 and not first.get("is_sink"):
            a = first["audio_rows"]
            rows = min(sink_rows, first["video_rows"])
            kept.append({"audio_rows": 0, "video_rows": rows, "layers": self._slice(first, a, a + rows), "is_sink": True})
        elif first.get("is_sink"):
            kept.append(first)
        recent = self.entries[-recent_chunks:] if recent_chunks > 0 else []
        recent = [e for e in recent if e is not first]
        self.entries = kept + recent
        self._recount()

    def drop_audio(self):
        removed = 0
        for entry in self.entries:
            a = entry["audio_rows"]
            if a > 0:
                entry["layers"] = self._slice(entry, a, a + entry["video_rows"])
                removed += a
                entry["audio_rows"] = 0
        self._recount()
        return removed

    def clear(self):
        self.entries = []
        self._staged = None
        self.bytes = 0

    def history(self, layer):
        """Concatenated bf16 keys/values [rows, heads, dim] for one layer, or (None, None)."""
        if not self.entries:
            return None, None
        keys, values = [], []
        for entry in self.entries:
            item = entry["layers"][layer]
            if self.fp8:
                kq, ks, vq, vs = (t.to(self.compute_device, non_blocking=True) for t in item)
                keys.append(kq.to(torch.bfloat16) * ks.to(torch.bfloat16))
                values.append(vq.to(torch.bfloat16) * vs.to(torch.bfloat16))
            else:
                k, v = (t.to(self.compute_device, non_blocking=True) for t in item)
                keys.append(k)
                values.append(v)
        if len(keys) == 1:
            return keys[0], values[0]
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)


# ---------------------------------------------------------------------------
# streaming attention: condition rows see condition rows; media rows see
# condition rows + clean history + themselves
# ---------------------------------------------------------------------------

class StreamingAttention:
    def __init__(self, cache):
        self.cache = cache
        self.block = None
        self.layer = 0
        self.n_cond = 0
        self.commit = False
        self.kernel_calls = 0

    def __call__(self, x, rope_freqs=None, transformer_options={}):
        attn = self.block.attn
        s = x.shape[0]
        heads, dim = attn.heads, attn.head_dim
        q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
        v = v.view(s, heads, dim)
        if rope_freqs is not None:
            q = q.view(1, s, heads, dim)
            k = k.view(1, s, heads, dim)
            qw = mm.cast_to(attn.q_norm.weight, device=x.device)
            kw = mm.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, heads, dim))
            k = attn.k_norm(k.view(s, heads, dim))

        def heads_first(t):
            return t.transpose(0, 1).unsqueeze(0)

        nc = self.n_cond
        out = torch.empty(s, heads * dim, dtype=x.dtype, device=x.device)
        if nc > 0:
            cond = optimized_attention(heads_first(q[:nc]), heads_first(k[:nc]), heads_first(v[:nc]), heads,
                                       mask=None, skip_reshape=True, transformer_options=transformer_options)
            out[:nc] = cond[0]
            self.kernel_calls += 1
        k_hist, v_hist = self.cache.history(self.layer)
        k_parts = [k[:nc], k[nc:]] if k_hist is None else [k[:nc], k_hist, k[nc:]]
        v_parts = [v[:nc], v[nc:]] if v_hist is None else [v[:nc], v_hist, v[nc:]]
        k_media = torch.cat(k_parts, dim=0) if len(k_parts) > 1 else k_parts[0]
        v_media = torch.cat(v_parts, dim=0) if len(v_parts) > 1 else v_parts[0]
        media = optimized_attention(heads_first(q[nc:]), heads_first(k_media), heads_first(v_media), heads,
                                    mask=None, skip_reshape=True, transformer_options=transformer_options)
        out[nc:] = media[0]
        self.kernel_calls += 1
        if self.commit:
            self.cache.stage(self.layer, k[nc:], v[nc:])
        del k_hist, v_hist, k_media, v_media
        return attn.out_proj(out)


# ---------------------------------------------------------------------------
# incremental VAE decode + ffmpeg publication
# ---------------------------------------------------------------------------

class FFmpegWriter:
    def __init__(self, path, width, height, fps=FPS, crf=18, preset="medium"):
        binary = shutil.which("ffmpeg")
        if binary is None:
            raise RuntimeError("ffmpeg is not on PATH; it is required to save the streamed video")
        self.path = path
        self.frames = 0
        self.process = subprocess.Popen(
            [binary, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def write(self, frames_uint8):
        # frames_uint8: [T, H, W, 3] uint8 on CPU
        if self.process.poll() is not None:
            raise RuntimeError("ffmpeg exited early: " + self.process.stderr.read().decode(errors="replace"))
        self.process.stdin.write(frames_uint8.contiguous().numpy().tobytes())
        self.frames += int(frames_uint8.shape[0])

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg video encode failed ({code}): {error}")

    def abort(self):
        try:
            self.process.kill()
        except Exception:
            pass


def mux_audio(video_only, wav_path, output, seconds, frames=None):
    binary = shutil.which("ffmpeg")
    command = [binary, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", video_only, "-i", wav_path,
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-t", f"{seconds:.4f}"]
    if frames:
        command += ["-frames:v", str(int(frames))]
    command += ["-movflags", "+faststart", output]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg mux failed: " + completed.stderr.decode(errors="replace"))


class IncrementalH3Decoder:
    """Mirror of MiniMaxH3VideoVAE.decode_temporal that accepts latents as they finish."""

    def __init__(self, vae, total_latents, writer):
        self.vae = vae
        self.model = vae.first_stage_model
        self.stride = int(self.model.tokens_chunk_size)
        self.window = self.stride + int(self.model.token_overlap)
        self.chunk_dec = self.stride * int(self.model.vae_ratio_t)
        self.split_count = int(self.model.token_drop > 0) + 1
        pad_tokens, num_chunks = self.model._decode_temporal_chunks(total_latents)
        if pad_tokens:
            raise ValueError(f"unexpected VAE padding for a 2+5k latent stream ({total_latents} latents)")
        self.num_chunks = int(num_chunks)
        self.next_chunk = 0
        self.overlap = None
        self.writer = writer
        self.frames_written = 0
        self.seconds = 0.0

    def pending(self, available_latents, final):
        chunks = []
        index = self.next_chunk
        while index < self.num_chunks:
            if index * self.stride + self.window <= available_latents or (final and index == self.num_chunks - 1):
                chunks.append(index)
                index += 1
            else:
                break
        return chunks

    def decode(self, video_latent, available_latents, final=False):
        chunks = self.pending(available_latents, final)
        if not chunks:
            return 0
        started = time.perf_counter()
        model = self.model
        device = self.vae.device
        dtype = self.vae.vae_dtype
        sample = video_latent[:, :, :self.window].to(device=device, dtype=dtype)
        memory = self.vae.memory_used_decode(sample.shape, dtype)
        with comfy.model_prefetch.pause_malloc_graph():
            mm.load_models_gpu([self.vae.patcher], memory_required=memory)
        mean = model.latents_mean.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
        std = model.latents_std.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
        written = 0
        for index in chunks:
            start = index * self.stride
            clip = video_latent[:, :, start:start + self.window].to(device=device, dtype=dtype) * std + mean
            clip_dec = model._adaptive_decode(clip)
            for j in range(self.split_count):
                f0 = j * self.chunk_dec
                f1 = min(f0 + self.chunk_dec, clip_dec.shape[2])
                part = clip_dec[:, :, f0:f1][:, :, model.frame_pre_padding:]
                if j == 0:
                    if self.overlap is not None:
                        part = model.blend(self.overlap, part, model.frame_overlap, dim=-3)
                        self.overlap = None
                    written += self._write(model._finalize_pixels(part))
                else:
                    self.overlap = part.contiguous()
            if index == self.num_chunks - 1 and self.overlap is not None:
                written += self._write(model._finalize_pixels(self.overlap))
                self.overlap = None
            del clip_dec, clip
            self.next_chunk = index + 1
        self.seconds += time.perf_counter() - started
        return written

    def _write(self, pixels):
        frames = (pixels[0].permute(1, 2, 3, 0) * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu()
        self.writer.write(frames)
        self.frames_written += int(frames.shape[0])
        return int(frames.shape[0])


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

class H3StreamingRun:
    def __init__(self, model_patcher, positive, options, progress=None):
        self.mp = model_patcher
        self.dm = model_patcher.model.diffusion_model
        if type(self.dm).__name__ != "MiniMaxH3Model":
            raise ValueError(f"H3 Streaming Sampler needs a MiniMax H3 model, got {type(self.dm).__name__}")
        self.options = options
        self.device = model_patcher.load_device
        self.dtype = model_patcher.model.get_dtype_inference()
        self.positive = positive
        self.progress = progress
        self.transformer_options = dict(model_patcher.model_options.get("transformer_options", {}))
        self.transformer_options.pop("optimized_attention_override", None)
        self.transformer_options.pop("patches_replace", None)
        self.attention = None
        self.cache = None
        self.text_states = None
        self.text_tags = None
        self.cond_rows = None
        self.anchor = None
        self.timing = {"noisy_forwards": 0, "clean_forwards": 0, "dit_seconds": 0.0}

    # -- conditioning ------------------------------------------------------

    def prepare_conditioning(self, lat_h, lat_w, seed):
        cond = self.positive[0]
        cross_attn = cond[0]
        extra = cond[1]
        text = self.dm.preprocess_text_embeds(cross_attn.to(device=self.device, dtype=self.dtype))[0]
        self.text_states = text.contiguous()
        tags = extra.get("minimax_token_tags")
        if tags is None:
            tags = torch.ones(text.shape[0], dtype=torch.long)
        self.text_tags = tags.view(-1).to(torch.long).cpu()
        if self.text_tags.shape[0] != text.shape[0]:
            raise ValueError("MiniMax H3 text token tags do not match the text embeddings")
        rows = []
        aug = float(extra.get("minimax_visual_cond_noise_aug", h3.VISUAL_COND_TIMESTEP))
        for kf in extra.get("minimax_keyframes", []) or []:
            latent = kf.get("latent")
            if latent is None:
                continue
            if int(kf.get("resolved_frame_index", 0)) != 0:
                LOG.warning("[H3 Streaming] ignoring a keyframe anchored at frame %s; only the first frame is streamed", kf.get("resolved_frame_index"))
                continue
            if latent.shape[2] != 1:
                LOG.warning("[H3 Streaming] using only the first latent frame of a multi-frame guide")
                latent = latent[:, :, :1]
            if latent.shape[3] != lat_h or latent.shape[4] != lat_w:
                raise ValueError(f"first-frame latent {tuple(latent.shape)} does not match the target canvas {lat_h}x{lat_w} latents")
            r = h3.patchify_video(latent.to(torch.float32), self.dm.patch_size)
            if aug < 1.0:
                generator = torch.Generator("cpu").manual_seed(int(seed))
                noise = torch.randn(r.shape, generator=generator, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(self.device))
        self.cond_rows = torch.cat(rows, dim=0) if rows else None
        self.cond_aug = aug
        return self.text_states.shape[0]

    # -- one packed chunk forward ------------------------------------------

    def _t_embed(self, t_vals):
        dm = self.dm
        if dm.use_adaln_curves:
            table = mm.cast_to(dm.adaln_t_table, device=self.device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        return dm.time_embedder(t_vals).to(self.dtype)

    def build_layout(self, chunk, lat_h, lat_w, text_len, prompt_start, use_anchor):
        """Row segments + RoPE table for one chunk on the global timeline."""
        origin = float(text_len)
        frame, w_grid = h3._frame_grid(lat_h, lat_w)
        frame_rows = frame.shape[0]
        pos = []
        segments = []
        row = 0
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = prompt_start + torch.arange(text_len, dtype=torch.float64)
        pos.append(g)
        segments.append((row, row + text_len, "text"))
        row += text_len
        n_cond = 0
        if use_anchor and self.cond_rows is not None:
            n_cond = self.cond_rows.shape[0]
            frames_in_anchor = n_cond // frame_rows
            pos.append(h3._video_grid(frames_in_anchor, frame, origin))
            segments.append((row, row + n_cond, "cond"))
            row += n_cond
        audio_t = chunk["aud_stop"] - chunk["aud_start"]
        pos.append(h3._audio_grid(origin + chunk["aud_start"], audio_t, float(w_grid[0]), float(w_grid[-1])))
        segments.append((row, row + audio_t * 2, "audio"))
        row += audio_t * 2
        video_t = chunk["lat_stop"] - chunk["lat_start"]
        times = torch.tensor([origin + FRAME_RESCALE * frames_before(chunk["lat_start"] + j) for j in range(video_t)], dtype=torch.float64)
        vg = torch.empty(video_t, frame_rows, 3, dtype=torch.float64)
        vg[:, :, 0] = times[:, None]
        vg[:, :, 1:] = frame[None]
        pos.append(vg.reshape(-1, 3))
        segments.append((row, row + video_t * frame_rows, "video"))
        row += video_t * frame_rows
        position_ids = torch.cat(pos)
        rope = h3.rope_rotation_table(self.dm.rope_freqs(position_ids, self.device), self.dtype)
        return {"segments": segments, "rope": rope, "seq_len": row, "n_condition": text_len + n_cond,
                "frame_rows": frame_rows, "audio_t": audio_t, "video_t": video_t}

    def forward(self, layout, video_rows, audio_rows, t_video, t_audio, commit=False):
        dm = self.dm
        device, dtype = self.device, self.dtype
        cond_t = 1.0 if commit else max(t_video, self.cond_aug)
        unique_t = sorted({float(t_video), float(t_audio), float(cond_t)})
        t_row = {t: i for i, t in enumerate(unique_t)}
        mod_segments = []
        for a, b, kind in layout["segments"]:
            if kind == "text":
                base = t_row[float(t_video)] * 3
                tags = self.text_tags.tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, base + int(tags[run_start])))
                        run_start = i
            elif kind == "cond":
                mod_segments.append((a, b, t_row[float(cond_t)] * 3 + 0))
            elif kind == "audio":
                mod_segments.append((a, b, t_row[float(t_audio)] * 3 + 2))
            else:
                mod_segments.append((a, b, t_row[float(t_video)] * 3 + 0))

        all_video_rows = video_rows if self.cond_rows is None or layout["n_condition"] == self.text_states.shape[0] \
            else torch.cat([self.cond_rows, video_rows], dim=0)
        video_embed = dm.video_patch_proj(all_video_rows.to(torch.float32)).to(dtype)
        audio_embed = dm.audio_patch_proj(audio_rows.to(torch.float32)).to(dtype)
        hidden = torch.empty(layout["seq_len"], dm.hidden_size, dtype=dtype, device=device)
        voff = 0
        for a, b, kind in layout["segments"]:
            n = b - a
            if kind == "text":
                hidden[a:b] = self.text_states
            elif kind in ("cond", "video"):
                hidden[a:b] = video_embed[voff:voff + n]
                voff += n
            else:
                hidden[a:b] = audio_embed
        t_emb = self._t_embed(torch.tensor(unique_t, dtype=torch.float32, device=device))
        rope = layout["rope"]
        to = self.transformer_options
        self.attention.n_cond = layout["n_condition"]
        self.attention.commit = commit
        if commit:
            self.cache.begin_commit()
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dm.blocks), device, to)
        started = time.perf_counter()
        try:
            for i, block in enumerate(dm.blocks):
                comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, malloc_scope="block")
                to["block_index"] = i
                self.attention.block = block
                self.attention.layer = i
                hidden = h3.DiTBlock.forward(block, hidden, t_emb, mod_segments, rope, transformer_options=to, attention=self.attention)
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        except BaseException:
            if commit:
                self.cache.rollback()
            raise
        finally:
            self.attention.commit = False
        if commit:
            self.timing["clean_forwards"] += 1
            self.timing["dit_seconds"] += time.perf_counter() - started
            return None
        va, vb, _ = next(s for s in layout["segments"] if s[2] == "video")
        aa, ab, _ = next(s for s in layout["segments"] if s[2] == "audio")
        video_seg = (va, vb, t_row[float(t_video)])
        audio_seg = (aa, ab, t_row[float(t_audio)])
        sigma_v = torch.tensor(1.0 - t_video, dtype=torch.float32, device=device)
        velocity, _ = dm.final_layer(hidden, t_emb, video_seg, audio_seg, sigma_v, self.sample_sigmas, (self.shift_video, self.shift_audio))
        self.timing["noisy_forwards"] += 1
        self.timing["dit_seconds"] += time.perf_counter() - started
        return velocity.to(torch.float32)

    # -- the streaming loop --------------------------------------------------

    def run(self, video_latent, audio_latent, plan, seed, steps, shift_video, shift_audio, decoder=None,
            decode_interval_latents=0, log=print):
        o = self.options
        dm = self.dm
        device = self.device
        self.shift_video, self.shift_audio = float(shift_video), float(shift_audio)
        sig_v = select_sigmas(steps, shift_video)
        sig_a = select_sigmas(steps, shift_audio)
        self.sample_sigmas = torch.tensor(sig_v, dtype=torch.float32, device=device)
        lat_h, lat_w = video_latent.shape[3], video_latent.shape[4]
        frame_rows = (lat_h // 2) * (lat_w // 2)
        text_len = self.text_states.shape[0]
        origin = float(text_len)
        heads, head_dim = dm.blocks[0].attn.heads, dm.blocks[0].attn.head_dim
        self.cache = KVCache(len(dm.blocks), o["kv_cache_dtype"], "cpu" if o["kv_cache_device"].startswith("cpu") else device, device,
                             pinned=o["kv_cache_device"] == "cpu pinned")
        self.attention = StreamingAttention(self.cache)
        clean_audio_rows_all = h3.pack_audio(audio_latent.to(torch.float32))  # [2*T, 32] channel-major
        audio_t_total = audio_latent.shape[-1]
        anchor_stats = None
        noised_audio = o["audio_mode"] == AUDIO_MODES[0]
        sink_rows = int(o["sink_latents"]) * frame_rows
        recent = int(o["context_chunks"])
        audio_reset = int(o["audio_reset_requests"])
        decoded_upto = 0
        last_decode_latent = 0
        total_forwards = len(plan) * (steps + 1)
        done_forwards = 0
        stream_started = time.perf_counter()
        current_request = -1
        prompt_start = 0.0
        for chunk in plan:
            mm.throw_exception_if_processing_interrupted()
            chunk_started = time.perf_counter()
            if chunk["request"] != current_request:
                current_request = chunk["request"]
                if o["text_position"] == TEXT_MODES[0]:
                    prompt_start = origin + FRAME_RESCALE * frames_before(chunk["lat_start"]) - text_len
                else:
                    prompt_start = 0.0
                if audio_reset > 0 and current_request > 0 and current_request % audio_reset == 0:
                    removed = self.cache.drop_audio()
                    log(f"[H3 Streaming] request {current_request}: dropped {removed} audio KV rows (TaoMate audio reset)")
            use_anchor = o["first_frame_anchor"] == ANCHOR_MODES[0] or (o["first_frame_anchor"] == ANCHOR_MODES[1] and chunk["request"] == 0)
            layout = self.build_layout(chunk, lat_h, lat_w, text_len, prompt_start, use_anchor)
            generator = torch.Generator("cpu").manual_seed(int(seed) + 1000003 * chunk["index"])
            video_t = chunk["lat_stop"] - chunk["lat_start"]
            noise_v = torch.randn(1, 24, video_t, lat_h, lat_w, generator=generator, dtype=torch.float32)
            x_v = h3.patchify_video(noise_v, dm.patch_size).to(device)
            a0, a1 = chunk["aud_start"], chunk["aud_stop"]
            clean_a = clean_audio_rows_all.view(2, audio_t_total, -1)[:, a0:a1].reshape(-1, clean_audio_rows_all.shape[-1]).to(device)
            eps_a = torch.randn(clean_a.shape, generator=generator, dtype=torch.float32).to(device)
            for i in range(steps):
                mm.throw_exception_if_processing_interrupted()
                sv, sv_next, sa = sig_v[i], sig_v[i + 1], sig_a[i]
                if noised_audio:
                    x_a = (1.0 - sa) * clean_a + sa * eps_a
                    t_a = 1.0 - sa
                else:
                    x_a = clean_a
                    t_a = 1.0
                velocity = self.forward(layout, x_v, x_a, 1.0 - sv, t_a)
                x_v = x_v + (sv - sv_next) * velocity
                del velocity
                done_forwards += 1
                if self.progress is not None:
                    self.progress.update_absolute(done_forwards, total_forwards)
            # prefix normalization to the first chunk's statistics
            if o["prefix_normalization"]:
                mean = x_v.mean(dim=0, keepdim=True)
                std = x_v.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
                if anchor_stats is None:
                    anchor_stats = (mean, std)
                else:
                    x_v = (x_v - mean) / std * anchor_stats[1] + anchor_stats[0]
            # clean commit forward
            self.forward(layout, x_v, clean_a, 1.0, 1.0, commit=True)
            self.cache.commit(audio_rows=clean_a.shape[0], video_rows=x_v.shape[0])
            self.cache.retain(sink_rows, recent)
            done_forwards += 1
            # store the clean chunk
            clean_video = h3.unpatchify_video(x_v, video_t, lat_h // 2, lat_w // 2, dm.latents_dim, dm.patch_size)
            video_latent[:, :, chunk["lat_start"]:chunk["lat_stop"]] = clean_video.to(video_latent.device, video_latent.dtype)
            preview = self._preview(clean_video[0, :, -1]) if self.progress is not None else None
            if self.progress is not None:
                self.progress.update_absolute(done_forwards, total_forwards, preview)
            del x_v, clean_a, eps_a, layout
            elapsed = time.perf_counter() - chunk_started
            free = mm.get_free_memory(device) / (1024 ** 3)
            log(f"[H3 Streaming] chunk {chunk['index'] + 1}/{len(plan)} (req {chunk['request']}, latents {chunk['lat_start']}-{chunk['lat_stop']}, "
                f"frames {chunk['frame_start']}-{chunk['frame_stop']}, audio {chunk['aud_start']}-{chunk['aud_stop']}): {elapsed:.1f}s, "
                f"KV rows {self.cache.rows} ({self.cache.bytes / 1024 ** 3:.2f} GB), free VRAM {free:.1f} GB")
            # incremental decode / publication
            if decoder is not None:
                final = chunk is plan[-1]
                if final or (decode_interval_latents > 0 and chunk["lat_stop"] - last_decode_latent >= decode_interval_latents):
                    frames = decoder.decode(video_latent, chunk["lat_stop"], final=final)
                    if frames:
                        last_decode_latent = chunk["lat_stop"]
                        log(f"[H3 Streaming] decoded and encoded {frames} frames (total {decoder.frames_written})")
                    if not final:
                        mm.load_models_gpu([self.mp], memory_required=self.memory_estimate)
        self.timing["stream_seconds"] = time.perf_counter() - stream_started
        self.timing["attention_kernel_calls"] = self.attention.kernel_calls
        self.timing["kv_rows_final"] = self.cache.rows
        self.cache.clear()
        return video_latent

    def _preview(self, latent_frame):
        try:
            from PIL import Image

            fmt = comfy.latent_formats.MiniMaxH3Video()
            factors = torch.tensor(fmt.latent_rgb_factors, dtype=torch.float32, device=latent_frame.device)
            bias = torch.tensor(fmt.latent_rgb_factors_bias, dtype=torch.float32, device=latent_frame.device)
            rgb = torch.einsum("chw,cr->hwr", latent_frame.to(torch.float32), factors) + bias
            rgb = ((rgb.clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(rgb)
            return ("JPEG", image, 512)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class SEH3StreamingSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 FL2VA model with the TaoMate 3-step LoRA applied (LoraLoaderModelOnly, strength 1.0)."}),
                "positive": ("CONDITIONING", {"tooltip": "MiniMax H3 FL2VA conditioning: prompt + optional first frame (MiniMax H3 Image to Video / SECourses Auto node)."}),
                "latent": ("LATENT", {"tooltip": "The AV latent of the same conditioning node; only its canvas (width/height) is used, the duration follows the audio."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE (encodes the soundtrack the video must follow)."}),
                "audio": ("AUDIO", {"tooltip": "The soundtrack to lip-sync. Any length: the video is generated chunk by chunk for the whole audio."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip": "3 = the TaoMate distilled schedule (sigma grid indices 0/16/33/49). Other values space the 50-step shifted grid evenly."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "chunk_groups": ("STRING", {"default": TAOMATE_GROUPS, "tooltip": "17-frame groups per chunk, repeating per 5 second request. TaoMate: 2,2,2,1 (39/34/34/17 frames). Smaller chunks lower latency and VRAM, larger chunks give more temporal context."}),
                "sink_latents": ("INT", {"default": 7, "min": 0, "max": 64, "tooltip": "Leading video latents of the very first chunk kept forever as an identity sink (TaoMate keeps its whole first chunk = 12; 7 = prefix + first 17-frame group fits 32 GB at 480p with the first-frame anchor doing the rest). 0 disables the sink."}),
                "context_chunks": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "Most recent clean chunks whose audio+video keys/values stay in the cache (TaoMate: 2)."}),
                "kv_cache_dtype": (KV_DTYPES, {"default": KV_DTYPES[0], "tooltip": "fp8 halves the cache versus bf16 with per-row/head scaling."}),
                "kv_cache_device": (KV_DEVICES, {"default": KV_DEVICES[0], "tooltip": "auto: gpu when weights + cache + activations fit, otherwise cpu pinned. gpu: fastest when it fits. cpu pinned / cpu: keep the history in system RAM and stream it per layer (unbounded by VRAM; pinned transfers are several times faster)."}),
                "weights": (WEIGHT_MODES, {"default": WEIGHT_MODES[0], "tooltip": "resident: load the DiT the classic way so its weights stay on the GPU next to the KV cache when everything fits (no per-step weight streaming). dynamic: ComfyUI's on-demand weight paging, which evicts weights as the cache grows."}),
                "audio_mode": (AUDIO_MODES, {"default": AUDIO_MODES[0], "tooltip": "How the given soundtrack enters each step: noised to the step's audio sigma (what the TaoMate student saw in training) or clean at t=1."}),
                "first_frame_anchor": (ANCHOR_MODES, {"default": ANCHOR_MODES[0], "tooltip": "Where the first-frame image rows stay visible as a clean condition."}),
                "text_position": (TEXT_MODES, {"default": TEXT_MODES[0]}),
                "prefix_normalization": ("BOOLEAN", {"default": True, "tooltip": "Affine-normalize each clean chunk's latent statistics to the first chunk (TaoMate); prevents brightness/contrast drift over long streams."}),
                "audio_reset_requests": ("INT", {"default": 12, "min": 0, "max": 1000, "tooltip": "Drop audio history from the cache every N requests (TaoMate: 12 = every 60 s). 0 never drops."}),
                "save_video": ("BOOLEAN", {"default": True, "tooltip": "Decode finished latents incrementally with the video VAE and write the MP4 (with the original audio) while generating."}),
                "filename_prefix": ("STRING", {"default": "video/H3_Streaming/H3_Stream"}),
                "decode_interval_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 3600.0, "step": 1.0, "tooltip": "Decode/encode in batches of this many seconds of finished video (each batch swaps the DiT and the video VAE on the GPU). 0 = only at the end."}),
                "video_crf": ("INT", {"default": 17, "min": 0, "max": 40}),
            },
            "optional": {
                "video_vae": ("VAE", {"tooltip": "MiniMax H3 video VAE; required when save_video is on."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("latent", "audio", "video", "video_path", "report")
    FUNCTION = "sample"
    OUTPUT_NODE = True
    CATEGORY = "SECourses/MiniMax H3 Lip Synch"
    DESCRIPTION = ("TaoMate-H3 style streaming generation for MiniMax H3: continuous audio + image -> long lip-synced video "
                   "with the 3-step LoRA, a persistent clean KV cache and incremental VAE decode + MP4 publication.")

    def sample(self, model, positive, latent, audio_vae, audio, seed, steps, shift_video, shift_audio, chunk_groups,
               sink_latents, context_chunks, kv_cache_dtype, kv_cache_device, weights, audio_mode, first_frame_anchor, text_position,
               prefix_normalization, audio_reset_requests, save_video, filename_prefix, decode_interval_seconds, video_crf,
               video_vae=None, unique_id=None):
        if h3 is None:
            raise RuntimeError("This ComfyUI version does not ship the native MiniMax H3 model; update ComfyUI.")
        if not isinstance(audio, dict) or "waveform" not in audio:
            raise ValueError("H3 Streaming Sampler needs a ComfyUI AUDIO input (the soundtrack to follow).")
        if save_video and video_vae is None:
            raise ValueError("save_video is enabled but no video_vae is connected.")
        groups = parse_groups(chunk_groups)

        samples = latent["samples"]
        video_shape = samples.unbind()[0].shape if getattr(samples, "is_nested", False) else samples.shape
        if len(video_shape) != 5 or video_shape[1] != 24:
            raise ValueError("The latent input must be a MiniMax H3 AV latent (video [B,24,T,H/16,W/16]).")
        lat_h, lat_w = int(video_shape[3]), int(video_shape[4])
        if lat_h % 2 or lat_w % 2:
            raise ValueError("MiniMax H3 needs a canvas whose width and height are multiples of 32.")
        width, height = lat_w * 16, lat_h * 16

        # soundtrack -> latent grid
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", DEFAULT_SAMPLE_RATE))
        hop = int(getattr(audio_vae, "downscale_ratio", DEFAULT_HOP))
        waveform = normalize_waveform(audio, sample_rate)
        audio_seconds = waveform.shape[-1] / sample_rate
        if audio_seconds < 1.0:
            raise ValueError(f"The soundtrack is only {audio_seconds:.2f}s; at least one second is needed.")
        frames = align_frame_count(round(audio_seconds * FPS))
        total_latents = latents_for_frames(frames)
        audio_t = round(frames / FPS * AUDIO_LATENT_RATE)
        plan = build_chunk_plan(total_latents, groups)
        log = lambda text: print(text, flush=True)
        log(f"[H3 Streaming] {audio_seconds:.2f}s audio -> {frames} frames ({frames / FPS:.2f}s), {total_latents} video latents, "
            f"{audio_t} audio latents, {len(plan)} chunks, canvas {width}x{height}, steps {steps}")

        fitted = fit_waveform(waveform, audio_t * hop)
        encoded = audio_vae.encode(fitted.unsqueeze(0).movedim(1, -1))
        if encoded.ndim != 4 or encoded.shape[1] != 32 or encoded.shape[2] != 2:
            raise ValueError(f"The audio VAE produced {tuple(encoded.shape)}; connect the MiniMax H3 audio VAE.")
        if encoded.shape[-1] != audio_t:
            encoded = torch.nn.functional.pad(encoded[..., :audio_t], (0, max(0, audio_t - encoded.shape[-1])))
        audio_latent = encoded.to(torch.float32).cpu()
        del encoded

        options = {
            "kv_cache_dtype": kv_cache_dtype, "kv_cache_device": kv_cache_device, "audio_mode": audio_mode,
            "first_frame_anchor": first_frame_anchor, "text_position": text_position, "prefix_normalization": bool(prefix_normalization),
            "sink_latents": int(sink_latents), "context_chunks": int(context_chunks), "audio_reset_requests": int(audio_reset_requests),
        }
        progress = comfy.utils.ProgressBar(len(plan) * (int(steps) + 1), node_id=unique_id) if unique_id is not None else comfy.utils.ProgressBar(len(plan) * (int(steps) + 1))
        if weights == WEIGHT_MODES[0] and getattr(model, "is_dynamic", lambda: False)():
            try:
                model = model.clone(disable_dynamic=True)
                log("[H3 Streaming] using a non-dynamic model load so the weights stay resident next to the KV cache")
            except Exception as error:  # older/newer patcher without the delegate path
                LOG.warning("[H3 Streaming] could not create a non-dynamic model clone (%s); using dynamic loading", error)
        run = H3StreamingRun(model, positive, options, progress=progress)

        # memory plan: model + KV cache + activations
        frame_rows = (lat_h // 2) * (lat_w // 2)
        max_chunk_latents = max(c["lat_stop"] - c["lat_start"] for c in plan)
        max_chunk_rows = max_chunk_latents * frame_rows + 2 * max(c["aud_stop"] - c["aud_start"] for c in plan)
        history_rows = min(int(sink_latents), max_chunk_latents) * frame_rows + int(context_chunks) * max_chunk_rows
        heads = model.model.diffusion_model.blocks[0].attn.heads
        head_dim = model.model.diffusion_model.blocks[0].attn.head_dim
        layers = len(model.model.diffusion_model.blocks)
        kv_bytes_per_row = layers * 2 * heads * head_dim * (1 if kv_cache_dtype == "fp8_e4m3" else 2) + layers * 2 * heads * 2
        peak_rows = history_rows + max_chunk_rows  # the new chunk is staged before the oldest one is dropped
        peak_cache_bytes = peak_rows * kv_bytes_per_row
        text_len_estimate = int(positive[0][0].shape[1]) + (frame_rows if first_frame_anchor != ANCHOR_MODES[2] else 0)
        seq = max_chunk_rows + text_len_estimate
        activation_bytes = seq * 200_000 + (history_rows + seq) * heads * head_dim * 2 * 2 * 2 + 768 * 1024 ** 2
        device = model.load_device
        total_vram = mm.get_total_memory(device)
        margin = 2 * 1024 ** 3
        budget = total_vram - model.model_size() - activation_bytes - margin
        if kv_cache_device == "auto":
            kv_cache_device = "gpu" if peak_cache_bytes <= budget else "cpu pinned"
            log(f"[H3 Streaming] auto KV cache placement: peak {peak_cache_bytes / 1024 ** 3:.2f} GB vs GPU budget {budget / 1024 ** 3:.2f} GB "
                f"(total {total_vram / 1024 ** 3:.1f} GB - weights {model.model_size() / 1024 ** 3:.1f} GB - activations - margin) -> {kv_cache_device}")
        options["kv_cache_device"] = kv_cache_device
        cache_bytes = peak_cache_bytes if kv_cache_device == "gpu" else 0
        run.memory_estimate = cache_bytes + activation_bytes
        log(f"[H3 Streaming] memory plan: KV cache up to {history_rows} rows (peak {peak_rows}) = {peak_cache_bytes / 1024 ** 3:.2f} GB ({kv_cache_dtype}, {kv_cache_device}), "
            f"activations ~{activation_bytes / 1024 ** 3:.2f} GB per forward of {seq} tokens")
        mm.load_models_gpu([model], memory_required=run.memory_estimate)
        free = mm.get_free_memory(model.load_device)
        log(f"[H3 Streaming] model loaded ({model.loaded_size() / 1024 ** 3:.2f} of {model.model_size() / 1024 ** 3:.2f} GB weights on the GPU), "
            f"free VRAM {free / 1024 ** 3:.2f} GB")
        if kv_cache_device == "gpu" and cache_bytes + activation_bytes > free:
            LOG.warning("[H3 Streaming] the planned KV cache + activations (%.2f GB) exceed free VRAM (%.2f GB); lower the resolution, "
                        "sink_latents/context_chunks, or set kv_cache_device=cpu pinned", (cache_bytes + activation_bytes) / 1024 ** 3, free / 1024 ** 3)

        text_len = run.prepare_conditioning(lat_h, lat_w, seed)
        log(f"[H3 Streaming] text rows {text_len}, first-frame rows {0 if run.cond_rows is None else run.cond_rows.shape[0]}, "
            f"chunk rows up to {max_chunk_rows}, KV history up to {history_rows} rows")

        video_latent = torch.zeros(1, 24, total_latents, lat_h, lat_w, dtype=torch.float32)
        writer = decoder = None
        video_path = ""
        temp_video = temp_wav = None
        output_dir = folder_paths.get_output_directory()
        if save_video:
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, width, height)
            os.makedirs(full_output_folder, exist_ok=True)
            base = f"{filename}_{counter:05}_"
            video_path = os.path.join(full_output_folder, base + ".mp4")
            temp_video = os.path.join(full_output_folder, base + ".video_only.mp4")
            temp_wav = os.path.join(full_output_folder, base + ".audio.wav")
            writer = FFmpegWriter(temp_video, width, height, FPS, crf=int(video_crf))
            decoder = IncrementalH3Decoder(video_vae, total_latents, writer)
        interval_latents = 0
        if decode_interval_seconds > 0:
            interval_latents = max(GROUP_LATENTS, int(round(decode_interval_seconds * FPS / GROUP_FRAMES)) * GROUP_LATENTS)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                run.run(video_latent, audio_latent, plan, seed, int(steps), shift_video, shift_audio, decoder=decoder,
                        decode_interval_latents=interval_latents, log=log)
            if writer is not None:
                writer.close()
                write_wav(temp_wav, fit_waveform(waveform, int(round(audio_seconds * sample_rate))), sample_rate)
                mux_audio(temp_video, temp_wav, video_path, audio_seconds, frames=int(round(audio_seconds * FPS)))
                for path in (temp_video, temp_wav):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        total = time.perf_counter() - started
        out_audio = {"waveform": fit_waveform(waveform, int(round(frames / FPS * sample_rate))).unsqueeze(0), "sample_rate": sample_rate}
        report = {
            "audio_seconds": round(audio_seconds, 3), "frames": frames, "video_latents": total_latents, "audio_latents": audio_t,
            "canvas": [width, height], "chunks": len(plan), "steps": int(steps), "sigmas_video": [round(s, 4) for s in select_sigmas(steps, shift_video)],
            "sigmas_audio": [round(s, 4) for s in select_sigmas(steps, shift_audio)], "chunk_groups": list(groups),
            "options": {**options, "weights": weights}, "timing": {**run.timing, "total_seconds": round(total, 2), "decode_seconds": round(decoder.seconds, 2) if decoder else 0.0,
                                            "seconds_per_video_second": round(total / (frames / FPS), 3)},
            "video_path": video_path, "frames_written": decoder.frames_written if decoder else 0,
        }
        text = json.dumps(report, indent=2)
        log("[H3 Streaming] done: " + json.dumps({"total_seconds": report["timing"]["total_seconds"], "dit_seconds": round(run.timing["dit_seconds"], 2),
                                                  "frames": frames, "video": video_path}))
        out_latent = {"samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent))}
        video_output = None
        if video_path and os.path.isfile(video_path):
            try:
                from comfy_api.latest import InputImpl

                video_output = InputImpl.VideoFromFile(video_path)
            except Exception:
                video_output = None
        return {"ui": {"text": [text]}, "result": (out_latent, out_audio, video_output, video_path, text)}


NODE_CLASS_MAPPINGS = {"SEH3StreamingSampler": SEH3StreamingSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"SEH3StreamingSampler": "H3 Streaming Sampler - TaoMate 3-Step (Audio + Image -> Long Video)"}
