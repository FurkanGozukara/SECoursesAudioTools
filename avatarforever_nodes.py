"""Native AvatarForever image + audio / text + audio streaming nodes."""

import json
import math
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.model_management as mm
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import InputImpl
from comfy_extras.nodes_audio import VAEEncodeAudio

from .avatarforever_model import avatar_forward
from .h3_streaming_nodes import FFmpegWriter, mux_audio, write_wav

SIGMAS = "1.0, 0.98125, 0.909375, 0.421875, 0.0"
DEFAULT_MODEL = "avatarforever-ltx-2.3-22b-INT8-ConvRot-HQ.safetensors"


def chunk_plan(total, audio_steps, chunk_size):
    ranges = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]
    boundaries = [0] + [min(audio_steps, math.ceil(end * audio_steps / total)) for _, end in ranges]
    return [(start, end, boundaries[i], boundaries[i + 1]) for i, (start, end) in enumerate(ranges)]


def selected_chunks(index, history_count, sink):
    first = 0 if history_count == -1 else max(0, index - history_count)
    result = list(range(first, index + 1))
    return [0] + result if sink and first else result


def parse_sigmas(text):
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) < 2 or values[-1] != 0 or not all(math.isfinite(v) for v in values):
        raise ValueError("AvatarForever sigmas need at least two finite values, ending in 0.")
    if any(a <= b or b < 0 for a, b in zip(values, values[1:])):
        raise ValueError("AvatarForever sigmas must decrease to 0.")
    return values


def prepare_audio(audio, fps, start, duration, lead_in):
    rate = int(audio["sample_rate"])
    source = audio["waveform"].detach().cpu()
    if source.shape[0] != 1:
        raise ValueError("AvatarForever takes one soundtrack per run.")
    begin = round(start * rate)
    end = min(source.shape[-1], begin + round(duration * rate)) if duration > 0 else source.shape[-1]
    source = source[..., begin:end]
    if source.shape[-1] == 0:
        raise ValueError("The selected audio range is empty; reduce Audio Start Seconds.")
    source = F.pad(source, (round(lead_in * rate), 0))
    seconds = source.shape[-1] / rate
    frames = max(9, math.ceil((seconds * fps - 1) / 8) * 8 + 1)
    padded = F.pad(source, (0, max(0, round(frames / fps * rate) - source.shape[-1])))
    return {"waveform": padded, "sample_rate": rate}, seconds, frames


def decode_to_writer(vae, latents, writer, tiled, spatial_tile, spatial_overlap, temporal_tile, temporal_overlap, frame_processor=None):
    """Rolling temporal tiles with native LTX index geometry; no full RGB movie in RAM."""
    tile = max(2, temporal_tile // 8)
    overlap = min(tile - 1, max(1, temporal_overlap // 8))
    step = tile - overlap
    total = latents.shape[2]
    pending = None
    for start in range(0, total, step):
        mm.throw_exception_if_processing_interrupted()
        end = min(total, start + tile)
        clip = latents[:, :, start:end]
        if tiled:
            size = max(2, spatial_tile // 32)
            pixels = vae.decode_tiled(clip, tile_x=size, tile_y=size,
                                      overlap=min(size - 1, spatial_overlap // 32), tile_t=tile, overlap_t=overlap)
        else:
            pixels = vae.decode(clip)
        pixels = pixels.reshape(-1, *pixels.shape[-3:]).cpu()
        # A tile beginning at latent k starts at pixel 8*k (native tiled VAE).
        # Each tile emits 8*n-7 frames, hence adjacent tiles share 8*overlap-7.
        shared = min(8 * overlap - 7, pixels.shape[0])
        if pending is not None:
            weight = torch.linspace(0, 1, shared + 2)[1:-1].reshape(-1, 1, 1, 1)
            pixels[:shared] = pending[:shared] * (1 - weight) + pixels[:shared] * weight
            pending = None
        final = end == total
        keep = 0 if final else 8 * overlap - 7
        if keep:
            pending = pixels[-keep:].clone()
            pixels = pixels[:-keep]
        pixels = (pixels * 255).round().clamp_(0, 255).to(torch.uint8)
        if frame_processor is not None:
            pixels = frame_processor(pixels)
        writer.write(pixels)
        del pixels
        if final:
            break


class SEAvatarForeverLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),
                                           {"default": DEFAULT_MODEL})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "SECourses/AvatarForever"
    DESCRIPTION = "Native LTX loader plus the checkpoint's three trained AvatarForever channel-conditioning tensors. No downloads."

    def load(self, unet_name):
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        names = ["video_channel_condition_proj.weight", "video_channel_condition_gate.weight", "video_channel_condition_gate.bias"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            prefix = next((p for p in ("model.diffusion_model.", "diffusion_model.", "") if p + names[0] in keys), None)
            if prefix is None or any(prefix + name not in keys for name in names):
                raise ValueError("Select an AvatarForever checkpoint containing its channel-condition projection and gate.")
            weights = tuple(handle.get_tensor(prefix + name).clone() for name in names)
        model = nodes.UNETLoader().load_unet(unet_name, "default")[0]
        model = model.clone()
        model.model_options["avatarforever_weights"] = weights
        model.model_options["avatarforever_checkpoint"] = unet_name
        return (model,)


class SEAvatarForeverSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "video_vae": ("VAE",),
            "audio_vae": ("VAE",), "audio": ("AUDIO",),
            "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 32}),
            "match_image_aspect": ("BOOLEAN", {"default": True, "tooltip": "With an image, preserve its aspect at the Width x Height pixel budget (rounded to 32). Prevents portrait faces being cropped into landscape."}),
            "fps": ("FLOAT", {"default": 25, "min": 1, "max": 120}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "use_image": ("BOOLEAN", {"default": True, "tooltip": "Off: audio + text, even when an image is connected."}),
            "image_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
            "image_compression": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "LTX input-image compression. 0 disables it."}),
            "audio_start_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 100000}),
            "duration_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 = entire soundtrack. Otherwise select this many seconds from Audio Start."}),
            "lead_in_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 10}),
            "sigmas": ("STRING", {"default": SIGMAS, "tooltip": "Official four-step schedule. CFG is 1; no negative prompt."}),
            "chunk_size": ("INT", {"default": 4, "min": 1, "max": 128, "tooltip": "Video latent frames per AR chunk. Official default 4."}),
            "history_chunks": ("INT", {"default": 1, "min": -1, "max": 10000, "tooltip": "Recent chunks retained. -1 = all history (memory grows with duration)."}),
            "sink_first_chunk": ("BOOLEAN", {"default": True}),
            "relative_positions": ("BOOLEAN", {"default": True}),
            "forever_cache": ("BOOLEAN", {"default": False, "tooltip": "Official approximate history-feature reuse after step 1 of each chunk. Faster, may change output. Released CLI default is off."}),
            "cache_device": (["auto", "gpu", "cpu"], {"default": "auto"}),
            "channel_condition": ("BOOLEAN", {"default": True, "tooltip": "Learned image conditioning on every new chunk; without image derives identity from generated chunk 0."}),
            "channel_mode": (["gated", "add"], {"default": "gated"}),
            "first_frame_prefix": ("BOOLEAN", {"default": False, "tooltip": "Reuse the generated first latent frame and aligned audio if chunk 0 is no longer in history."}),
            "prefix_position": (["prepend", "append"], {"default": "prepend"}),
            "resident_weights": ("BOOLEAN", {"default": True, "tooltip": "Use ComfyUI's non-dynamic load for stable performance beside the history cache."}),
            "tiled_vae": ("BOOLEAN", {"default": True}),
            "spatial_tile": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
            "spatial_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 32}),
            "temporal_tile": ("INT", {"default": 128, "min": 16, "max": 2048, "step": 8}),
            "temporal_overlap": ("INT", {"default": 32, "min": 8, "max": 512, "step": 8}),
            "crf": ("INT", {"default": 17, "min": 0, "max": 51}),
            "encoding_preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "fast"}),
            "filename_prefix": ("STRING", {"default": "video/AvatarForever/Avatar"}),
        }, "optional": {
            "image": ("IMAGE",),
            "mouth_enhancement": ("BOOLEAN", {"default": False, "tooltip": "Aligned CodeFormer mouth-only restoration before video export. The unified preset enables it. Off retains the original generation path."}),
            "mouth_fidelity": ("FLOAT", {"default": .9, "min": 0, "max": 1, "step": .01, "tooltip": "CodeFormer fidelity; 0.9 is the selected comparison recipe."}),
            "mouth_blend": ("FLOAT", {"default": .7, "min": 0, "max": 1, "step": .01, "tooltip": "Feathered mouth-only blend. 0 leaves every frame unchanged."}),
            "mouth_model": ("STRING", {"default": "codeformer.pth", "tooltip": "Existing file under models/facerestore_models. No automatic download."}),
            "mouth_detector": ("STRING", {"default": "models/buffalo_l/det_10g.onnx", "tooltip": "Existing SCRFD detector under models/insightface. No automatic download."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "sample"
    OUTPUT_NODE = True
    CATEGORY = "SECourses/AvatarForever"
    DESCRIPTION = "Image + audio or text + audio to any-length AR avatar video. Uses native LTX-2.3/Gemma 3, trained channel conditioning and optional ForeverCache. Source audio is preserved."

    def sample(self, model, positive, video_vae, audio_vae, audio, width, height, fps, seed,
               use_image=True, image_strength=1.0, image_compression=0, audio_start_seconds=0,
               duration_seconds=0, lead_in_seconds=0, sigmas=SIGMAS, chunk_size=4, history_chunks=1,
               sink_first_chunk=True, relative_positions=True, forever_cache=False, cache_device="auto",
               channel_condition=True, channel_mode="gated", first_frame_prefix=False, prefix_position="prepend",
               resident_weights=True, tiled_vae=True, spatial_tile=512, spatial_overlap=64,
               temporal_tile=128, temporal_overlap=32, crf=17, encoding_preset="fast",
               filename_prefix="video/AvatarForever/Avatar", image=None, unique_id=None, match_image_aspect=True,
               mouth_enhancement=False, mouth_fidelity=.9, mouth_blend=.7,
               mouth_model="codeformer.pth", mouth_detector="models/buffalo_l/det_10g.onnx"):
        started = time.perf_counter()
        if mouth_enhancement and mouth_blend != 0:
            from .codeformer_mouth import CodeFormerMouth, model_path
            model_path("facerestore_models", mouth_model)
            model_path("insightface", mouth_detector)
        if "avatarforever_weights" not in model.model_options:
            raise ValueError("Connect the AvatarForever Model Loader; the normal UNET loader drops its image-conditioning weights.")
        for key, value in (("cache_device", cache_device), ("channel_mode", channel_mode),
                           ("prefix_position", prefix_position), ("encoding_preset", encoding_preset)):
            if value not in self.INPUT_TYPES()["required"][key][0]:
                raise ValueError(f"Invalid AvatarForever {key}: {value}")
        schedule = parse_sigmas(sigmas)
        if width < 32 or height < 32 or fps <= 0 or chunk_size < 1 or history_chunks < -1:
            raise ValueError("AvatarForever needs positive dimensions, FPS and chunk size; history must be -1 or greater.")
        width, height = math.ceil(width / 32) * 32, math.ceil(height / 32) * 32
        if match_image_aspect and use_image and image is not None:
            area, aspect = width * height, image.shape[2] / image.shape[1]
            width = max(64, round(math.sqrt(area * aspect) / 32) * 32)
            height = max(64, round(math.sqrt(area / aspect) / 32) * 32)
        soundtrack, seconds, frames = prepare_audio(audio, fps, audio_start_seconds, duration_seconds, lead_in_seconds)
        total = (frames - 1) // 8 + 1
        # Encode a margin, then keep the native audio-token count for the video timeline.
        encode_audio = {**soundtrack, "waveform": F.pad(soundtrack["waveform"], (0, soundtrack["sample_rate"]))}
        audio_latent = VAEEncodeAudio.execute(audio_vae, encode_audio).args[0]["samples"].float().cpu()
        dm = model.model.diffusion_model
        ap = dm.a_patchifier
        audio_steps = round(frames / fps * ap.sample_rate / ap.hop_length / ap.audio_latent_downsample_factor)
        if audio_latent.shape[1] != dm.num_audio_channels or audio_latent.shape[-1] != dm.audio_frequency_bins:
            raise ValueError("Connect an LTX-2.3 audio VAE.")
        audio_latent = audio_latent[:, :, :audio_steps].contiguous()
        if audio_latent.shape[2] < audio_steps:
            audio_latent = torch.nn.functional.pad(audio_latent, (0, 0, 0, audio_steps - audio_latent.shape[2]))
        anchor = None
        if use_image and image is not None:
            pixels = comfy.utils.common_upscale(image[:1, :, :, :3].movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            if image_compression:
                from comfy_extras.nodes_lt import LTXVPreprocess
                pixels = LTXVPreprocess.execute(pixels, image_compression).args[0]
            anchor = video_vae.encode(pixels).float().cpu()[:, :, :1].clone()
        video = torch.zeros((1, dm.in_channels, total, height // 32, width // 32), dtype=torch.float32)
        plan = chunk_plan(total, audio_steps, chunk_size)
        frame_tokens = (height // 32) * (width // 32)
        device = model.load_device
        dtype = model.model.get_dtype_inference()
        if resident_weights:
            model = model.clone(disable_dynamic=True)
        else:
            model = model.clone()
        max_history = total if history_chunks == -1 else min(total, (history_chunks + int(sink_first_chunk)) * chunk_size + 1)
        cache_bytes = max_history * frame_tokens * dm.inner_dim * 4 * len(dm.transformer_blocks)
        activation_bytes = (max_history + chunk_size) * frame_tokens * 180000 + 512 * 1024**2
        if cache_device == "auto":
            cache_device = "gpu" if cache_bytes + activation_bytes + model.model_size() + 2 * 1024**3 < mm.get_total_memory(device) else "cpu"
        memory = activation_bytes + (cache_bytes if forever_cache and cache_device == "gpu" else 0)
        mm.load_models_gpu([model], memory_required=memory)
        dm = model.model.diffusion_model
        context = dm.preprocess_text_embeds(positive[0][0].to(device=device, dtype=dtype),
                    unprocessed=positive[0][1].get("unprocessed_ltxav_embeds", False))
        attention_mask = positive[0][1].get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        weights = tuple(w.to(device=device, dtype=dtype) for w in model.model_options["avatarforever_weights"])
        generator = torch.Generator("cpu").manual_seed(seed)
        progress = comfy.utils.ProgressBar(len(plan) * (len(schedule) - 1), node_id=unique_id)
        completed = 0
        channel = anchor if channel_condition else None
        print(f"[AvatarForever] {seconds:.2f}s -> {frames} frames, {len(plan)} chunks, {width}x{height}, cache={forever_cache}/{cache_device}", flush=True)
        dit_started = time.perf_counter()
        for index, (start, end, astart, aend) in enumerate(plan):
            mm.throw_exception_if_processing_interrupted()
            chunk_started = time.perf_counter()
            selected = selected_chunks(index, history_chunks, sink_first_chunk)
            vi = [j for k in selected for j in range(plan[k][0], plan[k][1])]
            ai = [j for k in selected for j in range(plan[k][2], plan[k][3])]
            vpos = list(range(len(vi))) if relative_positions else vi.copy()
            apos = list(range(len(ai))) if relative_positions else ai.copy()
            dynamic_prefix = first_frame_prefix and index > 0 and 0 not in selected
            prefix_audio = math.ceil(audio_steps / total)
            if dynamic_prefix:
                if prefix_position == "prepend":
                    vi, ai = [0] + vi, list(range(prefix_audio)) + ai
                    vpos, apos = [0] + vpos, list(range(prefix_audio)) + apos
                else:
                    vi, ai = vi + [0], ai + list(range(prefix_audio))
                    vpos, apos = vpos + [0], apos + list(range(prefix_audio))
            vstart = vi.index(start)
            aoffset = ai.index(astart)
            vsel = slice(vstart * frame_tokens, (vstart + end - start) * frame_tokens)
            asel = slice(aoffset, aoffset + aend - astart)
            current = torch.randn((1, dm.in_channels, end - start, height // 32, width // 32), generator=generator).to(device)
            fixed = anchor.to(device) if index == 0 and anchor is not None else None
            if fixed is not None:
                current[:, :, :1] = fixed * image_strength + current[:, :, :1] * (1 - image_strength)
            cache = {} if forever_cache else None
            for sigma, next_sigma in zip(schedule, schedule[1:]):
                window = video[:, :, vi].to(device)
                window[:, :, vstart:vstart + end - start] = current
                mask = torch.zeros((1, len(vi), 1), device=device)
                mask[:, vstart:vstart + end - start] = sigma
                if fixed is not None:
                    mask[:, vstart] *= 1 - image_strength
                times = mask.repeat_interleave(frame_tokens, dim=1)
                state = {"sigma": torch.tensor([sigma], device=device), "current": (vsel, asel),
                         "weights": weights, "channel": channel.to(device=device, dtype=dtype) if channel is not None else None,
                         "channel_mode": channel_mode, "video_positions": vpos, "audio_positions": apos,
                         "cache": cache, "cache_device": device if cache_device == "gpu" else torch.device("cpu")}
                opts = {**model.model_options.get("transformer_options", {}), "avatarforever": state}
                velocity = avatar_forward(dm, [window.to(dtype), audio_latent[:, :, ai].to(device=device, dtype=dtype)],
                              (times, torch.zeros((1, len(ai), 1), device=device)), context,
                              attention_mask=attention_mask, frame_rate=fps, transformer_options=opts)[0]
                velocity = velocity[:, :, vstart:vstart + end - start].float()
                if fixed is not None:
                    prediction = current[:, :, :1] - sigma * velocity[:, :, :1]
                    prediction = prediction * (1 - image_strength) + fixed * image_strength
                    velocity[:, :, :1] = (current[:, :, :1] - prediction) / sigma
                current = current + (next_sigma - sigma) * velocity
                if fixed is not None:
                    # Official post_process_latent blends the prediction AND update.
                    current[:, :, :1] = current[:, :, :1] * (1 - image_strength) + fixed * image_strength
                completed += 1
                progress.update_absolute(completed)
            video[:, :, start:end] = current.cpu()
            if index == 0 and channel_condition and channel is None:
                channel = video[:, :, :1].clone()
            print(f"[AvatarForever] chunk {index + 1}/{len(plan)}: {time.perf_counter() - chunk_started:.2f}s", flush=True)
            del cache, state, opts, current, window, velocity
        dit_seconds = time.perf_counter() - dit_started
        del context, weights, channel
        output_dir = folder_paths.get_output_directory()
        out_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, width, height)
        os.makedirs(out_folder, exist_ok=True)
        base = os.path.join(out_folder, f"{filename}_{counter:05}_")
        output, silent, wav = base + ".mp4", base + ".video_only.mp4", base + ".audio.wav"
        mouth = None
        if mouth_enhancement and mouth_blend != 0:
            mouth = CodeFormerMouth(mouth_model, mouth_detector, mouth_fidelity, mouth_blend)
        writer = FFmpegWriter(silent, width, height, fps, crf, encoding_preset)
        try:
            decode_to_writer(video_vae, video, writer, tiled_vae, spatial_tile, spatial_overlap, temporal_tile, temporal_overlap,
                             mouth.process if mouth is not None else None)
            writer.close()
            write_wav(wav, soundtrack["waveform"][0], soundtrack["sample_rate"])
            mux_audio(silent, wav, output, seconds, frames=math.ceil(seconds * fps))
        except BaseException:
            writer.abort()
            raise
        else:
            os.remove(silent)
            os.remove(wav)
        report = {"checkpoint": model.model_options["avatarforever_checkpoint"], "width": width, "height": height,
                  "seed": seed, "audio_start_seconds": audio_start_seconds, "lead_in_seconds": lead_in_seconds,
                  "fps": fps, "audio_seconds": seconds, "generated_frames": frames, "saved_frames": math.ceil(seconds * fps),
                  "chunks": len(plan), "sigmas": schedule, "forever_cache": forever_cache, "cache_device": cache_device,
                  "chunk_size": chunk_size, "history_chunks": history_chunks, "sink_first_chunk": sink_first_chunk,
                  "relative_positions": relative_positions, "channel_mode": channel_mode,
                  "first_frame_prefix": first_frame_prefix, "prefix_position": prefix_position,
                  "image_strength": image_strength, "resident_weights": resident_weights,
                  "channel_condition": channel_condition, "image": anchor is not None, "dit_seconds": round(dit_seconds, 2),
                  "mouth_enhancement": mouth.report if mouth is not None else {"enabled": False},
                  "total_seconds": round(time.perf_counter() - started, 2), "path": output}
        text = json.dumps(report, indent=2)
        with open(base + ".json", "w", encoding="utf-8") as handle:
            handle.write(text)
        return {"ui": {"text": [text], "images": [{"filename": os.path.basename(output),
                    "subfolder": subfolder, "type": "output"}], "animated": (True,)},
                "result": (InputImpl.VideoFromFile(output), text)}


NODE_CLASS_MAPPINGS = {"SEAvatarForeverLoader": SEAvatarForeverLoader, "SEAvatarForeverSampler": SEAvatarForeverSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"SEAvatarForeverLoader": "AvatarForever Model Loader (Native INT8 / BF16 / FP8)",
                            "SEAvatarForeverSampler": "AvatarForever - Audio + Optional Image to Long Video"}
