"""Audio-driven H3 continuation using native per-stream denoise masks.

The obvpm timeline's useful distinction is preserving clean latents at a join,
rather than repeatedly encoding the previous video's pixels. This controller
automates that method for a complete soundtrack. It uses native ComfyUI sampling
and the existing SECourses incremental decoder; no alternate model or kernel.
"""

import json
import logging
import math
import os
import time

import torch

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import folder_paths
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, RandomNoise, SamplerCustomAdvanced

from .h3_streaming_nodes import (
    FPS, IncrementalH3Decoder, FFmpegWriter, audio_boundary,
    fit_waveform, latents_for_frames, mux_audio, normalize_waveform, write_wav,
)

LOG = logging.getLogger("SECoursesH3LongLipSync")


def continuation_plan(seconds, window_frames=243, overlap_frames=39):
    """Shared 24-fps/40-Hz grid: 51k+39-frame windows, phase-zero starts.

    A 243-frame window minus 39 held frames advances exactly 204 frames / 340
    audio ticks / 60 video latents. No rounded per-clip durations accumulate.
    """
    if window_frames not in (141, 192, 243, 294, 345):
        raise ValueError("window_frames must be 141, 192, 243, 294 or 345")
    if overlap_frames not in (39, 90) or overlap_frames >= window_frames:
        raise ValueError("overlap_frames must be 39 or 90 and shorter than the window")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("The driving audio must have a positive, finite duration")
    target = math.ceil(seconds * FPS)
    groups_per_window = (window_frames - overlap_frames) // 51
    total_groups = max(1, math.ceil((target - overlap_frames) / 51))
    count = math.ceil(total_groups / groups_per_window)
    min_groups = math.ceil((141 - overlap_frames) / 51)
    total_groups = max(total_groups, count * min_groups)
    groups, extra = divmod(total_groups, count)
    start = 0
    plan = []
    for index in range(count):
        advance = 51 * (groups + (index < extra))
        frames = overlap_frames + advance
        end = start + frames
        s = (start // 17) * 5
        e = s + latents_for_frames(frames)
        plan.append({"start_frame": start, "end_frame": end, "start_latent": s, "end_latent": e,
                     "audio_start": audio_boundary(start), "audio_end": audio_boundary(end),
                     "held_latents": 0 if not plan else latents_for_frames(overlap_frames)})
        start += advance
    return plan


def window_conditioning(positive, encoded_audio, continuation):
    # An FL2VA first-frame anchor belongs to the first window. Later windows get
    # their pose from the held latent overlap. Ref2VA identity references remain.
    result = []
    for embedding, meta in positive:
        values = dict(meta)
        keyframes = [] if continuation else [dict(k) for k in values.get("minimax_keyframes", []) if "audio_latent" not in k]
        keyframes.append({"resolved_frame_index": 0, "audio_latent": encoded_audio})
        values["minimax_keyframes"] = keyframes
        result.append([embedding, values])
    return result


class MouthPassWriter:
    """Restores the mouth of every decoded uint8 chunk before the encoder receives it."""

    def __init__(self, writer, mouth):
        self.writer, self.mouth = writer, mouth

    def write(self, frames_uint8):
        return self.writer.write(self.mouth.process(frames_uint8.contiguous().clone()))

    def __getattr__(self, name):
        return getattr(self.writer, name)


class SEH3LongLipSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Native H3 FL2VA or Ref2VA with the matching Turbo LoRA and sigma shifts already applied."}),
            "positive": ("CONDITIONING",),
            "latent": ("LATENT", {"tooltip": "Uses the canvas from the native H3 encoder. Duration follows the entire audio."}),
            "audio": ("AUDIO",),
            "audio_vae": ("VAE",),
            "video_vae": ("VAE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
            "window_frames": (["243", "141", "192", "294", "345"], {"tooltip": "243 = 10.125-second sampling windows. Lower uses less memory; larger windows offer more context. All stay within the trained 15-second span."}),
            "overlap_frames": (["39", "90"], {"tooltip": "39 = 1.625 seconds held exactly in latent space at every join. 90 = 3.75 seconds, slower with more context."}),
            "filename_prefix": ("STRING", {"default": "video/H3_Long_Lip_Sync/Avatar"}),
            "video_crf": ("INT", {"default": 17, "min": 0, "max": 40}),
        }, "optional": {
            "mouth_pass": ("BOOLEAN", {"default": False, "tooltip": "Aligned CodeFormer mouth-only restoration of every decoded chunk before encoding. Off keeps the original frames."}),
            "mouth_fidelity": ("FLOAT", {"default": .9, "min": 0, "max": 1, "step": .01, "tooltip": "CodeFormer fidelity; 0.9 is the tested recipe."}),
            "mouth_blend": ("FLOAT", {"default": .7, "min": 0, "max": 1, "step": .01, "tooltip": "Feathered mouth-only blend. 0 leaves every frame unchanged."}),
            "mouth_model": ("STRING", {"default": "codeformer.pth", "tooltip": "Existing file under models/facerestore_models. No automatic download."}),
            "mouth_detector": ("STRING", {"default": "models/buffalo_l/det_10g.onnx", "tooltip": "Existing SCRFD detector under models/insightface. No automatic download."}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "report")
    FUNCTION = "generate"
    CATEGORY = "SECourses/MiniMax H3 Lip Synch"
    OUTPUT_NODE = True
    DESCRIPTION = ("One soundtrack to a continuous avatar: native H3 windows with clean latent overlap and locked source audio. "
                   "Avoids repeated pixel re-encoding. It reduces one source of drift; it does not guarantee indefinite identity or perfect lip sync.")

    def generate(self, model, positive, latent, audio, audio_vae, video_vae, seed, steps,
                 window_frames="243", overlap_frames="39", filename_prefix="video/H3_Long_Lip_Sync/Avatar", video_crf=17,
                 mouth_pass=False, mouth_fidelity=.9, mouth_blend=.7,
                 mouth_model="codeformer.pth", mouth_detector="models/buffalo_l/det_10g.onnx"):
        mouth_pass = bool(mouth_pass) and mouth_blend != 0
        if mouth_pass:
            # Fail before sampling, not after a long generation, when a mouth model is missing.
            from .codeformer_mouth import model_path
            model_path("facerestore_models", mouth_model)
            model_path("insightface", mouth_detector)
        if type(model.get_model_object("diffusion_model")).__name__ != "MiniMaxH3Model":
            raise ValueError("H3 Long Lip Sync requires a native MiniMax H3 model")
        from comfy.ldm.minimax import model as h3
        if not hasattr(h3, "mask_row_values"):
            raise RuntimeError("Update ComfyUI: H3 latent continuation needs native per-token denoise masks")
        if not getattr(latent["samples"], "is_nested", False):
            raise ValueError("Connect the joint video/audio latent from a native H3 encoder")
        source_video, _ = latent["samples"].unbind()
        if source_video.shape[0] != 1:
            raise ValueError("H3 Long Lip Sync generates one soundtrack at a time")
        width, height = source_video.shape[-1] * 16, source_video.shape[-2] * 16
        rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        waveform = normalize_waveform(audio, rate)
        seconds = waveform.shape[-1] / rate
        plan = continuation_plan(seconds, int(window_frames), int(overlap_frames))
        total_steps = plan[-1]["end_latent"]
        total_ticks = plan[-1]["audio_end"]
        hop = int(getattr(audio_vae, "downscale_ratio", 800))
        start_all = time.perf_counter()
        encoded = audio_vae.encode(fit_waveform(waveform, total_ticks * hop).unsqueeze(0).movedim(1, -1)).cpu()
        if encoded.shape[-1] != total_ticks:
            raise RuntimeError(f"H3 audio VAE returned {encoded.shape[-1]} ticks, expected {total_ticks}")
        video = torch.zeros((1, 24, total_steps, height // 16, width // 16), dtype=torch.float32)
        sampler = comfy.samplers.ksampler("euler")
        sigmas = BasicScheduler.execute(model, "simple", int(steps), 1.0).args[0]
        report = {"method": "native masked latent continuation", "seconds": seconds, "canvas": [width, height],
                  "seed": seed, "steps": steps, "window_frames": int(window_frames), "overlap_frames": int(overlap_frames),
                  "audio_mode": "locked source + clean audio guide", "chunks": [], "timing": {}}
        progress = comfy.utils.ProgressBar(len(plan))
        sampling_start = time.perf_counter()
        for index, span in enumerate(plan):
            mm.throw_exception_if_processing_interrupted()
            start = time.perf_counter()
            s, e = span["start_latent"], span["end_latent"]
            av = encoded[..., span["audio_start"]:span["audio_end"]]
            v = video[:, :, s:e].clone()
            vm = torch.ones((1, 1, e - s, 1, 1), dtype=v.dtype)
            vm[:, :, :span["held_latents"]] = 0
            # Audio also informs the model as a clean guide, as in the accepted
            # short lip-sync preset. The final file always uses the source WAV.
            mask = comfy.nested_tensor.NestedTensor((vm, torch.zeros((1, 1, 2, av.shape[-1]), dtype=av.dtype)))
            work = {"samples": comfy.nested_tensor.NestedTensor((v, av)), "noise_mask": mask}
            cond = window_conditioning(positive, av, continuation=index > 0)
            guider = BasicGuider.execute(model, cond).args[0]
            noise = RandomNoise.execute((int(seed) + index) % (1 << 64)).args[0]
            LOG.info("[H3 Long Lip Sync] chunk %d/%d: frames %d..%d, audio ticks %d..%d", index + 1, len(plan),
                     span["start_frame"], span["end_frame"], span["audio_start"], span["audio_end"])
            out = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, work).args[0]
            generated, generated_audio = out["samples"].unbind()
            held = span["held_latents"]
            held_out = generated[:, :, :held].cpu()
            audio_out = generated_audio.cpu()
            overlap_error = (held_out - v[:, :, :held]).abs().max().item() if held else 0.0
            audio_error = (audio_out - av).abs().max().item()
            # Native latent normalization makes a float32 round trip. Validate
            # that only its rounding changed held samples, then retain the exact
            # original overlap below (never overwrite it with the round trip).
            if not torch.allclose(held_out, v[:, :, :held], atol=1e-5, rtol=1e-5) or not torch.allclose(audio_out, av, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"Native H3 masks changed held samples: overlap max error {overlap_error:g}, audio {audio_error:g}")
            video[:, :, s + held:e] = generated[:, :, held:].cpu()
            report["chunks"].append({**span, "seed": noise.seed, "seconds": time.perf_counter() - start,
                                     "overlap_retained_exactly": True, "encoded_audio_mask_preserved": True,
                                     "sampler_overlap_max_abs": overlap_error, "sampler_audio_max_abs": audio_error})
            progress.update(index + 1)
        report["timing"]["sampling_seconds"] = time.perf_counter() - sampling_start
        output_root = folder_paths.get_output_directory()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_root, width, height)
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, f"{filename}_{counter:05}_")
        # Trim decoded frames before H.264 encoding: trimming packets afterward
        # can drop a final B-frame while retaining a later presentation time.
        writer = FFmpegWriter(base + ".video_only.mp4", width, height, FPS, crf=int(video_crf),
                              frame_limit=math.ceil(seconds * FPS))
        mouth = None
        if mouth_pass:
            from .codeformer_mouth import CodeFormerMouth
            mouth = CodeFormerMouth(mouth_model, mouth_detector, mouth_fidelity, mouth_blend)
        decoder = IncrementalH3Decoder(video_vae, total_steps, writer if mouth is None else MouthPassWriter(writer, mouth))
        try:
            decoder.decode(video, total_steps, final=True)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        write_wav(base + ".source.wav", waveform, rate)
        mux_audio(base + ".video_only.mp4", base + ".source.wav", base + ".mp4", seconds, frames=math.ceil(seconds * FPS))
        report["timing"]["decode_seconds"] = decoder.seconds
        report["timing"]["total_seconds"] = time.perf_counter() - start_all
        report["raw_frames"] = decoder.frames_written
        report["output_frames"] = math.ceil(seconds * FPS)
        report["mouth_pass"] = mouth.report if mouth is not None else {"enabled": False}
        report["soundtrack"] = {"source": "complete driving audio, unchanged timing",
                               "intermediate": f"{rate} Hz normalized PCM WAV",
                               "output_codec": "AAC", "lossless_copy": False}
        report["video_path"] = base + ".mp4"
        report_text = json.dumps(report, indent=2)
        with open(base + ".json", "w", encoding="utf-8") as f:
            f.write(report_text)
        from comfy_api.input_impl import VideoFromFile
        output = VideoFromFile(base + ".mp4")
        return {"ui": {"images": [{"filename": os.path.basename(base + ".mp4"), "subfolder": subfolder,
                                     "type": "output", "format": "video/mp4"}], "text": [report_text]},
                "result": (output, base + ".mp4", report_text)}


NODE_CLASS_MAPPINGS = {"SEH3LongLipSync": SEH3LongLipSync}
NODE_DISPLAY_NAME_MAPPINGS = {"SEH3LongLipSync": "H3 Long Lip Sync — Latent Continuation"}
