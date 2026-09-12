"""LTX-2.5 audio-to-video helper nodes (SECourses).

These nodes remove the manual plumbing of the official Lightricks two-stage
audio-to-video graph:

* ``SELTX25LoadImageOptional`` - a Load Image node with a real "none" option.
  With an image the workflow becomes image + audio to video, without one it is
  audio + text to video.  It also owns the *target* video resolution and derives
  the half-resolution stage-1 canvas that the 2x latent upscaler needs.
* ``SELTX25AudioPrepare`` - trims the source audio (duration 0 = use the whole
  clip), derives the LTX ``8k+1`` frame count from the audio length, pads the
  audio with silence to the exact video length, encodes it with the LTX audio
  VAE and freezes the tokens (noise mask 0) so both stages keep the source audio.
* ``SELTX25ImageCondition`` - LTXVImgToVideoInplace + LTXVPreprocess in one node
  that simply passes the latent through when no image is connected.
* ``SEImageFitToSize`` - center-crops (or cover-resizes) decoded frames to the
  exact target size, so odd targets such as 1920x1080 work on the 32/64 px grid.
"""

import hashlib
import math
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers

LOG_PREFIX = "[SE LTX-2.5 A2V]"
NONE_OPTION = "none"

# LTX-2.5 native two-stage canvas: 960x544 -> 1920x1088 (1080p pixel budget).
LTX25_PIXEL_BUDGET = 1920 * 1080
LTX25_SPATIAL_MULTIPLE = 32  # video VAE spatial compression (latent grid)
LTX25_TARGET_MULTIPLE = 64  # keeps the half-resolution stage on the 32 px latent grid
LTX25_TEMPORAL_MULTIPLE = 8  # frame count must be 8k + 1
LTX25_RECOMMENDED_MAX_SECONDS = 20.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _list_input_images():
    input_dir = folder_paths.get_input_directory()
    try:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    except FileNotFoundError:
        return []
    files = folder_paths.filter_files_content_types(files, ["image"])
    return sorted(files)


def resolution_from_aspect(width, height, pixel_budget=LTX25_PIXEL_BUDGET, multiple=LTX25_TARGET_MULTIPLE):
    """Target video size for an image aspect ratio at the LTX-2.5 1080p pixel budget.

    16:9 and 9:16 sources map to the standard 1920x1080 / 1080x1920.  Everything
    else snaps both sides to ``multiple`` (64) so the half-resolution stage lands
    exactly on the 32 px latent grid and no crop is needed.  Mirrors the JS
    implementation in web/js/ltx25_a2v_auto_resolution.js.
    """
    width = float(width)
    height = float(height)
    if width <= 0 or height <= 0:
        return 1920, 1080
    aspect = width / height
    for standard_aspect, (tw, th) in ((16.0 / 9.0, (1920, 1080)), (9.0 / 16.0, (1080, 1920))):
        if abs(aspect / standard_aspect - 1.0) < 0.02:
            return tw, th

    ideal_w = math.sqrt(pixel_budget * aspect)
    best = None
    for w in sorted({math.floor(ideal_w / multiple) * multiple, math.ceil(ideal_w / multiple) * multiple}):
        if w < multiple:
            continue
        ideal_h = w / aspect
        for h in sorted({math.floor(ideal_h / multiple) * multiple, math.ceil(ideal_h / multiple) * multiple}):
            if h < multiple:
                continue
            aspect_error = round(abs(math.log((w / h) / aspect)), 6)
            pixel_error = abs(w * h - pixel_budget)
            key = (aspect_error, pixel_error, w)
            if best is None or key < best[0]:
                best = (key, int(w), int(h))
    if best is None:
        return 1920, 1080
    return best[1], best[2]


def stage1_size(target_width, target_height):
    """Half-resolution stage-1 canvas that, after the 2x latent upscale, covers the target."""
    gw = math.ceil(target_width / 2.0 / LTX25_SPATIAL_MULTIPLE) * LTX25_SPATIAL_MULTIPLE
    gh = math.ceil(target_height / 2.0 / LTX25_SPATIAL_MULTIPLE) * LTX25_SPATIAL_MULTIPLE
    return max(LTX25_SPATIAL_MULTIPLE * 2, gw), max(LTX25_SPATIAL_MULTIPLE * 2, gh)


def ltx_frame_count(seconds, fps, cover_tolerance_seconds=0.02):
    """Smallest 8k+1 frame count whose duration covers ``seconds`` of audio.

    If dropping one 8-frame block would cut less than ``cover_tolerance_seconds``
    from the tail (e.g. an 18.000 s clip at 24 fps), the shorter count is used.
    """
    seconds = max(0.0, float(seconds))
    fps = float(fps)
    needed_frames = seconds * fps
    k = math.ceil((needed_frames - 1.0) / LTX25_TEMPORAL_MULTIPLE - 1e-9)
    k = max(0, k)
    frames = LTX25_TEMPORAL_MULTIPLE * k + 1
    if k > 0:
        shorter = LTX25_TEMPORAL_MULTIPLE * (k - 1) + 1
        if shorter / fps >= seconds - cover_tolerance_seconds:
            frames = shorter
    return int(frames)


def _describe_aspect(width, height):
    g = math.gcd(int(width), int(height)) or 1
    a, b = int(width) // g, int(height) // g
    if a > 64 or b > 64:
        return f"{width / height:.3f}:1"
    return f"{a}:{b}"


# --------------------------------------------------------------------------- #
# 1. optional image + target resolution
# --------------------------------------------------------------------------- #
class SELTX25LoadImageOptional:
    """Load Image with a real 'none' choice + LTX-2.5 target resolution planner."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    [NONE_OPTION] + _list_input_images(),
                    {
                        "image_upload": True,
                        "default": NONE_OPTION,
                        "tooltip": "Optional first-frame image. Choose 'none' for audio + text to video. "
                        "Uploading or selecting an image switches to image + audio to video automatically.",
                    },
                ),
                "target_width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 0,
                        "max": 8192,
                        "step": 2,
                        "tooltip": "Final video width in pixels. The workflow generates at half size and upscales 2x. "
                        "0 = auto (1080p pixel budget, aspect ratio taken from the image).",
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 0,
                        "max": 8192,
                        "step": 2,
                        "tooltip": "Final video height in pixels. 0 = auto (1080p pixel budget, aspect ratio taken from the image).",
                    },
                ),
                "auto_resolution_from_image": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When you upload or pick an image, set target_width/target_height from its aspect ratio "
                        "at the LTX-2.5 1080p pixel budget (16:9 -> 1920x1080, 9:16 -> 1080x1920, 1:1 -> 1408x1408 ...). "
                        "You can still edit the values afterwards.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "target_width", "target_height", "stage1_width", "stage1_height", "info")
    OUTPUT_TOOLTIPS = (
        "The loaded image, or nothing when 'none' is selected (downstream SE nodes then skip image conditioning).",
        "Final video width (after 2x upscale and crop).",
        "Final video height (after 2x upscale and crop).",
        "Stage-1 generation width (half resolution on the 32 px latent grid).",
        "Stage-1 generation height (half resolution on the 32 px latent grid).",
        "Human readable summary of the resolution plan.",
    )
    FUNCTION = "load"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Optional image input for LTX-2.5 audio-to-video. 'none' = audio + text to video. "
        "Also plans the target resolution: enter the FINAL size you want, the workflow generates at half size "
        "and upscales 2x, then center-crops to the exact target."
    )

    def load(self, image, target_width, target_height, auto_resolution_from_image=True):
        image_tensor = None
        src_w = src_h = None
        label = "none  (audio + text to video)"

        if image and image != NONE_OPTION:
            image_path = folder_paths.get_annotated_filepath(image)
            pil_image = node_helpers.pillow(Image.open, image_path)
            try:
                pil_image.seek(0)  # animated files: first frame only
            except Exception:
                pass
            pil_image = node_helpers.pillow(ImageOps.exif_transpose, pil_image)
            rgb = pil_image.convert("RGB")
            src_w, src_h = rgb.size
            array = np.array(rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(array)[None,]
            image_tensor = image_tensor.to(
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype(),
            )
            label = f"{os.path.basename(image)}  {src_w}x{src_h} ({_describe_aspect(src_w, src_h)})  -> image + audio to video"

        tw, th = int(target_width), int(target_height)
        auto_used = False
        if tw <= 0 or th <= 0:
            auto_used = True
            if src_w and src_h:
                tw, th = resolution_from_aspect(src_w, src_h)
            else:
                tw, th = 1920, 1080

        tw = max(128, min(8192, tw - (tw % 2)))
        th = max(128, min(8192, th - (th % 2)))
        gw, gh = stage1_size(tw, th)
        uw, uh = gw * 2, gh * 2
        crop = "no crop needed" if (uw, uh) == (tw, th) else f"center crop -> {tw}x{th}"

        info = (
            f"image: {label}\n"
            f"target video: {tw}x{th}{'  (auto from image aspect)' if auto_used else ''}\n"
            f"stage 1: {gw}x{gh}  ->  2x latent upscale: {uw}x{uh}  ->  {crop}"
        )
        print(f"{LOG_PREFIX} {info.replace(chr(10), ' | ')}")
        return {"ui": {"text": [info]}, "result": (image_tensor, tw, th, gw, gh, info)}

    @classmethod
    def IS_CHANGED(cls, image, target_width, target_height, auto_resolution_from_image=True):
        if not image or image == NONE_OPTION:
            return NONE_OPTION
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image or image == NONE_OPTION:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


# --------------------------------------------------------------------------- #
# 2. audio -> frozen latent + frame count
# --------------------------------------------------------------------------- #
class SELTX25AudioPrepare:
    """Trim / pad the source audio, derive the LTX frame count and freeze the audio latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Source audio (speech, singing, music). It drives lip sync and motion timing."}),
                "audio_vae": ("VAE", {"tooltip": "LTX-2.5 audio VAE (ltx-2.5-audio-vae-bf16.safetensors)."}),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": "Video frame rate. LTX-2.5 is trained around 24/25 fps."},
                ),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "Skip this many seconds from the start of the audio."},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.01,
                        "tooltip": "How many seconds of audio to use (video length follows it). 0 = everything after start_seconds.",
                    },
                ),
                "max_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.5,
                        "tooltip": "Safety cap on the video length in seconds (0 = no cap). LTX-2.5 is officially tuned for clips up to ~20 s; "
                        "longer clips need a lot more VRAM/time.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("frozen_audio_latent", "audio", "frames", "fps", "video_seconds", "info")
    OUTPUT_TOOLTIPS = (
        "Encoded source audio with noise mask 0 (frozen). Concat it with the video latent in BOTH stages.",
        "Trimmed audio, padded with silence to exactly the video length. Feed it to Create Video.",
        "LTX frame count (8k + 1) that covers the audio.",
        "Frame rate pass-through (connect to LTXV Conditioning and Create Video).",
        "Exact video length in seconds (frames / fps).",
        "Human readable summary.",
    )
    FUNCTION = "prepare"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Trims the audio (duration 0 = whole clip), derives the LTX-2.5 8k+1 frame count from the audio length, "
        "pads the audio with silence to the exact video length, encodes it with the audio VAE and freezes the tokens."
    )

    def prepare(self, audio, audio_vae, fps, start_seconds, duration_seconds, max_duration_seconds=0.0):
        if audio is None or audio.get("waveform") is None:
            raise ValueError("SELTX25AudioPrepare: no audio input.")
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 3:
            raise ValueError("SELTX25AudioPrepare: expected waveform shape [batch, channels, samples].")
        if waveform.shape[0] > 1:
            print(f"{LOG_PREFIX} audio batch of {waveform.shape[0]} received, using the first clip")
            waveform = waveform[:1]

        fps = float(fps)
        total_samples = waveform.shape[-1]
        source_seconds = total_samples / sample_rate

        start_sample = int(round(max(0.0, float(start_seconds)) * sample_rate))
        if start_sample >= total_samples:
            raise ValueError(
                f"SELTX25AudioPrepare: start_seconds ({float(start_seconds):.2f} s) is beyond the end of the audio ({source_seconds:.2f} s)."
            )
        if float(duration_seconds) > 0.0:
            end_sample = min(total_samples, start_sample + int(round(float(duration_seconds) * sample_rate)))
        else:
            end_sample = total_samples
        capped = False
        if float(max_duration_seconds) > 0.0:
            cap_sample = start_sample + int(round(float(max_duration_seconds) * sample_rate))
            if cap_sample < end_sample:
                end_sample = cap_sample
                capped = True
        if end_sample - start_sample < int(0.05 * sample_rate):
            raise ValueError("SELTX25AudioPrepare: the selected audio range is shorter than 0.05 s.")

        segment = waveform[..., start_sample:end_sample]
        audio_seconds = segment.shape[-1] / sample_rate

        frames = ltx_frame_count(audio_seconds, fps)
        video_seconds = frames / fps
        target_samples = int(round(video_seconds * sample_rate))
        if segment.shape[-1] < target_samples:
            pad = torch.zeros(
                (segment.shape[0], segment.shape[1], target_samples - segment.shape[-1]),
                dtype=segment.dtype,
                device=segment.device,
            )
            segment = torch.cat((segment, pad), dim=-1)
        elif segment.shape[-1] > target_samples:
            segment = segment[..., :target_samples]

        mux_audio = {"waveform": segment.contiguous(), "sample_rate": sample_rate}

        # Match the latent length the model would get from LTXV Empty Latent Audio for this frame count.
        expected = None
        first_stage = getattr(audio_vae, "first_stage_model", None)
        if first_stage is not None and hasattr(first_stage, "num_of_latents_from_frames"):
            try:
                expected = int(first_stage.num_of_latents_from_frames(frames, fps))
            except Exception:
                expected = None

        # Encode with the core LTX audio VAE path (resamples to the VAE rate internally).  A short
        # silence margin is appended for the encode only, so the encoder never comes up one latent
        # short; the extra latents are trimmed to the expected count below (real encoded silence).
        from comfy_extras.nodes_audio import VAEEncodeAudio

        encode_waveform = segment
        if expected is not None:
            margin = torch.zeros(
                (segment.shape[0], segment.shape[1], int(round(0.25 * sample_rate))),
                dtype=segment.dtype,
                device=segment.device,
            )
            encode_waveform = torch.cat((segment, margin), dim=-1)
        encoded = VAEEncodeAudio.execute(audio_vae, {"waveform": encode_waveform.contiguous(), "sample_rate": sample_rate})
        samples = encoded.args[0]["samples"] if hasattr(encoded, "args") else encoded[0]["samples"]
        noise_mask = torch.zeros_like(samples)
        if expected is not None and expected > 0 and samples.shape[2] != expected:
            if samples.shape[2] > expected:
                samples = samples[:, :, :expected]
                noise_mask = noise_mask[:, :, :expected]
            else:
                pad_len = expected - samples.shape[2]
                pad = torch.zeros_like(samples[:, :, :1]).repeat(1, 1, pad_len, *([1] * (samples.dim() - 3)))
                samples = torch.cat((samples, pad), dim=2)
                noise_mask = torch.cat((noise_mask, torch.ones_like(pad)), dim=2)

        frozen_latent = {"samples": samples, "noise_mask": noise_mask}

        info_lines = [
            f"audio: {source_seconds:.2f} s source, using {audio_seconds:.2f} s from {float(start_seconds):.2f} s"
            + ("  (capped by max_duration_seconds)" if capped else ""),
            f"video: {frames} frames @ {fps:g} fps = {video_seconds:.2f} s"
            + (f"  (+{video_seconds - audio_seconds:.2f} s silence tail)" if video_seconds > audio_seconds + 1e-3 else ""),
            f"audio latent: {tuple(samples.shape)} frozen (noise mask 0)",
        ]
        if video_seconds > LTX25_RECOMMENDED_MAX_SECONDS:
            info_lines.append(
                f"warning: {video_seconds:.1f} s is above the ~{LTX25_RECOMMENDED_MAX_SECONDS:g} s LTX-2.5 sweet spot; expect heavy VRAM use"
            )
        info = "\n".join(info_lines)
        print(f"{LOG_PREFIX} {info.replace(chr(10), ' | ')}")
        return {"ui": {"text": [info]}, "result": (frozen_latent, mux_audio, int(frames), fps, float(video_seconds), info)}


# --------------------------------------------------------------------------- #
# 3. optional image conditioning (in-place first frame)
# --------------------------------------------------------------------------- #
class SELTX25ImageCondition:
    """LTXVPreprocess + LTXVImgToVideoInplace that passes through when no image is connected."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "LTX-2.5 video VAE."}),
                "latent": ("LATENT", {"tooltip": "Video latent (empty latent for stage 1, upscaled latent for stage 2)."}),
                "strength": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How hard the first frame is locked to the image (official: 0.7 stage 1, 1.0 stage 2)."},
                ),
                "img_compression": (
                    "INT",
                    {"default": 18, "min": 0, "max": 100, "tooltip": "LTXV Preprocess compression applied to the image before encoding (official: 18 stage 1, 0 stage 2). 0 = off."},
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional first frame. Leave unconnected / 'none' upstream for text + audio to video."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "condition"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Encodes the (optional) image into the first latent frame with the given strength. "
        "When no image arrives the latent is returned untouched, so the same graph works for audio-only generation."
    )

    def condition(self, vae, latent, strength, img_compression, image=None):
        if image is None:
            print(f"{LOG_PREFIX} no input image -> skipping image conditioning (audio + text to video)")
            return (latent,)

        from comfy_extras.nodes_lt import get_noise_mask, preprocess

        samples = latent["samples"].clone()
        _, height_scale, width_scale = vae.downscale_index_formula
        _, _, _, latent_height, latent_width = samples.shape
        width = latent_width * width_scale
        height = latent_height * height_scale

        pixels = image[:1, :, :, :3]
        if pixels.shape[1] != height or pixels.shape[2] != width:
            pixels = comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        if int(img_compression) > 0:
            pixels = torch.stack([preprocess(pixels[i], int(img_compression)) for i in range(pixels.shape[0])])

        encoded = vae.encode(pixels[:, :, :, :3])
        samples[:, :, : encoded.shape[2]] = encoded

        noise_mask = get_noise_mask(latent)
        noise_mask[:, :, : encoded.shape[2]] = 1.0 - float(strength)

        out = dict(latent)
        out["samples"] = samples
        out["noise_mask"] = noise_mask
        print(f"{LOG_PREFIX} image conditioning applied at {width}x{height}, strength {float(strength):g}, compression {int(img_compression)}")
        return (out,)


# --------------------------------------------------------------------------- #
# 4. fit decoded frames to the exact target size
# --------------------------------------------------------------------------- #
class SEImageFitToSize:
    """Center-crop frames to an exact size; cover-resize first if a side is too small."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded frames."}),
                "width": ("INT", {"default": 1920, "min": 16, "max": 16384, "step": 2, "tooltip": "Exact output width."}),
                "height": ("INT", {"default": 1080, "min": 16, "max": 16384, "step": 2, "tooltip": "Exact output height."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "fit"
    CATEGORY = "SECourses/Image"
    DESCRIPTION = "Center-crops frames to exactly width x height (e.g. 1920x1088 -> 1920x1080). Upscales first only if a side is smaller than the target."

    def fit(self, images, width, height):
        width = int(width)
        height = int(height)
        _, h, w, _ = images.shape
        if h == height and w == width:
            return (images,)
        if w < width or h < height:
            scale = max(width / w, height / h)
            new_w = max(width, int(math.ceil(w * scale)))
            new_h = max(height, int(math.ceil(h * scale)))
            print(f"{LOG_PREFIX} frames {w}x{h} smaller than target {width}x{height}: cover-resizing to {new_w}x{new_h} before crop")
            images = comfy.utils.common_upscale(images.movedim(-1, 1), new_w, new_h, "lanczos", "disabled").movedim(1, -1)
            _, h, w, _ = images.shape
        x0 = (w - width) // 2
        y0 = (h - height) // 2
        print(f"{LOG_PREFIX} center crop {w}x{h} -> {width}x{height} (offset {x0},{y0})")
        return (images[:, y0 : y0 + height, x0 : x0 + width, :],)


NODE_CLASS_MAPPINGS = {
    "SELTX25LoadImageOptional": SELTX25LoadImageOptional,
    "SELTX25AudioPrepare": SELTX25AudioPrepare,
    "SELTX25ImageCondition": SELTX25ImageCondition,
    "SEImageFitToSize": SEImageFitToSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SELTX25LoadImageOptional": "SE LTX-2.5 Input Image (optional) + Target Resolution",
    "SELTX25AudioPrepare": "SE LTX-2.5 Audio -> Frozen Latent + Frames",
    "SELTX25ImageCondition": "SE LTX-2.5 Image Conditioning (auto skip)",
    "SEImageFitToSize": "SE Fit Frames To Exact Size (center crop)",
}
