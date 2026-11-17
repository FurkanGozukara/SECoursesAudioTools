"""LongCat-Video-Avatar 1.6 nodes: audio (+ image, + text) to long lip-synced video.

Native ComfyUI port of meituan-longcat/LongCat-Video (commit 58eb336,
run_demo_avatar_single_audio_to_video.py with --model_type avatar-v1.6 --use_distill
--mix_steps 4 --mix_alpha 1.0). The official recipe, reproduced here:

* Whisper large-v3 encoder features (5 layer groups) at 25 fps, from the separated vocals;
* 93-frame segments, each next one continues from the last 13 frames (re-encoded) plus the
  first segment's first latent as a reference placed at temporal position 10;
* 8 distilled steps on the 1.5 grid with flow shift 7: the DMD LoRA runs the first 4 steps,
  the OPSD LoRA the last 4 (mix_steps 4, mix_alpha 1.0), text and audio guidance 1 (one pass);
* 480p (or 720p) aspect buckets; 25 fps output with the original soundtrack.
"""

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.latent_formats
import comfy.model_base
import comfy.model_management as mm
import comfy.model_patcher
import comfy.storage
import comfy.supported_models_base
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl

from .longcat_avatar_model import LongCatVideoAvatarTransformer
from .avatarforever_nodes import AvatarPreview, send_panel_preview
from .h3_streaming_nodes import FFmpegWriter, mux_audio, write_wav

LOG = "[LongCat-Avatar]"
DEFAULT_MODEL = "LongCat_Video_Avatar_1.6_INT8_ConvRot.safetensors"
DEFAULT_DMD = "LongCat_Video_Avatar_1.6_DMD_LoRA.safetensors"
DEFAULT_OPSD = "LongCat_Video_Avatar_1.6_OPSD_LoRA.safetensors"
DEFAULT_WHISPER = "LongCat_Whisper_Large_V3_fp16.safetensors"
FPS = 25
NEGATIVE = ("Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
            "messy background, three legs, many people in the background, walking backwards")

# Official aspect buckets (longcat_video/utils/bukcet_config.py): ratio h/w -> (h, w).
BUCKETS = {
    "480p": {0.26: (320, 1216), 0.31: (352, 1120), 0.38: (384, 1024), 0.43: (416, 960), 0.52: (448, 864),
             0.58: (480, 832), 0.67: (512, 768), 0.74: (544, 736), 0.86: (576, 672), 0.95: (608, 640),
             1.05: (640, 608), 1.17: (672, 576), 1.29: (704, 544), 1.35: (736, 544), 1.50: (768, 512),
             1.67: (800, 480), 1.73: (832, 480), 2.00: (896, 448), 2.31: (960, 416), 2.58: (992, 384),
             2.75: (1056, 384), 3.09: (1088, 352), 3.70: (1184, 320), 3.80: (1216, 320), 3.90: (1248, 320),
             4.00: (1280, 320)},
    "720p": {0.25: (480, 1920), 0.29: (512, 1792), 0.32: (544, 1696), 0.36: (576, 1600), 0.40: (608, 1504),
             0.49: (672, 1376), 0.54: (704, 1312), 0.59: (736, 1248), 0.69: (800, 1152), 0.74: (832, 1120),
             0.82: (864, 1056), 0.88: (896, 1024), 0.94: (928, 992), 1.00: (960, 960), 1.07: (992, 928),
             1.14: (1024, 896), 1.22: (1056, 864), 1.31: (1088, 832), 1.35: (1120, 832), 1.44: (1152, 800),
             1.70: (1248, 736), 2.00: (1344, 672), 2.05: (1376, 672), 2.47: (1504, 608), 2.53: (1536, 608),
             2.83: (1632, 576), 3.06: (1664, 544), 3.12: (1696, 544), 3.62: (1856, 512), 3.93: (1888, 480),
             4.00: (1920, 480)},
}
SIZE_MODES = ["480p bucket (official)", "720p bucket (official)", "custom width x height"]


# ----------------------------------------------------------------------------- model loading

class LongCatAvatarConfig(comfy.supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}
    latent_format = comfy.latent_formats.Wan21
    sampling_settings = {"shift": 7.0, "multiplier": 1000}
    memory_usage_factor = 2.0
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return LongCatAvatarModel(self, device=device)


class LongCatAvatarModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, comfy.model_base.ModelType.FLOW, device=device,
                         unet_model=LongCatVideoAvatarTransformer)


ARCH_KEYS = ("in_channels", "out_channels", "hidden_size", "depth", "num_heads", "caption_channels", "mlp_ratio",
             "adaln_tembed_dim", "frequency_embedding_size", "patch_size", "text_tokens_zero_pad", "audio_window",
             "audio_block", "audio_channel", "intermediate_dim", "output_dim", "context_tokens", "vae_scale",
             "class_range", "class_interval")
DEFAULT_ARCH = {"in_channels": 16, "out_channels": 16, "hidden_size": 4096, "depth": 48, "num_heads": 32,
                "caption_channels": 4096, "mlp_ratio": 4, "adaln_tembed_dim": 512, "frequency_embedding_size": 256,
                "patch_size": [1, 2, 2], "text_tokens_zero_pad": True, "audio_window": 5, "audio_block": 5,
                "audio_channel": 1280, "intermediate_dim": 512, "output_dim": 768, "context_tokens": 32,
                "vae_scale": 4, "class_range": 24, "class_interval": 4}


def load_longcat_model(path, weight_dtype="default", disable_dynamic=False):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    metadata = metadata or {}
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
            break
    if "blocks.0.audio_cross_attn.kv_linear.weight" not in sd or "audio_proj.proj1_vf.weight" not in sd:
        raise ValueError("This file is not a LongCat-Video-Avatar transformer (audio cross-attention weights are missing).")
    arch = dict(DEFAULT_ARCH)
    try:
        arch.update({k: v for k, v in json.loads(metadata.get("config", "{}")).items() if k in ARCH_KEYS})
    except (TypeError, ValueError):
        pass
    arch["depth"] = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    proj1 = sd["audio_proj.proj1.weight"]
    if proj1.shape[1] == 5 * 12 * 768:  # LongCat-Avatar 1.0 (wav2vec2 features)
        arch.update(audio_block=12, audio_channel=768)
    model_config = LongCatAvatarConfig(arch)
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant:
        model_config.quant_config = quant
    load_device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    parameters = comfy.utils.calculate_parameters(sd)
    supported = list(model_config.supported_inference_dtypes)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if weight_dtype in dtypes and not quant:
        unet_dtype = dtypes[weight_dtype]
    else:
        unet_dtype = mm.unet_dtype(model_params=parameters, supported_dtypes=supported,
                                   weight_dtype=None if quant else comfy.utils.weight_dtype(sd))
    manual_cast = mm.unet_manual_cast(None if quant else unet_dtype, load_device, supported)
    model_config.set_inference_dtype(unet_dtype, manual_cast, device=load_device)
    model = model_config.get_model(sd, "")
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device,
                            fast_disk=comfy.storage.state_dict_fast_disk(sd))
    if not mm.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(sd, "", assign=patcher.is_dynamic())
    return patcher


class SELongCatAvatarLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"default": DEFAULT_MODEL,
                          "tooltip": "LongCat-Video-Avatar 1.6 transformer. INT8 ConvRot (default) is about twice as fast as BF16 and needs about half the memory."}),
            "weight_dtype": (["default", "bf16", "fp16"], {"default": "default",
                             "tooltip": "Compute type for the BF16 file. Quantized files (INT8 ConvRot) always use their stored format."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "SECourses/LongCat Avatar"
    DESCRIPTION = "Loads LongCat-Video-Avatar 1.6 (BF16 or native INT8 ConvRot) as a normal ComfyUI model: low-VRAM offload, dynamic VRAM and LoRA loaders work with it. No downloads."

    def load(self, unet_name, weight_dtype="default"):
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        patcher = load_longcat_model(path, weight_dtype)
        patcher.model_options["longcat_checkpoint"] = unet_name
        return (patcher,)


class SELongCatWhisperLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_encoder_name": (folder_paths.get_filename_list("audio_encoders"),
                                                    {"default": DEFAULT_WHISPER})}}

    RETURN_TYPES = ("AUDIO_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "SECourses/LongCat Avatar"
    DESCRIPTION = "Whisper large-v3 encoder for LongCat-Avatar's audio features. Loads only the encoder from a full Whisper file. No downloads."

    def load(self, audio_encoder_name):
        import comfy.audio_encoders.audio_encoders as audio_encoders
        path = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder_name)
        sd = comfy.utils.load_torch_file(path)
        if any(k.startswith("model.encoder.") for k in sd):
            sd = {k: v for k, v in sd.items() if k.startswith("model.encoder.")}
        elif any(k.startswith("encoder.") for k in sd):
            sd = {"model." + k: v for k, v in sd.items() if k.startswith("encoder.")}
        encoder = audio_encoders.load_audio_encoder_from_sd(sd)
        if not hasattr(encoder.model, "encoder") or not hasattr(encoder.model.encoder, "layers"):
            raise ValueError("Select a Whisper large-v3 file.")
        return (encoder,)


# ----------------------------------------------------------------------------- recipe helpers

def official_sigmas(distill=True, steps=8, shift=7.0, enhance_hf=False):
    """Timesteps/sigmas of the official pipeline (get_timesteps_sigmas for avatar-v1.5/1.6 + scheduler shift)."""
    if distill:
        idx = torch.arange(1, 9, dtype=torch.float32)
        idx = (idx * (1000 // 8)).round().long()
        idx = 1000 - idx
        base = torch.flip(torch.linspace(0, 1, 1000), [0])
        sigmas = torch.flip(base[idx], [0]).float()
    else:
        sigmas = torch.linspace(1, 0.001, steps)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * 1000
    if enhance_hf and not distill:
        tail = torch.from_numpy(np.linspace(500, 0, 10, dtype=np.float32, endpoint=False))
        timesteps = torch.cat([timesteps[timesteps > 500], tail])
        sigmas = timesteps / 1000
    return timesteps.float(), torch.cat([sigmas.float(), torch.zeros(1)])


def bucket_size(image_hw, mode):
    table = BUCKETS["720p" if mode.startswith("720p") else "480p"]
    ratio = image_hw[0] / image_hw[1]
    key = sorted(table, key=lambda r: abs(r - ratio))[0]
    return table[key]


def fit_image(image, width, height):
    """Resize to cover and center-crop (no distortion). image [1, H, W, 3] in 0..1 -> same layout."""
    pixels = image[:1, :, :, :3].movedim(-1, 1)
    pixels = comfy.utils.common_upscale(pixels, width, height, "bicubic", "center")
    return pixels.movedim(1, -1).clamp(0, 1)


def mono_16k(audio):
    import comfy.audio
    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    waveform = waveform.float().cpu()
    mono = waveform.mean(dim=0, keepdim=True)
    rate = int(audio["sample_rate"])
    if rate != 16000:
        mono = comfy.audio.resample(mono.unsqueeze(0), rate, 16000)[0]
    return mono[0]


@torch.no_grad()
def whisper_embedding(audio_encoder, speech, fps=FPS):
    """Official get_audio_embedding_whisper. speech: 1-D float tensor at 16 kHz. Returns [T, 5, 1280] float32."""
    MEL_CHUNK, ENC_CHUNK, ENC_FPS = 750 * 640, 3000, 50
    video_length = int(speech.shape[0] / 16000 * fps)
    mm.load_model_gpu(audio_encoder.patcher)
    device = audio_encoder.load_device
    whisper = audio_encoder.model
    mels = []
    for i in range(0, speech.shape[0], MEL_CHUNK):
        part = speech[i:i + MEL_CHUNK].reshape(1, 1, -1).to(device)
        mels.append(whisper.feature_extractor(part))
    mel = torch.cat(mels, dim=-1)
    enc = []
    dtype = audio_encoder.dtype
    for i in range(0, mel.shape[-1], ENC_CHUNK):
        _, hidden = whisper.encoder(mel[:, :, i:i + ENC_CHUNK].to(dtype))
        enc.append(torch.stack([h.float() for h in hidden], dim=2))
    prompts = torch.cat(enc, dim=1)[:, :video_length * 2]

    def interp(features):
        features = features.transpose(1, 2)
        return F.interpolate(features, size=video_length, align_corners=True, mode="linear").transpose(1, 2)

    groups = [prompts[:, :, 0:8].mean(2), prompts[:, :, 8:16].mean(2), prompts[:, :, 16:24].mean(2),
              prompts[:, :, 24:32].mean(2), prompts[:, :, 32]]
    emb = torch.stack([interp(g) for g in groups], dim=2)[0]
    return emb.cpu()


def window_audio(full, start, frames):
    """Official center_indices window (5 frames, stride 1): [1, frames, 5, 5, 1280]."""
    indices = torch.arange(5) - 2
    centers = torch.arange(start, start + frames).unsqueeze(1) + indices.unsqueeze(0)
    centers = centers.clamp(0, full.shape[0] - 1)
    return full[centers][None]


class StepLoRA:
    """Official LoRA network with per-step multipliers (DMD for the first steps, OPSD after).

    Applied at run time through forward hooks, like the official multi_lora_forward, so it works on
    BF16 and INT8 weights without merging.
    """

    def __init__(self, diffusion_model, paths, device, dtype):
        self.modules = {}
        self.multipliers = [1.0] * len(paths)
        self.handles = []
        self.files = []
        for index, path in enumerate(paths):
            loaded = 0
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                names = sorted({k[:-len(".lora_down.weight")] for k in keys if k.endswith(".lora_down.weight")})
                for name in names:
                    module_name = name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                    if module_name.startswith("diffusion_model."):
                        module_name = module_name[len("diffusion_model."):]
                    module = diffusion_model
                    try:
                        for part in module_name.split("."):
                            module = module[int(part)] if part.isdigit() else getattr(module, part)
                    except (AttributeError, IndexError):
                        raise ValueError(f"{os.path.basename(path)}: LoRA target {module_name} is not in the LongCat model.")
                    down = handle.get_tensor(name + ".lora_down.weight").to(device=device, dtype=dtype)
                    ups = sorted([k for k in keys if k.startswith(name + ".lora_up.blocks.")],
                                 key=lambda k: int(k.split(".lora_up.blocks.")[1].split(".")[0]))
                    if ups:
                        up = [handle.get_tensor(k).to(device=device, dtype=dtype) for k in ups]
                    else:
                        up = handle.get_tensor(name + ".lora_up.weight").to(device=device, dtype=dtype)
                    if name + ".alpha_scale" in keys:
                        scale = float(handle.get_tensor(name + ".alpha_scale").float().item())
                    elif name + ".alpha" in keys:
                        scale = float(handle.get_tensor(name + ".alpha").float().item()) / down.shape[0]
                    else:
                        scale = 1.0
                    self.modules.setdefault(module, []).append((index, down, up, scale))
                    loaded += 1
            if not loaded:
                raise ValueError(f"{os.path.basename(path)} holds no LoRA weights.")
            self.files.append((os.path.basename(path), loaded))

    def _hook(self, module, inputs, output):
        x = inputs[0]
        total = None
        for index, down, up, scale in self.modules[module]:
            mult = self.multipliers[index]
            if mult == 0.0:
                continue
            lx = F.linear(x.to(down.dtype), down)
            if isinstance(up, list):
                parts = lx.chunk(len(up), dim=-1)
                lx = torch.cat([F.linear(p, u) for p, u in zip(parts, up)], dim=-1)
            else:
                lx = F.linear(lx, up)
            delta = lx.to(output.dtype) * mult * scale
            total = delta if total is None else total + delta
        return output if total is None else output + total

    def attach(self):
        for module in self.modules:
            self.handles.append(module.register_forward_hook(self._hook))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def nbytes(self):
        total = 0
        for entries in self.modules.values():
            for _, down, up, _ in entries:
                total += down.numel() * down.element_size()
                ups = up if isinstance(up, list) else [up]
                total += sum(u.numel() * u.element_size() for u in ups)
        return total


def context_and_mask(conditioning, device, dtype, length=512):
    context = conditioning[0][0].to(device=device, dtype=torch.float32)
    if context.shape[1] < length:
        context = F.pad(context, (0, 0, 0, length - context.shape[1]))
    context = context[:, :length]
    mask = conditioning[0][1].get("attention_mask")
    if mask is not None and mask.shape[-1] >= length:
        mask = mask.reshape(mask.shape[0], -1)[:, :length].to(device=device, dtype=torch.float32)
    else:
        mask = (context.abs().sum(-1) > 0).float()
    return context.to(dtype), mask


def preview_ids(prompt, sampler_id):
    return [node_id for node_id, node in (prompt or {}).items()
            if node.get("class_type") in ("SELongCatAvatarPreview", "SEAvatarForeverPreview")
            and node.get("inputs", {}).get("video") == [str(sampler_id), 0]]


class SELongCatAvatarPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "SECourses/LongCat Avatar"
    DESCRIPTION = "Connect to the LongCat Avatar video output. Shows live sampling previews, then the saved video. Saves no extra copy."

    def show(self, video):
        from .avatarforever_nodes import SEAvatarForeverPreview
        return SEAvatarForeverPreview().show(video)


# ----------------------------------------------------------------------------- inputs adapter

DEFAULT_SCENE = ("The person in the image talks to the camera with natural facial expressions, clear lip movements, "
                 "subtle head movements and natural hand gestures.")
SINGING_SCENE = ("The person in the image sings to the camera with expressive facial expressions, clear lip movements, "
                 "subtle head movements and natural hand gestures.")


def _entry_path(entry):
    if entry.get("source") == "batch_folder":
        root = os.path.realpath(entry["batch_root"])
        path = os.path.realpath(entry["path"])
        if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
            raise ValueError(f"Folder batch file is outside its folder or missing: {path}")
        return path
    return folder_paths.get_annotated_filepath(entry["file"])


def _load_image_file(path):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]


def _load_audio_file(path):
    from comfy_extras.nodes_audio import load as load_audio
    waveform, rate = load_audio(path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": rate}


class SELongCatAvatarInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene": ("STRING", {"forceInput": True, "tooltip": "From the prompt box. Empty = a neutral talking-to-camera description."}),
            "references": ("SECOURSES_REF_PACK",),
            "performance": (["speech", "singing"], {"default": "speech"}),
        }, "optional": {"audio": ("AUDIO",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "AUDIO", "IMAGE")
    RETURN_NAMES = ("prompt", "audio", "image")
    FUNCTION = "build"
    CATEGORY = "SECourses/LongCat Avatar"
    DESCRIPTION = ("Joins the prompt box, the soundtrack and the optional image for LongCat Avatar. Folder batches pair "
                   "name.txt with name.png/jpg and name.wav/mp3 automatically; @image1 in the prompt box can also be the image.")

    def build(self, scene, references, performance="speech", audio=None, image=None):
        references = references if isinstance(references, dict) else {}
        if references.get("init_audio"):
            audio = _load_audio_file(_entry_path(references["init_audio"]))
        if audio is None:
            for entry in references.get("audios") or []:
                audio = _load_audio_file(_entry_path(entry))
                break
        if audio is None:
            raise ValueError("LongCat Avatar needs a soundtrack: upload it in 1. SOUNDTRACK (or pair name.wav with name.txt in a folder batch).")
        if references.get("init_image"):
            image = _load_image_file(_entry_path(references["init_image"]))
        elif image is None:
            for entry in references.get("images") or []:
                image = _load_image_file(_entry_path(entry))
                break
        text = (scene or "").strip()
        for token in ("@image1", "@audio1"):
            text = text.replace(token, "the person" if token == "@image1" else "the voice")
        if not text:
            text = SINGING_SCENE if performance == "singing" else DEFAULT_SCENE
        else:
            low = text.lower()
            verb = "sing" if performance == "singing" else ("speak", "talk", "say")
            if not any(v in low for v in ((verb,) if isinstance(verb, str) else verb)):
                text = text.rstrip(". ") + (". The person is singing." if performance == "singing" else ". The person is speaking to the camera.")
        return {"ui": {"text": [text]}, "result": (text, audio, image)}


# ----------------------------------------------------------------------------- sampler

class SELongCatAvatarSampler:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING", {"tooltip": "UMT5-XXL text (CLIP Loader type wan). Describe the person, the place and that the person is speaking or singing."}),
            "vae": ("VAE", {"tooltip": "Wan 2.1 VAE."}),
            "audio_encoder": ("AUDIO_ENCODER", {"tooltip": "Whisper large-v3 (LongCat Whisper Loader)."}),
            "audio": ("AUDIO",),
            "size_mode": (SIZE_MODES, {"default": SIZE_MODES[0], "tooltip": "Official buckets pick the size nearest to the image aspect. Custom uses Width x Height (rounded to 16; the image is center-cropped to it)."}),
            "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 16}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "dmd_lora": (loras, {"default": DEFAULT_DMD, "tooltip": "Official DMD LoRA (loraA). Runs the first Mix Steps."}),
            "opsd_lora": (loras, {"default": DEFAULT_OPSD, "tooltip": "Official OPSD LoRA (loraB). Runs the remaining steps."}),
            "distilled_8_steps": ("BOOLEAN", {"default": True, "label_on": "ENABLED (official 1.6, 8 steps)", "label_off": "DISABLED (50-step base model with guidance)",
                                              "tooltip": "Official LongCat-Avatar 1.6 recipe: 8 distilled steps with the DMD + OPSD LoRAs, guidance 1. Off runs the base model with Steps and guidance (about 20x slower)."}),
            "mix_steps": ("INT", {"default": 4, "min": 0, "max": 8, "tooltip": "Official 4: steps 1-4 use the DMD LoRA (x Mix Alpha), steps 5-8 the OPSD LoRA."}),
            "mix_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "DMD weight during the mixed steps (official 1.0). More DMD = steadier structure."}),
            "steps": ("INT", {"default": 50, "min": 4, "max": 100, "tooltip": "Only with the 8-step mode off."}),
            "text_guidance": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Only with the 8-step mode off (official 4)."}),
            "audio_guidance": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Only with the 8-step mode off (official 4). Higher may exaggerate the mouth."}),
            "audio_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
            "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "0 = the whole audio. The video length follows the audio."}),
            "vocal_separation": ("BOOLEAN", {"default": True, "label_on": "ENABLED (official)", "label_off": "DISABLED",
                                             "tooltip": "Official Kim_Vocal_2 vocal separation before the Whisper features (music and noise do not move the mouth). The saved video keeps the original audio."}),
            "condition_cache": ("BOOLEAN", {"default": True, "label_on": "ENABLED (exact, faster)", "label_off": "DISABLED (normal)",
                                            "tooltip": "Computes the reference and previous-frame tokens once per LoRA phase instead of every step. Same result, faster long videos. Switches itself off when it does not fit in free VRAM (large sizes)."}),
            "attention": (["auto", "sage", "pytorch"], {"default": "auto", "tooltip": "auto uses ComfyUI's attention (SageAttention with our launchers). sage forces SageAttention if installed."}),
            "torch_compile": ("BOOLEAN", {"default": False, "label_on": "ENABLED", "label_off": "DISABLED (normal)",
                                          "tooltip": "Compiles each transformer block (first run slower, later runs faster). Needs Triton."}),
            "frames_per_segment": ("INT", {"default": 93, "min": 33, "max": 189, "step": 4, "tooltip": "Official 93 frames (3.72 s) per segment."}),
            "context_frames": ("INT", {"default": 13, "min": 5, "max": 45, "step": 4, "tooltip": "Official 13 frames carried into the next segment."}),
            "ref_img_index": ("INT", {"default": 10, "min": -64, "max": 64, "tooltip": "Official 10: time position of the identity reference in later segments."}),
            "mask_frame_range": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "Official 0. >0 stops frames near the reference position from copying it (fewer repeated gestures)."}),
            "crf": ("INT", {"default": 17, "min": 0, "max": 51}),
            "encoding_preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "fast"}),
            "filename_prefix": ("STRING", {"default": "video/LongCat_Avatar/LongCat"}),
        }, "optional": {
            "image": ("IMAGE", {"tooltip": "Identity / first frame. None = audio + text."}),
            "negative": ("CONDITIONING", {"tooltip": "Only with the 8-step mode off. Default: the official negative prompt must be encoded and connected."}),
            "mouth_enhancement": ("BOOLEAN", {"default": False, "label_on": "ENABLED (CodeFormer mouth)", "label_off": "DISABLED",
                                              "tooltip": "Aligned CodeFormer mouth-only restoration of every frame before export."}),
            "mouth_fidelity": ("FLOAT", {"default": .9, "min": 0, "max": 1, "step": .01}),
            "mouth_blend": ("FLOAT", {"default": .7, "min": 0, "max": 1, "step": .01}),
            "mouth_model": ("STRING", {"default": "codeformer.pth"}),
            "mouth_detector": ("STRING", {"default": "models/buffalo_l/det_10g.onnx"}),
            "live_preview": (["auto", "latent", "off"], {"default": "latent"}),
            "save_original_too": ("BOOLEAN", {"default": False, "tooltip": "With mouth enhancement on, also save the video without it (for comparisons)."}),
        }, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "sample"
    OUTPUT_NODE = True
    CATEGORY = "SECourses/LongCat Avatar"
    DESCRIPTION = "LongCat-Video-Avatar 1.6: audio + optional image + text to a lip-synced video of any length (official 8-step DMD/OPSD recipe, segment continuation). Saves an MP4 with the original audio."

    def sample(self, model, positive, vae, audio_encoder, audio, size_mode, width, height, seed, dmd_lora, opsd_lora,
               distilled_8_steps=True, mix_steps=4, mix_alpha=1.0, steps=50, text_guidance=4.0, audio_guidance=4.0,
               audio_start_seconds=0.0, duration_seconds=0.0, vocal_separation=True, condition_cache=True,
               attention="auto", torch_compile=False, frames_per_segment=93, context_frames=13, ref_img_index=10,
               mask_frame_range=0, crf=17, encoding_preset="fast", filename_prefix="video/LongCat_Avatar/LongCat",
               image=None, negative=None, mouth_enhancement=False, mouth_fidelity=.9, mouth_blend=.7,
               mouth_model="codeformer.pth", mouth_detector="models/buffalo_l/det_10g.onnx", live_preview="latent",
               save_original_too=False, unique_id=None, prompt=None):
        started = time.perf_counter()
        if size_mode not in SIZE_MODES or attention not in ("auto", "sage", "pytorch"):
            raise ValueError("LongCat Avatar: invalid size mode or attention choice (the workflow widgets may be shifted).")
        if (frames_per_segment - 1) % 4 or (context_frames - 1) % 4 or context_frames >= frames_per_segment:
            raise ValueError("Frames per segment and context frames must be 4n+1, with fewer context frames than segment frames.")
        mouth = None
        if mouth_enhancement and mouth_blend != 0:
            from .codeformer_mouth import model_path
            model_path("facerestore_models", mouth_model)
            model_path("insightface", mouth_detector)
        dmd_path = folder_paths.get_full_path_or_raise("loras", dmd_lora) if distilled_8_steps else None
        opsd_path = folder_paths.get_full_path_or_raise("loras", opsd_lora) if distilled_8_steps else None
        use_cfg = (not distilled_8_steps) and (text_guidance > 1.0 or audio_guidance > 1.0)
        if use_cfg and negative is None:
            raise ValueError("Connect a negative conditioning (the official negative prompt) when guidance is above 1.")

        # ---- audio
        rate = int(audio["sample_rate"])
        waveform = audio["waveform"]
        if waveform.dim() == 3:
            waveform = waveform[:1]
        else:
            waveform = waveform.unsqueeze(0)
        begin = round(audio_start_seconds * rate)
        end = waveform.shape[-1] if duration_seconds <= 0 else min(waveform.shape[-1], begin + round(duration_seconds * rate))
        clip = waveform[..., begin:end].float().cpu()
        if clip.shape[-1] == 0:
            raise ValueError("The selected audio range is empty.")
        seconds = clip.shape[-1] / rate
        out_frames = max(1, math.ceil(seconds * FPS))
        step_frames = frames_per_segment - context_frames
        segments = 1 + max(0, math.ceil((out_frames - frames_per_segment) / step_frames))
        gen_frames = frames_per_segment + (segments - 1) * step_frames
        soundtrack = {"waveform": clip, "sample_rate": rate}
        separation = {"enabled": False}
        if vocal_separation:
            # Official separate_vocals: the source audio at its own rate -> Kim_Vocal_2 -> 16 kHz mono vocals.
            t0 = time.perf_counter()
            from .longcat_vocal_separator import separate_vocals_16k
            speech, separation = separate_vocals_16k(clip[0], rate, duration=gen_frames / FPS + 1.0)
            separation["seconds"] = round(time.perf_counter() - t0, 2)
        else:
            speech = mono_16k(soundtrack)
        need = int(math.ceil((gen_frames / FPS + 1.0) * 16000))
        if speech.shape[0] < need:
            speech = F.pad(speech, (0, need - speech.shape[0]))
        t0 = time.perf_counter()
        full_audio = whisper_embedding(audio_encoder, speech)
        audio_seconds = time.perf_counter() - t0
        if torch.isnan(full_audio).any():
            raise ValueError("Broken audio embedding (NaN).")

        # ---- size
        if size_mode.startswith("custom"):
            W, H = max(256, round(width / 16) * 16), max(256, round(height / 16) * 16)
        elif image is not None:
            H, W = bucket_size((image.shape[1], image.shape[2]), size_mode)
        else:
            H, W = bucket_size((height, width), size_mode)
        latent_h, latent_w = H // 8, W // 8
        frames_latent = (frames_per_segment - 1) // 4 + 1
        cond_latents_n = (context_frames - 1) // 4 + 1

        # ---- model
        device = mm.get_torch_device()
        work = model.clone()
        dm_dtype = work.model.diffusion_model.compute_dtype() if hasattr(work.model.diffusion_model, "compute_dtype") else torch.bfloat16
        lora = None
        lora_bytes = 0
        if distilled_8_steps:
            lora = StepLoRA(work.model.diffusion_model, [dmd_path, opsd_path], device, dm_dtype)
            lora_bytes = lora.nbytes()
        tokens = (frames_latent + 1) * (latent_h // 2) * (latent_w // 2)
        activation = int(tokens * 4096 * 2 * 9 + 3 * 1024 ** 3)
        mm.load_models_gpu([work], memory_required=activation + lora_bytes)
        dm = work.model.diffusion_model
        dtype = dm.compute_dtype()
        latent_format = work.model.latent_format
        previous_attention = None
        import comfy.ldm.modules.attention as comfy_attention
        if attention == "sage":
            if getattr(comfy_attention, "SAGE_ATTENTION_IS_AVAILABLE", False) or hasattr(comfy_attention, "attention_sage"):
                previous_attention = comfy_attention.optimized_attention
                comfy_attention.optimized_attention = comfy_attention.attention_sage
        elif attention == "pytorch":
            previous_attention = comfy_attention.optimized_attention
            comfy_attention.optimized_attention = comfy_attention.attention_pytorch
        compiled = []
        if torch_compile:
            for i, block in enumerate(dm.blocks):
                compiled.append((i, block))
                dm.blocks[i] = torch.compile(block, fullgraph=False, dynamic=False)

        # The exact condition cache keeps keys and values of the reference + previous-frame tokens for every block.
        # Use it only when it fits in free VRAM beside the weights and activations (it would push weights out).
        cache_bytes = (cond_latents_n + 1) * (latent_h // 2) * (latent_w // 2) * dm.hidden_size * 2 * 2 * len(dm.blocks)
        cache_used = False
        if condition_cache and segments > 1:
            free = mm.get_free_memory(device)
            cache_used = cache_bytes < 0.6 * max(0, free - activation)
            if not cache_used:
                print(f"{LOG} condition cache off for this size: needs {cache_bytes / 1024 ** 3:.1f} GB, "
                      f"{free / 1024 ** 3:.1f} GB free", flush=True)
        condition_cache = cache_used
        text_y, text_mask = context_and_mask(positive, device, dtype)
        text_y = dm.text_embed(text_y, text_mask)
        neg_y = None
        if use_cfg:
            neg_y, neg_mask = context_and_mask(negative, device, dtype)
            neg_y = dm.text_embed(neg_y, neg_mask)

        if distilled_8_steps:
            timesteps, sigmas = official_sigmas(True)
        else:
            timesteps, sigmas = official_sigmas(False, steps)
        timesteps, sigmas = timesteps.to(device), sigmas.to(device)
        n_steps = len(timesteps)
        generator = torch.Generator("cpu").manual_seed(seed)
        progress = comfy.utils.ProgressBar(segments * n_steps, node_id=unique_id)
        panel_ids = preview_ids(prompt, unique_id)
        # Wan latent RGB preview of the newest frame (the LTX tiny VAE of AvatarPreview's "auto" does not apply).
        preview = AvatarPreview("latent", latent_format) if live_preview != "off" else None

        # ---- output
        output_dir = folder_paths.get_output_directory()
        out_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, W, H)
        os.makedirs(out_folder, exist_ok=True)
        base = os.path.join(out_folder, f"{filename}_{counter:05}_")
        output, silent, wav = base + ".mp4", base + ".video_only.mp4", base + ".audio.wav"
        original_writer = None
        if mouth_enhancement and mouth_blend != 0:
            from .codeformer_mouth import CodeFormerMouth
            mouth = CodeFormerMouth(mouth_model, mouth_detector, mouth_fidelity, mouth_blend)
            if save_original_too:
                original_writer = FFmpegWriter(base + "no_mouth.video_only.mp4", W, H, FPS, crf, encoding_preset, frame_limit=out_frames)
        writer = FFmpegWriter(silent, W, H, FPS, crf, encoding_preset, frame_limit=out_frames)

        report = {"checkpoint": model.model_options.get("longcat_checkpoint"), "width": W, "height": H, "seed": seed,
                  "fps": FPS, "audio_seconds": round(seconds, 3), "segments": segments, "frames_per_segment": frames_per_segment,
                  "context_frames": context_frames, "generated_frames": gen_frames, "saved_frames": out_frames,
                  "recipe": "official 1.6 distilled 8 steps (DMD/OPSD)" if distilled_8_steps else f"base {n_steps} steps, guidance text {text_guidance} audio {audio_guidance}",
                  "mix_steps": mix_steps, "mix_alpha": mix_alpha, "timesteps": [round(float(t), 3) for t in timesteps],
                  "ref_img_index": ref_img_index, "mask_frame_range": mask_frame_range, "condition_cache": condition_cache,
                  "condition_cache_gb": round(cache_bytes / 1024 ** 3, 2),
                  "attention": attention, "torch_compile": torch_compile, "vocal_separation": separation,
                  "image": image is not None, "loras": lora.files if lora is not None else []}
        print(f"{LOG} {seconds:.2f}s audio -> {out_frames} frames, {segments} segment(s) of {frames_per_segment}, {W}x{H}, "
              f"{'8-step DMD/OPSD' if distilled_8_steps else str(n_steps) + ' steps'}, cache={condition_cache}", flush=True)
        dit_seconds = 0.0
        decode_seconds = 0.0
        segment_times = []
        try:
            if lora is not None:
                lora.attach()
            ref_latent = None
            prev_frames = None  # uint8 [context, H, W, 3] of the previous segment's last frames
            image_latent = None
            if image is not None:
                pixels = fit_image(image, W, H)
                image_latent = latent_format.process_in(vae.encode(pixels[:, :, :, :3]).float()[:, :, :1]).to(device)
            written = 0
            for seg in range(segments):
                mm.throw_exception_if_processing_interrupted()
                seg_started = time.perf_counter()
                audio_win = window_audio(full_audio, seg * step_frames, frames_per_segment).to(device)
                noise = torch.randn((1, 16, frames_latent, latent_h, latent_w), generator=generator, dtype=torch.float32).to(device)
                latents = noise
                num_cond, num_ref = 0, 0
                cond_video_latents = None
                if seg == 0:
                    if image_latent is not None:
                        latents[:, :, :1] = image_latent
                        num_cond = 1
                else:
                    cond_pixels = prev_frames.float() / 255.0
                    t0 = time.perf_counter()
                    cond_video_latents = latent_format.process_in(vae.encode(cond_pixels).float()).to(device)
                    decode_seconds += time.perf_counter() - t0
                    latents[:, :, :cond_latents_n] = cond_video_latents
                    num_cond = cond_latents_n
                    latents = torch.cat([ref_latent, latents], dim=2)
                    num_cond += 1
                    num_ref = 1
                T = latents.shape[2]
                # The VAE encode/decode between segments can evict transformer weights: load them back first.
                mm.load_models_gpu([work], memory_required=activation + lora_bytes)
                tok = dm.audio_tokens(audio_win, T, num_ref)  # [1, T, 32, 768]
                tok_zero = dm.audio_tokens(torch.zeros_like(audio_win), T, num_ref) if use_cfg else None
                text_cache, neg_cache = {}, {}
                cond_cache = {} if (condition_cache and num_cond > 0) else None
                cond_cache_neg = {} if (condition_cache and num_cond > 0 and use_cfg) else None
                phase = None
                t_seg = time.perf_counter()
                for i in range(n_steps):
                    mm.throw_exception_if_processing_interrupted()
                    if lora is not None:
                        mixed = i < mix_steps
                        new_phase = "mix" if mixed else "opsd"
                        if new_phase != phase:
                            lora.multipliers = [mix_alpha if mixed else 0.0, (1.0 - mix_alpha) if mixed else 1.0]
                            phase = new_phase
                            if cond_cache is not None:
                                cond_cache = {}
                            if cond_cache_neg is not None:
                                cond_cache_neg = {}
                    t = timesteps[i]
                    tstep = t.expand(1).unsqueeze(-1).repeat(1, T).to(dtype)
                    tstep[:, :num_cond] = 0
                    x_in = latents.to(dtype)
                    out = dm(x_in, tstep, text_y, tok, num_cond=num_cond, num_ref=num_ref, ref_img_index=ref_img_index,
                             mask_frame_range=mask_frame_range, text_kv_cache=text_cache, cond_cache=cond_cache)
                    if use_cfg:
                        out_uncond_text = dm(x_in, tstep, neg_y, tok, num_cond=num_cond, num_ref=num_ref,
                                             ref_img_index=ref_img_index, mask_frame_range=mask_frame_range,
                                             text_kv_cache=neg_cache, cond_cache=cond_cache_neg)
                        out_uncond = dm(x_in, tstep, neg_y, tok_zero, num_cond=num_cond, num_ref=num_ref,
                                        ref_img_index=ref_img_index, mask_frame_range=mask_frame_range,
                                        text_kv_cache=neg_cache, cond_cache=None)
                        out = out_uncond + text_guidance * (out - out_uncond_text) + audio_guidance * (out_uncond_text - out_uncond)
                    noise_pred = -out.float()
                    s, s_next = sigmas[i], sigmas[i + 1]
                    preview_image = None
                    if preview is not None:
                        # predicted clean newest frame: x0 = x - sigma * velocity
                        preview_image = preview.decode(latents[:, :, -1:].float() - s * noise_pred[:, :, -1:])
                    latents[:, :, num_cond:] = latents[:, :, num_cond:].float() + (s_next - s) * noise_pred[:, :, num_cond:]
                    done = seg * n_steps + i + 1
                    if panel_ids:
                        if preview_image is not None:
                            send_panel_preview(panel_ids, preview_image)
                        progress.update_absolute(done)
                    else:
                        progress.update_absolute(done, preview=preview_image)
                dit_seconds += time.perf_counter() - t_seg
                del text_cache, neg_cache, cond_cache, cond_cache_neg
                if num_ref:
                    latents = latents[:, :, num_ref:]
                    num_cond -= num_ref
                if cond_video_latents is not None:
                    latents[:, :, :cond_latents_n] = cond_video_latents
                if seg == 0:
                    ref_latent = latents[:, :, :1].clone()
                t0 = time.perf_counter()
                pixels = vae.decode(latent_format.process_out(latents.float()))
                pixels = pixels.reshape(-1, *pixels.shape[-3:])[:frames_per_segment].cpu()
                decode_seconds += time.perf_counter() - t0
                frames_u8 = (pixels.clamp(0, 1) * 255).to(torch.uint8)  # official astype(uint8)
                del pixels
                new = frames_u8 if seg == 0 else frames_u8[context_frames:]
                prev_frames = frames_u8[-context_frames:].clone()
                new = new[:max(0, out_frames - written)]
                if len(new):
                    if original_writer is not None:
                        original_writer.write(new)
                    writer.write(mouth.process(new) if mouth is not None else new)
                    written += len(new)
                segment_times.append(round(time.perf_counter() - seg_started, 2))
                print(f"{LOG} segment {seg + 1}/{segments}: {segment_times[-1]:.1f}s", flush=True)
                del latents, noise, tok, tok_zero, frames_u8, new
            writer.close()
            write_wav(wav, soundtrack["waveform"][0], soundtrack["sample_rate"])
            mux_audio(silent, wav, output, seconds, frames=out_frames)
            if original_writer is not None:
                original_writer.close()
                mux_audio(base + "no_mouth.video_only.mp4", wav, base + "no_mouth.mp4", seconds, frames=out_frames)
                os.remove(base + "no_mouth.video_only.mp4")
        except BaseException:
            writer.abort()
            if original_writer is not None:
                original_writer.abort()
            raise
        finally:
            if lora is not None:
                lora.detach()
            if previous_attention is not None:
                comfy_attention.optimized_attention = previous_attention
            for i, block in compiled:
                dm.blocks[i] = block
        for path in (silent, wav):
            if os.path.exists(path):
                os.remove(path)
        report.update({"dit_seconds": round(dit_seconds, 2), "vae_seconds": round(decode_seconds, 2),
                       "audio_feature_seconds": round(audio_seconds, 2), "segment_seconds": segment_times,
                       "mouth_enhancement": mouth.report if mouth is not None else {"enabled": False},
                       "total_seconds": round(time.perf_counter() - started, 2), "path": output})
        text = json.dumps(report, indent=2)
        with open(base + ".json", "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"{LOG} done in {report['total_seconds']:.1f}s -> {output}", flush=True)
        return {"ui": {} if panel_ids else {"text": [text], "images": [{"filename": os.path.basename(output),
                    "subfolder": subfolder, "type": "output"}], "animated": (True,)},
                "result": (InputImpl.VideoFromFile(output), text)}


NODE_CLASS_MAPPINGS = {
    "SELongCatAvatarInputs": SELongCatAvatarInputs,
    "SELongCatAvatarLoader": SELongCatAvatarLoader,
    "SELongCatWhisperLoader": SELongCatWhisperLoader,
    "SELongCatAvatarSampler": SELongCatAvatarSampler,
    "SELongCatAvatarPreview": SELongCatAvatarPreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SELongCatAvatarInputs": "LongCat Avatar Inputs (prompt box + audio + image)",
    "SELongCatAvatarLoader": "LongCat Avatar 1.6 Model Loader (INT8 ConvRot / BF16)",
    "SELongCatWhisperLoader": "LongCat Avatar Whisper Loader",
    "SELongCatAvatarSampler": "LongCat Avatar 1.6 - Audio + Image + Text to Long Video",
    "SELongCatAvatarPreview": "LongCat Avatar Live Preview & Final Video",
}
