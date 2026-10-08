"""Mel-Band RoFormer vocal separation without torchaudio.

One-for-one replacements for kijai/ComfyUI-MelBandRoFormer's loader and sampler: same checkpoint,
chunking and outputs, but the input is resampled to 44.1 kHz with ComfyUI's comfy.audio.resample.
"""

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .melband_roformer_model import MelBandRoformer

SAMPLE_RATE = 44100
CHUNK = 352800  # 8 s, the training segment length
STEP = CHUNK // 2
FADE = CHUNK // 10
BORDER = CHUNK - STEP

MODEL_CONFIG = dict(
    dim=384,
    depth=6,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_bands=60,
    dim_head=64,
    heads=8,
    sample_rate=SAMPLE_RATE,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
)


def separate_vocals(model, waveform, device):
    """[2, L] at 44.1 kHz -> vocal stem [2, L]; overlapping 8 s chunks blended with linear fades."""
    length = waveform.shape[-1]
    padded = length > 2 * BORDER
    if padded:
        waveform = F.pad(waveform, (BORDER, BORDER), mode="reflect")
    waveform = waveform.to(device)
    total = waveform.shape[-1]

    window = torch.ones(CHUNK, device=device)
    window[:FADE] *= torch.linspace(0, 1, FADE, device=device)
    window[-FADE:] *= torch.linspace(1, 0, FADE, device=device)

    vocals = torch.zeros_like(waveform, dtype=torch.float32)
    counter = torch.zeros_like(waveform, dtype=torch.float32)
    starts = range(0, total, STEP)
    pbar = comfy.utils.ProgressBar(len(starts))
    for start in starts:
        part = waveform[:, start:start + CHUNK]
        part_length = part.shape[-1]
        if part_length < CHUNK:
            if part_length > CHUNK // 2 + 1:
                part = F.pad(part, (0, CHUNK - part_length), mode="reflect")
            else:
                part = F.pad(part, (0, CHUNK - part_length))

        out = model(part.unsqueeze(0))[0]

        chunk_window = window.clone()
        if start == 0:
            chunk_window[:FADE] = 1
        elif start + CHUNK >= total:
            chunk_window[-FADE:] = 1

        vocals[..., start:start + part_length] += out[..., :part_length] * chunk_window[:part_length]
        counter[..., start:start + part_length] += chunk_window[:part_length]
        pbar.update(1)

    vocals = vocals / counter
    if padded:
        vocals = vocals[..., BORDER:-BORDER]
    return vocals


class SEMelBandRoFormerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "MelBandRoformer vocal checkpoint (for example MelBandRoformer_fp32.safetensors) in models/diffusion_models."}),
            },
        }

    RETURN_TYPES = ("MELROFORMERMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "audio"
    DESCRIPTION = "Loads the Mel-Band RoFormer vocal separation checkpoint used by kijai's MelBandRoFormer nodes. Needs no torchaudio."

    def load(self, model_name):
        model = MelBandRoformer(**MODEL_CONFIG).eval()
        model.load_state_dict(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("diffusion_models", model_name)), strict=True)
        return (model,)


class SEMelBandRoFormerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MELROFORMERMODEL",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "instruments")
    FUNCTION = "separate"
    CATEGORY = "audio"
    DESCRIPTION = "Splits audio into vocals and instruments at 44.1 kHz stereo; mono input is duplicated to both channels. Needs no torchaudio."

    def separate(self, model, audio):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        if sample_rate != SAMPLE_RATE:
            import comfy.audio

            waveform = comfy.audio.resample(waveform, sample_rate, SAMPLE_RATE)

        device = mm.get_torch_device()
        model.to(device)
        vocals = torch.stack([separate_vocals(model, item, device) for item in waveform])
        model.to(mm.unet_offload_device())

        vocals = vocals.cpu()
        instruments = waveform.cpu() - vocals
        return ({"waveform": vocals, "sample_rate": SAMPLE_RATE}, {"waveform": instruments, "sample_rate": SAMPLE_RATE})


NODE_CLASS_MAPPINGS = {
    "SEMelBandRoFormerLoader": SEMelBandRoFormerLoader,
    "SEMelBandRoFormerSampler": SEMelBandRoFormerSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SEMelBandRoFormerLoader": "SE Mel-Band RoFormer Model Loader",
    "SEMelBandRoFormerSampler": "SE Mel-Band RoFormer Sampler",
}
