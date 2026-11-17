"""Official LongCat-Video-Avatar vocal separation (Kim_Vocal_2 MDX-Net) for the LongCat Avatar sampler.

The separator itself is longcat_kim_vocal.KimVocalSeparator, a numpy/torch/onnxruntime port of
audio-separator 0.30.2 as used by the official separate_vocals (83-87 dB SNR against it on CPU).
The model file Kim_Vocal_2.onnx is read from the audio_encoders model folder (LongCat Core Bundle).
"""
import os
import threading

import numpy as np
import torch

import comfy.model_management as mm

MODEL_NAME = "Kim_Vocal_2.onnx"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "longcat_vocal")
_LOCK = threading.Lock()
_CACHE = {}


def _model_file():
    from .codeformer_mouth import model_path
    try:
        return model_path("audio_encoders", MODEL_NAME)
    except Exception as error:
        raise FileNotFoundError(
            f"{MODEL_NAME} is missing from the audio_encoders model folder. Download the LongCat-Video-Avatar 1.6 Core Bundle "
            f"with the SwarmUI Model Downloader, or switch Vocal Separation off.") from error


def _separator(provider):
    from .longcat_kim_vocal import KimVocalSeparator
    key = provider
    if key not in _CACHE:
        device = mm.get_torch_device()
        device_id = device.index if getattr(device, "type", "") == "cuda" and device.index is not None else 0
        _CACHE[key] = KimVocalSeparator(_model_file(), model_data_dir=DATA_DIR, provider=provider, device_id=device_id)
    return _CACHE[key]


def separate_vocals_16k(waveform, sample_rate, duration=None):
    """waveform [channels, samples] (torch or numpy) at sample_rate -> (16 kHz mono vocals float32 tensor, info)."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().float().cpu().numpy()
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    with _LOCK:
        info = {"enabled": True, "model": MODEL_NAME}
        provider = "auto" if mm.get_torch_device().type == "cuda" else "cpu"
        try:
            separator = _separator(provider)
            vocals = separator.separate(waveform, int(sample_rate), duration=duration)
        except Exception as error:  # out of GPU memory or no CUDA provider: the CPU path gives the same result
            if provider == "cpu":
                raise
            info["cuda_error"] = str(error)[:200]
            separator = _separator("cpu")
            vocals = separator.separate(waveform, int(sample_rate), duration=duration)
        info["provider"] = separator.provider
    if not np.isfinite(vocals).all() or float(np.abs(vocals).max(initial=0.0)) == 0.0:
        raise ValueError("Vocal separation found no voice in this audio. Switch Vocal Separation off for instrumental or silent audio.")
    return torch.from_numpy(np.ascontiguousarray(vocals)), info
