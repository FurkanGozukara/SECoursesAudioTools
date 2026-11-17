"""Kim_Vocal_2 (UVR MDX-Net) vocal separation, reproducing LongCat-Video-Avatar's separate_vocals.

The official LongCat-Video-Avatar pipeline (longcat_video/audio_process/vocal_separator.py) runs
python-audio-separator 0.30.2's MDXSeparator with Kim_Vocal_2.onnx before its Whisper features:

    librosa.load(path, sr=44100, mono=False)           # soxr_hq resampling, mono -> duplicated to stereo
    normalize(peak 0.9, only lowers)                    # spec_utils.normalize(max_peak=0.9, min_peak=-0.9)
    optional crop to ceil(duration*44100) + chunk_size samples
    _demix_batched(batch_size=2)                        # hann-weighted overlap-add of 5.92 s chunks, overlap 0.25
        run_model: torch.stft(n_fft 7680, hop 1024) -> zero bins 0..2 -> onnx -> torch.istft
    source *= compensate (1.009)
    normalize(peak 0.9, only lowers)
    librosa.to_mono (channel mean) -> librosa.resample 44100 -> 16000 (soxr_hq) -> crop to the duration

This module does the same with numpy, torch and onnxruntime only. The one part that cannot be
bit-identical without libsoxr is the resampling: it is replaced by a linear-phase Kaiser windowed-sinc
resampler built to libsoxr's HQ specification (see README.md for the measured difference).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from fractions import Fraction
from typing import Optional, Sequence, Union

import numpy as np
import torch

MODEL_SAMPLE_RATE = 44100
OUTPUT_SAMPLE_RATE = 16000
_HASH_BYTES = 10000 * 1024  # audio-separator get_model_hash: md5 of the last 10,240,000 bytes

# libsoxr SOXR_HQ (20-bit precision) as used by python-soxr quality "HQ" / librosa res_type "soxr_hq":
#   passband_end = 1 - 0.05 / TO_3dB(rej), rej = 20 * 20*log10(2); stopband_begin = 1 (relative to the lower Nyquist),
#   linear phase. The filter below meets that spec with a Kaiser window. Tuned values: see README.md.
SOXR_HQ_PRECISION_BITS = 20
# Kaiser attenuation that best matches soxr 1.1.0 HQ output (grid search against soxr: 128 dB -> 79 dB SNR,
# 136 dB -> 86.5 dB SNR on 44.1k -> 16k speech; libsoxr's own nominal value is (20 + 1) * 6.02 = 126.4 dB).
SOXR_LIKE_ATT_DB = 136.0


def _soxr_passband_end(bits: int = SOXR_HQ_PRECISION_BITS) -> float:
    rej = bits * 20.0 * math.log10(2.0)
    to_3db = (1.6e-6 * rej - 7.5e-4) * rej + 0.646
    return 1.0 - 0.05 / to_3db


def _kaiser_beta(att_db: float) -> float:
    if att_db > 50:
        return 0.1102 * (att_db - 8.7)
    if att_db >= 21:
        return 0.5842 * (att_db - 21) ** 0.4 + 0.07886 * (att_db - 21)
    return 0.0


_KERNEL_CACHE: dict = {}


def _resample_kernel(sr_in: int, sr_out: int, passband_end: float, stopband_begin: float, att_db: float):
    key = (sr_in, sr_out, passband_end, stopband_begin, att_db)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    ratio = Fraction(sr_out, sr_in)
    up, down = ratio.numerator, ratio.denominator  # output m sits at input time m*down/up
    nyq = min(sr_in, sr_out) / 2.0
    fp, fs = passband_end * nyq, stopband_begin * nyq
    fc = 0.5 * (fp + fs)  # -6 dB point
    df = fs - fp
    beta = _kaiser_beta(att_db)
    half_seconds = 0.5 * (att_db - 7.95) / (2.285 * 2.0 * math.pi * df)
    hw = half_seconds * sr_in  # half width in input samples
    jmin = math.floor(-hw)
    jmax = math.ceil((up - 1) * down / up + hw)
    j = torch.arange(jmin, jmax + 1, dtype=torch.float64)
    p = torch.arange(up, dtype=torch.float64)
    t = (p * down / up)[:, None] - j[None, :]  # [up, width], time from tap to output, input samples
    c = 2.0 * fc / sr_in
    h = c * torch.special.sinc(c * t)
    u = (t / hw).clamp(-1.0, 1.0)
    w = torch.special.i0(beta * torch.sqrt(1.0 - u * u)) / torch.special.i0(torch.tensor(beta, dtype=torch.float64))
    w = torch.where(t.abs() <= hw, w, torch.zeros_like(w))
    kernel = (h * w)[:, None, :].contiguous()  # [up, 1, width]
    out = (kernel, up, down, jmin)
    _KERNEL_CACHE[key] = out
    return out


def resample_soxr_hq_like(
    x: np.ndarray,
    sr_in: int,
    sr_out: int,
    passband_end: Optional[float] = None,
    stopband_begin: float = 1.0,
    att_db: Optional[float] = None,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Resample along the last axis like librosa.resample(res_type='soxr_hq', fix=True).

    Output length is ceil(n * sr_out / sr_in); edges are zero-padded like a flushed libsoxr stream.
    Computed in float64 and returned in the input dtype.
    """
    sr_in, sr_out = int(sr_in), int(sr_out)
    if sr_in == sr_out:
        return x
    if passband_end is None:
        passband_end = _soxr_passband_end()
    if att_db is None:
        att_db = SOXR_LIKE_ATT_DB
    kernel, up, down, jmin = _resample_kernel(sr_in, sr_out, passband_end, stopband_begin, att_db)
    width = kernel.shape[-1]
    lead = x.shape[:-1]
    n = x.shape[-1]
    n_out = int(math.ceil(n * sr_out / sr_in))
    frames = int(math.ceil(n_out / up))
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64)).reshape(-1, 1, n).to(device)
    left = -jmin
    total = (frames - 1) * down + width
    right = max(0, total - left - n)
    xt = torch.nn.functional.pad(xt, (left, right))
    y = torch.nn.functional.conv1d(xt, kernel.to(device), stride=down)  # [rows, up, frames]
    y = y.transpose(1, 2).reshape(xt.shape[0], -1)[:, :n_out]
    return y.cpu().numpy().reshape(*lead, n_out).astype(x.dtype, copy=False)


def _normalize(wave: np.ndarray, max_peak: float, min_peak: Optional[float] = None) -> np.ndarray:
    """audio_separator.separator.uvr_lib_v5.spec_utils.normalize (in place)."""
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave *= max_peak / maxv
    elif min_peak is not None and maxv < min_peak:
        wave *= min_peak / maxv
    return wave


def model_hash(model_path: str) -> str:
    size = os.path.getsize(model_path)
    with open(model_path, "rb") as f:
        if size >= _HASH_BYTES:
            f.seek(size - _HASH_BYTES, os.SEEK_SET)
        return hashlib.md5(f.read()).hexdigest()


def resolve_model_data(model_path: str, data_dir: Optional[str] = None) -> dict:
    """audio-separator Separator.load_model_data_using_hash: md5 lookup in mdx_model_data.json, then vr_model_data.json."""
    data_dir = data_dir or os.path.dirname(os.path.abspath(model_path))
    h = model_hash(model_path)
    for name in ("mdx_model_data.json", "vr_model_data.json"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                table = json.load(f)
            if h in table:
                data = dict(table[h])
                data["_hash"] = h
                data["_source"] = name
                return data
    raise ValueError(f"parameters for MD5 hash {h} not found in {data_dir} mdx/vr model data")


def _to_channels_first(waveform) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().to("cpu", torch.float32).numpy()
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 3:  # ComfyUI AUDIO: [batch, channels, samples]; the first item is used
        x = x[0]
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected [channels, samples], got shape {x.shape}")
    return x


class KimVocalSeparator:
    """Reproduces LongCat-Video-Avatar separate_vocals(Separator(...).load_model('Kim_Vocal_2.onnx'))."""

    def __init__(
        self,
        model_path: str,
        model_data_dir: Optional[str] = None,
        provider: str = "auto",  # "auto" | "cuda" | "cpu"
        device_id: int = 0,
        segment_size: int = 256,
        overlap: float = 0.25,
        batch_size: int = 2,  # official separate_vocals default (not the Separator's mdx batch_size)
        hop_length: int = 1024,
        enable_denoise: bool = False,
        normalization_threshold: float = 0.9,
        sample_rate: int = MODEL_SAMPLE_RATE,
        cuda_mem_limit_bytes: Optional[int] = None,
        cuda_options: Optional[dict] = None,  # extra CUDAExecutionProvider options, e.g. {"use_tf32": "0"}
        release_gpu_memory: bool = True,
    ):
        import onnxruntime as ort

        self.model_path = model_path
        self.model_data = resolve_model_data(model_path, model_data_dir)
        self.compensate = float(self.model_data["compensate"])
        self.dim_f = int(self.model_data["mdx_dim_f_set"])
        self.dim_t = 2 ** int(self.model_data["mdx_dim_t_set"])
        self.n_fft = int(self.model_data["mdx_n_fft_scale_set"])
        self.primary_stem = self.model_data.get("primary_stem", "Vocals")
        if segment_size != self.dim_t:
            raise ValueError(f"segment_size {segment_size} != model dim_t {self.dim_t} (audio-separator would fall back to onnx2torch)")
        self.segment_size = segment_size
        self.overlap = overlap
        self.batch_size = max(1, int(batch_size))
        self.hop_length = hop_length
        self.enable_denoise = enable_denoise
        self.normalization_threshold = normalization_threshold
        self.sample_rate = sample_rate
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim

        available = ort.get_available_providers()
        want_cuda = provider in ("auto", "cuda") and "CUDAExecutionProvider" in available
        if provider == "cuda" and not want_cuda:
            raise RuntimeError(f"CUDAExecutionProvider not available in onnxruntime ({available})")
        if want_cuda:
            opts = {"device_id": int(device_id), "cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
            if cuda_mem_limit_bytes:
                opts["gpu_mem_limit"] = int(cuda_mem_limit_bytes)
            opts.update({k: str(v) for k, v in (cuda_options or {}).items()})
            providers: Sequence = [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.provider = self.session.get_providers()[0]
        # Return the CUDA arena to the driver after the last chunk of every separate() call
        self._last_run_options = None
        if self.provider == "CUDAExecutionProvider" and release_gpu_memory:
            ro = ort.RunOptions()
            ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{int(device_id)}")
            self._last_run_options = ro
        self.input_name = self.session.get_inputs()[0].name
        # audio-separator runs the STFT on torch_device: cuda when torch sees a GPU, else cpu
        if self.provider == "CUDAExecutionProvider" and torch.cuda.is_available():
            self.torch_device = torch.device("cuda", int(device_id))
        else:
            self.torch_device = torch.device("cpu")
        self.window = torch.hann_window(self.n_fft, periodic=True)

    # --- uvr_lib_v5.stft.STFT ------------------------------------------------------------
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        batch, ch, t = x.shape
        spec = torch.stft(x.reshape(-1, t), n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(x.device), center=True, return_complex=True)
        spec = torch.view_as_real(spec).permute(0, 3, 1, 2)  # [b*c, 2, F, frames]
        spec = spec.reshape(batch, ch, 2, -1, spec.shape[-1]).reshape(batch, ch * 2, -1, spec.shape[-1])
        return spec[..., : self.dim_f, :]

    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        batch, c, f, frames = spec.shape
        n_bins = self.n_fft // 2 + 1
        spec = torch.cat([spec, torch.zeros(batch, c, n_bins - f, frames, device=spec.device, dtype=spec.dtype)], -2)
        spec = spec.reshape(batch, c // 2, 2, n_bins, frames).reshape(-1, 2, n_bins, frames).permute(0, 2, 3, 1)
        spec = spec[..., 0] + spec[..., 1] * 1.0j
        wave = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(spec.device), center=True)
        return wave.reshape(batch, 2, -1)

    def _model_run(self, spek: torch.Tensor, run_options=None) -> np.ndarray:
        return self.session.run(None, {self.input_name: spek.cpu().numpy()}, run_options)[0]

    def _run_model(self, mix: torch.Tensor, last: bool = False) -> np.ndarray:
        """MDXSeparator.run_model (is_match_mix=False)."""
        spek = self._stft(mix.to(self.torch_device))
        spek[:, :, :3, :] *= 0
        ro = self._last_run_options if last else None
        if self.enable_denoise:
            spec_pred = self._model_run(-spek) * -0.5 + self._model_run(spek, ro) * 0.5
        else:
            spec_pred = self._model_run(spek, ro)
        return self._istft(torch.tensor(spec_pred).to(self.torch_device)).cpu().detach().numpy()

    def _demix_batched(self, mix: np.ndarray) -> np.ndarray:
        """longcat_video.audio_process.vocal_separator._demix_batched."""
        chunk_size, overlap, trim = self.chunk_size, self.overlap, self.trim
        gen_size = chunk_size - 2 * trim
        pad = gen_size + trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate((np.zeros((2, trim), dtype=np.float32), mix, np.zeros((2, pad), dtype=np.float32)), axis=1)
        step = int((1 - overlap) * chunk_size)
        starts = list(range(0, mixture.shape[-1], step))
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros_like(result)
        for g in range(0, len(starts), self.batch_size):
            parts, meta = [], []
            for start in starts[g: g + self.batch_size]:
                end = min(start + chunk_size, mixture.shape[-1])
                actual = end - start
                window = np.tile(np.hanning(actual)[None, None, :], (1, 2, 1)) if overlap else None
                part = mixture[:, start:end]
                if actual != chunk_size:
                    part = np.pad(part, ((0, 0), (0, chunk_size - actual)))
                parts.append(part)
                meta.append((start, end, actual, window))
            mix_wave = torch.from_numpy(np.stack(parts)).to(self.torch_device, dtype=torch.float32)
            with torch.no_grad():
                batch_result = self._run_model(mix_wave, last=g + self.batch_size >= len(starts))
            for i, (start, end, actual, window) in enumerate(meta):
                chunk = batch_result[i: i + 1, ..., :actual]
                if window is not None:
                    chunk *= window
                    divider[..., start:end] += window
                else:
                    divider[..., start:end] += 1
                result[..., start:end] += chunk
        with np.errstate(invalid="ignore", divide="ignore"):
            source = (result / divider)[0, :, trim:-trim]
        source = source[:, : mix.shape[-1]]
        source *= self.compensate
        return source

    def prepare_mix(self, waveform, sample_rate: int) -> np.ndarray:
        """librosa.load(mono=False, sr=44100) + CommonSeparator.prepare_mix -> float32 [2, n] at 44.1 kHz."""
        x = _to_channels_first(waveform)
        if x.shape[0] > 2:  # the official path would fail on >2 channels; downmix like librosa.to_mono
            x = x.mean(axis=0, keepdims=True)
        if int(sample_rate) != self.sample_rate:
            x = resample_soxr_hq_like(x, int(sample_rate), self.sample_rate)
        x = x.astype(np.float32, copy=False)
        if x.shape[0] == 1:
            x = np.asfortranarray([x[0], x[0]])
        return np.ascontiguousarray(x)

    def separate(self, waveform, sample_rate: int, duration: Optional[float] = None,
                 output_sample_rate: int = OUTPUT_SAMPLE_RATE) -> np.ndarray:
        """Vocals as float32 mono at output_sample_rate, exactly the array separate_vocals() returns."""
        mix = self.prepare_mix(waveform, sample_rate)
        mix = _normalize(mix, self.normalization_threshold, -self.normalization_threshold)
        target_samples = None
        if duration is not None:
            target_samples = min(math.ceil(duration * self.sample_rate), mix.shape[-1])
            mix = mix[..., : min(target_samples + self.chunk_size, mix.shape[-1])]
        source = self._demix_batched(mix)
        source = _normalize(source, self.normalization_threshold, -self.normalization_threshold)
        mono = np.mean(source, axis=0)  # librosa.to_mono
        if self.sample_rate != output_sample_rate:
            mono = resample_soxr_hq_like(mono, self.sample_rate, output_sample_rate)
        if target_samples is not None:
            mono = mono[: math.ceil(target_samples * output_sample_rate / self.sample_rate)]
        return mono.astype(np.float32, copy=False)
