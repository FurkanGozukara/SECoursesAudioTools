"""Mel-Band RoFormer source separation network (inference only).

Adapted from ZFTurbo/Music-Source-Separation-Training models/bs_roformer/mel_band_roformer.py at
commit d2f6ca1 (MIT), the code the MelBandRoformer vocal checkpoint by Kimberley Jensen was trained
with, itself based on lucidrains/BS-RoFormer (MIT). The training loss and dropout are removed, and
the two helpers it imported are reimplemented here: librosa's mel filter bank (ISC) and the part of
rotary-embedding-torch (MIT) the model uses. Licenses: docs/MelBandRoFormer-LICENSE.txt
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from torch import nn


def slaney_mel_filter_bank(sample_rate, n_fft, n_mels):
    """librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels) with its default Slaney scale and norm."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    max_hz = float(sample_rate) / 2
    max_mel = max_hz / f_sp
    if max_hz >= min_log_hz:
        max_mel = min_log_mel + np.log(max_hz / min_log_hz) / logstep
    mels = np.linspace(0.0, max_mel, n_mels + 2)
    mel_f = f_sp * mels
    log_t = mels >= min_log_mel
    mel_f[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


class RotaryEmbedding(nn.Module):
    """rotary-embedding-torch's RotaryEmbedding(dim) applied with rotate_queries_or_keys (sequence on dim -2)."""

    def __init__(self, dim):
        super().__init__()
        self.freqs = nn.Parameter(torch.empty(dim // 2), requires_grad=False)

    def rotate_queries_or_keys(self, t):
        positions = torch.arange(t.shape[-2], device=t.device, dtype=t.dtype).type(self.freqs.dtype)
        freqs = torch.outer(positions, self.freqs).repeat_interleave(2, dim=-1)
        pairs = t.unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (t * freqs.cos() + rotated * freqs.sin()).type(t.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(dim_inner, dim),
            nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, rotary_embed):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, *, dim, depth, dim_head, heads, rotary_embed):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, rotary_embed=rotary_embed),
                FeedForward(dim=dim),
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])

        for dim_in in dim_inputs:
            self.to_features.append(nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim),
            ))

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            outs.append(to_feature(split_input))

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden, depth):
    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if ind < len(dims) - 2:
            net.append(nn.Tanh())

    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.to_freqs = nn.ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            self.to_freqs.append(nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            ))

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []
        for band_features, mlp in zip(x, self.to_freqs):
            outs.append(mlp(band_features))

        return torch.cat(outs, dim=-1)


class MelBandRoformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo,
            num_stems,
            time_transformer_depth,
            freq_transformer_depth,
            num_bands,
            dim_head,
            heads,
            sample_rate,
            stft_n_fft,
            stft_hop_length,
            stft_win_length,
            stft_normalized,
            mask_estimator_depth,
    ):
        super().__init__()

        self.audio_channels = 2 if stereo else 1

        self.layers = nn.ModuleList([])

        transformer_kwargs = dict(dim=dim, heads=heads, dim_head=dim_head)

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs),
            ]))

        self.stft_win_length = stft_win_length
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = stft_n_fft // 2 + 1

        # binary mel bands as in the paper; the first and last bins are forced in like upstream
        mel_filter_bank = torch.from_numpy(slaney_mel_filter_bank(sample_rate, stft_n_fft, num_bands))
        mel_filter_bank[0][0] = 1.
        mel_filter_bank[-1, -1] = 1.
        freqs_per_band = mel_filter_bank > 0

        freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('num_bands_per_freq', freqs_per_band.sum(dim=0), persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_band.sum(dim=1).tolist())

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            self.mask_estimators.append(MaskEstimator(dim=dim, dim_inputs=freqs_per_bands_with_complex, depth=mask_estimator_depth))

    def forward(self, raw_audio):
        """[batch, channels, samples] -> separated [batch, channels, samples] (one stem) or [batch, stems, channels, samples]."""
        device = raw_audio.device
        batch, channels, _ = raw_audio.shape

        raw_audio, batch_audio_channel_packed_shape = pack([raw_audio], '* t')

        stft_window = torch.hann_window(self.stft_win_length, device=device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack(stft_repr, batch_audio_channel_packed_shape, '* f t c')[0]
        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        batch_arange = torch.arange(batch, device=device)[..., None]

        x = stft_repr[batch_arange, self.freq_indices]

        x = rearrange(x, 'b f t c -> b t (f c)')

        x = self.band_split(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x = time_transformer(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

        num_stems = len(self.mask_estimators)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        # average the estimated masks over the overlapping mel bands

        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1])

        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio
