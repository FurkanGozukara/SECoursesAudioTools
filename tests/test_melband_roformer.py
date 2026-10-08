"""Mel-Band RoFormer port and torchaudio-free resampling checks; CPU only, no checkpoint needed.

Run from the ComfyUI directory:  python -m unittest custom_nodes.SECoursesAudioTools.tests.test_melband_roformer
"""
import math
import sys
import unittest

import numpy as np
import torch

from custom_nodes.SECoursesAudioTools import h3_streaming_nodes as hs
from custom_nodes.SECoursesAudioTools import melband_roformer_model as mm
from custom_nodes.SECoursesAudioTools import melband_roformer_nodes as mn


class MelFilterBankTests(unittest.TestCase):
    def test_band_layout_matches_checkpoint(self):
        # band_split.to_features.{i}.1.weight of MelBandRoformer_fp32.safetensors is [384, 4 * freqs_in_band]
        expected = [7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 9, 9, 9, 10, 10, 11, 13, 13, 13, 15, 16, 17,
                    19, 20, 20, 22, 24, 26, 28, 29, 31, 33, 36, 39, 41, 44, 47, 50, 54, 57, 61, 66, 71, 76, 80, 86, 93,
                    99, 105, 113, 122, 130]
        weights = mm.slaney_mel_filter_bank(44100, 2048, 60)
        self.assertEqual(weights.dtype, np.float32)
        bands = weights > 0
        bands[0, 0] = True
        bands[-1, -1] = True
        self.assertEqual(bands.sum(axis=1).tolist(), expected)
        self.assertTrue(bands.any(axis=0).all())


class RotaryTests(unittest.TestCase):
    def test_pairs_are_rotated_by_position(self):
        rotary = mm.RotaryEmbedding(8)
        rotary.freqs.data.copy_(1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8)))
        t = torch.randn(2, 3, 5, 8)
        out = rotary.rotate_queries_or_keys(t)
        for pos in range(5):
            for pair in range(4):
                angle = pos * rotary.freqs[pair].item()
                x, y = t[..., pos, 2 * pair], t[..., pos, 2 * pair + 1]
                torch.testing.assert_close(out[..., pos, 2 * pair], x * math.cos(angle) - y * math.sin(angle), rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(out[..., pos, 2 * pair + 1], y * math.cos(angle) + x * math.sin(angle), rtol=1e-5, atol=1e-5)


class ChunkBlendTests(unittest.TestCase):
    def test_identity_model_reconstructs_input(self):
        identity = lambda batch: batch  # noqa: E731
        for length in (1000, mn.CHUNK // 2, mn.CHUNK + 12345, 3 * mn.CHUNK + 777):
            waveform = torch.randn(2, length)
            out = mn.separate_vocals(identity, waveform, torch.device("cpu"))
            self.assertEqual(out.shape, waveform.shape)
            torch.testing.assert_close(out, waveform, rtol=1e-5, atol=1e-5)


class NoTorchaudioTests(unittest.TestCase):
    def test_h3_normalize_waveform_resamples_without_torchaudio(self):
        saved = sys.modules.get("torchaudio", ...)
        sys.modules["torchaudio"] = None
        try:
            audio = {"waveform": torch.randn(1, 1, 48000), "sample_rate": 48000}
            out = hs.normalize_waveform(audio, 32000)
        finally:
            if saved is ...:
                del sys.modules["torchaudio"]
            else:
                sys.modules["torchaudio"] = saved
        self.assertEqual(tuple(out.shape), (2, 32000))


if __name__ == "__main__":
    unittest.main()
