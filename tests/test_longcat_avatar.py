"""LongCat-Video-Avatar 1.6 port: a tiny random model must give the same velocity with and without the condition cache and
with and without row chunking, in every mode; the official 1.6 sigmas and buckets. CPU only, no model files. Run from the
ComfyUI root:
python -m pytest -q custom_nodes/SECoursesAudioTools/tests/test_longcat_avatar.py -p no:cacheprovider"""
import importlib
import sys
import types
import unittest
from pathlib import Path

import torch

import comfy.ldm.modules.attention as comfy_attention
import comfy.ops

PACKAGE = "secourses_audio_tools_under_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE] = package
lcm = importlib.import_module(f"{PACKAGE}.longcat_avatar_model")


def tiny_model():
    torch.manual_seed(0)
    model = lcm.LongCatVideoAvatarTransformer(
        hidden_size=256, depth=3, num_heads=4, caption_channels=64, mlp_ratio=4, adaln_tembed_dim=64,
        audio_window=5, audio_block=5, audio_channel=32, intermediate_dim=32, output_dim=48, context_tokens=4,
        dtype=torch.float32, operations=comfy.ops.manual_cast).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return model


class LongCatPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_attention = comfy_attention.optimized_attention
        comfy_attention.optimized_attention = comfy_attention.attention_pytorch
        cls.model = tiny_model()
        T, H, W = 7, 8, 6
        cls.T = T
        cls.x = torch.randn(1, 16, T, H, W)
        cls.text = cls.model.text_embed(torch.randn(1, 20, 64), torch.cat([torch.ones(1, 12), torch.zeros(1, 8)], 1))
        cls.audio = torch.randn(1, (T - 1) * 4 + 1, 5, 5, 32)

    @classmethod
    def tearDownClass(cls):
        comfy_attention.optimized_attention = cls.saved_attention

    def run_model(self, num_cond, num_ref, cache=None):
        tok = self.model.audio_tokens(self.audio, self.T, num_ref)
        t = torch.full((1, self.T), 900.0)
        t[:, :num_cond] = 0
        return self.model(self.x, t, self.text, tok, num_cond=num_cond, num_ref=num_ref,
                          ref_img_index=10 if num_ref else None, mask_frame_range=0, text_kv_cache={}, cond_cache=cache)

    def test_modes_and_cache(self):
        with torch.no_grad():
            for num_cond, num_ref in ((0, 0), (1, 0), (3, 1)):
                plain = self.run_model(num_cond, num_ref)
                self.assertTrue(torch.isfinite(plain).all())
                if num_cond:
                    cache = {}
                    first = self.run_model(num_cond, num_ref, cache)
                    second = self.run_model(num_cond, num_ref, cache)
                    self.assertLess((plain[:, :, num_cond:] - first[:, :, num_cond:]).abs().max().item(), 1e-5)
                    self.assertLess((plain[:, :, num_cond:] - second[:, :, num_cond:]).abs().max().item(), 1e-5)

    def test_chunking_is_exact(self):
        with torch.no_grad():
            plain = self.run_model(3, 1)
            saved = lcm.ROW_CHUNK
            lcm.ROW_CHUNK = 5
            try:
                chunked = self.run_model(3, 1)
            finally:
                lcm.ROW_CHUNK = saved
            self.assertEqual((plain - chunked).abs().max().item(), 0.0)

    def test_rope_matches_rotate_half(self):
        cos, sin, mats = lcm.rope_3d_cos_sin(128, (3, 4, 5), torch.device("cpu"), 10, 1)
        x = torch.randn(1, 2, 60, 128)
        reference = x * cos + lcm._rotate_half(x) * sin
        self.assertLess((lcm.apply_rope(x, cos, sin, mats) - reference).abs().max().item(), 1e-5)


class LongCatRecipeTests(unittest.TestCase):
    def test_official_sigmas_and_buckets(self):
        nodes = importlib.import_module(f"{PACKAGE}.longcat_avatar_nodes")
        timesteps, sigmas = nodes.official_sigmas(True)
        expected = [1.0, 0.979978, 0.954488, 0.920936, 0.874781, 0.807277, 0.699158, 0.497992, 0.0]
        self.assertEqual(len(sigmas), 9)
        for got, want in zip(sigmas.tolist(), expected):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(nodes.bucket_size((1536, 1024), "480p bucket (official)"), (768, 512))
        self.assertEqual(nodes.bucket_size((480, 832), "480p bucket (official)"), (480, 832))


if __name__ == "__main__":
    unittest.main()
