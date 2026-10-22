"""MOSS-Image Edit reference branch: the patched native model must match a straight-line version of ChenkinLab's
algorithm (references at timestep 0 attending only to each other; text and target attending to themselves plus the
references' keys and values in every block), with the reference cache reused across steps. Run from the ComfyUI root:
python -m pytest -q custom_nodes/SECoursesAudioTools/tests/test_moss_image_edit.py -p no:cacheprovider"""
import importlib
import math
import sys
import types
import unittest
from pathlib import Path

import torch

import comfy.ops
import comfy.patcher_extension
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import MOSSImage, situ

PACKAGE = "secourses_audio_tools_under_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE] = package
edit = importlib.import_module(f"{PACKAGE}.moss_image_edit_nodes")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def official_reference_size(width, height, max_edge):
    # ChenkinLab ComfyUI-Moss-Image-Edit _reference_tensor: scale down only, round each side to 16 (half up)
    limit = max_edge // 16 * 16
    scale = min(1.0, limit / max(width, height))
    return tuple(min(limit, max(16, int(math.floor(v * scale / 16 + 0.5)) * 16)) for v in (width, height))


def tiny_model(fused_qkv=False):
    torch.manual_seed(0)
    model = MOSSImage(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=24, latent_channels=8, fused_qkv=fused_qkv,
                      dtype=torch.float32, device=DEVICE, operations=comfy.ops.disable_weight_init)
    for name, p in model.named_parameters():
        torch.nn.init.normal_(p, 0.0, 0.3 if "norm" not in name else 0.1)
        if "norm" in name:
            p.data += 1
    return model


def rotate(x, coords, d_axis):
    """Split-half RoPE from (r, f, h, w) coordinates, written out: pairs (x[i], x[i + D/2]) with axis-major frequencies."""
    inv = torch.cat([10000.0 ** (-torch.arange(0, d_axis, 2, device=x.device, dtype=torch.float64) / d_axis) for _ in range(4)])
    axis = torch.arange(4, device=x.device).repeat_interleave(d_axis // 2)
    angles = (coords.double()[:, axis] * inv).float()
    cos, sin = torch.cat((angles.cos(), angles.cos()), -1), torch.cat((angles.sin(), angles.sin()), -1)
    a, b = x.chunk(2, -1)
    return x * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]


def target_coords(h, w, text_len, device):
    rows = (torch.arange(h, device=device) - (h - 1) / 2).repeat_interleave(w)
    cols = (torch.arange(w, device=device) - (w - 1) / 2).repeat(h)
    image = torch.stack((torch.zeros_like(rows), torch.full_like(rows, text_len), rows, cols), -1)
    text = torch.stack((torch.zeros(text_len, device=device), torch.arange(text_len, device=device), torch.zeros(text_len, device=device), torch.zeros(text_len, device=device)), -1)
    return torch.cat((image, text)).float()


def reference_coords(sizes, device):
    out = []
    for index, (h, w) in enumerate(sizes):
        rows = (torch.arange(h, device=device) - (h - 1) / 2).repeat_interleave(w)
        cols = (torch.arange(w, device=device) - (w - 1) / 2).repeat(h)
        out.append(torch.stack((torch.full_like(rows, index + 1), torch.zeros_like(rows), rows, cols), -1))
    return torch.cat(out).float()


def oracle(model, x, timestep, context, refs):
    """ChenkinLab's [text, ref, target] edit forward written out with the model's own modules."""
    B, _, H, W = x.shape
    heads, d_axis = model.layers[0].num_heads, model.rope_axes_dim[0]

    def modulation(t):
        vec = model.time_embedding(timestep_embedding(t, 256))
        return model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)

    def qkv(block, y, coords):
        q, k, v = edit.qkv(block, y)
        shape = (*y.shape[:2], heads, -1)
        q = rotate(block.q_norm(q.reshape(shape)), coords, d_axis)
        k = rotate(block.k_norm(k.reshape(shape)), coords, d_axis)
        return q.transpose(1, 2), k.transpose(1, 2), v.reshape(shape).transpose(1, 2)

    def finish(block, x, attention, mod):
        x = x + block.proj["o"](attention.transpose(1, 2).flatten(2)) * mod[2]
        y = block.norm2(x) * (1 + mod[4]) + mod[3]
        hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + block.proj["ffn_down"](situ(hidden, gate)) * mod[5]

    pe = model.rope(H, W, context.shape[1], x.device)
    body = torch.cat((model.video_embedder(x.flatten(2).transpose(1, 2)), model.text_embedder(context, pe[:, H * W:])), 1)
    body_coords = target_coords(H, W, context.shape[1], x.device)
    ref = torch.cat([model.video_embedder(r.flatten(2).transpose(1, 2)) for r in refs], 1)
    ref_coords = reference_coords([r.shape[-2:] for r in refs], x.device)
    mod, mod0 = modulation(timestep), modulation(torch.zeros(1, device=x.device))
    for block in model.layers:
        rq, rk, rv = qkv(block, block.norm1(ref) * (1 + mod0[1]) + mod0[0], ref_coords)
        q, k, v = qkv(block, block.norm1(body) * (1 + mod[1]) + mod[0], body_coords)
        k = torch.cat((k, rk.expand(B, -1, -1, -1)), 2)
        v = torch.cat((v, rv.expand(B, -1, -1, -1)), 2)
        body = finish(block, body, torch.nn.functional.scaled_dot_product_attention(q, k, v), mod)
        ref = finish(block, ref, torch.nn.functional.scaled_dot_product_attention(rq, rk, rv), mod0)
    head_shift, head_scale = model.video_head_modulation(model.time_embedding(timestep_embedding(timestep, 256))).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    out = model.final_linear_video(model.final_norm_video(body[:, :H * W]) * (1 + head_scale) + head_shift)
    return out.transpose(1, 2).reshape(B, -1, H, W)


def patched_forward(model, run, x, timestep, context):
    transformer_options = {
        edit.KEY: run,
        "patches_replace": {"dit": {("single_block", i): edit.edit_block(model, i) for i in range(len(model.layers))}},
        "wrappers": {comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL: {edit.KEY: [edit.diffusion_model_wrapper]}},
    }
    return model(x, timestep, context, transformer_options=transformer_options)


class MossImageEditTests(unittest.TestCase):
    def test_reference_size_matches_the_official_rule(self):
        for size in [(768, 768), (1728, 1152), (481, 640), (2496, 1664), (100, 3000), (1023, 17), (10, 10)]:
            for edge in (512, 1024, 1536, 520):
                self.assertEqual(edit.reference_size(*size, edge), official_reference_size(*size, edge), (size, edge))

    def test_patched_model_matches_the_official_algorithm_and_reuses_the_cache(self):
        for fused in (False, True):
            model = tiny_model(fused)
            torch.manual_seed(1)
            context = torch.randn(2, 5, 24, device=DEVICE)
            refs = [torch.randn(1, 8, 3, 3, device=DEVICE), torch.randn(1, 8, 2, 4, device=DEVICE)]
            run = edit.EditRun(refs)
            for t in (0.9, 0.35):
                x = torch.randn(2, 8, 4, 6, device=DEVICE)
                timestep = torch.full((2,), t, device=DEVICE)
                with torch.no_grad():
                    expected = oracle(model, x, timestep, context, refs)
                    got = patched_forward(model, run, x.clone(), timestep, context)
                torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-4)
            self.assertEqual(len(run.cache), len(model.layers))

    def test_without_references_the_model_is_unchanged(self):
        node = edit.SEMossImageEditReferences()
        sentinel = object()
        self.assertEqual(node.apply(sentinel, None, {"prompt": "x", "images": []}, 1024, 832, 1216), (sentinel, 832, 1216))


if __name__ == "__main__":
    unittest.main()
