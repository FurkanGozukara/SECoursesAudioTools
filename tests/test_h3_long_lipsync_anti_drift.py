"""Anti-drift switches of H3 Long Lip Sync on tiny synthetic latents: CPU only, no model or sampler.

Run from the ComfyUI directory:  python -m unittest custom_nodes.SECoursesAudioTools.tests.test_h3_long_lipsync_anti_drift
"""
import json
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
from custom_nodes.SECoursesAudioTools import h3_long_lipsync as ll

RATE, HOP, STEPS, SEED = 320, 8, 4, 3
SECONDS = 14.0  # three 141-frame windows, 39 held frames = 12 held latents
APPLY_MODEL = comfy.patcher_extension.WrappersMP.APPLY_MODEL


class MiniMaxH3Model:
    pass


class FakeModel:
    def __init__(self, wrappers=None):
        self.wrappers = wrappers or {}

    def get_model_object(self, name):
        return MiniMaxH3Model()

    def clone(self):
        return FakeModel({kind: {key: list(w) for key, w in group.items()} for kind, group in self.wrappers.items()})

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


def fake_sample(v, vm, seed):
    """Masked sampling stand-in: rows masked 0 return unchanged, fractional rows mix, free rows get seeded content."""
    content = torch.randn(v.shape, generator=torch.Generator().manual_seed(seed))
    return v * (1 - vm) + content * vm


def previous_loop(plan):
    """The window loop before the anti-drift switches, driven by the same sampling stand-in."""
    video = torch.zeros((1, 24, plan[-1]["end_latent"], 2, 2))
    windows = []
    for index, span in enumerate(plan):
        s, e = span["start_latent"], span["end_latent"]
        v = video[:, :, s:e].clone()
        vm = torch.ones((1, 1, e - s, 1, 1), dtype=v.dtype)
        vm[:, :, :span["held_latents"]] = 0
        generated = fake_sample(v, vm, SEED + index)
        held = span["held_latents"]
        video[:, :, s + held:e] = generated[:, :, held:]
        windows.append((v, vm))
    return video, windows


def run_node(module=ll, sample=fake_sample, **switches):
    calls, decoded, model = [], {}, FakeModel()

    def sampler_execute(noise, guider, sampler, sigmas, work):
        (v, a), (vm, _) = work["samples"].unbind(), work["noise_mask"].unbind()
        generated = sample(v, vm, noise.seed)
        calls.append(SimpleNamespace(v=v.clone(), vm=vm.clone(), generated=generated.clone(), model=guider.model))
        return SimpleNamespace(args=({"samples": comfy.nested_tensor.NestedTensor((generated, a.clone()))},))

    class Decoder:
        seconds, frames_written = 0.0, 0

        def __init__(self, vae, total_latents, writer):
            decoded["total"] = total_latents

        def decode(self, video, total_latents, final):
            decoded["video"] = video.clone()

    writer = SimpleNamespace(close=lambda: None, abort=lambda: None)
    audio_vae = SimpleNamespace(audio_sample_rate=RATE, downscale_ratio=HOP, encode=lambda w: torch.randn(
        (1, 32, 2, w.shape[1] // HOP), generator=torch.Generator().manual_seed(7)))
    latent = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 10)))}
    audio = {"waveform": torch.zeros(1, 1, int(SECONDS * RATE)), "sample_rate": RATE}
    with tempfile.TemporaryDirectory() as output, patch.object(module.folder_paths, "get_output_directory", return_value=output), patch.multiple(
            module, BasicScheduler=SimpleNamespace(execute=lambda m, name, steps, d: SimpleNamespace(args=(torch.linspace(1, 0, steps + 1),))),
            BasicGuider=SimpleNamespace(execute=lambda m, cond: SimpleNamespace(args=(SimpleNamespace(model=m),))),
            RandomNoise=SimpleNamespace(execute=lambda seed: SimpleNamespace(args=(SimpleNamespace(seed=seed),))),
            SamplerCustomAdvanced=SimpleNamespace(execute=sampler_execute), IncrementalH3Decoder=Decoder,
            FFmpegWriter=lambda *args, **kwargs: writer, write_wav=lambda *args: None, mux_audio=lambda *args, **kwargs: None):
        result = module.SEH3LongLipSync().generate(model, [["embedding", {}]], latent, audio, audio_vae, None, SEED, STEPS,
                                                   window_frames="141", overlap_frames="39", **switches)
    return calls, decoded, json.loads(result["result"][2]), model


class AntiDriftTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = ll.continuation_plan(SECONDS, 141, 39)
        cls.reference, cls.windows = previous_loop(cls.plan)

    def test_inputs_are_appended_after_existing_widgets_and_off_by_default(self):
        optional = ll.SEH3LongLipSync.INPUT_TYPES()["optional"]
        self.assertEqual(list(optional), ["decode_vae", "mouth_pass", "mouth_fidelity", "mouth_blend", "mouth_model", "mouth_detector",
                                          "context_noise", "keystone", "capstone_from_step"])
        self.assertEqual([optional[k][1]["default"] for k in ("context_noise", "keystone", "capstone_from_step")], [0.0, False, 0])

    def test_switches_off_reproduce_the_previous_loop(self):
        self.assertEqual(len(self.plan), 3)
        for switches in ({}, {"context_noise": 0.0, "keystone": False, "capstone_from_step": 0}):
            calls, decoded, report, model = run_node(**switches)
            self.assertTrue(torch.equal(decoded["video"], self.reference))
            for call, (v, vm) in zip(calls, self.windows, strict=True):
                self.assertTrue(torch.equal(call.v, v) and torch.equal(call.vm, vm))
                self.assertIs(call.model, model)
            self.assertEqual([chunk["anti_drift"] for chunk in report["chunks"]], [[], [], []])

    def test_context_noise_ramps_toward_the_join_and_keeps_the_overlap(self):
        calls, decoded, report, _ = run_node(context_noise=0.8)
        self.assertTrue(torch.equal(calls[0].vm, torch.ones_like(calls[0].vm)))
        expected = torch.tensor([0.0] + [0.8 * (1 - i / 10) for i in range(11)])
        for call in calls[1:]:
            ramp = call.vm[0, 0, :, 0, 0]
            self.assertTrue(torch.allclose(ramp[:12], expected, atol=1e-6))
            self.assertEqual((ramp[0].item(), ramp[11].item()), (0.0, 0.0))  # latent 0 and the join latent stay exact
            self.assertTrue(torch.equal(ramp[12:], torch.ones_like(ramp[12:])))
            self.assertFalse(torch.equal(call.generated[:, :, 1:11], call.v[:, :, 1:11]))
        # The partly resampled overlap never reaches the video: every held latent is the original.
        self.assertTrue(torch.equal(decoded["video"], self.reference))
        self.assertEqual([chunk["anti_drift"] for chunk in report["chunks"]], [[], ["context noise 0.8"], ["context noise 0.8"]])

    def test_a_changed_locked_row_still_fails_with_context_noise(self):
        def corrupt_join(v, vm, seed):
            out = fake_sample(v, vm, seed)
            if vm[0, 0, 0, 0, 0] == 0:
                out[:, :, 11] += 1.0
            return out
        with self.assertRaisesRegex(RuntimeError, "changed held samples"):
            run_node(sample=corrupt_join, context_noise=0.8)

    def test_keystone_locks_frame_zero_and_keeps_frame_count(self):
        calls, decoded, report, _ = run_node(keystone=True)
        for call, (v, vm), span in zip(calls[1:], self.windows[1:], self.plan[1:]):
            self.assertTrue(torch.equal(call.v[:, :, 0], self.reference[:, :, 0]))
            self.assertFalse(torch.equal(self.reference[:, :, span["start_latent"]], self.reference[:, :, 0]))
            self.assertTrue(torch.equal(call.v[:, :, 1:], v[:, :, 1:]) and torch.equal(call.vm, vm))
        self.assertEqual(decoded["total"], self.plan[-1]["end_latent"])
        self.assertTrue(torch.equal(decoded["video"], self.reference))
        self.assertEqual([chunk["anti_drift"] for chunk in report["chunks"]], [[], ["keystone"], ["keystone"]])

    def test_capstone_wrapper_attaches_and_detaches_one_locked_latent(self):
        video, audio, anchor = torch.randn(1, 24, 7, 2, 2), torch.randn(1, 32, 2, 12), torch.randn(1, 24, 1, 2, 2)
        x, shapes = comfy.utils.pack_latents([video, audio])
        mask = torch.ones(1, 1, 7, 2, 2)
        mask[:, :, :3] = 0
        sigmas = torch.linspace(1, 0, STEPS + 1)
        wrapper, seen = ll.capstone_wrapper(anchor, sigmas, 3), []

        def executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs):
            seen.append((x, kwargs["latent_shapes"], kwargs["denoise_mask"]))
            return x * 2

        for step in range(STEPS):
            out = wrapper(executor, x, sigmas[step:step + 1], None, None, None, {}, latent_shapes=shapes, denoise_mask=mask)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.equal(out, x * 2))
            seen_x, seen_shapes, seen_mask = seen[-1]
            if step < 2:
                self.assertIs(seen_x, x)
                continue
            seen_video, seen_audio = comfy.utils.unpack_latents(seen_x, seen_shapes)
            self.assertEqual(seen_video.shape[2], 8)
            self.assertTrue(torch.equal(seen_video[:, :, :7], video) and torch.equal(seen_video[:, :, 7:], anchor))
            self.assertTrue(torch.equal(seen_audio, audio))
            self.assertTrue(torch.equal(seen_mask[:, :, :7], mask) and not seen_mask[:, :, 7:].any())

    def test_capstone_runs_from_the_second_window_with_the_first_windows_anchor(self):
        calls, decoded, report, model = run_node(capstone_from_step=3)
        self.assertIs(calls[0].model, model)
        self.assertIs(calls[1].model, calls[2].model)
        self.assertEqual(model.wrappers, {})
        (wrapper,) = calls[1].model.wrappers[APPLY_MODEL]["se_h3_capstone"]
        cap = 5 * ((self.plan[0]["end_latent"] - 1) // 5)
        window = calls[1].v
        x, shapes = comfy.utils.pack_latents([window, torch.zeros(1, 32, 2, 5)])
        seen = []
        wrapper(lambda x, *args, **kwargs: seen.append(x) or torch.zeros_like(x), x, torch.tensor([0.0]), None, None, None, {},
                latent_shapes=shapes, denoise_mask=torch.zeros(1, 1, window.shape[2], 2, 2))
        attached = comfy.utils.unpack_latents(seen[0], [window.shape[:2] + (window.shape[2] + 1,) + window.shape[3:], shapes[1]])[0]
        self.assertTrue(torch.equal(attached[:, :, -1:], decoded["video"][:, :, cap:cap + 1]))
        self.assertEqual(decoded["total"], self.plan[-1]["end_latent"])
        self.assertTrue(torch.equal(decoded["video"], self.reference))
        self.assertEqual([chunk["anti_drift"] for chunk in report["chunks"]], [[], ["capstone from step 3/4"], ["capstone from step 3/4"]])

    def test_capstone_step_above_the_schedule_does_nothing(self):
        calls, decoded, report, model = run_node(capstone_from_step=STEPS + 1)
        self.assertTrue(all(call.model is model for call in calls))
        self.assertEqual([chunk["anti_drift"] for chunk in report["chunks"]], [[], [], []])
