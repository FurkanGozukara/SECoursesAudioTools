"""CPU checks for long-video coverage, bounded history and temporal decoding."""
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from custom_nodes.SECoursesAudioTools import avatarforever_nodes as af
from custom_nodes.SECoursesAudioTools.codeformer_mouth import blend_mouth, mouth_mask, CodeFormerMouth, TEMPLATE


class AvatarForeverTests(unittest.TestCase):
    def test_requested_durations_cover_audio_without_gaps(self):
        for seconds in (0.01, 10, 30, 60, 300):
            audio = {"sample_rate": 1000, "waveform": torch.zeros(1, 1, round(seconds * 1000))}
            padded, actual, frames = af.prepare_audio(audio, 25, 0, 0, 0)
            self.assertEqual(actual, seconds)
            self.assertEqual((frames - 1) % 8, 0)
            self.assertGreaterEqual(frames, math.ceil(seconds * 25))
            self.assertGreaterEqual(padded["waveform"].shape[-1], seconds * 1000)
            total, audio_steps = (frames - 1) // 8 + 1, round(frames / 25 * 25)
            for size in (1, 4, 11):
                plan = af.chunk_plan(total, audio_steps, size)
                self.assertEqual([i for s, e, _, _ in plan for i in range(s, e)], list(range(total)))
                self.assertEqual([i for _, _, s, e in plan for i in range(s, e)], list(range(audio_steps)))

    def test_trim_lead_in_and_no_repeated_audio(self):
        audio = {"sample_rate": 100, "waveform": torch.arange(1000).reshape(1, 1, -1).float()}
        out, seconds, _ = af.prepare_audio(audio, 25, 2, 3, 0.5)
        self.assertEqual(seconds, 3.5)
        torch.testing.assert_close(out["waveform"][..., :50], torch.zeros(1, 1, 50))
        torch.testing.assert_close(out["waveform"][..., 50:350], audio["waveform"][..., 200:500])
        _, seconds, _ = af.prepare_audio(audio, 25, 8, 300, 0)
        self.assertEqual(seconds, 2)
        with self.assertRaisesRegex(ValueError, "empty"):
            af.prepare_audio(audio, 25, 20, 0, 0)

    def test_default_history_is_bounded_for_five_minutes(self):
        for index in range(236):
            selected = af.selected_chunks(index, 1, True)
            self.assertIn(0, selected)
            self.assertEqual(selected[-1], index)
            self.assertLessEqual(len(selected), 3)  # sink, previous, current
            self.assertEqual(len(set(selected)), len(selected))
        self.assertEqual(af.selected_chunks(6, 0, False), [6])
        self.assertEqual(af.selected_chunks(6, -1, False), list(range(7)))

    def test_released_schedule_and_invalid_schedules(self):
        self.assertEqual(af.parse_sigmas(af.SIGMAS), [1, .98125, .909375, .421875, 0])
        for value in ("1", "1,.5", "1,nan,0", "1,inf,0", "1,.5,.6,0", "1,0,0", "1,-1,0"):
            with self.assertRaises(ValueError):
                af.parse_sigmas(value)

    def test_temporal_tiles_preserve_global_timeline(self):
        class Vae:
            def decode(self, clip):
                start, count = int(clip[0, 0, 0, 0, 0]), clip.shape[2]
                return (torch.arange(start * 8, start * 8 + count * 8 - 7) % 251 / 255).reshape(-1, 1, 1, 1)

            def decode_tiled(self, clip, **kwargs):
                assert kwargs["overlap"] < kwargs["tile_x"]
                return self.decode(clip)

        class Writer:
            def __init__(self):
                self.frames = []

            def write(self, frames):
                self.frames.append(frames)

        for count in (1, 2, 16, 17, 32, 189, 939):
            for tiled in (False, True):
                for tile, overlap in ((16, 8), (128, 32), (128, 128)):
                    writer = Writer()
                    af.decode_to_writer(Vae(), torch.arange(count).reshape(1, 1, count, 1, 1),
                                        writer, tiled, 64, 1024, tile, overlap)
                    actual = torch.cat(writer.frames).flatten()
                    expected = (torch.arange(count * 8 - 7) % 251).to(torch.uint8)
                    torch.testing.assert_close(actual, expected)

    def test_mouth_processor_receives_each_export_frame_once(self):
        class Vae:
            def decode(self, clip):
                return torch.zeros(clip.shape[2] * 8 - 7, 2, 2, 3)

        class Writer:
            frames = 0
            def write(self, frames):
                assert frames.dtype == torch.uint8
                assert torch.all(frames == 7)
                self.frames += len(frames)

        calls = []
        def process(frames):
            calls.append(len(frames))
            return frames.add_(7)

        writer = Writer()
        af.decode_to_writer(Vae(), torch.zeros(1, 1, 32, 1, 1), writer, False, 512, 64, 128, 32, process)
        self.assertEqual(writer.frames, 249)
        self.assertEqual(sum(calls), writer.frames)
        self.assertGreater(len(calls), 1)

    def test_mouth_roi_matches_full_canvas_and_preserves_surroundings(self):
        rng = np.random.default_rng(42)
        source = rng.integers(30, 220, (768, 512, 3), dtype=np.uint8)
        crop = rng.integers(40, 210, (512, 512, 3), dtype=np.uint8)
        restored = (crop.astype(np.int16) + rng.integers(-30, 31, crop.shape)).astype(np.uint8)
        center, width = np.array([256., 365.]), 110.
        mask = mouth_mask(center, width) * .7
        delta = (restored.astype(np.float32) - crop.astype(np.float32)) * mask
        for angle, offset in ((0, (0, 0)), (27, (100, -90)), (-18, (-200, 180))):
            transform = cv2.getRotationMatrix2D((256, 256), angle, .9).astype(np.float32)
            transform[:, 2] += offset
            full_delta = cv2.warpAffine(delta, cv2.invertAffineTransform(transform), (512, 768))
            expected = np.clip(np.rint(source.astype(np.float32) + full_delta), 0, 255).astype(np.uint8)
            actual = source.copy()
            blend_mouth(actual, crop, restored, transform, center, width, .7)
            difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            self.assertLessEqual(difference.max(), 2)
            self.assertLess(difference.mean(), .005)
            np.testing.assert_array_equal(actual[np.all(full_delta == 0, axis=2)], source[np.all(full_delta == 0, axis=2)])

    def test_no_face_and_zero_strength_leave_frames_unchanged(self):
        class Detector:
            def detect(self, image):
                return np.empty((0, 5)), None

        enhancer = object.__new__(CodeFormerMouth)
        enhancer.strength, enhancer.batch_size = .7, 4
        enhancer.detector = Detector()
        enhancer.report = {'frames': 0, 'detection_seconds': 0., 'processing_seconds': 0.}
        frames = torch.randint(0, 256, (5, 16, 16, 3), dtype=torch.uint8)
        original = frames.clone()
        self.assertIs(enhancer.process(frames), frames)
        torch.testing.assert_close(frames, original)
        self.assertEqual(enhancer.report['frames'], 5)
        enhancer.strength = 0
        enhancer.detector = None
        self.assertIs(enhancer.process(frames), frames)
        torch.testing.assert_close(frames, original)

    def test_mouth_worker_finishes_batches_and_propagates_errors(self):
        class Detector:
            def detect(self, image):
                if image[0, 0, 0] % 3 == 0:
                    return np.empty((0, 5)), None
                return np.array([[0, 0, 32, 32, 1.]]), TEMPLATE[None]

        enhancer = object.__new__(CodeFormerMouth)
        enhancer.strength, enhancer.fidelity, enhancer.batch_size = .7, .9, 4
        enhancer.device = torch.device('cpu')
        enhancer.detector = Detector()
        enhancer.patcher = SimpleNamespace(model=lambda batch, **kwargs: (torch.zeros_like(batch),))
        enhancer.report = dict.fromkeys(('frames', 'faces', 'detection_seconds', 'restoration_seconds',
                                         'compositing_seconds', 'processing_seconds'), 0.)
        caller = threading.get_ident()
        visited = []

        def composite(pixels, crops, restored, transforms, centers, widths, indices):
            self.assertNotEqual(threading.get_ident(), caller)
            for index in indices:
                visited.append(index)
                pixels[index, 0, 0] = 100 + index
            return 0.

        enhancer._composite = composite
        frames = torch.arange(10, dtype=torch.uint8).reshape(10, 1, 1, 1).expand(10, 32, 32, 3).clone()
        with patch.object(af.mm, 'load_models_gpu'):
            self.assertIs(enhancer.process(frames), frames)
            self.assertEqual(visited, [1, 2, 4, 5, 7, 8])
            self.assertEqual(frames[:, 0, 0, 0].tolist(), [0, 101, 102, 3, 104, 105, 6, 107, 108, 9])
            with patch.object(enhancer, '_composite', side_effect=RuntimeError('compositing failed')):
                with self.assertRaisesRegex(RuntimeError, 'compositing failed'):
                    enhancer.process(torch.ones(4, 32, 32, 3, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
