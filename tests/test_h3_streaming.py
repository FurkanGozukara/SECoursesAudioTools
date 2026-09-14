"""Geometry, schedule and cache-retention checks for the MiniMax H3 streaming sampler.

Run from the ComfyUI directory:  python -m unittest custom_nodes.SECoursesAudioTools.tests.test_h3_streaming
"""
import unittest

import torch

from custom_nodes.SECoursesAudioTools import h3_streaming_nodes as hs


class ChunkPlanTests(unittest.TestCase):
    def test_taomate_five_second_plan(self):
        plan = hs.build_chunk_plan(37, (2, 2, 2, 1))
        frames = [(c["frame_start"], c["frame_stop"]) for c in plan]
        audio = [(c["aud_start"], c["aud_stop"]) for c in plan]
        self.assertEqual(frames, [(0, 39), (39, 73), (73, 107), (107, 124)])
        self.assertEqual(audio, [(0, 65), (65, 122), (122, 178), (178, 207)])
        self.assertEqual([c["lat_stop"] - c["lat_start"] for c in plan], [12, 10, 10, 5])

    def test_long_stream_covers_every_frame_and_audio_latent(self):
        for seconds in (14.454422, 120.0, 300.0, 3.0):
            frames = hs.align_frame_count(round(seconds * hs.FPS))
            latents = hs.latents_for_frames(frames)
            audio_t = round(frames / hs.FPS * hs.AUDIO_LATENT_RATE)
            plan = hs.build_chunk_plan(latents, (2, 2, 2, 1))
            self.assertEqual(plan[0]["lat_start"], 0)
            self.assertEqual(plan[-1]["lat_stop"], latents)
            self.assertEqual(plan[-1]["frame_stop"], frames)
            self.assertEqual(plan[-1]["aud_stop"], audio_t)
            for previous, current in zip(plan, plan[1:]):
                self.assertEqual(previous["lat_stop"], current["lat_start"])
                self.assertEqual(previous["aud_stop"], current["aud_start"])
            requests = {c["request"] for c in plan}
            self.assertEqual(requests, set(range(plan[-1]["request"] + 1)))

    def test_video_time_positions_match_comfy_grid(self):
        if hs.h3 is None:
            self.skipTest("ComfyUI without MiniMax H3")
        times = [hs.FRAME_RESCALE * hs.frames_before(i) for i in range(37)]
        grid = hs.h3._video_t_grid(37, 0.0).tolist()
        for a, b in zip(times, grid):
            self.assertAlmostEqual(a, b, places=9)

    def test_parse_groups(self):
        self.assertEqual(hs.parse_groups("2,2,2,1"), (2, 2, 2, 1))
        self.assertEqual(hs.parse_groups(" 1; 1 "), (1, 1))
        with self.assertRaises(ValueError):
            hs.parse_groups("0")


class ScheduleTests(unittest.TestCase):
    def test_taomate_three_step_sigmas(self):
        video = hs.select_sigmas(3, 12.0)
        audio = hs.select_sigmas(3, 3.0)
        self.assertEqual(len(video), 4)
        self.assertAlmostEqual(video[0], 1.0)
        self.assertAlmostEqual(video[1], 0.9612, places=3)
        self.assertAlmostEqual(video[2], 0.8533, places=3)
        self.assertEqual(video[3], 0.0)
        self.assertAlmostEqual(audio[1], 0.8609, places=3)
        self.assertAlmostEqual(audio[2], 0.5926, places=3)

    def test_other_step_counts_span_the_grid(self):
        for steps in (1, 2, 4, 6):
            indices = hs.state_indices(steps)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], 49)
            self.assertEqual(len(indices), steps + 1)


class KVCacheTests(unittest.TestCase):
    def _fill(self, cache, layers, audio_rows, video_rows):
        cache.begin_commit()
        for layer in range(layers):
            rows = audio_rows + video_rows
            k = torch.randn(rows, 2, 8)
            v = torch.randn(rows, 2, 8)
            cache.stage(layer, k, v)
        cache.commit(audio_rows, video_rows)

    def test_retention_keeps_video_sink_and_recent_chunks(self):
        cache = hs.KVCache(2, "bf16", "cpu", torch.device("cpu"))
        for _ in range(4):
            self._fill(cache, 2, audio_rows=4, video_rows=10)
            cache.retain(sink_rows=6, recent_chunks=2)
        self.assertEqual(len(cache.entries), 3)
        self.assertTrue(cache.entries[0]["is_sink"])
        self.assertEqual(cache.entries[0]["audio_rows"], 0)
        self.assertEqual(cache.entries[0]["video_rows"], 6)
        self.assertEqual(cache.rows, 6 + 2 * 14)
        key, value = cache.history(0)
        self.assertEqual(tuple(key.shape), (6 + 28, 2, 8))
        removed = cache.drop_audio()
        self.assertEqual(removed, 8)
        self.assertEqual(cache.rows, 6 + 20)

    def test_fp8_roundtrip_is_close(self):
        cache = hs.KVCache(1, "fp8_e4m3", "cpu", torch.device("cpu"))
        cache.begin_commit()
        k = torch.randn(16, 2, 8) * 3.0
        v = torch.randn(16, 2, 8)
        cache.stage(0, k, v)
        cache.commit(0, 16)
        key, value = cache.history(0)
        self.assertLess((key.float() - k).abs().max().item() / k.abs().max().item(), 0.1)
        self.assertLess((value.float() - v).abs().max().item() / v.abs().max().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
