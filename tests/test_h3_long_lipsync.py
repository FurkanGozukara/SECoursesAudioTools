"""Grid and conditioning invariants without starting ComfyUI or allocating a GPU."""
import ast
from contextlib import nullcontext
from fractions import Fraction
import math
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest
import wave

import torch


def load_functions():
    root = Path(__file__).resolve().parents[1]
    scope = {"math": math, "FPS": 24, "Fraction": Fraction, "AUDIO_LATENT_RATE": 40,
             "VIDEO_PREFIX_LATENTS": 2, "GROUP_LATENTS": 5, "GROUP_FRAMES": 17}
    functions = {"latents_for_frames", "round_half_even", "audio_boundary", "continuation_plan", "window_conditioning"}
    for filename in ("h3_streaming_nodes.py", "h3_long_lipsync.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        exec(compile(tree, filename, "exec"), scope)
    return scope


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_functions()

    def test_grid_covers_audio_without_accumulated_rounding(self):
        for window in (141, 192, 243, 294, 345):
            for overlap in (39, 90):
                for seconds in (.1, 3.01, 10, 15, 30, 45, 60, 301.123):
                    spans = self.functions["continuation_plan"](seconds, window, overlap)
                    self.assertGreaterEqual(spans[-1]["end_frame"], math.ceil(seconds * 24))
                    self.assertLess(spans[-1]["end_frame"] - math.ceil(seconds * 24), window)
                    for i, span in enumerate(spans):
                        s, e = span["start_frame"], span["end_frame"]
                        self.assertEqual(s % 51, 0)
                        self.assertEqual((e - s - 5) % 17, 0)
                        self.assertLessEqual(e - s, window)
                        self.assertEqual(span["audio_start"], s * 5 // 3)
                        self.assertEqual(span["audio_end"], e * 5 // 3)
                        if i:
                            prev = spans[i - 1]
                            self.assertEqual(prev["end_frame"] - s, overlap)
                            self.assertEqual(prev["end_latent"] - span["start_latent"], span["held_latents"])
                            self.assertEqual(prev["audio_end"] - span["audio_start"], overlap * 5 // 3)

    def test_invalid_duration_or_grid_is_rejected(self):
        for args in ((0,), (-1,), (float("nan"),), (float("inf"),), (10, 240), (10, 243, 40)):
            with self.assertRaises(ValueError):
                self.functions["continuation_plan"](*args)

    def test_continuation_keeps_identity_references_and_replaces_audio_guide(self):
        metadata = {"minimax_keyframes": [{"latent": "first image"}, {"audio_latent": "old"}],
                    "minimax_refs": ["identity"], "unrelated": 7}
        positive = [["embedding", metadata]]
        first = self.functions["window_conditioning"](positive, "new", False)
        following = self.functions["window_conditioning"](positive, "later", True)
        self.assertEqual(len(first[0][1]["minimax_keyframes"]), 2)
        self.assertEqual(following[0][1]["minimax_keyframes"], [{"resolved_frame_index": 0, "audio_latent": "later"}])
        self.assertEqual(following[0][1]["minimax_refs"], ["identity"])
        self.assertEqual(metadata["minimax_keyframes"][1]["audio_latent"], "old")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_frame_limit_preserves_complete_cfr_packet_grid(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "h3_streaming_nodes.py").read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                     and n.name in ("FFmpegWriter", "mux_audio")]
        scope = {"FPS": 24, "shutil": shutil, "subprocess": subprocess}
        exec(compile(tree, "h3_streaming_nodes.py", "exec"), scope)
        # Keep the small encoder artifacts available for failed-run diagnosis.
        with nullcontext(tempfile.mkdtemp(prefix="se_h3_cfr_")) as folder:
            folder = Path(folder)
            for frames in (25, 240, 720):
                video, audio, output = [str(folder / f"{frames}{suffix}") for suffix in (".video.mp4", ".wav", ".mp4")]
                writer = scope["FFmpegWriter"](video, 64, 64, frame_limit=frames)
                for start in range(0, frames + 51, 17):
                    batch = torch.empty((17, 64, 64, 3), dtype=torch.uint8)
                    for i in range(17): batch[i].fill_((start + i) % 256)
                    writer.write(batch)
                writer.close()
                self.assertEqual(writer.frames, frames)
                with wave.open(audio, "wb") as wav:
                    wav.setparams((2, 2, 32000, 0, "NONE", "not compressed"))
                    wav.writeframes(bytes(math.ceil(frames / 24 * 32000) * 4))
                scope["mux_audio"](video, audio, output, frames / 24, frames=frames)
                probe = json.loads(subprocess.check_output([
                    "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "packet=pts,duration", "-of", "json", output],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
                packets = sorted(probe["packets"], key=lambda p: p["pts"])
                self.assertEqual(len(packets), frames)
                step = packets[0]["duration"]
                self.assertEqual([p["pts"] for p in packets], [i * step for i in range(frames)])


if __name__ == "__main__":
    unittest.main()
