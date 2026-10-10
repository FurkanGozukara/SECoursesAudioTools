from unittest import TestCase
from unittest.mock import patch

import torch

from custom_nodes.SECoursesAudioTools import h3_long_lipsync as ll


class MouthPassWriterTests(TestCase):
    def test_restores_a_copy_before_encoding(self):
        written = []

        class Writer:
            closed = False

            def write(self, frames):
                written.append(frames.clone())

            def close(self):
                self.closed = True

        class Mouth:
            def process(self, frames):
                frames[:, 0, 0] = 7
                return frames

        decoded = torch.zeros(5, 2, 3, 3, dtype=torch.uint8)
        writer = Writer()
        wrapped = ll.MouthPassWriter(writer, Mouth())
        wrapped.write(decoded)
        self.assertTrue(torch.equal(decoded, torch.zeros_like(decoded)))
        self.assertEqual(int(written[0][:, 0, 0].min()), 7)
        wrapped.close()
        self.assertTrue(writer.closed)

    def test_mouth_inputs_are_optional_and_off_by_default(self):
        optional = ll.SEH3LongLipSync.INPUT_TYPES()["optional"]
        self.assertEqual(set(optional), {"mouth_pass", "mouth_fidelity", "mouth_blend", "mouth_model", "mouth_detector"})
        self.assertFalse(optional["mouth_pass"][1]["default"])

    def test_missing_mouth_model_fails_before_sampling(self):
        with patch("custom_nodes.SECoursesAudioTools.codeformer_mouth.model_path", side_effect=FileNotFoundError("codeformer.pth")):
            with self.assertRaises(FileNotFoundError):
                ll.SEH3LongLipSync().generate(None, None, None, None, None, None, 0, 4, mouth_pass=True)
