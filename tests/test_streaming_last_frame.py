import unittest

import torch

from custom_nodes.SECoursesAudioTools import (
    SEStreamingLastFrameRecorder,
    SEStreamingLastFrameReferenceImage,
    _STREAMING_LAST_FRAME_CACHE,
)


class FakeMetaBatch:
    unique_id = "7001"


def prompt(requeue):
    return {
        "7001": {
            "class_type": "VHS_BatchManager",
            "inputs": {"requeue": requeue},
        }
    }


class StreamingLastFrameTests(unittest.TestCase):
    def setUp(self):
        _STREAMING_LAST_FRAME_CACHE.clear()
        self.meta_batch = FakeMetaBatch()
        self.provider = SEStreamingLastFrameReferenceImage()
        self.recorder = SEStreamingLastFrameRecorder()

    def test_requeued_chunk_uses_recorded_last_frame(self):
        initial = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        frames = torch.stack(
            (
                torch.full((2, 2, 3), 0.25, dtype=torch.float32),
                torch.full((2, 2, 3), 0.75, dtype=torch.float32),
            )
        )

        first = self.provider.select(initial, True, 1.0, self.meta_batch, prompt(0))[0]
        self.assertTrue(torch.equal(first, initial))

        passthrough = self.recorder.record(frames, True, 1.0, self.meta_batch, prompt(0))[0]
        self.assertIs(passthrough, frames)

        next_reference = self.provider.select(initial, True, 1.0, self.meta_batch, prompt(1))[0]
        self.assertTrue(torch.equal(next_reference, frames[-1:]))

    def test_disabled_or_zero_chunk_uses_input_image(self):
        initial = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        frames = torch.ones((2, 2, 2, 3), dtype=torch.float32)
        self.recorder.record(frames, True, 1.0, self.meta_batch, prompt(0))

        disabled = self.provider.select(initial + 2, False, 1.0, self.meta_batch, prompt(1))[0]
        self.assertTrue(torch.equal(disabled, initial + 2))

        self.recorder.record(frames, True, 1.0, self.meta_batch, prompt(0))
        chunk_off = self.provider.select(initial + 3, True, 0.0, self.meta_batch, prompt(1))[0]
        self.assertTrue(torch.equal(chunk_off, initial + 3))


if __name__ == "__main__":
    unittest.main()
