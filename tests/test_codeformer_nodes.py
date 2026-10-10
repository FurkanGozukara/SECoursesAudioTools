from fractions import Fraction
from unittest import TestCase
from unittest.mock import patch

import torch
from comfy_api.latest import InputImpl, Types
from custom_nodes.SECoursesAudioTools import codeformer_nodes as cn


class CodeFormerVideoTests(TestCase):
    def test_disabled_does_not_read_video_or_load_models(self):
        video = object()
        with patch.object(cn, "CodeFormerMouth", side_effect=AssertionError("must not load")):
            self.assertIs(cn.SECodeFormerMouthVideo().restore(video, enabled=False)[0], video)
            self.assertIs(cn.SECodeFormerMouthVideo().restore(video, mouth_blend=0)[0], video)

    def test_finishing_preserves_source_audio_timing_alpha_and_untouched_pixels(self):
        images = torch.full((19, 4, 5, 3), .123456)
        before = images.clone()
        audio = {"waveform": torch.arange(99).reshape(1, 1, -1).float(), "sample_rate": 24000}
        alpha = torch.full((19, 4, 5), .5)
        components = Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(24000, 1001),
                                           alpha=alpha, metadata={"test": "unchanged"})
        video = InputImpl.VideoFromComponents(components)
        counts = []

        class Mouth:
            report = {"test": True}
            def process(self, frames):
                counts.append(len(frames))
                frames[:, 1, 2] = 204
                return frames

        with (patch.object(cn, "CodeFormerMouth", return_value=Mouth()) as factory,
              patch.object(cn.folder_paths, "get_save_image_path", return_value=("unused", "mouth", 1, "", "mouth")),
              patch.object(InputImpl.VideoFromComponents, "save_to") as save):
            result = cn.SECodeFormerMouthVideo().restore(video)
            output = result['result'][0].get_components()
        save.assert_called_once()
        self.assertEqual(result['ui']['images'][0]['filename'], 'mouth_00001_.mp4')
        factory.assert_called_once_with("codeformer.pth", "models/buffalo_l/det_10g.onnx", .9, .7)
        self.assertEqual(counts, [16, 3])
        self.assertIs(output.audio, audio)
        self.assertIs(output.alpha, alpha)
        self.assertEqual(output.frame_rate, components.frame_rate)
        self.assertEqual(output.metadata, components.metadata)
        self.assertTrue(torch.equal(images, before))
        expected = before.clone()
        expected[:, 1, 2] = .8
        self.assertTrue(torch.equal(output.images, expected))
