"""Guard gallery numbering, optional input, and exact unmasked compositing."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("qwen21", Path(__file__).parents[1] / "qwen_image21_nodes.py")
qwen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen)


class QwenImage21Tests(unittest.TestCase):
    def test_init_does_not_renumber_gallery_tokens(self):
        self.assertEqual(qwen.translate_prompt("Keep @init, add @image1 and @IMAGE2.", 2, True),
                         "Keep <image1>, add <image2> and <image3>.")
        self.assertEqual(qwen.translate_prompt("Keep @image1.", 1, False), "Keep <image1>.")

    def test_stale_reference_fails_instead_of_editing_wrong_image(self):
        with self.assertRaisesRegex(ValueError, "no matching"):
            qwen.translate_prompt("Use @image2", 1, False)
        with self.assertRaisesRegex(ValueError, "needs an image"):
            qwen.translate_prompt("Use @init", 1, False)

    def test_optional_canvas_requires_no_placeholder_file(self):
        self.assertEqual(qwen.SEQwenImage21Canvas().load_image(qwen.NO_IMAGE), (None, None))
        self.assertTrue(qwen.SEQwenImage21Canvas.VALIDATE_INPUTS(qwen.NO_IMAGE))

    def test_inpaint_preserves_rgb_and_alpha_outside_mask(self):
        source = torch.rand(1, 32, 64, 3)
        generated = torch.rand(1, 32, 64, 4)
        mask = torch.zeros(1, 32, 64)
        mask[:, 8:24, 16:48] = 1
        mask[:, 7, 16:48] = 0.5
        result, = qwen.SEQwenImage21Finish().finish(generated, {"canvas": source, "mask": mask})
        self.assertTrue(torch.equal(result[..., :3][mask == 0], source[mask == 0]))
        self.assertTrue(torch.all(result[..., 3][mask == 0] == 1))
        self.assertTrue(torch.equal(result[mask == 1], generated[mask == 1]))
        self.assertTrue(torch.allclose(result[..., :3][mask == .5],
                                       (source[mask == .5] + generated[..., :3][mask == .5]) / 2))

    def test_rgba_generation_passes_through_without_flattening(self):
        rgba = torch.rand(1, 32, 32, 4)
        self.assertIs(qwen.SEQwenImage21Finish().finish(rgba, {})[0], rgba)

    def test_modes_and_reference_limit_fail_before_encoding(self):
        args = dict(clip=None, vae=None, references={"images": []}, mode=qwen.MODES[1],
                    width=1024, height=1024, reference_resolution=1024, denoise=.85,
                    transparent=False, negative_prompt="")
        prepare = qwen.SEQwenImage21Prepare().prepare
        with self.assertRaisesRegex(ValueError, "requires an image"):
            prepare(**args)
        args.update(mode=qwen.MODES[0], references={"images": [{}] * 10}, init_image=torch.zeros(1,32,32,3))
        with self.assertRaisesRegex(ValueError, "10 images total"):
            prepare(**args)
        args.update(references={}, mode=qwen.MODES[2], mask=torch.zeros(1,32,32))
        with self.assertRaisesRegex(ValueError, "painted mask"):
            prepare(**args)


if __name__ == "__main__":
    unittest.main()
