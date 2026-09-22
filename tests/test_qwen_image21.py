"""Guard gallery numbering, optional input, and exact unmasked compositing."""
import base64
import io
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("qwen21", Path(__file__).parents[1] / "qwen_image21_nodes.py")
qwen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen)


class QwenImage21Tests(unittest.TestCase):
    def test_swarm_transport_retains_alpha_and_white_edit_mask(self):
        def encoded(mode, color):
            buffer = io.BytesIO()
            qwen.Image.new(mode, (64, 32), color).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        red = encoded("RGBA", (255, 0, 0, 64))
        blue = encoded("RGB", (0, 0, 255))
        pack, init, mask = qwen.SEQwenImage21SwarmInputs().load(
            "Keep @init, use @image1 then @image2", json.dumps([red, blue]), blue, encoded("L", 128))
        self.assertEqual(pack["image_tensors"][0].shape, (1, 32, 64, 4))
        self.assertAlmostEqual(pack["image_tensors"][0][0, 0, 0, 3].item(), 64 / 255)
        self.assertEqual(pack["image_tensors"][1][0, 0, 0, 2].item(), 1)
        self.assertEqual(init.shape, (1, 32, 64, 3))
        self.assertAlmostEqual(mask[0, 0, 0].item(), 128 / 255)
        self.assertEqual(qwen.SEQwenImage21SwarmInputs().load("text", "[]", "", ""),
                         ({"prompt": "text", "image_tensors": []}, None, None))

    def test_swarm_tensors_use_native_encoder_in_order(self):
        init, red, blue = (torch.rand(1, 32, 64, channels) for channels in (3, 4, 3))
        refs = {"prompt": "@init then @image1 then @image2", "image_tensors": [red, blue]}
        args = dict(clip=None, vae=None, references=refs, mode=qwen.MODES[0], width=64, height=32,
                    reference_resolution=0, denoise=.37, transparent=False, negative_prompt="blurry", init_image=init)
        result = SimpleNamespace(result=([], [], {"samples": torch.zeros(1, 64, 2, 4)}))
        with patch("comfy_extras.nodes_qwen.TextEncodeQwenImage21.execute", return_value=result) as encode:
            output = qwen.SEQwenImage21Prepare().prepare(**args)
            self.assertEqual(encode.call_args.args[1:3], ("<image1> then <image2> then <image3>", "blurry"))
            self.assertEqual(encode.call_args.kwargs["resolution"], 0)
            images = encode.call_args.kwargs["images"]
            self.assertTrue(torch.equal(images["image_1"], init))
            self.assertIs(images["image_2"], red)
            self.assertIs(images["image_3"], blue)
            self.assertEqual(output[3], 1)
        args["references"] = {"image_tensors": [red] * 10}
        with self.assertRaisesRegex(ValueError, "10 images total"):
            qwen.SEQwenImage21Prepare().prepare(**args)

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
        self.assertEqual(qwen.SEQwenImage21Canvas().load_image(qwen.NO_IMAGE), (None, None, .85))
        self.assertTrue(qwen.SEQwenImage21Canvas.VALIDATE_INPUTS(qwen.NO_IMAGE))

    def test_canvas_denoise_preserves_native_image_and_mask(self):
        source, mask = torch.rand(1, 32, 32, 3), torch.rand(1, 32, 32)
        with patch.object(qwen.nodes.LoadImage, "load_image", return_value=(source, mask)):
            result = qwen.SEQwenImage21Canvas().load_image("painted.png", .37)
        self.assertIs(result[0], source)
        self.assertIs(result[1], mask)
        self.assertEqual(result[2], .37)
        self.assertEqual(qwen.SEQwenImage21Canvas().load_image(qwen.NO_IMAGE, .37), (None, None, .37))

    def test_prepare_uses_canvas_strength_only_for_img2img_and_inpaint(self):
        source = torch.rand(1, 32, 32, 3)
        mask = torch.zeros(1, 32, 32)
        mask[:, 8:24, 8:24] = .7
        encoded = torch.rand(1, 64, 2, 2)
        vae = SimpleNamespace(encode=lambda image: encoded)
        args = dict(clip=None, vae=vae, references={}, width=32, height=32,
                    reference_resolution=32, denoise=.37, transparent=False,
                    negative_prompt="", init_image=source, mask=mask)
        with patch("comfy_extras.nodes_qwen.TextEncodeQwenImage21.execute",
                   return_value=SimpleNamespace(result=([], [], {"samples": encoded}))):
            for mode in qwen.MODES:
                result = qwen.SEQwenImage21Prepare().prepare(mode=mode, **args)
                self.assertEqual(result[3], 1.0 if mode == qwen.MODES[0] else .37)
                self.assertEqual("noise_mask" in result[2], mode == qwen.MODES[2])
                if mode == qwen.MODES[2]:
                    self.assertTrue(torch.equal(result[2]["noise_mask"], (mask > 0).float()))
                    self.assertTrue(torch.equal(result[4]["mask"], mask))

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
