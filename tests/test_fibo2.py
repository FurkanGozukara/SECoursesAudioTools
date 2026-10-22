"""Fibo 2.0 bridge: prompts and negatives reach the native encoder as Bria's pipeline encodes them, masked edits grey out
like training, and the official dynamic shift keeps the caption's text images at sigma 0.9. Run from the ComfyUI root."""
import base64
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("fibo2_nodes", Path(__file__).parents[1] / "fibo2_nodes.py")
fibo2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fibo2)

# The model card's text-to-image negative (NEGATIVE_T2I), which Bria's pipeline encodes as written
NEGATIVE_T2I = '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights","lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"","artistic_style":""}'


class Fibo2PromptTests(unittest.TestCase):
    def test_json_caption_is_kept_apart_from_image_labels(self):
        caption = json.dumps({"short_description": "A {red|blue} mug (glossy:1.3)", "edit_instruction": "Put @image2 on the table of @image1",
                              "text_render": [{"text": "SALE [50%]"}]}, indent=2)
        prepared = fibo2.prepare_prompt(caption, seed=5, image_count=2)
        self.assertEqual(prepared, caption.replace("@image2", "<image_2>").replace("@image1", "<image_1>"))
        self.assertEqual(json.loads(prepared)["short_description"], "A {red|blue} mug (glossy:1.3)")

    def test_plain_prompt_choices_follow_the_seed_and_weights_are_removed(self):
        first = fibo2.prepare_prompt("a {red|green|blue} car (shiny:1.2) at (10:30), 4:3", seed=7)
        self.assertEqual(first, fibo2.prepare_prompt("a {red|green|blue} car (shiny:1.2) at (10:30), 4:3", seed=7))
        self.assertRegex(first, r"^a (red|green|blue) car shiny at \(10:30\), 4:3$")
        self.assertEqual(fibo2.prepare_prompt(r"keep \{a|b\} and (x:5)"), r"keep \{a|b\} and (x:5)")

    def test_image_labels_with_and_without_the_edit_canvas(self):
        self.assertEqual(fibo2.prepare_prompt("dress @image1 in @IMG2", image_count=2), "dress <image_1> in <image_2>")
        self.assertEqual(fibo2.prepare_prompt("dress @init in @image1", image_count=1, has_init=True), "dress <image_1> in <image_2>")
        with self.assertRaisesRegex(ValueError, "no matching gallery image"):
            fibo2.prepare_prompt("@image2", image_count=1)
        with self.assertRaisesRegex(ValueError, "edit canvas"):
            fibo2.prepare_prompt("keep @init", image_count=1)

    def test_negative_is_encoded_as_written(self):
        import comfy.text_encoders.fibo2 as te

        expected = te.TEMPLATE.format(NEGATIVE_T2I)
        self.assertEqual(fibo2.raw_negative(NEGATIVE_T2I), expected)
        # A pretty-printed copy of the same JSON becomes the same compact negative, empty fields and key order kept.
        self.assertEqual(fibo2.raw_negative(json.dumps(json.loads(NEGATIVE_T2I), indent=4)), expected)
        self.assertEqual(fibo2.raw_negative(""), "")
        self.assertEqual(fibo2.raw_negative("blurry, low quality"), "blurry, low quality")

    def test_tokens_of_the_negative_match_bria_and_not_the_cleaned_caption(self):
        import comfy.text_encoders.fibo2 as te

        tokenizer = te.Fibo2Tokenizer()

        def ids(text):
            return [t[0] for batch in tokenizer.tokenize_with_weights(text)["qwen3vl_4b"] for t in batch]

        bria = tokenizer.qwen3vl_4b.tokenizer(te.TEMPLATE.format(NEGATIVE_T2I))["input_ids"]
        self.assertEqual(ids(fibo2.raw_negative(NEGATIVE_T2I)), bria)
        self.assertNotEqual(ids(NEGATIVE_T2I), bria)


class Fibo2ImageTests(unittest.TestCase):
    def test_grey_out_matches_bria_training(self):
        image = torch.rand(1, 8, 12, 3)
        mask = torch.zeros(1, 8, 12)
        mask[0, 2:5, 3:9] = 1.0
        mask[0, 0, 0] = 128 / 255
        mask[0, 0, 1] = 127 / 255
        out = fibo2.grey_out(image, mask)
        self.assertTrue(torch.all(out[0, 2:5, 3:9] == 128 / 255))
        self.assertTrue(torch.all(out[0, 0, 0] == 128 / 255))
        self.assertTrue(torch.equal(out[0, 0, 1], image[0, 0, 1]))
        self.assertTrue(torch.equal(out[0, 6:], image[0, 6:]))
        # A mask of another size is scaled to the image first (nearest).
        small = torch.zeros(1, 4, 6)
        small[0, :2] = 1.0
        out = fibo2.grey_out(image, small)
        self.assertTrue(torch.all(out[0, :4] == 128 / 255))
        self.assertTrue(torch.equal(out[0, 4:], image[0, 4:]))

    def test_swarm_inputs_decode_attachments(self):
        def encoded(mode, color, size=(64, 32)):
            buffer = io.BytesIO()
            fibo2.Image.new(mode, size, color).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        pack, init, mask = fibo2.SEFibo2SwarmInputs().load(
            "Edit @init", json.dumps([encoded("RGBA", (255, 0, 0, 0))]), encoded("RGB", (0, 0, 255)), encoded("L", 255))
        # Transparent attachments are composited on white.
        self.assertTrue(torch.all(pack["image_tensors"][0] == 1.0))
        self.assertEqual(init.shape, (1, 32, 64, 3))
        self.assertEqual(mask.shape, (1, 32, 64))
        self.assertEqual(fibo2.SEFibo2SwarmInputs().load("text", "[]", "", encoded("L", 255)), ({"prompt": "text", "image_tensors": []}, None, None))


class Fibo2ShiftTests(unittest.TestCase):
    def test_official_shift_matches_bria_and_comfy_at_one_megapixel(self):
        from comfy_extras.nodes_fibo2 import TEXT_IMAGES_START

        self.assertAlmostEqual(fibo2.official_shift(1024, 1024), 1.15)
        self.assertAlmostEqual(fibo2.official_shift(1424, 800), 0.5 + 0.65 / 3840 * (89 * 50 - 256))
        self.assertAlmostEqual(fibo2.text_images_start(1.15), TEXT_IMAGES_START)
        for mu in (0.8, 1.15, 2.0):
            percent = fibo2.text_images_start(mu)
            sigma = math.exp(mu) / (math.exp(mu) + (1 / (1 - percent) - 1))
            self.assertAlmostEqual(sigma, 0.9)

    def test_remap_moves_only_the_text_image_switch(self):
        from comfy_extras.nodes_fibo2 import TEXT_IMAGES_START

        cond = [[1, {"end_percent": TEXT_IMAGES_START - 1e-6}], [2, {"start_percent": TEXT_IMAGES_START}], [3, {"start_percent": 0.5}]]
        out = fibo2.remap_text_images(cond, 2.0)
        start = fibo2.text_images_start(2.0)
        self.assertAlmostEqual(out[0][1]["end_percent"], start - 1e-6)
        self.assertAlmostEqual(out[1][1]["start_percent"], start)
        self.assertEqual(out[2][1], {"start_percent": 0.5})
        self.assertEqual(cond[0][1]["end_percent"], TEXT_IMAGES_START - 1e-6)

    def test_sampling_node_patches_the_shift_from_the_latent(self):
        latent = {"samples": torch.zeros(1, 128, 96, 96)}
        calls = []

        class FakeModelSamplingFlux:
            def patch(self, model, max_shift, base_shift, width, height):
                calls.append((max_shift, base_shift, width, height))
                return ("patched",)

        with patch("comfy_extras.nodes_model_advanced.ModelSamplingFlux", FakeModelSamplingFlux):
            model, _ = fibo2.SEFibo2Sampling().apply("model", [], latent, fibo2.SHIFT_MODES[0])
            self.assertEqual((model, calls), ("patched", [(1.15, 0.5, 1536, 1536)]))
            self.assertEqual(fibo2.SEFibo2Sampling().apply("model", [], latent, fibo2.SHIFT_MODES[1]), ("model", []))


class Fibo2EncodeTests(unittest.TestCase):
    def run_encode(self, **kwargs):
        seen = {}

        def execute(clip, prompt, negative_prompt, vae=None, images=None, width=0, height=0, batch_size=1):
            seen.update(prompt=prompt, negative=negative_prompt, images=images, width=width, height=height, batch=batch_size)
            w, h = (width or 1024) // 16, (height or 1024) // 16
            return SimpleNamespace(result=("positive", "negative", {"samples": torch.zeros(batch_size, 128, h, w)}))

        args = dict(clip=None, vae=None, prompt="", negative_prompt="", images=[], size_mode=fibo2.SIZE_MODES[0],
                    width=1024, height=1024, batch_size=1)
        args.update(kwargs)
        with patch("comfy_extras.nodes_fibo2.TextEncodeFibo2.execute", execute):
            return fibo2.encode(**args), seen

    def test_text_to_image_uses_the_resolution(self):
        (_, _, latent, restore), seen = self.run_encode(prompt="a fox", width=1424, height=800, batch_size=2)
        self.assertEqual((seen["width"], seen["height"], seen["batch"], seen["images"]), (1424, 800, 2, {}))
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 50, 89))
        self.assertIsNone(restore)

    def test_edit_image_order_and_size_modes(self):
        from comfy_extras.nodes_fibo2 import trained_size

        canvas, gallery = torch.rand(1, 48, 64, 3), torch.rand(1, 32, 32, 3)
        (_, _, _, restore), seen = self.run_encode(prompt="put @image1 next to @init", images=[gallery], edit_image=canvas)
        self.assertEqual(seen["prompt"], "put <image_2> next to <image_1>")
        # Below 1 MP the auto mode uses Bria's trained size for the first image's aspect ratio.
        self.assertEqual((seen["width"], seen["height"]), trained_size(64 / 48))
        self.assertTrue(torch.equal(seen["images"]["image_1"], canvas))
        self.assertTrue(torch.equal(seen["images"]["image_2"], gallery))
        self.assertIsNone(restore)
        _, seen = self.run_encode(prompt="edit", images=[gallery], size_mode=fibo2.SIZE_MODES[1])
        self.assertEqual((seen["width"], seen["height"]), (0, 0))
        _, seen = self.run_encode(prompt="edit", images=[gallery], size_mode=fibo2.SIZE_MODES[2], width=832, height=1248)
        self.assertEqual((seen["width"], seen["height"]), (832, 1248))

    def test_auto_edit_size_keeps_one_to_four_megapixels(self):
        self.assertEqual(fibo2.edit_size(1728, 1152), (1728, 1152))
        self.assertEqual(fibo2.edit_size(1730, 1159), (1728, 1152))
        self.assertEqual(fibo2.edit_size(2048, 2048), (2048, 2048))
        width, height = fibo2.edit_size(5000, 3333)
        self.assertLessEqual(width * height, 2048 * 2048)
        self.assertEqual((width % 16, height % 16), (0, 0))
        self.assertAlmostEqual(width / height, 5000 / 3333, delta=0.02)
        _, seen = self.run_encode(prompt="edit", images=[torch.rand(1, 1152, 1728, 3)])
        self.assertEqual((seen["width"], seen["height"]), (1728, 1152))

    def test_masked_edit_greys_out_and_returns_the_restore_crop(self):
        canvas = torch.rand(1, 64, 64, 3)
        mask = torch.zeros(1, 64, 64)
        mask[0, 16:48, 16:48] = 1.0
        (_, _, latent, restore), seen = self.run_encode(prompt="replace the mug", edit_image=canvas, edit_mask=mask,
                                                         negative_prompt=NEGATIVE_T2I)
        self.assertTrue(torch.all(seen["images"]["image_1"][0, 16:48, 16:48] == 128 / 255))
        self.assertTrue(seen["negative"].startswith("<|im_start|>user\n{\"short_description\":\"\""))
        self.assertEqual(tuple(restore["canvas"].shape), (1, 1024, 1024, 3))
        self.assertEqual(tuple(restore["mask"].shape), (1024, 1024))
        with self.assertRaisesRegex(ValueError, "single image"):
            self.run_encode(prompt="x", images=[torch.rand(1, 8, 8, 3)], edit_image=canvas, edit_mask=mask)

    def test_restore_unmasked_pixels(self):
        images = torch.zeros(2, 4, 4, 3)
        restore = {"canvas": torch.ones(1, 4, 4, 3), "mask": torch.zeros(4, 4)}
        restore["mask"][:2] = 1.0
        out = fibo2.SEFibo2RestoreUnmasked().restore(images, restore)[0]
        self.assertTrue(torch.all(out[:, :2] == 0) and torch.all(out[:, 2:] == 1))
        self.assertIs(fibo2.SEFibo2RestoreUnmasked().restore(images, restore, enabled=False)[0], images)
        self.assertIs(fibo2.SEFibo2RestoreUnmasked().restore(images, None)[0], images)


if __name__ == "__main__":
    unittest.main()
