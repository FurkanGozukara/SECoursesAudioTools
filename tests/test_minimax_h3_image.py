"""MiniMax H3 Image gallery bridge: prompt labels, MiniMax's size rounding, the official enhancer message and the answer
parsing. CPU only, no model files. Run from the ComfyUI root:
python -m pytest -q custom_nodes/SECoursesAudioTools/tests/test_minimax_h3_image.py -p no:cacheprovider"""
import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from comfy_api.latest import io

PACKAGE = "secourses_audio_tools_under_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE] = package
h3i = importlib.import_module(f"{PACKAGE}.minimax_h3_image_nodes")


def image(width, height):
    return torch.rand(1, height, width, 3)


class PromptTests(unittest.TestCase):
    def test_labels_choices_and_weights(self):
        text = h3i.prepare_prompt("Put @image2 on @IMG1, a {red|red} (glossy:1.3) car", 7, 2)
        self.assertEqual(text, "Put <Picture 2> on <Picture 1>, a red glossy car")
        with self.assertRaisesRegex(ValueError, "no matching"):
            h3i.prepare_prompt("@image3", 0, 2)

    def test_choices_follow_the_seed(self):
        a = [h3i.prepare_prompt("{a|b|c|d|e|f}", seed, 0) for seed in range(20)]
        self.assertEqual(a, [h3i.prepare_prompt("{a|b|c|d|e|f}", seed, 0) for seed in range(20)])
        self.assertGreater(len(set(a)), 1)


class SizeTests(unittest.TestCase):
    def test_minimax_size_tables(self):
        # MiniMax's API tables: 4K area and 2K area
        cases = {(1, 1, 4096): (4096, 4096), (4, 3, 4096): (4736, 3552), (3, 4, 4096): (3552, 4736), (3, 2, 4096): (5024, 3360),
                 (16, 9, 4096): (5472, 3072), (21, 9, 4096): (6272, 2688), (3, 1, 4096): (7104, 2368),
                 (1, 1, 2048): (2048, 2048), (4, 3, 2048): (2368, 1760), (16, 9, 2048): (2720, 1536), (3, 1, 2048): (3552, 1184)}
        for (w, h, side), expected in cases.items():
            with self.subTest(ratio=f"{w}:{h}", side=side):
                self.assertEqual(h3i.size_for(w / h, side * side), expected)

    def test_ratio_parsing_and_clamp(self):
        self.assertAlmostEqual(h3i.parse_ratio("16:9"), 16 / 9)
        self.assertIsNone(h3i.parse_ratio("adaptive"))
        self.assertIsNone(h3i.parse_ratio("0:9"))
        self.assertEqual(h3i.size_for(10.0, 4096 * 4096), h3i.size_for(3.0, 4096 * 4096))
        self.assertEqual(h3i.reduced_ratio(6000, 4000), "3:2")


class AnswerTests(unittest.TestCase):
    def test_last_json_object_wins(self):
        answer = 'draft {"expanded_caption": "old"} final {"expanded_caption": "A \\"red\\" fox.", "aspect_ratio": "3:2"}'
        self.assertEqual(h3i.parse_answer(answer), {"expanded_caption": 'A "red" fox.', "aspect_ratio": "3:2"})

    def test_broken_json_falls_back_to_the_caption_field(self):
        self.assertEqual(h3i.parse_answer('{"expanded_caption": "A calm lake.", "aspect_ratio": '), {"expanded_caption": "A calm lake."})
        self.assertIsNone(h3i.parse_answer("no answer"))
        self.assertIsNone(h3i.parse_answer('{"expanded_caption": "  "}'))


class EnhancerMessageTests(unittest.TestCase):
    def test_official_prompt_files_unchanged(self):
        # sha256 of MiniMaxAI/MiniMax-H3-Image-PE prompt/manifest.json (LF line endings)
        expected = {"t2i_system_prompt.txt": "818ab3f652ec36b3b489b55c29f072f87e14af703cd11a4f3bff75f423d0beac",
                    "t2i_task_prompt.txt": "890213ad61e143bbd5b7906a802430296e82ca5c61ba16df717dcef5477bbcf4",
                    "edit_task_prompt.txt": "c5695dbfc07a9e65555cef67d92467c613371cc2039b96f057022246a30a57b2"}
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(h3i.read_pe(name).encode("utf-8")).hexdigest(), digest)
        self.assertIn("input_text: {}", h3i.read_pe("t2i_task_prompt.txt"))

    def test_edit_message_puts_each_label_before_its_image(self):
        text = h3i.edit_message('Make it "night"', [image(600, 400), image(512, 512)], "adaptive")
        self.assertTrue(text.startswith("<|im_start|>user\n" + h3i.read_pe("edit_task_prompt.txt")))
        self.assertTrue(text.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n"))
        self.assertEqual(text.count(h3i.IMAGE_PAD), 2)
        first = text.index("- <Picture 1> [aspect_ratio 3:2]")
        second = text.index("- <Picture 2> [aspect_ratio 1:1]")
        pads = [i for i in range(len(text)) if text.startswith(h3i.IMAGE_PAD, i)]
        self.assertLess(first, pads[0])
        self.assertLess(pads[0], second)
        self.assertLess(second, pads[1])
        self.assertIn('User Editing Instruction (JSON string):\n"Make it \\"night\\""', text)
        self.assertIn("User Aspect Ratio Selection:\nadaptive", text)


class FakeEnhancer(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_generate(**kwargs):
            self.calls.append(kwargs)
            ratio_key = "target_aspect_ratio" if kwargs.get("images") else "aspect_ratio"
            return io.NodeOutput('{"expanded_caption": "A detailed caption.", "%s": "16:9"}' % ratio_key, "thinking")

        patcher = patch("comfy_extras.nodes_textgen.TextGenerateV2.execute", side_effect=fake_generate)
        patcher.start()
        self.addCleanup(patcher.stop)


class NodeTests(FakeEnhancer):
    def test_lazy_enhancer_clip(self):
        node = h3i.SEH3ImagePromptEnhance()
        self.assertEqual(node.check_lazy_status({}, h3i.ENHANCE_MODES[0], 0), ["enhancer_clip"])
        self.assertEqual(node.check_lazy_status({}, h3i.ENHANCE_MODES[1], 0), ["enhancer_clip"])
        self.assertEqual(node.check_lazy_status({}, h3i.ENHANCE_MODES[2], 0), [])

    def test_fast_mode_turns_reasoning_off(self):
        h3i.SEH3ImagePromptEnhance().run({"prompt": "a fox"}, h3i.ENHANCE_MODES[1], 5, enhancer_clip=object())
        self.assertFalse(self.calls[0]["thinking"])
        refs = {"prompt": "Make @image1 snowy", "image_tensors": [image(64, 64)]}
        h3i.SEH3ImagePromptEnhance().run(refs, h3i.ENHANCE_MODES[1], 5, enhancer_clip=object())
        self.assertTrue(self.calls[1]["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_off_passes_the_prompt(self):
        pack, prompt = h3i.SEH3ImagePromptEnhance().run({"prompt": "A {fox|fox} (sleeping:1.2)"}, h3i.ENHANCE_MODES[2], 3)
        self.assertEqual(prompt, "A fox sleeping")
        self.assertIsNone(pack["h3_image_aspect"])
        self.assertEqual(self.calls, [])

    def test_text_to_image_uses_the_official_t2i_request(self):
        pack, prompt = h3i.SEH3ImagePromptEnhance().run({"prompt": "a fox"}, h3i.ENHANCE_MODES[0], 5, enhancer_clip=object())
        self.assertEqual((prompt, pack["h3_image_aspect"]), ("A detailed caption.", "16:9"))
        call = self.calls[0]
        self.assertTrue(call["prompt"].endswith("input_text: a fox\noutput_text:\n"))
        self.assertTrue(call["thinking"])
        self.assertEqual(call["system_prompt"], h3i.read_pe("t2i_system_prompt.txt").rstrip("\n"))
        self.assertEqual(call["sampling_mode"]["seed"], 5)
        self.assertEqual(call["max_length"], h3i.MAX_TOKENS)

    def test_edit_uses_the_gallery_images(self):
        refs = {"prompt": "Turn @image1 into winter", "image_tensors": [image(3000, 2000), image(640, 480)]}
        pack, prompt = h3i.SEH3ImagePromptEnhance().run(refs, h3i.ENHANCE_MODES[0], 1, enhancer_clip=object())
        call = self.calls[0]
        self.assertIn('"Turn <Picture 1> into winter"', call["prompt"])
        self.assertEqual(list(call["images"]), ["image1", "image2"])
        self.assertLessEqual(call["images"]["image1"].shape[1] * call["images"]["image1"].shape[2], 1_000_000 + 4000)
        self.assertEqual(pack["h3_image_aspect"], "16:9")

    def test_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "Write a prompt"):
            h3i.SEH3ImagePromptEnhance().run({"prompt": "  "}, h3i.ENHANCE_MODES[2], 0)

    def test_unparsable_answer_keeps_the_prompt(self):
        with patch("comfy_extras.nodes_textgen.TextGenerateV2.execute", return_value=io.NodeOutput("", "cut off")) as generate:
            pack, prompt = h3i.SEH3ImagePromptEnhance().run({"prompt": "a fox"}, h3i.ENHANCE_MODES[0], 0, enhancer_clip=object())
        self.assertEqual((prompt, pack["h3_image_aspect"]), ("a fox", None))
        self.assertEqual([c.kwargs["thinking"] for c in generate.call_args_list], [True, False])

    def test_reasoning_without_answer_falls_back_to_fast(self):
        answers = [io.NodeOutput("", "ran out of tokens"), io.NodeOutput('{"expanded_caption": "Fast caption.", "target_aspect_ratio": "3:2"}', "")]
        refs = {"prompt": "Make @image1 snowy", "image_tensors": [image(64, 64)]}
        with patch("comfy_extras.nodes_textgen.TextGenerateV2.execute", side_effect=answers) as generate:
            pack, prompt = h3i.SEH3ImagePromptEnhance().run(refs, h3i.ENHANCE_MODES[0], 0, enhancer_clip=object())
        self.assertEqual((prompt, pack["h3_image_aspect"]), ("Fast caption.", "3:2"))
        self.assertEqual([c.kwargs["thinking"] for c in generate.call_args_list], [True, False])


class SwarmNodeTests(FakeEnhancer):
    def refs(self, *images):
        return {"ref_items": [{"type": "image", "data": i} for i in images], "ref_blocks": [], "picture_label": "Picture %d: "}

    def test_lazy_enhancer_clip(self):
        node = h3i.SEH3ImageSwarmEnhance()
        self.assertEqual(node.check_lazy_status("p", h3i.ENHANCE_MODES[0], 0, 4096, 4096, h3i.SIZE_MODES[0]), ["enhancer_clip"])
        self.assertEqual(node.check_lazy_status("p", h3i.ENHANCE_MODES[2], 0, 4096, 4096, h3i.SIZE_MODES[0]), [])

    def test_text_to_image_takes_the_enhancer_ratio(self):
        out = h3i.SEH3ImageSwarmEnhance().run("a fox", h3i.ENHANCE_MODES[0], 5, 4096, 4096, h3i.SIZE_MODES[0], enhancer_clip=object())
        self.assertEqual(out, ("A detailed caption.", 5472, 3072))
        self.assertNotIn("images", self.calls[0])

    def test_edit_reads_swarm_references(self):
        out = h3i.SEH3ImageSwarmEnhance().run("Turn @image1 into winter", h3i.ENHANCE_MODES[0], 1, 4096, 4096, h3i.SIZE_MODES[1],
                                               enhancer_clip=object(), minimax_refs=self.refs(image(1536, 1024), image(1024, 1024)))
        self.assertEqual(out, ("A detailed caption.", 4096, 4096))
        self.assertIn('"Turn <Picture 1> into winter"', self.calls[0]["prompt"])
        self.assertEqual(list(self.calls[0]["images"]), ["image1", "image2"])

    def test_off_keeps_image_1_shape(self):
        out = h3i.SEH3ImageSwarmEnhance().run("Make @image1 (snowy:1.2)", h3i.ENHANCE_MODES[2], 0, 4096, 4096, h3i.SIZE_MODES[0],
                                               minimax_refs=self.refs(image(1536, 1024)))
        self.assertEqual(out, ("Make <Picture 1> snowy", *h3i.size_for(1.5, 4096 * 4096)))
        self.assertEqual(self.calls, [])
        with self.assertRaisesRegex(ValueError, "no matching"):
            h3i.SEH3ImageSwarmEnhance().run("@image2", h3i.ENHANCE_MODES[2], 0, 4096, 4096, h3i.SIZE_MODES[0], minimax_refs=self.refs(image(64, 64)))


class TextToImageOnlyTests(unittest.TestCase):
    def test_images_stop_a_text_to_image_only_run(self):
        node = h3i.SEH3ImagePromptEnhance()
        refs = {"prompt": "p", "image_tensors": [image(64, 64)]}
        with self.assertRaisesRegex(ValueError, "text to image only"):
            node.run(refs, h3i.ENHANCE_MODES[2], 0, text_to_image_only=True)

    def test_default_and_empty_gallery_run_unchanged(self):
        node = h3i.SEH3ImagePromptEnhance()
        refs = {"prompt": "p", "image_tensors": [image(64, 64)]}
        self.assertEqual(node.run(refs, h3i.ENHANCE_MODES[2], 0)[1], "p")
        self.assertEqual(node.run({"prompt": "p"}, h3i.ENHANCE_MODES[2], 0, text_to_image_only=True)[1], "p")
        self.assertEqual(node.check_lazy_status({"prompt": "p"}, h3i.ENHANCE_MODES[2], 0, text_to_image_only=True), [])


class EncodeTests(unittest.TestCase):
    def run_encode(self, refs, width, height, mode, batch=1):
        seen = {}

        def fake_conditioning(clip, prompt, width, height, vae=None, images=None):
            seen.update(prompt=prompt, width=width, height=height, images=list(images or {}))
            return io.NodeOutput([[torch.zeros(1), {}]], {"samples": torch.zeros(1, 24, 1, height // 16, width // 16)})

        with patch("comfy_extras.nodes_minimax_h3.MiniMaxH3ImageConditioning.execute", side_effect=fake_conditioning):
            _, latent = h3i.SEH3ImageGalleryEncode().encode(None, None, refs, width, height, mode, batch)
        return seen, latent

    def test_auto_uses_the_enhancer_ratio_at_the_megapixels(self):
        seen, latent = self.run_encode({"prompt": "p", "h3_image_aspect": "16:9"}, 4096, 4096, h3i.SIZE_MODES[0])
        self.assertEqual((seen["width"], seen["height"]), (5472, 3072))
        self.assertEqual(tuple(latent["samples"].shape), (1, 24, 1, 192, 342))

    def test_auto_edit_without_ratio_keeps_image_1_shape(self):
        seen, _ = self.run_encode({"prompt": "p", "image_tensors": [image(600, 400)], "h3_image_aspect": None}, 2048, 2048, h3i.SIZE_MODES[0])
        self.assertEqual((seen["width"], seen["height"]), h3i.size_for(1.5, 2048 * 2048))
        self.assertEqual(seen["images"], ["image_1"])

    def test_exact_and_batch(self):
        seen, latent = self.run_encode({"prompt": "p", "h3_image_aspect": "16:9"}, 3072, 2048, h3i.SIZE_MODES[1], batch=2)
        self.assertEqual((seen["width"], seen["height"]), (3072, 2048))
        self.assertEqual(latent["samples"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
