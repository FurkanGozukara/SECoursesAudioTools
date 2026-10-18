"""MOSS-Image prompt bridge: JSON layouts stay exact, drafts compile like the official tool, prompt syntax is safe."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("moss_image_nodes", Path(__file__).parents[1] / "moss_image_nodes.py")
moss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moss)

# Draft from OpenMOSS skills/moss-image-prompt-enhancer/references/layout.md; the document gives the cup's compiled box.
OFFICIAL_DRAFT = {
    "width": 1536, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
    "summary": {"meta": ["product campaign", "studio photography"], "canvas": "single", "caption": "A warm studio campaign for a handmade ceramic cup."},
    "items": [
        {"label": "object", "caption": "An ivory ceramic cup.", "bbox": [176, 224, 800, 864]},
        {"label": "text", "caption": "Large dark-green Chinese title.", "text": "把日常，过成诗。", "bbox": [928, 272, 1408, 448]},
        {"label": "text", "caption": "Smaller supporting copy.", "text": "一杯温暖\n一刻从容", "bbox": [928, 512, 1376, 656]},
    ],
}

OFFICIAL_COMPILED = '{"summary":{"meta":["product campaign","studio photography"],"canvas":"single","caption":"A warm studio campaign for a handmade ceramic cup."},"items":[{"label":"object","caption":"An ivory ceramic cup.","bbox":[-288,-592,352,32]},{"label":"text","caption":"Large dark-green Chinese title.","text":"把日常，过成诗。","bbox":[-240,160,-64,640]},{"label":"text","caption":"Smaller supporting copy.","text":"一杯温暖\\n一刻从容","bbox":[0,160,144,608]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'

class FakeClip:
    def __init__(self):
        self.seen = []

    def tokenize(self, text):
        self.seen.append(text)
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [[tokens["text"], {"pooled_output": None}]]


class MossImagePromptTests(unittest.TestCase):
    def prepare(self, text, **kwargs):
        return moss.SEMossImageGalleryPrompt().prepare({"prompt": text}, **kwargs)

    def test_official_draft_compiles_like_the_reference_tool(self):
        prompt, is_json = self.prepare(json.dumps(OFFICIAL_DRAFT, ensure_ascii=False, indent=2))
        layout = json.loads(prompt)
        self.assertTrue(is_json)
        self.assertEqual(layout["items"][0]["bbox"], [-288, -592, 352, 32])
        self.assertEqual(layout["bbox_coord_space"], "abs_centralized")
        self.assertEqual(layout["bbox_format"], "yxyx")
        self.assertEqual(layout["items"][2]["text"], "一杯温暖\n一刻从容")
        # Byte-identical to OpenMOSS compile_prompt.py on this draft (checked 2026-10-17).
        self.assertEqual(prompt, OFFICIAL_COMPILED)

    def test_draft_is_scaled_to_a_different_output_size(self):
        prompt, _ = self.prepare(json.dumps(OFFICIAL_DRAFT), width=768, height=512)
        self.assertEqual(json.loads(prompt)["items"][0]["bbox"], [-144, -296, 176, 16])

    def test_compiled_layout_and_structured_json_pass_through(self):
        layout = {"summary": {"caption": "Poster {red|blue} (red:1.5) [a:b:0.5]"}, "items": [
            {"label": "text", "caption": "Title", "text": "SALE [50%]", "bbox": [-400, -300, -300, 300]}],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
        prompt, is_json = self.prepare(json.dumps(layout, indent=1), seed=3)
        self.assertTrue(is_json)
        self.assertEqual(json.loads(prompt), layout)
        structured = {"scene_type": "comic", "gen_brief": "Panels {a|b}"}
        self.assertEqual(json.loads(self.prepare(json.dumps(structured))[0]), structured)

    def test_caption_only_draft_becomes_plain_text(self):
        self.assertEqual(self.prepare(json.dumps({"caption": "A red cup"})), ("A red cup", False))

    def test_choices_follow_the_seed_and_weights_are_removed(self):
        first = self.prepare("a {red|green|blue} cup", seed=7)[0]
        self.assertEqual(first, self.prepare("a {red|green|blue} cup", seed=7)[0])
        self.assertIn(first, ("a red cup", "a green cup", "a blue cup"))
        self.assertEqual(self.prepare("a (glossy red teapot:1.05) and ((blue: 1.2) cup:0.9)")[0],
                         "a glossy red teapot and blue cup")
        # Times, ratios, room numbers and large numbers are not weights.
        for keep in ("open (9:30)", "ratio (16:9)", "(Room:101)", "plain (parentheses) stay"):
            self.assertEqual(self.prepare(keep)[0], keep)

    def test_gallery_media_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "text-to-image"):
            moss.SEMossImageGalleryPrompt().prepare({"prompt": "x", "images": [{}]})

    def test_invalid_draft_reports_the_item(self):
        draft = dict(OFFICIAL_DRAFT, items=[{"label": "object", "caption": "x", "bbox": [0, 0, 2000, 10]}])
        with self.assertRaisesRegex(ValueError, r"items\[0\]\.bbox"):
            self.prepare(json.dumps(draft))


class MossImageEncodeTests(unittest.TestCase):
    def test_json_and_plain_text_encode_verbatim(self):
        clip = FakeClip()
        text = json.dumps({"summary": {"caption": "[a:b:0.5]"}, "items": []})
        self.assertEqual(moss.SEMossImageTextEncode().encode(clip, text, 50)[0][0][0], text)
        self.assertEqual(moss.SEMossImageTextEncode().encode(clip, 'A sign reading "[OPEN]" and [note: hi]', 50)[0][0][0],
                         'A sign reading "[OPEN]" and [note: hi]')

    def test_schedules_use_swarm_with_other_brackets_escaped(self):
        calls = []

        class Swarm:
            def encode(self, clip, steps, prompt, *size):
                calls.append((steps, prompt))
                return ("swarm",)

        with patch.dict(sys.modules, {"nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={"SwarmClipTextEncodeAdvanced": Swarm})}):
            out = moss.SEMossImageTextEncode().encode(FakeClip(), "a [red:blue:0.5] cup, sign [OPEN], [cat|dog]", 30)
        self.assertEqual(out, ("swarm",))
        self.assertEqual(calls, [(30, "a [red:blue:0.5] cup, sign \\[OPEN\\], [cat|dog]")])

    def test_schedule_detection(self):
        for text, expected in (("[a:b:0.5]", True), ("[a:10]", True), ("[x|y]", True), ("a <break> b", True),
                               ("[note: hi]", False), ("[OPEN]", False), ("[a:b]", False)):
            self.assertEqual(moss.schedule_prompt(text)[0], expected, text)


class MossImageDeBandTests(unittest.TestCase):
    def model(self, path):
        return SimpleNamespace(cached_patcher_init=(object, (path, {})))

    def run_base(self, path):
        model = self.model(path)
        with patch.object(moss, "model_variant", return_value="Base"), patch("builtins.print") as printed:
            out = moss.SEMossImageSFTDeBand().apply(model, True, "None", 1.0)
        self.assertIs(out[0], model)
        return " ".join(str(call.args[0]) for call in printed.call_args_list)

    def test_file_name_comes_from_the_loader_call(self):
        self.assertEqual(moss.model_file_name(self.model("G:/models/diffusion_models/moss_image_sft_int8_convrot.safetensors")),
                         "moss_image_sft_int8_convrot.safetensors")
        self.assertIsNone(moss.model_file_name(SimpleNamespace()))

    def test_base_weights_under_an_sft_name_are_reported(self):
        messages = self.run_base("diffusion_models/moss_image_sft_int8_convrot.safetensors")
        self.assertIn("holds the Base weights, not SFT", messages)
        self.assertIn("SFT de-band skipped", messages)

    def test_real_base_file_has_no_warning(self):
        messages = self.run_base("diffusion_models/moss_image_base_int8_convrot.safetensors")
        self.assertNotIn("WARNING", messages)
        self.assertIn("SFT de-band skipped", messages)


if __name__ == "__main__":
    unittest.main()
