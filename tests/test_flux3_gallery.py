"""Reference order, prompt aliases and scheduled conditioning must survive the bridge."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

spec = importlib.util.spec_from_file_location("flux3_gallery", Path(__file__).parents[1] / "flux3_nodes.py")
flux3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flux3)


class Flux3GalleryTests(unittest.TestCase):
    def test_aliases_preserve_advanced_prompting(self):
        refs = {"images": [{}, {}], "prompt": "@image1 with @IMG2 (red:1.1) [red:blue:0.5] <break> <obj_1>"}
        self.assertEqual(flux3.SEFlux3GalleryPrompt().prepare(refs)[0],
                         "<ref_image_0> with <ref_image_1> (red:1.1) [red:blue:0.5] <break> <obj_1>")
        refs['prompt'] = '@image3'
        with self.assertRaisesRegex(ValueError, 'no matching'):
            flux3.SEFlux3GalleryPrompt().prepare(refs)

    def test_reference_index_boundaries_and_native_tokens(self):
        node = flux3.SEFlux3GalleryPrompt()
        refs = {'images': [{}] * 10}
        for alias in ('image', 'IMG', 'picture', 'PIC'):
            with self.subTest(alias=alias):
                refs['prompt'] = f'@{alias}1 @{alias}2 @{alias}10 <ref_image_0> <ref_image_9> <obj_1>'
                self.assertEqual(node.prepare(refs)[0],
                    '<ref_image_0> <ref_image_1> <ref_image_9> <ref_image_0> <ref_image_9> <obj_1>')
        for invalid in (0, 11):
            refs['prompt'] = f'@image{invalid}'
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'no matching'):
                node.prepare(refs)

    def test_empty_gallery_is_identity(self):
        conditioning = [[torch.zeros(1), {'start_percent': .5}]]
        result = flux3.SEFlux3GalleryReferences().apply(conditioning, {}, None, 1)
        self.assertIs(result[0], conditioning)

    def test_prompt_choices_are_seeded_and_keep_schedules(self):
        pack = {'prompt': '{red|{blue|green}} bowl [red:blue:0.5] <break>'}
        node = flux3.SEFlux3GalleryPrompt()
        first = node.prepare(pack, 123)[0]
        self.assertEqual(first, node.prepare(pack, 123)[0])
        self.assertNotIn('{', first)
        self.assertIn('[red:blue:0.5] <break>', first)
        self.assertEqual(node.prepare({'prompt': r'\{red|blue}'}, 123)[0], r'\{red|blue}')

    def test_invalid_media_and_reference_count(self):
        for refs in ({'videos': [{}]}, {'audios': [{}]}, {'images': [{}] * 11}):
            with self.assertRaises(ValueError):
                flux3.SEFlux3GalleryPrompt().prepare(refs)

    def test_order_aspect_alpha_and_every_schedule_entry(self):
        with TemporaryDirectory() as folder:
            a, b = [Path(folder) / name for name in ('wide.png', 'tall.png')]
            Image.new('RGB', (128, 64), 'red').save(a)
            Image.new('RGBA', (64, 128), (0, 0, 255, 0)).save(b)
            refs = {'images': [a, b]}
            original = [[torch.zeros(1), {'start_percent': 0.0}], [torch.ones(1), {'start_percent': .5}]]
            fake_gallery = SimpleNamespace(_resolve_reference_entry=lambda entry: entry)
            fake_class = SimpleNamespace(__module__='gallery')
            with patch.dict(flux3.nodes.NODE_CLASS_MAPPINGS, SECoursesReferenceGallery=fake_class), patch.object(flux3.importlib, 'import_module', return_value=fake_gallery):
                output = flux3.SEFlux3GalleryReferences().apply(original, refs, SimpleNamespace(encode=lambda image: image), .125)[0]
            self.assertNotIn('reference_latents', original[0][1])
            for before, after in zip(original, output):
                self.assertEqual(before[1]['start_percent'], after[1]['start_percent'])
                red, white = after[1]['reference_latents']
                self.assertEqual(red.shape, (1, 256, 512, 3))
                self.assertEqual(white.shape, (1, 512, 256, 3))
                self.assertTrue(torch.all(red[..., 0] == 1))
                self.assertTrue(torch.all(white == 1))


if __name__ == '__main__':
    unittest.main()
