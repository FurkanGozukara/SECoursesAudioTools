import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PACKAGE = "secourses_audio_tools_under_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE] = package
control = importlib.import_module(f"{PACKAGE}.moss_image_control_nodes")


class ControlTests(unittest.TestCase):
    def test_no_maps_and_zero_strength_do_not_load_or_change_the_model(self):
        model, positive, negative = object(), object(), object()
        with patch.object(control.comfy.utils, "load_torch_file", side_effect=AssertionError("unexpected load")):
            for kwargs in ({}, {"strength": 0, "depth_map": torch.ones(1, 16, 16, 3)}):
                result = control.SEMossImageColorDepthControl().apply(model, positive, negative, None, **kwargs)
                self.assertEqual(result, (model, positive, negative))

    def test_color_ids_survive_resizing_and_schedule_ranges_are_preserved(self):
        maps = []

        class Vae:
            def encode(self, image):
                maps.append(image.clone())
                return image.movedim(-1, 1)

        positive = [[torch.zeros(1, 2, 3), {"start_percent": 0.25, "end_percent": 0.75}]]
        image = torch.zeros(1, 2, 2, 3)
        image[:, :, 1, 0] = 1
        with patch.object(control.folder_paths, "get_full_path_or_raise", return_value="control"), \
             patch.object(control.comfy.utils, "load_torch_file", return_value={}), \
             patch.object(control.comfy.sd, "load_lora_for_models", return_value=("patched", None)):
            model, pos, neg = control.SEMossImageColorDepthControl().apply(
                "original", positive, positive, Vae(), width=16, height=32, color_map=image, end_percent=0.5)
        self.assertEqual(model, "patched")
        self.assertEqual(maps[0].shape, (1, 32, 16, 3))
        self.assertEqual(set(maps[0].unique().tolist()), {0, 1})
        self.assertTrue(any("concat_latent_image" in item[1] for item in pos))
        self.assertTrue(all(0.25 <= item[1]["start_percent"] <= item[1]["end_percent"] <= 0.75 for item in pos))
        controlled = [item[1] for item in pos if "concat_latent_image" in item[1]]
        self.assertEqual(controlled[0]["start_percent"], 0.25)
        self.assertEqual(controlled[0]["end_percent"], 0.5)


if __name__ == "__main__":
    unittest.main()
