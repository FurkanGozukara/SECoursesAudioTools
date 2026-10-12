"""CPU checks for the Musubi Trainer LoRA name aliases (musubi_lora_keys.py)."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import comfy.lora
import comfy.model_base

from custom_nodes.SECoursesAudioTools import musubi_lora_keys as mk


class MusubiLoraKeyTests(unittest.TestCase):
    def test_lycoris_targets_get_the_trainer_names_including_slices(self):
        target = ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight", (0, 0, 64))
        key_map = {
            "lycoris_transformer_blocks_0_img_mlp_gate_layer": target,
            "lora_unet_transformer_blocks_0_attn_to_q": "diffusion_model.transformer_blocks.0.attn.to_q.weight",
            "lycoris_transformer_blocks_0_attn_to_q": "something_else",
        }
        added = mk.add_musubi_names(SimpleNamespace(), key_map)
        self.assertEqual(added, 1)
        self.assertEqual(key_map["lora_unet_transformer_blocks_0_img_mlp_gate_layer"], target)
        # a name ComfyUI maps itself is never replaced
        self.assertEqual(key_map["lora_unet_transformer_blocks_0_attn_to_q"], "diffusion_model.transformer_blocks.0.attn.to_q.weight")

    def test_hidream_o1_gets_the_model_prefix(self):
        class FakeHiDreamO1:
            pass

        key_map = {"lora_unet_language_model_layers_0_mlp_down_proj": "diffusion_model.language_model.layers.0.mlp.down_proj.weight"}
        with patch.object(comfy.model_base, "HiDreamO1", FakeHiDreamO1, create=True):
            mk.add_musubi_names(FakeHiDreamO1(), key_map)
        self.assertEqual(
            key_map["lora_unet_model_language_model_layers_0_mlp_down_proj"],
            "diffusion_model.language_model.layers.0.mlp.down_proj.weight",
        )

    def test_install_wraps_the_key_map_once(self):
        calls = []

        def fake_keys(model, key_map={}):
            calls.append(key_map)
            key_map["lycoris_a_b"] = "diffusion_model.a.b.weight"
            return key_map

        with patch.object(comfy.lora, "model_lora_keys_unet", fake_keys), patch.object(comfy.lora, mk._MARK, False, create=True):
            mk.install()
            mk.install()
            result = comfy.lora.model_lora_keys_unet(SimpleNamespace(), {})
            self.assertIs(comfy.lora.model_lora_keys_unet.__wrapped__, fake_keys)
        self.assertEqual(result["lora_unet_a_b"], "diffusion_model.a.b.weight")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
