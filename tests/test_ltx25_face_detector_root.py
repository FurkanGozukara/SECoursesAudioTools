import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from custom_nodes.SECoursesAudioTools import ltx25_a2v_nodes as a2v


def make_pack(insightface_root, *files):
    pack = Path(insightface_root) / "models" / "buffalo_l"
    pack.mkdir(parents=True, exist_ok=True)
    for name in files:
        (pack / name).write_bytes(b"onnx")


class FaceDetectorRootTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.comfy_models = base / "ComfyUI" / "models"
        self.swarm_models = base / "SwarmUI" / "Models"
        self.folders = {"diffusion_models": [str(self.comfy_models / "diffusion_models"), str(self.swarm_models / "diffusion_models")],
                        "checkpoints": [str(self.swarm_models / "Stable-Diffusion")]}

    def tearDown(self):
        self.tmp.cleanup()

    def root(self):
        def get_folder_paths(key):
            return list(self.folders[key])
        with (patch.object(a2v.folder_paths, "models_dir", str(self.comfy_models)),
              patch.object(a2v.folder_paths, "get_folder_paths", side_effect=get_folder_paths)):
            return a2v._insightface_detector_root()

    def test_detector_next_to_swarm_models_is_used_despite_empty_comfy_folder(self):
        make_pack(self.comfy_models / "insightface")
        make_pack(self.swarm_models / "insightface", "det_10g.onnx")
        self.assertEqual(Path(self.root()), self.swarm_models / "insightface")

    def test_comfy_folder_is_preferred(self):
        make_pack(self.comfy_models / "insightface", "det_10g.onnx")
        make_pack(self.swarm_models / "insightface", "det_10g.onnx")
        self.assertEqual(Path(self.root()), self.comfy_models / "insightface")

    def test_registered_insightface_folder_is_searched(self):
        extra = Path(self.tmp.name) / "extra" / "insightface"
        make_pack(extra, "det_10g.onnx")
        self.folders["insightface"] = [str(extra)]
        self.assertEqual(Path(self.root()), extra)

    def test_no_detector_keeps_the_haar_cascade(self):
        make_pack(self.comfy_models / "insightface")
        self.assertIsNone(self.root())
