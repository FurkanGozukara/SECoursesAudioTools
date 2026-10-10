"""Optional mouth-only finishing for an existing ComfyUI video or a batch of frames."""

from dataclasses import replace
import json
import os

import torch
import folder_paths
from comfy_api.latest import InputImpl, Types

from .codeformer_mouth import CodeFormerMouth


def restore_mouth_frames(images, mouth):
    """Run the mouth pass over float RGB(A) frames in 16-frame chunks on a CPU copy. Pixels the pass does not
    change keep their original float values, so the rest of every frame is bit-identical."""
    images = images.detach().to(device="cpu", copy=True)
    for frames in images.split(16):
        rgb = frames[..., :3]
        pixels = rgb.mul(255).clamp(0, 255).to(torch.uint8)
        original = pixels.clone()
        restored = mouth.process(pixels)
        rgb.copy_(torch.where(restored != original, restored.to(rgb.dtype).div(255), rgb))
    return images


class SECodeFormerMouthImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "Restore only the mouth region. Off returns the frames unchanged without loading restoration models."}),
            "fidelity": ("FLOAT", {"default": .9, "min": 0, "max": 1, "step": .01, "tooltip": "CodeFormer fidelity; 0.9 is the tested recipe."}),
            "mouth_blend": ("FLOAT", {"default": .7, "min": 0, "max": 1, "step": .01, "tooltip": "Feathered mouth-only blend. 0 leaves every frame unchanged."}),
            "model": ("STRING", {"default": "codeformer.pth", "tooltip": "Existing file under models/facerestore_models. No automatic download."}),
            "detector": ("STRING", {"default": "models/buffalo_l/det_10g.onnx", "tooltip": "Existing SCRFD detector under models/insightface. No automatic download."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "restore"
    CATEGORY = "SECourses/video"
    DESCRIPTION = ("Restores only the mouth of every frame with aligned CodeFormer, for frame pipelines such as SwarmUI's. "
                   "Frame count, size and every pixel outside the restored mouth stay the same.")

    def restore(self, images, enabled=True, fidelity=.9, mouth_blend=.7,
                model="codeformer.pth", detector="models/buffalo_l/det_10g.onnx"):
        if not enabled or mouth_blend == 0:
            return (images,)
        mouth = CodeFormerMouth(model, detector, fidelity, mouth_blend)
        restored = restore_mouth_frames(images, mouth)
        print("[CodeFormer Mouth] " + json.dumps(mouth.report))
        return (restored,)


class SECodeFormerMouthVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "Restore only the mouth region. Off returns the original video without loading restoration models."}),
            "fidelity": ("FLOAT", {"default": .9, "min": 0, "max": 1, "step": .01}),
            "mouth_blend": ("FLOAT", {"default": .7, "min": 0, "max": 1, "step": .01}),
            "model": ("STRING", {"default": "codeformer.pth"}),
            "detector": ("STRING", {"default": "models/buffalo_l/det_10g.onnx"}),
            "filename_prefix": ("STRING", {"default": "video/MiniMax_H3_Lip_Synch_Mouth"}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("restored_video",)
    FUNCTION = "restore"
    OUTPUT_NODE = True
    CATEGORY = "SECourses/video"
    DESCRIPTION = "Restores the mouth and saves a separate MP4. Preserves the source video, frame count, frame rate and audio."

    def restore(self, video, enabled=True, fidelity=.9, mouth_blend=.7,
                model="codeformer.pth", detector="models/buffalo_l/det_10g.onnx",
                filename_prefix="video/MiniMax_H3_Lip_Synch_Mouth", prompt=None, extra_pnginfo=None):
        if not enabled or mouth_blend == 0:
            return (video,)
        source = video.get_components()
        mouth = CodeFormerMouth(model, detector, fidelity, mouth_blend)
        images = restore_mouth_frames(source.images, mouth)
        result = InputImpl.VideoFromComponents(replace(source, images=images), bit_depth=video.get_bit_depth(),
                                               color_space=video.get_color_space() or "sRGB")
        width, height = result.get_dimensions()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        filename = f"{filename}_{counter:05}_.mp4"
        metadata = {**(extra_pnginfo or {}), "mouth_restoration": mouth.report}
        if prompt is not None:
            metadata["prompt"] = prompt
        result.save_to(os.path.join(folder, filename), format=Types.VideoContainer.MP4,
                       codec=Types.VideoCodec.H264, metadata=metadata)
        print("[CodeFormer Mouth] " + json.dumps(mouth.report))
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder,
                                    "type": "output", "format": "video/mp4"}]}, "result": (result,)}


NODE_CLASS_MAPPINGS = {"SECodeFormerMouthVideo": SECodeFormerMouthVideo, "SECodeFormerMouthImages": SECodeFormerMouthImages}
NODE_DISPLAY_NAME_MAPPINGS = {"SECodeFormerMouthVideo": "CodeFormer Mouth Pass (Video)",
                              "SECodeFormerMouthImages": "CodeFormer Mouth Pass (Images)"}
