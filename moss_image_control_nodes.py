"""SECourses gallery and control-map adapters for native MOSS-Image conditioning."""

import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy_extras.nodes_moss_image import MOSSImageControl

from .fibo2_nodes import NO_IMAGE, SEFibo2EditCanvas, gallery_images
from .moss_image_edit_nodes import reference_size

CONTROL_LORA = "moss_image_color_depth_control.safetensors"


class SEMossImageControlMap(SEFibo2EditCanvas):
    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["image"][1]["tooltip"] = (
            "Optional control map. Color: one flat color per region (colors are IDs). "
            "Depth: near white, far black. None disables this map."
        )
        return schema

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = "Load an optional color-region or depth map. No image is required when the other map supplies the control."

    def load_image(self, image):
        return (super().load_image(image)[0],)


class SEMossImageColorDepthControl:
    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        normalized = {name.replace("\\", "/"): name for name in names}
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "vae": ("VAE",),
            "control_lora": (names, {"default": normalized.get(CONTROL_LORA, CONTROL_LORA)}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                "tooltip": "Scales the maps' latent contribution. Zero bypasses control and its LoRA."}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "0.5 for synthetic maps: establish placement first, then finish natural surfaces. 1.0 for real-image maps."}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
        }, "optional": {"color_map": ("IMAGE",), "depth_map": ("IMAGE",)}}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = ("Native MOSS color/depth control with the community adapter from darask0. "
        "Maps follow the output resolution; color regions use nearest-neighbor resizing. "
        "No maps or zero strength bypasses both the adapter and control. Gallery prompts and schedules stay intact.")

    def apply(self, model, positive, negative, vae, control_lora=CONTROL_LORA, lora_strength=1.0,
              strength=1.0, start_percent=0.0, end_percent=0.5, width=1024, height=1024,
              color_map=None, depth_map=None):
        if strength == 0 or (color_map is None and depth_map is None):
            return (model, positive, negative)
        if start_percent > end_percent:
            raise ValueError("MOSS control start must not exceed its end.")
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", control_lora), safe_load=True)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, lora_strength, 0)
        maps = []
        for image, method in ((color_map, "nearest-exact"), (depth_map, "bilinear")):
            if image is not None:
                image = image[..., :3]
                if image.shape[1:3] != (height, width):
                    image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)
            maps.append(image)
        result = MOSSImageControl.execute(positive, negative, vae, *maps, strength, start_percent, end_percent).result
        return (model, result[0], result[1])


class SEMossImageNativeEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "references": ("SECOURSES_REF_PACK",),
            "reference_max_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
            "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
        }}

    RETURN_TYPES = ("SE_MOSS_EDIT_REFS", "INT", "INT")
    RETURN_NAMES = ("references", "width", "height")
    FUNCTION = "apply"
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = ("ChenkinLab Moss-Image-Edit using ComfyUI's native reference branch. "
        "Keeps gallery prompts, scheduling and the first image's edit size; no model-block replacements. "
        "Load MOSS_Image_Edit_LoRA on the model first.")

    def apply(self, vae, references, reference_max_edge=1024, width=1024, height=1024):
        images = gallery_images(references)
        if not images:
            return ({}, width, height)
        latents, sizes = [], []
        for image in images:
            size = reference_size(image.shape[2], image.shape[1], reference_max_edge)
            if size != (image.shape[2], image.shape[1]):
                image = comfy.utils.common_upscale(image.movedim(-1, 1), *size, "lanczos", "disabled").movedim(1, -1)
            latents.append(vae.encode(image[..., :3]))
            sizes.append(size)
        values = {"reference_latents": latents, "reference_latents_method": "index_timestep_zero_text_start"}
        return (values, *sizes[0])


class SEMossImageApplyEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                             "references": ("SE_MOSS_EDIT_REFS",)}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "SECourses/MOSS-Image"

    def apply(self, positive, negative, references):
        return (node_helpers.conditioning_set_values(positive, references),
                node_helpers.conditioning_set_values(negative, references))


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (SEMossImageControlMap, SEMossImageColorDepthControl, SEMossImageNativeEdit, SEMossImageApplyEdit)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEMossImageControlMap": "MOSS-Image Optional Control Map",
    "SEMossImageColorDepthControl": "MOSS-Image Color + Depth Control",
    "SEMossImageNativeEdit": "MOSS-Image Native Edit References",
    "SEMossImageApplyEdit": "MOSS-Image Apply Native Edit Conditioning",
}
