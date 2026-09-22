"""Small gallery/canvas adapters around ComfyUI's native Qwen Image 2.1 nodes."""

import base64
import importlib
import io
import json
import re

import numpy as np
from PIL import Image, ImageOps
import torch

import comfy.model_management
import comfy.utils
import nodes


MODES = ["Generate / reference edit", "Image to image", "Inpaint masked area"]
NO_IMAGE = "(none - disabled)"


def translate_prompt(prompt, reference_count, has_init):
    """Gallery numbering remains stable when a separate init canvas is present."""
    def replace(match):
        number = int(match.group(1))
        if not 1 <= number <= reference_count:
            raise ValueError(f"@image{number} has no matching gallery image. Update the prompt after removing/reordering cards.")
        return f"<image{number + int(has_init)}>"
    prompt = re.sub(r"@(?:image|img|picture|pic)(\d+)\b", replace, prompt, flags=re.I)
    if re.search(r"@init\b", prompt, re.I):
        if not has_init:
            raise ValueError("@init needs an image in the optional init canvas node.")
        prompt = re.sub(r"@init\b", "<image1>", prompt, flags=re.I)
    return prompt


def resize_image(image, width, height):
    return comfy.utils.common_upscale(image[:1].movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)


class SEQwenImage21Canvas(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        schema = nodes.LoadImage.INPUT_TYPES()
        values, options = schema["required"]["image"]
        schema["required"]["image"] = ([NO_IMAGE, *values], {**options, "tooltip": "Optional init canvas. Select none to disable. Right-click the image preview > Open in Mask Editor for inpainting; painted mask = area to change."})
        schema["optional"] = {"denoise": ("FLOAT", {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength: lower keeps more of the original; 1.0 fully redraws. Inpaint applies it inside the painted mask; image-to-image applies it to the whole image. Generate/reference edit always uses 1.0."})}
        return schema

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("IMAGE", "MASK", "denoise")
    CATEGORY = "SECourses/Qwen Image 2.1"
    DESCRIPTION = "Optional init image and native ComfyUI mask editor. None disables the canvas. For transparent reference images, use the gallery, which retains RGBA."

    def load_image(self, image, denoise=0.85):
        pixels, mask = (None, None) if image == NO_IMAGE else super().load_image(image)
        return pixels, mask, float(denoise)

    @classmethod
    def IS_CHANGED(cls, image, denoise=0.85):
        return NO_IMAGE if image == NO_IMAGE else nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image, denoise=0.85):
        return True if image == NO_IMAGE else nodes.LoadImage.VALIDATE_INPUTS(image)


class SEQwenImage21Prepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "vae": ("VAE",), "references": ("SECOURSES_REF_PACK",),
            "mode": (MODES,),
            "width": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32, "tooltip": "Text-only canvas. With references/init, output follows the first image's aspect ratio."}),
            "height": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32}),
            "reference_resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32, "tooltip": "Reference / edit size (pixel-area target): resizes each input to about this value squared while preserving its aspect ratio, rounded to 32 pixels. 1024 is about 1 MP; 2048 is about 4 MP and costs more VRAM/time. 0 keeps each input's original size rounded to 32. With an init/reference, output size follows the init/first reference. Text-only uses the canvas width/height instead."}),
            "denoise": ("FLOAT", {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Only img2img/inpaint. Higher redraws more. Generate/reference edit always uses 1.0."}),
            "transparent": ("BOOLEAN", {"default": False, "tooltip": "Request a transparent background: adds the official RGBA wording to the prompt. Describe an isolated subject/cutout and save as PNG. This asks the model to generate alpha; it is not a background-removal tool or the paint mask. Inpaint still preserves the original outside the mask."}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Ignored by the sampler at the recommended CFG 1.0. Raise CFG only intentionally."}),
        }, "optional": {"init_image": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "FLOAT", "SE_QWEN21_COMPOSITE")
    RETURN_NAMES = ("positive", "negative", "latent", "denoise", "preserve_unmasked")
    FUNCTION = "prepare"
    CATEGORY = "SECourses/Qwen Image 2.1"
    DESCRIPTION = "Native Qwen 2.1 conditioning with 0-10 images from the SECourses gallery. @image1 addresses the first gallery card; @init addresses the separate canvas. Inpainting uses a native latent noise mask plus exact unmasked-pixel compositing after decode."

    def prepare(self, clip, vae, references, mode, width, height, reference_resolution,
                denoise, transparent, negative_prompt, init_image=None, mask=None):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        if mode not in MODES:
            raise ValueError(f"Unknown Qwen 2.1 mode: {mode}")
        if references.get("videos") or references.get("audios"):
            raise ValueError("Qwen Image 2.1 accepts images only. Remove video/audio cards from the gallery.")
        entries = references.get("images", [])
        tensors = references.get("image_tensors", [])
        reference_count = len(entries) + len(tensors)
        has_init = init_image is not None
        if reference_count + int(has_init) > 10:
            raise ValueError("Qwen Image 2.1 supports up to 10 images total, including the optional init canvas.")
        if mode != MODES[0] and not has_init:
            raise ValueError(f"{mode} requires an image in the optional init canvas node.")
        if mode == MODES[2] and (mask is None or not torch.any(mask > 0)):
            raise ValueError("Inpaint requires a painted mask. Open the init image in Mask Editor and paint the area to change.")
        if any(int(value) < 32 or int(value) > 4096 or int(value) % 32 for value in (width, height)):
            raise ValueError("Canvas width/height must be multiples of 32 between 32 and 4096.")
        prompt = translate_prompt(references.get("prompt", ""), reference_count, has_init)
        if transparent:
            prompt = "This is an RGBA image with transparency. " + prompt + " The image has alpha channel and the background is transparent."
        images = [init_image[:1]] if has_init else []
        if entries:
            gallery = importlib.import_module(nodes.NODE_CLASS_MAPPINGS["SECoursesReferenceGallery"].__module__)
            for entry in entries:
                with Image.open(gallery._resolve_reference_entry(entry)) as source:
                    source = ImageOps.exif_transpose(source)
                    has_alpha = "A" in source.getbands() or "transparency" in source.info
                    pixels = np.array(source.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32) / 255.0
                    images.append(torch.from_numpy(pixels).unsqueeze(0))
        images.extend(tensors)
        result = TextEncodeQwenImage21.execute(
            clip, prompt, negative_prompt, vae=vae, resolution=int(reference_resolution),
            images={f"image_{i + 1}": image for i, image in enumerate(images)},
        )
        positive, negative, latent = result.result
        preserve = {}
        if not images:
            latent = {"samples": torch.zeros((1, 64, height // 16, width // 16), device=comfy.model_management.intermediate_device())}
        if mode != MODES[0]:
            height, width = (int(v) * 16 for v in latent["samples"].shape[-2:])
            canvas = resize_image(init_image, width, height)
            latent = {"samples": vae.encode(canvas)}
            if mode == MODES[2]:
                edit_mask = torch.nn.functional.interpolate(mask[:1].reshape(1, 1, *mask.shape[-2:]).float(), size=(height, width), mode="bilinear", align_corners=False).squeeze(1).clamp(0, 1)
                # Fractional masks reinsert the original at every sampling step.
                # Allow the painted area to denoise; feather only the final blend.
                latent["noise_mask"] = (edit_mask > 0).float()
                preserve = {"canvas": canvas, "mask": edit_mask}
        else:
            denoise = 1.0
        return positive, negative, latent, float(denoise), preserve


class SEQwenImage21Finish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "preserve_unmasked": ("SE_QWEN21_COMPOSITE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "finish"
    CATEGORY = "SECourses/Qwen Image 2.1"
    DESCRIPTION = "Passes generation/RGBA through unchanged. In mask mode restores all pixels outside the mask from the resized init canvas, including opaque alpha."

    def finish(self, images, preserve_unmasked):
        if not preserve_unmasked:
            return (images,)
        canvas = preserve_unmasked["canvas"].to(images)
        if images.shape[-1] == 4 and canvas.shape[-1] == 3:
            canvas = torch.cat((canvas, torch.ones_like(canvas[..., :1])), dim=-1)
        mask = preserve_unmasked["mask"].to(images).unsqueeze(-1)
        return (images * mask + canvas * (1 - mask),)


class SEQwenImage21SwarmInputs:
    """Decode Swarm's ordered attachments without flattening reference alpha."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True}),
            "references_json": ("STRING", {"default": "[]"}),
            "init_base64": ("STRING", {"default": ""}),
            "mask_base64": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("SECOURSES_REF_PACK", "IMAGE", "MASK")
    FUNCTION = "load"
    CATEGORY = "SECourses/Qwen Image 2.1"

    def load(self, prompt, references_json, init_base64, mask_base64):
        def decode(value, mask=False):
            if value.startswith("data:"):
                value = value.split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as source:
                source = ImageOps.exif_transpose(source)
                mode = "L" if mask else ("RGBA" if "A" in source.getbands() or "transparency" in source.info else "RGB")
                pixels = np.array(source.convert(mode), dtype=np.float32) / 255.0
            return torch.from_numpy(pixels).unsqueeze(0)

        references = {"prompt": prompt, "image_tensors": [decode(value) for value in json.loads(references_json)]}
        init = decode(init_base64) if init_base64 else None
        mask = decode(mask_base64, mask=True) if mask_base64 else None
        return references, init, mask


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SEQwenImage21Canvas, SEQwenImage21Prepare, SEQwenImage21Finish, SEQwenImage21SwarmInputs)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEQwenImage21Canvas": "Qwen 2.1 Optional Init Image + Mask",
    "SEQwenImage21Prepare": "Qwen 2.1 Gallery + Canvas",
    "SEQwenImage21Finish": "Qwen 2.1 Preserve Unmasked Pixels",
}
