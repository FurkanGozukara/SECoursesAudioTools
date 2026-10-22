"""Connect the SECourses prompt gallery and SwarmUI to native Fibo 2.0 conditioning.

Fibo 2.0 (Bria) reads a structured JSON caption (Bria's VGL) or plain text with Qwen3-VL 4B. ComfyUI's native
TextEncodeFibo2 cleans the caption, wraps edits, draws the caption's text images and adds the reference latents.
These nodes feed it gallery images and a masked-edit canvas, and follow Bria's pipeline (diffusers
BriaFibo2Pipeline) in the two places where ComfyUI's defaults differ from it:
- Bria encodes the negative prompt as written. TextEncodeFibo2 cleans a JSON negative and drops its empty fields
  (163 instead of 181 tokens for the model card's text-to-image negative).
- Bria shifts the sigmas by the image size (FLUX's dynamic shift, 0.5 at 256 tokens to 1.15 at 4096 tokens of
  16x16 pixels). ComfyUI uses 1.15 at every size, Bria's value only at about 1024x1024.
"""

import base64
import importlib
import io
import json
import math
import random
import re

import numpy as np
from PIL import Image, ImageOps
import torch

import comfy.utils
import nodes

SIZE_MODES = ["Auto: edits keep the first image size (1-4 MP)", "Bria default: edits at the trained ~1 MP size", "Always use the resolution inputs"]
SHIFT_MODES = ["Official: dynamic shift by image size (Bria)", "ComfyUI default: fixed shift 1.15"]
NO_IMAGE = "(none - disabled)"
# Edits of images from 1 to 4 MP keep their size (tested up to 4 MP); smaller ones get Bria's trained ~1 MP size
MIN_EDIT_AREA, MAX_EDIT_AREA = 1024 * 1024, 2048 * 2048
# FLUX's dynamic shift with Bria's scheduler settings
BASE_SEQ_LEN, MAX_SEQ_LEN, BASE_SHIFT, MAX_SHIFT = 256, 4096, 0.5, 1.15
# A generation shows Qwen3-VL the images of its caption's text from this sigma on (diffusers text_images_max_noise_level)
TEXT_IMAGES_SIGMA = 0.9

# A1111/Comfy weights such as (red car:1.2). A digit right before the colon is not a weight, so (10:30) and (4:3) stay.
WEIGHT = re.compile(r"(?<!\\)\(([^()]*?[^\d\s():])\s*:\s*((?:\d+(?:\.\d*)?|\.\d+))\s*\)")
CHOICE = re.compile(r"(?<!\\)\{([^{}]*\|[^{}]*)\}")
IMAGE_REF = re.compile(r"@(?:image|img|picture|pic)(\d+)\b", re.I)
INIT_REF = re.compile(r"@init\b", re.I)


def parse_json_prompt(text):
    """The decoded JSON object when the whole prompt is a JSON object, else None."""
    stripped = (text or "").strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def image_labels(prompt, image_count, has_init=False):
    """@image1, @image2, ... follow the gallery order and become Fibo 2.0's <image_1>, <image_2>, ...
    With an image in the edit canvas, @init is <image_1> and the gallery continues from <image_2>."""
    offset = int(has_init)

    def replace(match):
        number = int(match.group(1))
        if not 1 <= number <= image_count:
            raise ValueError(f"@image{number} has no matching gallery image. Update the prompt after removing or reordering cards.")
        return f"<image_{number + offset}>"

    prompt = IMAGE_REF.sub(replace, prompt)
    if INIT_REF.search(prompt):
        if not has_init:
            raise ValueError("@init needs an image in the edit canvas node.")
        prompt = INIT_REF.sub("<image_1>", prompt)
    return prompt


def prepare_prompt(text, seed=0, image_count=0, has_init=False):
    """A JSON caption keeps every character apart from @image labels: TextEncodeFibo2 cleans it as training did.
    Plain prompts get seeded {a|b} choices and lose (word:1.2) weight syntax, which Qwen3-VL would read literally."""
    text = text if isinstance(text, str) else ""
    if parse_json_prompt(text) is None:
        rng = random.Random(seed)
        while CHOICE.search(text):
            text = CHOICE.sub(lambda match: rng.choice(match.group(1).split("|")), text)
        previous = None
        while previous != text:
            previous, text = text, WEIGHT.sub(lambda match: match.group(1) if float(match.group(2)) <= 3 else match.group(0), text)
    return image_labels(text, image_count, has_init)


def raw_negative(text):
    """Bria encodes the negative prompt as written. TextEncodeFibo2 would clean a JSON negative, so it gets the chat turn
    itself, which the tokenizer takes as it is; the JSON is only made compact (key order and empty fields kept).
    An empty negative stays empty: TextEncodeFibo2 then uses the null caption, Bria's default negative."""
    value = parse_json_prompt(text)
    if value is None:
        return text if isinstance(text, str) else ""
    import comfy.text_encoders.fibo2

    return comfy.text_encoders.fibo2.TEMPLATE.format(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def gallery_images(references):
    """Gallery images in card order as [1, H, W, 3] tensors; transparency is composited on white.
    SwarmUI attachments arrive already decoded in image_tensors."""
    if references.get("videos") or references.get("audios"):
        raise ValueError("Fibo 2.0 takes image references only. Remove the video/audio cards from the gallery.")
    images = []
    entries = references.get("images", [])
    if entries:
        gallery = importlib.import_module(nodes.NODE_CLASS_MAPPINGS["SECoursesReferenceGallery"].__module__)
        for entry in entries:
            with Image.open(gallery._resolve_reference_entry(entry)) as source:
                images.append(pil_to_tensor(source))
    images.extend(references.get("image_tensors", []))
    return images


def pil_to_tensor(source):
    source = ImageOps.exif_transpose(source).convert("RGBA")
    background = Image.new("RGBA", source.size, "white")
    pixels = np.array(Image.alpha_composite(background, source).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def grey_out(image, mask):
    """Bria's masked edit, as its training did it: the masked region (the mask made binary at 128 of 255) becomes flat
    (128, 128, 128) grey at the image's own size, before the image is resized."""
    mask = mask[:1].reshape(1, 1, *mask.shape[-2:]).float()
    if mask.shape[-2:] != image.shape[1:3]:
        mask = torch.nn.functional.interpolate(mask, size=image.shape[1:3], mode="nearest")
    masked = (mask[0, 0] > 0.5).to(image).unsqueeze(-1)
    return image[:1, ..., :3] * (1 - masked) + (128 / 255) * masked


def edit_size(width, height):
    """Result size of an edit in the auto mode: the image's own size in multiples of 16 from 1 to 4 MP, scaled down above
    4 MP, and Bria's trained size for its aspect ratio (about 1 MP) below 1 MP."""
    if width * height < MIN_EDIT_AREA:
        from comfy_extras.nodes_fibo2 import trained_size

        return trained_size(width / height)
    scale = min(1.0, math.sqrt(MAX_EDIT_AREA / (width * height)))
    return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)


def official_shift(width, height):
    """mu of Bria's dynamic shift for a width x height result."""
    tokens = (height // 16) * (width // 16)
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return BASE_SHIFT + slope * (tokens - BASE_SEQ_LEN)


def text_images_start(mu):
    """Sampling percent at which the flux time shift mu reaches sigma 0.9."""
    shifted = math.exp(mu) * (1 / TEXT_IMAGES_SIGMA - 1)
    return shifted / (1 + shifted)


def remap_text_images(conditioning, mu):
    """Move TextEncodeFibo2's switch to the caption with its text images (computed for shift 1.15) to sigma 0.9 at mu."""
    from comfy_extras.nodes_fibo2 import TEXT_IMAGES_START

    start = text_images_start(mu)
    out = []
    for cond, options in conditioning:
        options = options.copy()
        if math.isclose(options.get("end_percent", -1.0), TEXT_IMAGES_START - 1e-6, rel_tol=0, abs_tol=1e-9):
            options["end_percent"] = start - 1e-6
        if math.isclose(options.get("start_percent", -1.0), TEXT_IMAGES_START, rel_tol=0, abs_tol=1e-9):
            options["start_percent"] = start
        out.append([cond, options])
    return out


def encode(clip, vae, prompt, negative_prompt, images, size_mode, width, height, batch_size, seed=0, edit_image=None, edit_mask=None):
    """Native TextEncodeFibo2 with the gallery images, the optional edit canvas and its mask.
    Returns positive, negative, latent and the canvas/mask pair that restores unmasked pixels (None without a mask)."""
    from comfy_extras.nodes_fibo2 import TextEncodeFibo2

    if size_mode not in SIZE_MODES:
        raise ValueError(f"Unknown Fibo 2.0 size mode: {size_mode}")
    images = list(images)
    has_init = edit_image is not None
    masked = has_init and edit_mask is not None and bool(torch.any(edit_mask[:1] > 0.5))
    prompt = prepare_prompt(prompt, seed, len(images), has_init)
    if has_init:
        if masked and images:
            raise ValueError("A painted mask edits a single image, as Fibo 2.0 was trained. Remove the gallery images or clear the mask in the edit canvas.")
        images.insert(0, grey_out(edit_image, edit_mask) if masked else edit_image[:1, ..., :3])
    if images and size_mode == SIZE_MODES[0]:
        width, height = edit_size(images[0].shape[2], images[0].shape[1])
    elif images and size_mode == SIZE_MODES[1]:
        width = height = 0
    positive, negative, latent = TextEncodeFibo2.execute(
        clip, prompt, raw_negative(negative_prompt), vae=vae,
        images={f"image_{i + 1}": image for i, image in enumerate(images)},
        width=int(width), height=int(height), batch_size=int(batch_size)).result
    restore = None
    if masked:
        out_h, out_w = (int(v) * 16 for v in latent["samples"].shape[-2:])
        # The source and its mask cropped to the result exactly as TextEncodeFibo2 crops a single image.
        canvas = comfy.utils.common_upscale(edit_image[:1, ..., :3].movedim(-1, 1), out_w, out_h, "lanczos", "center").movedim(1, -1)
        mask = comfy.utils.common_upscale(edit_mask[:1].reshape(1, 1, *edit_mask.shape[-2:]).float(), out_w, out_h, "bilinear", "center")
        restore = {"canvas": canvas.clamp(0, 1), "mask": mask[0, 0].clamp(0, 1)}
    return positive, negative, latent, restore


class SEFibo2EditCanvas(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        schema = nodes.LoadImage.INPUT_TYPES()
        values, options = schema["required"]["image"]
        schema["required"]["image"] = ([NO_IMAGE, *values], {**options, "tooltip": "Optional image to edit with a painted mask. Right-click the preview > Open in Mask Editor and paint the area to change. None disables the canvas; the gallery images are then the images to edit."})
        return schema

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "SECourses/Fibo 2.0"
    DESCRIPTION = "Optional image to edit with ComfyUI's native Mask Editor (masked edit). It becomes <image_1> (@init in the prompt); gallery images follow it. A painted mask edits this image alone."

    def load_image(self, image):
        return (None, None) if image == NO_IMAGE else super().load_image(image)

    @classmethod
    def IS_CHANGED(cls, image):
        return NO_IMAGE if image == NO_IMAGE else nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True if image == NO_IMAGE else nodes.LoadImage.VALIDATE_INPUTS(image)


class SEFibo2GalleryEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "references": ("SECOURSES_REF_PACK",),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Used when CFG is above 1 (Dev). Empty = the null caption Fibo 2.0 was trained with, Bria's negative for edits. Text-to-image uses Bria's JSON negative. Encoded exactly as written, as in Bria's pipeline."}),
                "size_mode": (SIZE_MODES, {"tooltip": "Text-to-image always uses the resolution inputs. Auto: an edit keeps the first image's own size (multiples of 16) from 1 to 4 MP, so a 2 MP photo is edited at 2 MP; larger images are scaled down to 4 MP, smaller ones go up to Fibo 2.0's trained ~1 MP size. Bria default: every edit at the trained ~1 MP size for the first image's aspect ratio. Always use the resolution inputs: a single image to edit is center-cropped to their aspect ratio."}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "Images per prompt."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for {red|blue} random choices. Link the shared seed so a fixed seed repeats."}),
            },
            "optional": {
                "edit_image": ("IMAGE", {"tooltip": "Optional image to edit from the edit canvas. It becomes <image_1>; gallery images follow it."}),
                "edit_mask": ("MASK", {"tooltip": "Painted mask of the edit canvas: the region to change. It is greyed out as Fibo 2.0's training did."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "SE_FIBO2_RESTORE")
    RETURN_NAMES = ("positive", "negative", "latent", "restore_unmasked")
    FUNCTION = "encode"
    CATEGORY = "SECourses/Fibo 2.0"
    DESCRIPTION = ("Fibo 2.0 conditioning from the SECourses gallery with ComfyUI's native Text Encode Fibo 2.0: text-to-image without "
                   "images, editing with 1 to 4 images (more are accepted), @image1 -> <image_1>. JSON captions are passed through; "
                   "plain prompts get seeded {a|b} choices and lose (word:1.2) weights. The negative prompt is encoded as written, "
                   "like Bria's pipeline. A masked edit greys out the painted region of the edit canvas image.")

    def encode(self, clip, vae, references, negative_prompt, size_mode, width, height, batch_size, seed=0, edit_image=None, edit_mask=None):
        return encode(clip, vae, references.get("prompt", ""), negative_prompt, gallery_images(references), size_mode,
                      width, height, batch_size, seed, edit_image, edit_mask)


class SEFibo2Sampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "latent": ("LATENT",),
            "shift": (SHIFT_MODES, {"tooltip": "Official: Bria's resolution-dependent shift, 1.15 at about 1024x1024 and higher for larger images (about 2.0 at 1536x1536). ComfyUI default: 1.15 at every size."}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "apply"
    CATEGORY = "SECourses/Fibo 2.0"
    DESCRIPTION = ("Sigma shift of Bria's Fibo 2.0 pipeline: FLUX's dynamic shift from 0.5 at 256 tokens to 1.15 at 4096 tokens "
                   "(16x16 pixel patches), taken from the latent's size, applied with ComfyUI's ModelSamplingFlux. The caption's "
                   "text images then still start at sigma 0.9, as in Bria's pipeline.")

    def apply(self, model, positive, latent, shift):
        if shift not in SHIFT_MODES:
            raise ValueError(f"Unknown Fibo 2.0 shift mode: {shift}")
        if shift != SHIFT_MODES[0]:
            return (model, positive)
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux

        height, width = (int(v) * 16 for v in latent["samples"].shape[-2:])
        model = ModelSamplingFlux().patch(model, MAX_SHIFT, BASE_SHIFT, width, height)[0]
        return (model, remap_text_images(positive, official_shift(width, height)))


class SEFibo2RestoreUnmasked:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "restore_unmasked": ("SE_FIBO2_RESTORE",),
            "enabled": ("BOOLEAN", {"default": True, "label_on": "KEEP UNMASKED PIXELS: ON", "label_off": "KEEP UNMASKED PIXELS: OFF",
                "tooltip": "After a masked edit, copy every pixel outside the painted mask back from the source, so only the masked area can change. Off shows Fibo 2.0's own result. No effect without a mask."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "SECourses/Fibo 2.0"
    DESCRIPTION = "Masked edits: restores the source pixels outside the painted mask (soft mask edges blend). Other generations pass through unchanged."

    def restore(self, images, restore_unmasked, enabled=True):
        if not enabled or not restore_unmasked:
            return (images,)
        canvas = restore_unmasked["canvas"].to(images)
        mask = restore_unmasked["mask"].to(images).unsqueeze(-1)
        return (images * mask + canvas * (1 - mask),)


class SEFibo2SwarmInputs:
    """SwarmUI attachments for SEFibo2GalleryEncode: prompt images, and the init image and mask of a masked edit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True}),
            "references_json": ("STRING", {"default": "[]"}),
            "init_base64": ("STRING", {"default": ""}),
            "mask_base64": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("SECOURSES_REF_PACK", "IMAGE", "MASK")
    RETURN_NAMES = ("references", "edit_image", "edit_mask")
    FUNCTION = "load"
    CATEGORY = "SECourses/Fibo 2.0"

    def load(self, prompt, references_json, init_base64, mask_base64):
        def decode(value, mask=False):
            if value.startswith("data:"):
                value = value.split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as source:
                if not mask:
                    return pil_to_tensor(source)
                pixels = np.array(ImageOps.exif_transpose(source).convert("L"), dtype=np.float32) / 255.0
            return torch.from_numpy(pixels).unsqueeze(0)

        references = {"prompt": prompt, "image_tensors": [decode(value) for value in json.loads(references_json)]}
        init = decode(init_base64) if init_base64 else None
        mask = decode(mask_base64, mask=True) if init is not None and mask_base64 else None
        return references, init, mask


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SEFibo2EditCanvas, SEFibo2GalleryEncode, SEFibo2Sampling, SEFibo2RestoreUnmasked, SEFibo2SwarmInputs)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEFibo2EditCanvas": "Fibo 2.0 Optional Edit Image + Mask",
    "SEFibo2GalleryEncode": "Fibo 2.0 Gallery Encode",
    "SEFibo2Sampling": "Fibo 2.0 Official Sigma Shift",
    "SEFibo2RestoreUnmasked": "Fibo 2.0 Keep Unmasked Pixels",
    "SEFibo2SwarmInputs": "Fibo 2.0 SwarmUI Inputs",
}
