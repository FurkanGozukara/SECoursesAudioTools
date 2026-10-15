"""Connect the SECourses image/prompt gallery to native FLUX.3 conditioning."""

import importlib
import random
import re

import numpy as np
from PIL import Image, ImageOps
import torch

import node_helpers
import nodes
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels


def image_entries(references):
    if references.get("videos") or references.get("audios"):
        raise ValueError("FLUX.3 accepts image references only. Remove video/audio cards from the gallery.")
    entries = references.get("images", [])
    if len(entries) > 10:
        raise ValueError("FLUX.3 supports up to 10 reference images. Remove the extra gallery cards.")
    return entries


class SEFlux3GalleryPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"references": ("SECOURSES_REF_PACK",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "prepare"
    CATEGORY = "SECourses/FLUX.3"
    DESCRIPTION = "Gallery prompt with 1-based @image1 aliases translated to FLUX.3's 0-based <ref_image_0> labels. Keeps weights, schedules and object IDs for Swarm's advanced text encoder."

    def prepare(self, references, seed=0):
        count = len(image_entries(references))

        def replace(match):
            number = int(match.group(1))
            if not 1 <= number <= count:
                raise ValueError(f"@image{number} has no matching gallery image. Update the prompt after removing/reordering cards.")
            return f"<ref_image_{number - 1}>"

        prompt = re.sub(r"@(?:image|img|picture|pic)(\d+)\b", replace, references.get("prompt", ""), flags=re.I)
        rng = random.Random(seed)
        # Resolve choices here so added prompts and folder prompts work too.
        choices = re.compile(r"(?<!\\)\{([^{}]*\|[^{}]*)\}")
        while choices.search(prompt):
            prompt = choices.sub(lambda match: rng.choice(match.group(1).split("|")), prompt)
        return (prompt,)


class SEFlux3GalleryReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "references": ("SECOURSES_REF_PACK",),
            "vae": ("VAE",),
            "reference_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                "tooltip": "Size of EACH reference, preserving its aspect ratio and rounding to 16 pixels. 1.0 = about 1024 x 1024. Output size is set separately by Resolution Sync."}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "SECourses/FLUX.3"
    DESCRIPTION = "Optional 0-10 image references from the gallery, in card order. Empty gallery passes conditioning through. Appends each VAE latent to every scheduled conditioning entry, as native Set Reference Latent does."

    def apply(self, conditioning, references, vae, reference_megapixels):
        entries = image_entries(references)
        if not entries:
            return (conditioning,)
        gallery = importlib.import_module(nodes.NODE_CLASS_MAPPINGS["SECoursesReferenceGallery"].__module__)
        latents = []
        for entry in entries:
            with Image.open(gallery._resolve_reference_entry(entry)) as source:
                source = ImageOps.exif_transpose(source).convert("RGBA")
                # FLUX.3 is RGB; composite transparent references on white.
                background = Image.new("RGBA", source.size, "white")
                pixels = np.array(Image.alpha_composite(background, source).convert("RGB"), dtype=np.float32) / 255.0
            image = torch.from_numpy(pixels).unsqueeze(0)
            image = ImageScaleToTotalPixels.execute(image, "lanczos", reference_megapixels, 16).result[0]
            latents.append(vae.encode(image))
        return (node_helpers.conditioning_set_values(conditioning, {"reference_latents": latents}, append=True),)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SEFlux3GalleryPrompt, SEFlux3GalleryReferences)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEFlux3GalleryPrompt": "FLUX.3 Gallery Prompt",
    "SEFlux3GalleryReferences": "FLUX.3 Optional Gallery References",
}
