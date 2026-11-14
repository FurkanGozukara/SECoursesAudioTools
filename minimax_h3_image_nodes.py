"""Connect the SECourses prompt gallery to native MiniMax H3 Image conditioning, with the official prompt enhancer.

MiniMax H3 Image expects a detailed English description of the whole image. MiniMax's API expands short prompts with
its prompt enhancer before generating (always for edits); MiniMaxAI/MiniMax-H3-Image-PE publishes the instructions
(assets/minimax_h3_image_pe, unchanged) and how the edit message is assembled. Here a local Qwen3.5 9B runs them
through ComfyUI's native Generate Text, as the official ComfyUI templates do. The enhancer also picks the aspect ratio,
and the encode node uses it at the chosen megapixels with MiniMax's own size rounding (16:9 at 16 MP = 5472 x 3072).
"""

import json
import math
import random
import re
from pathlib import Path

import torch

from .fibo2_nodes import CHOICE, WEIGHT, gallery_images

PE_DIR = Path(__file__).resolve().parent / "assets" / "minimax_h3_image_pe"
ENHANCE_MODES = ["On: official enhancer with reasoning (best)", "On: fast enhancer without reasoning",
                 "Off: use my prompt exactly as written"]
SIZE_MODES = ["Auto: enhancer picks the aspect ratio (edits keep image 1's shape) at these megapixels",
              "Exact: always use the width and height above"]
IMAGE_REF = re.compile(r"@(?:image|img|picture|pic)(\d+)\b", re.I)
# MiniMax's API accepts aspect ratios from 1:3 to 3:1
MIN_RATIO, MAX_RATIO = 1 / 3, 3.0
CANVAS_MULTIPLE = 32
# Generate Text settings of the official request (prompt/t2i/request_params.json); max_length as in the ComfyUI templates
SAMPLING = {"sampling_mode": "on", "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
            "repetition_penalty": 1.0, "presence_penalty": 0.0}
MAX_TOKENS = 8192
VISION_START, IMAGE_PAD, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"


def read_pe(name):
    return (PE_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def prepare_prompt(text, seed, image_count):
    """Seeded {a|b} choices, (word:1.2) weights removed (the encoders read them literally), and the gallery's @image1 as
    the <Picture 1> label MiniMax H3 Image uses."""
    text = text if isinstance(text, str) else ""
    rng = random.Random(seed)
    while CHOICE.search(text):
        text = CHOICE.sub(lambda match: rng.choice(match.group(1).split("|")), text)
    previous = None
    while previous != text:
        previous, text = text, WEIGHT.sub(lambda match: match.group(1) if float(match.group(2)) <= 3 else match.group(0), text)

    def label(match):
        number = int(match.group(1))
        if not 1 <= number <= image_count:
            raise ValueError(f"@image{number} has no matching gallery image. Update the prompt after removing or reordering cards.")
        return f"<Picture {number}>"

    return IMAGE_REF.sub(label, text).strip()


def reduced_ratio(width, height):
    divisor = math.gcd(int(width), int(height)) or 1
    return f"{int(width) // divisor}:{int(height) // divisor}"


def parse_ratio(value):
    """'16:9' -> 1.777...; None for anything else."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*", str(value or ""))
    if not match or float(match.group(2)) == 0 or float(match.group(1)) == 0:
        return None
    return float(match.group(1)) / float(match.group(2))


def size_for(ratio, area):
    """MiniMax's size for an aspect ratio and a pixel area: each side rounded to the nearest multiple of 32
    (1:1 at 16 MP = 4096 x 4096, 4:3 = 4736 x 3552, 16:9 = 5472 x 3072, as in MiniMax's size tables)."""
    ratio = min(max(float(ratio), MIN_RATIO), MAX_RATIO)
    width = max(CANVAS_MULTIPLE, round(math.sqrt(area * ratio) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    height = max(CANVAS_MULTIPLE, round(math.sqrt(area / ratio) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return width, height


def parse_answer(text):
    """The enhancer's JSON object (the last one in the answer) as a dict, or None."""
    text = str(text or "")
    decoder = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("expanded_caption"), str) and value["expanded_caption"].strip():
            return value
    match = re.search(r'"expanded_caption"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if match:
        try:
            caption = json.loads(f'"{match.group(1)}"')
        except ValueError:
            return None
        return {"expanded_caption": caption} if caption.strip() else None
    return None


def scaled_for_vision(image, megapixels):
    """An image no larger than megapixels (aspect kept) for the enhancer's vision encoder."""
    height, width = image.shape[1:3]
    scale = min(1.0, math.sqrt(megapixels * 1e6 / (width * height)))
    if scale >= 1.0:
        return image[:1]
    size = (max(32, round(height * scale)), max(32, round(width * scale)))
    return torch.nn.functional.interpolate(image[:1].movedim(-1, 1), size=size, mode="area").movedim(1, -1)


def edit_message(instruction, images, aspect_selection, thinking=True):
    """The official edit enhancer message (MiniMax-H3-Image-PE README): the task prompt, the image count, each image's
    label and source ratio right before it, then the instruction and the target ratio. Raw chat text, so ComfyUI's
    tokenizer keeps this order and puts the images in the vision slots."""
    count = len(images)
    parts = [read_pe("edit_task_prompt.txt"),
             f"Reference Image Count:\n{count} input image(s), labeled <Picture 1> through <Picture {count}>."]
    for index, image in enumerate(images, start=1):
        parts.append(f"Reference Media:\n- <Picture {index}> [aspect_ratio {reduced_ratio(image.shape[2], image.shape[1])}]")
        parts.append(VISION_START + IMAGE_PAD + VISION_END)
    parts.append("\nUser Editing Instruction (JSON string):\n" + json.dumps(instruction, ensure_ascii=False) + "\n\n"
                 f"User Aspect Ratio Selection:\n{aspect_selection}\n\n"
                 "Apply the instructions above to the supplied inputs and reference images. "
                 "Output only the required two-field JSON object.")
    # An empty think block turns reasoning off, as the chat template does
    think = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return "<|im_start|>user\n" + "".join(parts) + "<|im_end|>\n<|im_start|>assistant\n" + think


def run_enhancer(clip, prompt, images, seed, thinking=True):
    """Runs the official enhancer: returns (caption, aspect ratio string or None, raw answer)."""
    from comfy_extras.nodes_textgen import TextGenerateV2

    sampling = dict(SAMPLING, seed=int(seed) % 0xffffffffffffffff)
    if images:
        vision_mp = min(1.0, 4.0 / len(images))
        vision = {f"image{i + 1}": scaled_for_vision(image, vision_mp) for i, image in enumerate(images)}
        result = TextGenerateV2.execute(clip=clip, prompt=edit_message(prompt, images, "adaptive", thinking), max_length=MAX_TOKENS,
                                        sampling_mode=sampling, thinking=thinking, images=vision)
        ratio_key = "target_aspect_ratio"
    else:
        result = TextGenerateV2.execute(clip=clip, prompt=read_pe("t2i_task_prompt.txt").format(prompt), max_length=MAX_TOKENS,
                                        sampling_mode=sampling, thinking=thinking, system_prompt=read_pe("t2i_system_prompt.txt").rstrip("\n"))
        ratio_key = "aspect_ratio"
    answer = result.args[0]
    parsed = parse_answer(answer)
    if parsed is None:
        return None, None, answer
    return parsed["expanded_caption"].strip(), parsed.get(ratio_key) or parsed.get("aspect_ratio"), answer


def enhance_prompt(enhance, clip, prompt, images, seed, node):
    """(prompt the model receives, enhancer aspect ratio or None) for an ENHANCE_MODES choice."""
    if enhance not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhancer mode: {enhance}")
    if enhance == ENHANCE_MODES[2]:
        return prompt, None
    caption, ratio, answer = run_enhancer(clip, prompt, images, seed, thinking=enhance == ENHANCE_MODES[0])
    if caption is None and enhance == ENHANCE_MODES[0]:
        # Seen on edits: the reasoning ran out of tokens or ended without the JSON answer. The fast mode still answers.
        print(f"[{node}] the reasoning enhancer gave no caption, so the fast enhancer writes it. Answer: {answer[-300:]!r}", flush=True)
        caption, ratio, answer = run_enhancer(clip, prompt, images, seed, thinking=False)
    if caption is None:
        print(f"[{node}] the enhancer gave no caption, so your prompt is used as written. Answer: {answer[-600:]!r}", flush=True)
        return prompt, None
    return caption, ratio


def result_size(size_mode, ratio, first_image, width, height):
    """Exact: width x height. Auto: that pixel area at the enhancer's ratio, else image 1's shape, else width x height's."""
    if size_mode not in SIZE_MODES:
        raise ValueError(f"Unknown size mode: {size_mode}")
    width, height = int(width), int(height)
    if size_mode == SIZE_MODES[1]:
        return width, height
    ratio = parse_ratio(ratio)
    if ratio is None:
        ratio = first_image.shape[2] / first_image.shape[1] if first_image is not None else width / height
    return size_for(ratio, width * height)


class SEH3ImagePromptEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": ("SECOURSES_REF_PACK",),
                "enhance": (ENHANCE_MODES, {"tooltip": "On (official, with reasoning): a local Qwen3.5 9B turns your idea (any language, a few words are enough) into the detailed English caption MiniMax H3 Image was trained on, with MiniMax's official instructions, and picks the aspect ratio. Edits get the official edit instructions with your images. Fast: the same without the reasoning, much quicker and slightly less careful. Off: your prompt goes to the model as written; use it for long detailed captions you wrote yourself."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed of the enhancer and of {red|blue} choices. Link the shared seed so a fixed seed repeats the same caption."}),
            },
            "optional": {
                "enhancer_clip": ("CLIP", {"lazy": True, "tooltip": "Qwen3.5 9B (Load CLIP, any type except moss_image). Loaded only when the enhancer is On."}),
                "text_to_image_only": ("BOOLEAN", {"default": False, "tooltip": "On in text-to-image-only presets (KJNodes empty image): gallery images stop the run with a message instead of being ignored. Use the Image Edit preset for images."}),
            },
        }

    RETURN_TYPES = ("SECOURSES_REF_PACK", "STRING")
    RETURN_NAMES = ("references", "prompt")
    FUNCTION = "run"
    CATEGORY = "SECourses/MiniMax H3 Image"
    DESCRIPTION = ("Prepares the gallery prompt for MiniMax H3 Image: seeded {a|b} choices, @image1 -> <Picture 1>, and with the "
                   "enhancer On the official MiniMax prompt enhancer on a local Qwen3.5 9B (text to image, or editing with up to "
                   "16 gallery images). The prompt output is the exact text the model receives.")

    def check_lazy_status(self, references, enhance, seed, enhancer_clip=None, text_to_image_only=False):
        return ["enhancer_clip"] if enhance != ENHANCE_MODES[2] and enhancer_clip is None else []

    def run(self, references, enhance, seed, enhancer_clip=None, text_to_image_only=False):
        images = gallery_images(references)
        if text_to_image_only and images:
            raise ValueError("This preset is text to image only (KJNodes empty image). Remove the gallery images, or use the "
                             "MiniMax H3 Image Image Edit preset to edit them.")
        prompt = prepare_prompt(references.get("prompt", ""), seed, len(images))
        if not prompt:
            raise ValueError("Write a prompt first: what to create, or for images, what to change.")
        prompt, ratio = enhance_prompt(enhance, enhancer_clip, prompt, images, seed, "SEH3ImagePromptEnhance")
        print(f"[SEH3ImagePromptEnhance] {len(images)} image(s), aspect {ratio or 'from size'}; prompt: {prompt[:400]!r}", flush=True)
        return (dict(references, prompt=prompt, h3_image_aspect=ratio), prompt)



class SEH3ImageGalleryEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The MiniMax H3 Qwen3-VL 32B encoder (Load CLIP type minimax)."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 Image VAE."}),
                "references": ("SECOURSES_REF_PACK", {"tooltip": "From MiniMax H3 Image Prompt Enhance."}),
                "width": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 32}),
                "size_mode": (SIZE_MODES, {"tooltip": "Auto: the result has the aspect ratio the enhancer chose (with the enhancer Off, image 1's shape for edits, the width and height above for text to image) at the megapixels of the width and height above, rounded like MiniMax's API. Exact: always the width and height above."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "Images per prompt."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "SECourses/MiniMax H3 Image"
    DESCRIPTION = ("MiniMax H3 Image conditioning from the gallery with ComfyUI's native MiniMax H3 Image Conditioning: text to image "
                   "without images, editing with 1 to 16 images (image 1 is the one edited). Sets the result size from the "
                   "enhancer's aspect ratio or image 1 at the chosen megapixels.")

    def encode(self, clip, vae, references, width, height, size_mode, batch_size=1):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

        images = gallery_images(references)
        width, height = result_size(size_mode, references.get("h3_image_aspect"), images[0] if images else None, width, height)
        positive, latent = MiniMaxH3ImageConditioning.execute(
            clip=clip, prompt=references.get("prompt", ""), width=width, height=height, vae=vae,
            images={f"image_{i + 1}": image for i, image in enumerate(images)}).args
        if batch_size > 1:
            latent = dict(latent, samples=latent["samples"].repeat(int(batch_size), *([1] * (latent["samples"].ndim - 1))))
        print(f"[SEH3ImageGalleryEncode] {width}x{height} ({width * height / 1e6:.1f} MP), {len(images)} image(s), batch {batch_size}", flush=True)
        return (positive, latent)


class SEH3ImageSwarmEnhance:
    """The SwarmUI side of the presets (SwarmUI_Premium_Extensions/MiniMaxH3Image): the same prompt preparation, official
    enhancer and Auto size as the gallery nodes, for SwarmUI's native MiniMax H3 Image workflow. Its outputs replace the
    text of SwarmUI's encode node and the size of its empty latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "SwarmUI's processed prompt; @image1 is the first prompt image."}),
                "enhance": (ENHANCE_MODES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 32}),
                "size_mode": (SIZE_MODES,),
            },
            "optional": {
                "enhancer_clip": ("CLIP", {"lazy": True}),
                "minimax_refs": ("MiniMaxReferences", {"tooltip": "SwarmUI's MiniMax H3 Image references (the prompt images)."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "width", "height")
    FUNCTION = "run"
    CATEGORY = "SECourses/MiniMax H3 Image"

    def check_lazy_status(self, prompt, enhance, seed, width, height, size_mode, enhancer_clip=None, minimax_refs=None):
        return ["enhancer_clip"] if enhance != ENHANCE_MODES[2] and enhancer_clip is None else []

    def run(self, prompt, enhance, seed, width, height, size_mode, enhancer_clip=None, minimax_refs=None):
        images = [item["data"] for item in (minimax_refs or {}).get("ref_items", []) if item["type"] == "image"]
        prompt = prepare_prompt(prompt, seed, len(images))
        if not prompt:
            raise ValueError("Write a prompt first: what to create, or for images, what to change.")
        prompt, ratio = enhance_prompt(enhance, enhancer_clip, prompt, images, seed, "SEH3ImageSwarmEnhance")
        width, height = result_size(size_mode, ratio, images[0] if images else None, width, height)
        print(f"[SEH3ImageSwarmEnhance] {len(images)} image(s), {width}x{height}; prompt: {prompt[:400]!r}", flush=True)
        return (prompt, width, height)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SEH3ImagePromptEnhance, SEH3ImageGalleryEncode, SEH3ImageSwarmEnhance)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEH3ImagePromptEnhance": "MiniMax H3 Image Prompt Enhance (Gallery)",
    "SEH3ImageGalleryEncode": "MiniMax H3 Image Gallery Encode",
    "SEH3ImageSwarmEnhance": "MiniMax H3 Image Prompt Enhance (SwarmUI)",
}
