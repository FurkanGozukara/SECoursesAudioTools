"""Connect the SECourses prompt gallery to native MOSS-Image conditioning.

MOSS-Image reads its prompt with Qwen3.5 as plain language, a tag list or a JSON layout
(summary + items with abs_centralized yxyx boxes). ComfyUI's Qwen3.5 tokenizer has prompt
weights disabled, so `(word:1.2)` would reach the model as literal text.
"""

import json
import math
import os
import random
import re

LAYOUT_LABELS = ("object", "character", "text")
# A1111/Comfy weights such as (red car:1.2). A digit right before the colon is not a weight,
# so times and ratios like (10:30) or (4:3) stay untouched.
WEIGHT = re.compile(r"(?<!\\)\(([^()]*?[^\d\s():])\s*:\s*((?:\d+(?:\.\d*)?|\.\d+))\s*\)")
CHOICE = re.compile(r"(?<!\\)\{([^{}]*\|[^{}]*)\}")
# Innermost [...] groups; only real schedules ([a:b:0.5], [a:0.5]) and alternation ([a|b]) go to Swarm's parser.
BRACKET = re.compile(r"(?<!\\)\[([^\[\]]*)\]")
FROM_TO = re.compile(r"^[^:|]*:[^:|]*:\s*(?:\d+(?:\.\d*)?|\.\d+)\s*$")
TO_AT = re.compile(r"^[^:|]*:\s*(?:\d+(?:\.\d*)?|\.\d+)\s*$")


def media_check(references):
    if references.get("images") or references.get("videos") or references.get("audios"):
        raise ValueError("MOSS-Image is a text-to-image model and does not use reference media. Remove the gallery cards, or use another model's preset for image editing.")


def parse_json_prompt(text):
    """Return the decoded JSON object when the whole prompt is a JSON object, else None."""
    stripped = (text or "").strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _require(condition, message):
    if not condition:
        raise ValueError(f"MOSS-Image layout: {message}")


def compile_draft(draft, width=None, height=None):
    """Official prompt-enhancer draft (abs/xyxy pixels with width and height) -> compact model JSON.

    Matches OpenMOSS skills/moss-image-prompt-enhancer/scripts/compile_prompt.py: boxes enclose
    fractional edges (floor left/top, ceil right/bottom) before subtracting the canvas centre.
    When the generation size differs from the draft canvas, boxes are scaled to the generation size.
    """
    canvas_w, canvas_h = draft.get("width"), draft.get("height")
    _require(type(canvas_w) is int and type(canvas_h) is int and canvas_w > 0 and canvas_h > 0,
             "a pixel draft needs integer width and height")
    out_w, out_h = (width or canvas_w), (height or canvas_h)
    sx, sy = out_w / canvas_w, out_h / canvas_h
    _require(isinstance(draft.get("summary"), dict) and str(draft["summary"].get("caption", "")).strip(),
             "summary.caption is required")
    items = draft.get("items")
    _require(isinstance(items, list) and items, "items must be a nonempty array")
    compiled = []
    for index, item in enumerate(items):
        _require(isinstance(item, dict), f"items[{index}] must be an object")
        _require(item.get("label") in LAYOUT_LABELS, f"items[{index}].label must be object, character or text")
        _require(str(item.get("caption", "")).strip(), f"items[{index}].caption is required")
        box = item.get("bbox")
        _require(isinstance(box, list) and len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box),
                 f"items[{index}].bbox must be [x_min, y_min, x_max, y_max]")
        x1, y1, x2, y2 = box
        _require(0 <= x1 < x2 <= canvas_w and 0 <= y1 < y2 <= canvas_h,
                 f"items[{index}].bbox must lie inside the {canvas_w}x{canvas_h} draft canvas")
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        result = dict(item)
        result["bbox"] = [math.floor(round(y1, 9)) - out_h // 2, math.floor(round(x1, 9)) - out_w // 2,
                          math.ceil(round(y2, 9)) - out_h // 2, math.ceil(round(x2, 9)) - out_w // 2]
        compiled.append(result)
    layout = {"summary": draft["summary"], "items": compiled, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def check_compiled_layout(layout, width=None, height=None):
    """Light validation of a compiled layout; returns notes for the console."""
    notes = []
    items = layout.get("items")
    if not isinstance(items, list):
        return notes
    if layout.get("bbox_format", "yxyx") != "yxyx" or layout.get("bbox_coord_space", "abs_centralized") != "abs_centralized":
        notes.append("layout uses an unusual bbox_format/bbox_coord_space; MOSS-Image was trained on abs_centralized yxyx boxes")
    if width and height:
        half_w, half_h = width / 2, height / 2
        for index, item in enumerate(items):
            box = item.get("bbox") if isinstance(item, dict) else None
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                y0, x0, y1, x1 = box
                if min(y0, y1) < -half_h - 1 or max(y0, y1) > half_h + 1 or min(x0, x1) < -half_w - 1 or max(x0, x1) > half_w + 1:
                    notes.append(f"item {index + 1} box {box} reaches outside the {width}x{height} canvas; "
                                 "set the output resolution the layout was written for")
    return notes


def prepare_prompt(text, seed=0, width=None, height=None):
    """Return (prompt, is_json). JSON prompts pass through verbatim apart from compact serialization."""
    value = parse_json_prompt(text)
    if value is not None:
        if set(value) == {"caption"} and isinstance(value["caption"], str):
            text = value["caption"]
        elif value.get("bbox_coord_space") == "abs" or value.get("bbox_format") == "xyxy":
            return compile_draft(value, width, height), True
        else:
            for note in check_compiled_layout(value, width, height):
                print(f"[SECourses MOSS-Image] {note}", flush=True)
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), True
    rng = random.Random(seed)
    # Resolve choices here so inline and folder batch prompts get them too.
    while CHOICE.search(text):
        text = CHOICE.sub(lambda match: rng.choice(match.group(1).split("|")), text)
    # Qwen3.5 reads (word:1.2) literally; keep the words, drop the weight syntax.
    def unweight(match):
        return match.group(1) if float(match.group(2)) <= 3 else match.group(0)

    previous = None
    while previous != text:
        previous, text = text, WEIGHT.sub(unweight, text)
    return text, False


def schedule_prompt(text):
    """Return (uses_schedule, text for Swarm's parser).

    Swarm's legacy parser drops the brackets of any [...] group and fails on [note: words], so groups that
    are not real schedules or alternations are escaped before it sees them; their text stays literal."""
    found = False

    def visit(match):
        nonlocal found
        inner = match.group(1)
        if FROM_TO.match(inner) or TO_AT.match(inner) or ("|" in inner and ":" not in inner):
            found = True
            return "\0" + inner + "\1"
        return "\2" + inner + "\3"

    previous = None
    while previous != text:
        previous, text = text, BRACKET.sub(visit, text)
    text = text.replace("\0", "[").replace("\1", "]").replace("\2", "\\[").replace("\3", "\\]")
    return found or "<break>" in text, text


class SEMossImageGalleryPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": ("SECOURSES_REF_PACK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Seed for {red|blue} random choices. Link the shared seed so a fixed seed repeats."}),
            },
            "optional": {
                "width": ("INT", {"forceInput": True, "tooltip": "Output width: scales pixel layout drafts and checks layout boxes."}),
                "height": ("INT", {"forceInput": True, "tooltip": "Output height: scales pixel layout drafts and checks layout boxes."}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "is_json")
    FUNCTION = "prepare"
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = ("Gallery prompt for MOSS-Image: resolves {a|b} choices with the seed, removes (word:1.2) weight syntax that "
                   "Qwen3.5 would read literally, and passes JSON prompts through unchanged (compact). Official pixel layout drafts "
                   "(width, height, abs, xyxy) are compiled to the model's centred yxyx boxes. Gallery media is rejected.")

    def prepare(self, references, seed=0, width=None, height=None):
        media_check(references)
        return prepare_prompt(references.get("prompt", ""), seed, width, height)


class SEMossImageTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Plain prompt, tags or a JSON layout. JSON is encoded exactly as written."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000,
                                  "tooltip": "Sampling steps. Only used by [from:to:when] / [a|b] prompt schedules."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = ("MOSS-Image text encoder. JSON layouts and plain prompts are encoded verbatim with the native MOSS chat template; "
                   "prompts with [from:to:when], [a|b] or <break> use SwarmUI's advanced encoder when it is installed.")

    def encode(self, clip, text, steps=50):
        text = text if isinstance(text, str) else ""
        if parse_json_prompt(text) is None:
            scheduled, protected = schedule_prompt(text)
            if scheduled:
                import nodes

                swarm = nodes.NODE_CLASS_MAPPINGS.get("SwarmClipTextEncodeAdvanced")
                if swarm is not None:
                    return swarm().encode(clip, steps, protected, 1024, 1024, 1024, 1024)
                print("[SECourses MOSS-Image] SwarmClipTextEncodeAdvanced is not installed; prompt schedules are encoded as plain text.", flush=True)
        return (clip.encode_from_tokens_scheduled(clip.tokenize(text)),)


# SHA-256 of final_norm_video.weight (BF16 bytes); identical in the BF16 and INT8 ConvRot files of a variant.
VARIANT_FINGERPRINTS = {
    "c895d3101a2da82393a32a4f8876383964b2920c1f472b65323c406b223b2454": "SFT",
    "441ae61b1be782d5059fa2c262424680616db9c7c1ec7ec12de9ae08ed43ecc6": "Base",
}
DEBAND_DEFAULT = "MOSS_Image_SFT_DeBand_L37-41.safetensors"


def model_variant(model):
    """'SFT', 'Base' or None (fine-tune, merge or unreadable) for a MOSS-Image MODEL."""
    import hashlib

    import torch

    try:
        weight = model.model.diffusion_model.final_norm_video.weight
        data = weight.detach().to(device="cpu", dtype=torch.bfloat16, copy=True).contiguous().view(torch.uint8)
        return VARIANT_FINGERPRINTS.get(hashlib.sha256(data.numpy().tobytes()).hexdigest())
    except Exception as error:  # meta/offloaded weights or another architecture
        print(f"[SECourses MOSS-Image] could not read the model variant: {error}", flush=True)
        return None


def model_file_name(model):
    """File name ComfyUI loaded the MODEL from (cached_patcher_init holds the loader call), or None."""
    try:
        return os.path.basename(str(model.cached_patcher_init[1][0]))
    except (AttributeError, TypeError, IndexError):
        return None


def lora_choices(default):
    import folder_paths

    names = folder_paths.get_filename_list("loras")
    normalized = {name.replace("\\", "/"): name for name in names}
    return ["None"] + names, normalized.get(default, "None")


class SEMossImageSFTDeBand:
    @classmethod
    def INPUT_TYPES(cls):
        choices, default = lora_choices(DEBAND_DEFAULT)
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "SFT DE-BAND: ON", "label_off": "SFT DE-BAND: OFF",
                                        "tooltip": "Removes the faint vertical color stripes MOSS-Image SFT draws on skies, fog, walls and backdrops. Applied only to the SFT model; Base and unknown models pass through unchanged."}),
                "fix_lora": (choices, {"default": default,
                                       "tooltip": "Main fix: reverts SFT layers 37-41 to the Base weights (MATLOWAI/MOSS-Image-SFT-DeBand, Apache-2.0)."}),
                "fix_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "extra_fix_lora": (choices, {"default": "None",
                                             "tooltip": "Optional extra fix for stubborn skies and wide images (layers 32-36). Cleaner, but changes the SFT look more."}),
                "extra_fix_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "SECourses/MOSS-Image"
    DESCRIPTION = ("SFT banding fix that recognizes the loaded MOSS-Image variant from its weights: applied to SFT (INT8 or BF16), "
                   "skipped for Base, fine-tunes and merges, so switching the model never double-applies it.")

    def apply(self, model, enabled, fix_lora, fix_strength, extra_fix_lora="None", extra_fix_strength=1.0):
        if not enabled:
            return (model,)
        variant = model_variant(model)
        name = model_file_name(model)
        if variant == "Base" and name and "sft" in name.lower():
            # ComfyUI's official MOSS-Image bounding-box template (templates 0.11.89) downloads Base under the SFT name.
            print(f"[SECourses MOSS-Image] WARNING: {name} holds the Base weights, not SFT. Delete it and download the SFT "
                  "file again (MOSS-Image Core Bundle).", flush=True)
        if variant != "SFT":
            reason = "the Base model has no SFT banding" if variant == "Base" else "the model is not the original MOSS-Image SFT"
            print(f"[SECourses MOSS-Image] SFT de-band skipped: {reason}.", flush=True)
            return (model,)
        import comfy.sd
        import comfy.utils
        import folder_paths

        for name, strength in ((fix_lora, fix_strength), (extra_fix_lora, extra_fix_strength)):
            if name in (None, "", "None") or strength == 0:
                continue
            lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return (model,)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SEMossImageGalleryPrompt, SEMossImageTextEncode, SEMossImageSFTDeBand)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SEMossImageGalleryPrompt": "MOSS-Image Gallery Prompt",
    "SEMossImageTextEncode": "MOSS-Image Text Encode",
    "SEMossImageSFTDeBand": "MOSS-Image SFT De-Band (auto)",
}
