import torch
import torch.nn.functional as F


def _validate_target(source_h, source_w, target_h, target_w):
    if target_h < source_h or target_w < source_w:
        raise ValueError(
            f"Target size {target_w}x{target_h} must be >= source size {source_w}x{source_h}."
        )


def _make_rect_mask(height, width, x0, y0, x1, y1, device):
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def _bbox_from_mask(mask):
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0 or xs.numel() == 0:
        raise ValueError("Source mask is empty.")
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    return x0, y0, x1, y1


def _expand_interval_to_multiple(start, end, limit, multiple):
    size = end - start
    if multiple <= 1 or size % multiple == 0:
        return start, end

    target = ((size + multiple - 1) // multiple) * multiple
    grow = target - size

    room_before = start
    room_after = limit - end

    grow_after = min(grow, room_after)
    end += grow_after
    grow -= grow_after

    grow_before = min(grow, room_before)
    start -= grow_before
    grow -= grow_before

    if grow > 0 and end < limit:
        extra = min(grow, limit - end)
        end += extra
        grow -= extra
    if grow > 0 and start > 0:
        extra = min(grow, start)
        start -= extra
        grow -= extra

    return start, end


def _region_rects(canvas_h, canvas_w, source_mask_2d, region, context, align_to=1):
    x0, y0, x1, y1 = _bbox_from_mask(source_mask_2d)

    full_mask = torch.zeros(
        (canvas_h, canvas_w), dtype=torch.float32, device=source_mask_2d.device
    )

    if region == "top":
        full_mask[:y0, :] = 1.0
        cx0, cy0, cx1, cy1 = 0, 0, canvas_w, min(canvas_h, y0 + context)
    elif region == "bottom":
        full_mask[y1:, :] = 1.0
        cx0, cy0, cx1, cy1 = 0, max(0, y1 - context), canvas_w, canvas_h
    elif region == "left":
        full_mask[y0:y1, :x0] = 1.0
        cx0, cy0, cx1, cy1 = 0, 0, min(canvas_w, x0 + context), canvas_h
    elif region == "right":
        full_mask[y0:y1, x1:] = 1.0
        cx0, cy0, cx1, cy1 = max(0, x1 - context), 0, canvas_w, canvas_h
    elif region == "top_left":
        full_mask[:y0, :x0] = 1.0
        cx0, cy0, cx1, cy1 = 0, 0, min(canvas_w, x0 + context), min(
            canvas_h, y0 + context
        )
    elif region == "top_center":
        full_mask[:y0, x0:x1] = 1.0
        cx0, cy0, cx1, cy1 = max(0, x0 - context), 0, min(
            canvas_w, x1 + context
        ), min(canvas_h, y0 + context)
    elif region == "top_right":
        full_mask[:y0, x1:] = 1.0
        cx0, cy0, cx1, cy1 = max(0, x1 - context), 0, canvas_w, min(
            canvas_h, y0 + context
        )
    elif region == "bottom_left":
        full_mask[y1:, :x0] = 1.0
        cx0, cy0, cx1, cy1 = 0, max(0, y1 - context), min(
            canvas_w, x0 + context
        ), canvas_h
    elif region == "bottom_center":
        full_mask[y1:, x0:x1] = 1.0
        cx0, cy0, cx1, cy1 = max(0, x0 - context), max(0, y1 - context), min(
            canvas_w, x1 + context
        ), canvas_h
    elif region == "bottom_right":
        full_mask[y1:, x1:] = 1.0
        cx0, cy0, cx1, cy1 = max(0, x1 - context), max(
            0, y1 - context
        ), canvas_w, canvas_h
    else:
        raise ValueError(f"Unsupported region '{region}'.")

    if align_to and align_to > 1:
        cx0, cx1 = _expand_interval_to_multiple(cx0, cx1, canvas_w, align_to)
        cy0, cy1 = _expand_interval_to_multiple(cy0, cy1, canvas_h, align_to)

    return full_mask, (cx0, cy0, cx1, cy1)


def _build_guide_image(region_image, guide_mask, guide_fill, neutral_value):
    if guide_fill == "replicate":
        return region_image

    mask = guide_mask.to(region_image.dtype).unsqueeze(0).unsqueeze(-1)
    if guide_fill == "neutral":
        fill = torch.full_like(region_image, float(neutral_value))
    elif guide_fill == "source_mean":
        counts = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        fill = (region_image * mask).sum(dim=(1, 2), keepdim=True) / counts
        fill = fill.expand_as(region_image)
    else:
        raise ValueError(f"Unsupported guide_fill '{guide_fill}'.")

    return region_image * mask + fill * (1.0 - mask)


def _apply_linear_feather(paste_mask, region, feather):
    if feather <= 0:
        return paste_mask
    if torch.count_nonzero(paste_mask) == 0:
        return paste_mask

    gx0, gy0, gx1, gy1 = _bbox_from_mask(paste_mask)
    out = paste_mask.clone()

    def ramp_up(length):
        return torch.linspace(
            1.0 / float(length + 1),
            1.0,
            steps=length,
            device=out.device,
            dtype=out.dtype,
        )

    def feather_top(length):
        length = min(length, gy1 - gy0)
        if length <= 0:
            return
        out[gy0 : gy0 + length, gx0:gx1] *= ramp_up(length).view(-1, 1)

    def feather_bottom(length):
        length = min(length, gy1 - gy0)
        if length <= 0:
            return
        out[gy1 - length : gy1, gx0:gx1] *= ramp_up(length).flip(0).view(-1, 1)

    def feather_left(length):
        length = min(length, gx1 - gx0)
        if length <= 0:
            return
        out[gy0:gy1, gx0 : gx0 + length] *= ramp_up(length).view(1, -1)

    def feather_right(length):
        length = min(length, gx1 - gx0)
        if length <= 0:
            return
        out[gy0:gy1, gx1 - length : gx1] *= ramp_up(length).flip(0).view(1, -1)

    if region == "top":
        feather_bottom(feather)
    elif region == "bottom":
        feather_top(feather)
    elif region == "left":
        feather_right(feather)
    elif region == "right":
        feather_left(feather)
    elif region == "top_left":
        feather_right(feather)
        feather_bottom(feather)
    elif region == "top_center":
        feather_left(feather)
        feather_right(feather)
        feather_bottom(feather)
    elif region == "top_right":
        feather_left(feather)
        feather_bottom(feather)
    elif region == "bottom_left":
        feather_top(feather)
        feather_right(feather)
    elif region == "bottom_center":
        feather_top(feather)
        feather_left(feather)
        feather_right(feather)
    elif region == "bottom_right":
        feather_top(feather)
        feather_left(feather)

    return out


class VideoOutpaintReplicateCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": (
                    "INT",
                    {"default": 1280, "min": 64, "max": 8192, "step": 32},
                ),
                "target_height": (
                    "INT",
                    {"default": 736, "min": 64, "max": 8192, "step": 32},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT")
    RETURN_NAMES = ("canvas", "outer_mask", "source_mask", "source_x", "source_y")
    FUNCTION = "prepare"
    CATEGORY = "image/outpaint"

    def prepare(self, image, target_width, target_height):
        batch, source_h, source_w, channels = image.shape
        _validate_target(source_h, source_w, target_height, target_width)

        x0 = (target_width - source_w) // 2
        y0 = (target_height - source_h) // 2
        x1 = x0 + source_w
        y1 = y0 + source_h

        padded = F.pad(
            image.movedim(-1, 1),
            (x0, target_width - x1, y0, target_height - y1),
            mode="replicate",
        ).movedim(1, -1)

        source_mask = _make_rect_mask(
            target_height, target_width, x0, y0, x1, y1, image.device
        )
        outer_mask = 1.0 - source_mask

        return (padded, outer_mask, source_mask, x0, y0)


class VideoOutpaintPrepareCanvasByPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "add_top": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "add_left": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "add_bottom": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "add_right": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "internal_align": (
                    "INT",
                    {"default": 32, "min": 1, "max": 512, "step": 1},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
        "INT",
        "INT",
        "IMAGE",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "canvas",
        "outer_mask",
        "source_mask",
        "source_x",
        "source_y",
        "original_canvas",
        "final_width",
        "final_height",
        "internal_width",
        "internal_height",
        "crop_x",
        "crop_y",
    )
    FUNCTION = "prepare"
    CATEGORY = "image/outpaint"

    def prepare(
        self,
        image,
        add_top,
        add_left,
        add_bottom,
        add_right,
        internal_align,
    ):
        batch, source_h, source_w, channels = image.shape

        final_width = source_w + int(add_left) + int(add_right)
        final_height = source_h + int(add_top) + int(add_bottom)

        if final_width <= 0 or final_height <= 0:
            raise ValueError("Final output size must be positive.")

        if internal_align <= 1:
            internal_width = final_width
            internal_height = final_height
        else:
            internal_width = (
                (final_width + int(internal_align) - 1) // int(internal_align)
            ) * int(internal_align)
            internal_height = (
                (final_height + int(internal_align) - 1) // int(internal_align)
            ) * int(internal_align)

        extra_right = internal_width - final_width
        extra_bottom = internal_height - final_height

        x0 = int(add_left)
        y0 = int(add_top)
        x1 = x0 + source_w
        y1 = y0 + source_h

        replicate_canvas = F.pad(
            image.movedim(-1, 1),
            (
                int(add_left),
                int(add_right) + extra_right,
                int(add_top),
                int(add_bottom) + extra_bottom,
            ),
            mode="replicate",
        ).movedim(1, -1)

        original_canvas = torch.zeros(
            (batch, internal_height, internal_width, channels),
            dtype=image.dtype,
            device=image.device,
        )
        original_canvas[:, y0:y1, x0:x1, :] = image

        source_mask = _make_rect_mask(
            internal_height, internal_width, x0, y0, x1, y1, image.device
        )
        outer_mask = 1.0 - source_mask

        crop_x = 0
        crop_y = 0

        return (
            replicate_canvas,
            outer_mask,
            source_mask,
            x0,
            y0,
            original_canvas,
            final_width,
            final_height,
            internal_width,
            internal_height,
            crop_x,
            crop_y,
        )


class VideoOutpaintRegionCrop:
    REGION_OPTIONS = [
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_center",
        "top_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas": ("IMAGE",),
                "source_mask": ("MASK",),
                "region": (cls.REGION_OPTIONS,),
                "context": (
                    "INT",
                    {"default": 320, "min": 32, "max": 4096, "step": 32},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "region_image",
        "generate_mask",
        "guide_mask",
        "paste_x",
        "paste_y",
        "region_width",
        "region_height",
    )
    FUNCTION = "crop_region"
    CATEGORY = "image/outpaint"

    def crop_region(self, canvas, source_mask, region, context):
        batch, canvas_h, canvas_w, _ = canvas.shape
        if source_mask.ndim == 3:
            source_mask_2d = source_mask[0]
        else:
            source_mask_2d = source_mask
        full_mask, (cx0, cy0, cx1, cy1) = _region_rects(
            canvas_h, canvas_w, source_mask_2d, region, context
        )

        region_image = canvas[:, cy0:cy1, cx0:cx1, :]
        generate_mask = full_mask[cy0:cy1, cx0:cx1]
        guide_mask = source_mask_2d[cy0:cy1, cx0:cx1].clone()

        return (
            region_image,
            generate_mask,
            guide_mask,
            cx0,
            cy0,
            cx1 - cx0,
            cy1 - cy0,
        )


class VideoOutpaintRegionCropAdvanced(VideoOutpaintRegionCrop):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas": ("IMAGE",),
                "source_mask": ("MASK",),
                "region": (cls.REGION_OPTIONS,),
                "context": (
                    "INT",
                    {"default": 320, "min": 32, "max": 4096, "step": 32},
                ),
                "guide_fill": (
                    ["replicate", "source_mean", "neutral"],
                    {"default": "source_mean"},
                ),
                "neutral_value": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "feather": (
                    "INT",
                    {"default": 48, "min": 0, "max": 512, "step": 1},
                ),
                "align_to": (
                    "INT",
                    {"default": 64, "min": 1, "max": 512, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "region_image",
        "guide_image",
        "generate_mask",
        "guide_mask",
        "paste_mask",
        "paste_x",
        "paste_y",
        "region_width",
        "region_height",
    )
    FUNCTION = "crop_region_advanced"
    CATEGORY = "image/outpaint"

    def crop_region_advanced(
        self,
        canvas,
        source_mask,
        region,
        context,
        guide_fill,
        neutral_value,
        feather,
        align_to,
    ):
        batch, canvas_h, canvas_w, _ = canvas.shape
        if source_mask.ndim == 3:
            source_mask_2d = source_mask[0]
        else:
            source_mask_2d = source_mask

        full_mask, (cx0, cy0, cx1, cy1) = _region_rects(
            canvas_h, canvas_w, source_mask_2d, region, context, align_to=align_to
        )

        region_image = canvas[:, cy0:cy1, cx0:cx1, :]
        generate_mask = full_mask[cy0:cy1, cx0:cx1]
        guide_mask = source_mask_2d[cy0:cy1, cx0:cx1].clone()

        guide_image = _build_guide_image(
            region_image, guide_mask, guide_fill, neutral_value
        )
        paste_mask = _apply_linear_feather(generate_mask, region, feather)

        return (
            region_image,
            guide_image,
            generate_mask,
            guide_mask,
            paste_mask,
            cx0,
            cy0,
            cx1 - cx0,
            cy1 - cy0,
        )
