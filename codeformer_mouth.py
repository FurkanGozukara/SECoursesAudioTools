"""Aligned CodeFormer mouth restoration for decoded video frames."""

import os
import time

import cv2
import numpy as np
import torch

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths


TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


def model_path(category, name):
    # Shared Swarm model folders are siblings of its configured model categories.
    roots = {folder_paths.models_dir}
    for key in ("diffusion_models", "checkpoints"):
        roots.update(os.path.dirname(path) for path in folder_paths.get_folder_paths(key))
    for root in sorted(roots):
        folder_paths.add_model_folder_path(category, os.path.join(root, category))
    return folder_paths.get_full_path_or_raise(category, name.replace("\\", "/"))


def mouth_mask(center, width):
    yy, xx = np.ogrid[:512, :512]
    radius = np.sqrt(((xx - center[0]) / max(40, width * .75)) ** 2
                     + ((yy - center[1] - 5) / max(24, width * .48)) ** 2)
    ramp = np.clip((1 - radius) / .38, 0, 1)
    return ((1 - np.cos(np.pi * ramp)) * .5).astype(np.float32)[..., None]


def blend_mouth(source, crop, restored, transform, center, width, strength):
    """Project only the mouth's support rectangle; source pixels outside stay intact."""
    mask = mouth_mask(center, width) * strength
    inverse = cv2.invertAffineTransform(transform)
    x_radius, y_radius = max(40, width * .75), max(24, width * .48)
    corners = np.array([[center[0] - x_radius - 2, center[1] + 5 - y_radius - 2],
                        [center[0] + x_radius + 2, center[1] + 5 - y_radius - 2],
                        [center[0] - x_radius - 2, center[1] + 5 + y_radius + 2],
                        [center[0] + x_radius + 2, center[1] + 5 + y_radius + 2]], np.float32)
    bounds = cv2.transform(corners[None], inverse)[0]
    height, image_width = source.shape[:2]
    left, top = np.maximum(np.floor(bounds.min(axis=0)).astype(int) - 2, 0)
    right, bottom = np.minimum(np.ceil(bounds.max(axis=0)).astype(int) + 3, [image_width, height])
    if right <= left or bottom <= top:
        return
    inverse[:, 2] -= [left, top]
    delta = (restored.astype(np.float32) - crop.astype(np.float32)) * mask
    delta = cv2.warpAffine(delta, inverse, (right - left, bottom - top), flags=cv2.INTER_LINEAR)
    region = source[top:bottom, left:right]
    region[:] = np.clip(np.rint(region.astype(np.float32) + delta), 0, 255).astype(np.uint8)


class CodeFormerMouth:
    def __init__(self, model_name="codeformer.pth", detector_name="models/buffalo_l/det_10g.onnx",
                 fidelity=.9, strength=.7, batch_size=4):
        # These packages are optional when the workflow's mouth toggle is off.
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD
        from spandrel_extra_arches.architectures.CodeFormer import CodeFormerArch

        checkpoint = model_path("facerestore_models", model_name)
        detector_path = model_path("insightface", detector_name)
        self.device = mm.get_torch_device()
        self.fidelity, self.strength, self.batch_size = fidelity, strength, batch_size
        state = comfy.utils.load_torch_file(checkpoint, safe_load=True)
        state = state["params_ema"] if "params_ema" in state else state
        model = CodeFormerArch().load(state).eval().model
        self.patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=self.device,
                                                           offload_device=mm.unet_offload_device())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = "CUDAExecutionProvider" if self.device.type == "cuda" else "CPUExecutionProvider"
        providers = [(provider, {"device_id": self.device.index or 0, "cudnn_conv_algo_search": "HEURISTIC"})] if self.device.type == "cuda" else [provider]
        if self.device.type == "cuda":
            mm.free_memory(1024 ** 3, self.device)
        session = ort.InferenceSession(detector_path, sess_options=options, providers=providers)
        if session.get_providers()[0] != provider:
            raise RuntimeError("CodeFormer face detection could not use CUDA. Install a matching onnxruntime-gpu package, or turn Mouth Enhancement off.")
        self.detector = SCRFD(model_file=detector_path, session=session)
        self.detector.prepare(ctx_id=0 if self.device.type == "cuda" else -1,
                              input_size=(640, 640), det_thresh=.5)
        self.report = {"enabled": True, "model": model_name, "fidelity": fidelity, "mouth_blend": strength,
                       "restoration_device": str(self.device), "restoration_dtype": "float32",
                       "detector": detector_name, "detector_provider": provider,
                       "frames": 0, "faces": 0, "detection_seconds": 0., "restoration_seconds": 0.,
                       "compositing_seconds": 0.}

    def process(self, frames):
        """Consume and return owned RGB uint8 CPU frames, with bounded face batches."""
        if self.strength == 0:
            return frames
        pixels = frames.numpy()
        loaded = False
        for first in range(0, len(pixels), self.batch_size):
            mm.throw_exception_if_processing_interrupted()
            crops, transforms, centers, widths, indices = [], [], [], [], []
            start = time.perf_counter()
            for index in range(first, min(first + self.batch_size, len(pixels))):
                boxes, landmarks = self.detector.detect(np.ascontiguousarray(pixels[index, ..., ::-1]))
                self.report["frames"] += 1
                if not len(boxes):
                    continue
                face = np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
                points = landmarks[face].astype(np.float32)
                transform = cv2.estimateAffinePartial2D(points, TEMPLATE, method=cv2.LMEDS)[0].astype(np.float32)
                aligned = cv2.transform(points[None], transform)[0]
                crops.append(cv2.warpAffine(pixels[index], transform, (512, 512), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REFLECT_101))
                transforms.append(transform)
                centers.append(aligned[3:5].mean(axis=0))
                widths.append(np.linalg.norm(aligned[3] - aligned[4]))
                indices.append(index)
            self.report["detection_seconds"] += time.perf_counter() - start
            if not crops:
                continue
            start = time.perf_counter()
            # Same conservative activation estimate as ComfyUI's upscale path, per face.
            if not loaded:
                mm.load_models_gpu([self.patcher], memory_required=512 * 512 * 3 * 4 * 384 * self.batch_size, force_full_load=True)
                loaded = True
            batch = torch.from_numpy(np.stack(crops)).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2).div_(127.5).sub_(1)
            restored = self.patcher.model(batch, weight=self.fidelity)[0]
            restored = restored.clamp(-1, 1).add(1).mul(127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()
            self.report["restoration_seconds"] += time.perf_counter() - start
            self.report["faces"] += len(crops)
            start = time.perf_counter()
            for i, index in enumerate(indices):
                blend_mouth(pixels[index], crops[i], restored[i], transforms[i], centers[i], widths[i], self.strength)
            self.report["compositing_seconds"] += time.perf_counter() - start
        return frames
