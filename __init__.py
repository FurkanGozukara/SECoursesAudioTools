import gc
import math
import os
import threading

import torch

from .video_outpaint import (
    VideoOutpaintPrepareCanvasByPadding,
    VideoOutpaintRegionCrop,
    VideoOutpaintRegionCropAdvanced,
    VideoOutpaintReplicateCanvas,
)


class SEAnyType(str):
    def __ne__(self, __value):
        return False


SE_ANY = SEAnyType("*")
VIDEO_EXTENSIONS = {"webm", "mp4", "mkv", "gif", "mov", "avi", "m4v"}


def _input_videos():
    import folder_paths

    input_dir = folder_paths.get_input_directory()
    try:
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1].lower() in VIDEO_EXTENSIONS
        ]
        return sorted(files)
    except FileNotFoundError:
        return []


def _fraction_to_float(value, fallback=0.0):
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _padded_ltx_frame_count(frame_count):
    frame_count = max(1, int(frame_count))
    return int(1 + math.ceil((frame_count - 1) / 8) * 8)


_STREAMING_LAST_FRAME_CACHE = {}
_STREAMING_LAST_FRAME_LOCK = threading.Lock()


def _coerce_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _streaming_last_frame_active(enabled, streaming_chunk_seconds):
    try:
        chunk_seconds = float(streaming_chunk_seconds)
    except (TypeError, ValueError):
        chunk_seconds = 0.0
    return _coerce_bool(enabled) and chunk_seconds > 0.0


def _streaming_cache_key(meta_batch=None):
    meta_id = getattr(meta_batch, "unique_id", None)
    if meta_id is not None:
        return f"meta_batch:{meta_id}"
    return "meta_batch:default"


def _prompt_requeue_index(prompt=None, meta_batch=None):
    if not isinstance(prompt, dict):
        return 0

    meta_id = getattr(meta_batch, "unique_id", None)
    if meta_id is not None:
        node = prompt.get(str(meta_id)) or prompt.get(meta_id)
        if isinstance(node, dict):
            try:
                return int(node.get("inputs", {}).get("requeue", 0))
            except (TypeError, ValueError):
                return 0

    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") == "VHS_BatchManager":
            try:
                return int(node.get("inputs", {}).get("requeue", 0))
            except (TypeError, ValueError):
                return 0
    return 0


def _clear_streaming_last_frame(key):
    with _STREAMING_LAST_FRAME_LOCK:
        _STREAMING_LAST_FRAME_CACHE.pop(key, None)


def _get_streaming_last_frame(key):
    with _STREAMING_LAST_FRAME_LOCK:
        frame = _STREAMING_LAST_FRAME_CACHE.get(key)
    return frame.clone() if frame is not None else None


def _store_streaming_last_frame(key, images):
    if images is None or not hasattr(images, "dim") or images.dim() < 4 or images.shape[0] <= 0:
        return False
    last_frame = images[-1:].detach().cpu().clone()
    with _STREAMING_LAST_FRAME_LOCK:
        _STREAMING_LAST_FRAME_CACHE[key] = last_frame
    return True


class PrependAudioSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "silence_seconds": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "prepend"
    CATEGORY = "audio"

    def prepend(self, audio, silence_seconds):
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))

        if waveform is None or sample_rate <= 0:
            raise ValueError("Invalid AUDIO input. Expected waveform and sample_rate.")

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 3:
            raise ValueError("Unsupported waveform shape. Expected [batch, channels, samples].")

        silence_samples = max(0, int(round(float(silence_seconds) * sample_rate)))
        if silence_samples == 0:
            return (audio,)

        silence = torch.zeros(
            (waveform.shape[0], waveform.shape[1], silence_samples),
            dtype=waveform.dtype,
            device=waveform.device,
        )

        out_audio = dict(audio)
        out_audio["waveform"] = torch.cat((silence, waveform), dim=2)
        out_audio["sample_rate"] = sample_rate
        return (out_audio,)


class LTXFramesFromAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "duration_seconds")
    FUNCTION = "calculate"
    CATEGORY = "audio"

    def calculate(self, audio, fps):
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))

        if waveform is None or sample_rate <= 0:
            raise ValueError("Invalid AUDIO input. Expected waveform and sample_rate.")

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 3:
            raise ValueError("Unsupported waveform shape. Expected [batch, channels, samples].")

        duration_seconds = float(waveform.shape[-1]) / float(sample_rate)
        raw_frames = duration_seconds * float(fps)

        # LTX expects frame counts that satisfy 4n + 1.
        frames = int(4 * math.ceil(max(0.0, (raw_frames - 1.0) / 4.0)) + 1)
        frames = max(1, frames)

        return (frames, duration_seconds)


class SEOptionalMetaBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN",),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
        }

    RETURN_TYPES = ("VHS_BatchManager",)
    RETURN_NAMES = ("meta_batch",)
    FUNCTION = "select"
    CATEGORY = "video"

    def select(self, enabled, meta_batch=None):
        return (meta_batch if enabled else None,)


class SEMetaBatchBypassGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN",),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "passthrough"
    CATEGORY = "video"

    def passthrough(self, image, enabled, meta_batch=None):
        if meta_batch is not None and not enabled:
            meta_batch.close_inputs()
            meta_batch.has_closed_inputs = True
        return (image,)


class SEStreamingLastFrameReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": False}),
                "streaming_chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.001,
                    },
                ),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select"
    CATEGORY = "video"

    def select(
        self,
        image,
        enabled=False,
        streaming_chunk_seconds=0.0,
        meta_batch=None,
        prompt=None,
        unique_id=None,
    ):
        key = _streaming_cache_key(meta_batch)
        requeue = _prompt_requeue_index(prompt, meta_batch)
        if not _streaming_last_frame_active(enabled, streaming_chunk_seconds):
            _clear_streaming_last_frame(key)
            return (image,)

        if requeue <= 0:
            _clear_streaming_last_frame(key)
            print("[SECoursesAudioTools] streaming last-frame reference: first chunk uses input image")
            return (image,)

        cached = _get_streaming_last_frame(key)
        if cached is None:
            print(
                "[SECoursesAudioTools] streaming last-frame reference: "
                "no cached frame found; using input image"
            )
            return (image,)

        if hasattr(image, "device"):
            cached = cached.to(device=image.device, dtype=image.dtype)
        print(
            "[SECoursesAudioTools] streaming last-frame reference: "
            f"using previous chunk last frame for requeue {requeue}"
        )
        return (cached,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class SEStreamingLastFrameRecorder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": False}),
                "streaming_chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.001,
                    },
                ),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "record"
    CATEGORY = "video"

    def record(
        self,
        images,
        enabled=False,
        streaming_chunk_seconds=0.0,
        meta_batch=None,
        prompt=None,
        unique_id=None,
    ):
        key = _streaming_cache_key(meta_batch)
        if not _streaming_last_frame_active(enabled, streaming_chunk_seconds):
            _clear_streaming_last_frame(key)
            return (images,)

        if _prompt_requeue_index(prompt, meta_batch) <= 0:
            _clear_streaming_last_frame(key)

        if _store_streaming_last_frame(key, images):
            print("[SECoursesAudioTools] streaming last-frame recorder: stored chunk last frame")
        return (images,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class SEAutoStreamingChunkSeconds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps_override": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                    },
                ),
                "requested_chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.001,
                    },
                ),
                "reencode_chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 100000.0,
                        "step": 0.1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("effective_chunk_seconds",)
    FUNCTION = "resolve"
    CATEGORY = "video"

    def resolve(self, fps_override, requested_chunk_seconds, reencode_chunk_seconds):
        requested = float(requested_chunk_seconds)
        print(
            "[SECoursesAudioTools] streaming chunk resolver: "
            f"fps_override={float(fps_override):g}, requested={requested:g}, effective={requested:g}"
        )
        return (requested,)


class SEMemoryFlushImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "empty_cache": ("BOOLEAN", {"default": True}),
                "gc_collect": ("BOOLEAN", {"default": True}),
                "unload_all_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "dependency_1": (SE_ANY,),
                "dependency_2": (SE_ANY,),
                "dependency_3": (SE_ANY,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "flush"
    CATEGORY = "video"

    def flush(
        self,
        image,
        empty_cache=True,
        gc_collect=True,
        unload_all_models=True,
        dependency_1=None,
        dependency_2=None,
        dependency_3=None,
    ):
        before = ""
        if torch.cuda.is_available():
            before = (
                f" cuda_alloc={torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                f" cuda_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            )
        print("[SECoursesAudioTools] memory flush before LTX load:" + before)

        dependency_1 = None
        dependency_2 = None
        dependency_3 = None

        if unload_all_models:
            import comfy.model_management

            comfy.model_management.unload_all_models()
        if gc_collect:
            gc.collect()
        if empty_cache:
            import comfy.model_management

            comfy.model_management.soft_empty_cache()
        after = ""
        if torch.cuda.is_available():
            after = (
                f" cuda_alloc={torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                f" cuda_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            )
        print("[SECoursesAudioTools] memory flush after LTX load gate:" + after)
        return (image,)


class SELowVRAMAudioVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Audio VAE checkpoint to load."},
                ),
            },
            "optional": {
                "dependencies": (
                    SE_ANY,
                    {"tooltip": "Connect any previous stage output to force sequential loading."},
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "load_audio_vae"
    CATEGORY = "audio"

    def load_audio_vae(self, ckpt_name, dependencies=None):
        import comfy.sd
        import comfy.utils
        import folder_paths

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        sd = comfy.utils.state_dict_prefix_replace(
            sd,
            {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
            filter_keys=True,
        )
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)


class SELoadVideoWithPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_input_videos(),),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_name", "video_path")
    FUNCTION = "load"
    CATEGORY = "video"

    def load(self, video):
        import folder_paths
        from comfy_api.latest import InputImpl

        video_path = folder_paths.get_annotated_filepath(video)
        return (InputImpl.VideoFromFile(video_path), video, video_path)

    @classmethod
    def IS_CHANGED(cls, video):
        import folder_paths

        video_path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        import folder_paths

        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


class SEVideoPathFromVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "get_path"
    CATEGORY = "video"

    def get_path(self, video):
        if not hasattr(video, "get_stream_source"):
            raise ValueError("VIDEO input does not expose a stream source.")
        source = video.get_stream_source()
        if not isinstance(source, str):
            raise ValueError("VIDEO source is not a file path; streaming path mode requires a file-backed video.")
        if not os.path.isfile(source):
            raise ValueError(f"VIDEO source path does not exist: {source}")
        return (source,)

    @classmethod
    def IS_CHANGED(cls, video):
        if not hasattr(video, "get_stream_source"):
            return float("nan")
        source = video.get_stream_source()
        if not isinstance(source, str) or not os.path.isfile(source):
            return float("nan")
        return os.path.getmtime(source)


class SEVideoStreamingInfoPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "fps_override": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                    },
                ),
                "source_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.001,
                    },
                ),
                "streaming_chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.001,
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "resolved_fps",
        "source_fps",
        "effective_frame_count",
        "effective_duration",
        "source_width",
        "source_height",
        "frame_load_cap",
        "frames_per_batch",
        "source_duration",
    )
    FUNCTION = "get_info"
    CATEGORY = "video"

    def get_info(
        self,
        video_path,
        fps_override=0.0,
        source_duration_seconds=0.0,
        streaming_chunk_seconds=10.0,
    ):
        import av

        video_path = str(video_path).strip().strip('"')
        if not video_path or not os.path.isfile(video_path):
            raise ValueError(f"Invalid video_path: {video_path}")

        with av.open(video_path, mode="r") as container:
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            if video_stream is None:
                raise ValueError(f"No video stream found in {video_path}")

            source_fps = _fraction_to_float(video_stream.average_rate, 0.0)
            if source_fps <= 0:
                source_fps = 1.0

            if video_stream.duration is not None and video_stream.time_base is not None:
                source_duration = float(video_stream.duration * video_stream.time_base)
            elif container.duration is not None:
                source_duration = float(container.duration / av.time_base)
            elif video_stream.frames:
                source_duration = float(video_stream.frames / source_fps)
            else:
                raise ValueError(f"Could not determine duration for {video_path}")

            source_width = int(video_stream.width)
            source_height = int(video_stream.height)
            source_frame_count = int(video_stream.frames or round(source_duration * source_fps))

        resolved_fps = float(fps_override) if float(fps_override) > 0 else source_fps
        duration_limit = float(source_duration_seconds)
        if duration_limit > 0:
            effective_duration = min(duration_limit, source_duration)
            frame_load_cap = max(1, int(round(effective_duration * resolved_fps)))
        else:
            effective_duration = source_duration
            frame_load_cap = 0

        if frame_load_cap > 0:
            effective_frame_count = frame_load_cap
        elif float(fps_override) > 0:
            effective_frame_count = max(1, int(round(effective_duration * resolved_fps)))
        else:
            effective_frame_count = max(1, source_frame_count)

        if float(streaming_chunk_seconds) > 0:
            raw_chunk_frames = max(1, int(round(float(streaming_chunk_seconds) * resolved_fps)))
            frames_per_batch = min(
                effective_frame_count,
                _padded_ltx_frame_count(raw_chunk_frames),
            )
        else:
            frames_per_batch = effective_frame_count

        frames_per_batch = max(1, int(frames_per_batch))

        return (
            float(resolved_fps),
            float(source_fps),
            int(effective_frame_count),
            float(effective_duration),
            int(source_width),
            int(source_height),
            int(frame_load_cap),
            frames_per_batch,
            float(source_duration),
        )

    @classmethod
    def IS_CHANGED(cls, video_path, **kwargs):
        video_path = str(video_path).strip().strip('"')
        if not video_path or not os.path.isfile(video_path):
            return float("nan")
        return os.path.getmtime(video_path)


NODE_CLASS_MAPPINGS = {
    "PrependAudioSilence": PrependAudioSilence,
    "LTXFramesFromAudio": LTXFramesFromAudio,
    "SEOptionalMetaBatch": SEOptionalMetaBatch,
    "SEMetaBatchBypassGate": SEMetaBatchBypassGate,
    "SEStreamingLastFrameReferenceImage": SEStreamingLastFrameReferenceImage,
    "SEStreamingLastFrameRecorder": SEStreamingLastFrameRecorder,
    "SEAutoStreamingChunkSeconds": SEAutoStreamingChunkSeconds,
    "SEMemoryFlushImage": SEMemoryFlushImage,
    "SELowVRAMAudioVAELoader": SELowVRAMAudioVAELoader,
    "SELoadVideoWithPath": SELoadVideoWithPath,
    "SEVideoPathFromVideo": SEVideoPathFromVideo,
    "SEVideoStreamingInfoPath": SEVideoStreamingInfoPath,
    "VideoOutpaintPrepareCanvasByPadding": VideoOutpaintPrepareCanvasByPadding,
    "VideoOutpaintReplicateCanvas": VideoOutpaintReplicateCanvas,
    "VideoOutpaintRegionCrop": VideoOutpaintRegionCrop,
    "VideoOutpaintRegionCropAdvanced": VideoOutpaintRegionCropAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrependAudioSilence": "Prepend Audio Silence",
    "LTXFramesFromAudio": "LTX Frames From Audio",
    "SEOptionalMetaBatch": "SE Optional Meta Batch",
    "SEMetaBatchBypassGate": "SE Meta Batch Bypass Gate",
    "SEStreamingLastFrameReferenceImage": "SE Streaming Last Frame Reference Image",
    "SEStreamingLastFrameRecorder": "SE Streaming Last Frame Recorder",
    "SEAutoStreamingChunkSeconds": "SE Auto Streaming Chunk Seconds",
    "SEMemoryFlushImage": "SE Memory Flush Image",
    "SELowVRAMAudioVAELoader": "SE Low VRAM Audio VAE Loader",
    "SELoadVideoWithPath": "SE Load Video With Path",
    "SEVideoPathFromVideo": "SE Video Path From Video",
    "SEVideoStreamingInfoPath": "SE Video Streaming Info Path",
    "VideoOutpaintPrepareCanvasByPadding": "Video Outpaint Prepare Canvas By Padding",
    "VideoOutpaintReplicateCanvas": "Video Outpaint Replicate Canvas",
    "VideoOutpaintRegionCrop": "Video Outpaint Region Crop",
    "VideoOutpaintRegionCropAdvanced": "Video Outpaint Region Crop Advanced",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
