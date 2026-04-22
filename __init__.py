import math

import torch

from .video_outpaint import (
    VideoOutpaintPrepareCanvasByPadding,
    VideoOutpaintRegionCrop,
    VideoOutpaintRegionCropAdvanced,
    VideoOutpaintReplicateCanvas,
)


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


NODE_CLASS_MAPPINGS = {
    "PrependAudioSilence": PrependAudioSilence,
    "LTXFramesFromAudio": LTXFramesFromAudio,
    "VideoOutpaintPrepareCanvasByPadding": VideoOutpaintPrepareCanvasByPadding,
    "VideoOutpaintReplicateCanvas": VideoOutpaintReplicateCanvas,
    "VideoOutpaintRegionCrop": VideoOutpaintRegionCrop,
    "VideoOutpaintRegionCropAdvanced": VideoOutpaintRegionCropAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrependAudioSilence": "Prepend Audio Silence",
    "LTXFramesFromAudio": "LTX Frames From Audio",
    "VideoOutpaintPrepareCanvasByPadding": "Video Outpaint Prepare Canvas By Padding",
    "VideoOutpaintReplicateCanvas": "Video Outpaint Replicate Canvas",
    "VideoOutpaintRegionCrop": "Video Outpaint Region Crop",
    "VideoOutpaintRegionCropAdvanced": "Video Outpaint Region Crop Advanced",
}
