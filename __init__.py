import torch


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


NODE_CLASS_MAPPINGS = {
    "PrependAudioSilence": PrependAudioSilence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrependAudioSilence": "Prepend Audio Silence",
}
