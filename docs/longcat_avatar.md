# LongCat-Video-Avatar 1.6 nodes

Audio + an optional photo + an optional scene prompt to a lip-synced talking or singing video of any length, in ComfyUI and
(through the LongCatAvatar extension) in SwarmUI. A native port of the official LongCat-Video-Avatar 1.6 inference
(meituan-longcat/LongCat-Video, commit 58eb336, `--model_type avatar-v1.6 --use_distill --mix_steps 4 --mix_alpha 1.0`).

## Nodes

| Node | What it does |
|---|---|
| `SELongCatAvatarLoader` | Loads the transformer (`LongCat_Video_Avatar_1.6_INT8_ConvRot.safetensors`, default; BF16 optional) as a normal ComfyUI model: low-VRAM offload, dynamic VRAM and LoRA loaders work with it. |
| `SELongCatWhisperLoader` | Loads only the encoder of `LongCat_Whisper_Large_V3_fp16.safetensors` (the official whisper-large-v3 file) as an AUDIO_ENCODER. |
| `SELongCatAvatarInputs` | Joins the SECourses prompt box (gallery), the soundtrack and the optional photo; folder batches pair `name.txt` with `name.png/jpg` and `name.wav/mp3`; adds "speaking/singing" when the prompt does not say it. |
| `SELongCatAvatarSampler` | The whole generation: vocal separation, Whisper features, 93-frame segments with continuation, the 8-step DMD/OPSD recipe, VAE decode, optional CodeFormer mouth enhancement, streaming H.264 export with the original audio, a JSON report. |
| `SELongCatAvatarPreview` | Live sampling previews, then the saved video. |

## Official recipe (reproduced)

- Audio: Kim_Vocal_2 vocal separation (official, always on in the official scripts; switchable here), then Whisper large-v3
  features in 30 s chunks, hidden states averaged in groups [0:8], [8:16], [16:24], [24:32] plus the final layer, resampled
  from 50 Hz to 25 fps; a 5-frame window per video frame. Loudness normalization is off for 1.6 (it was on for 1.5).
- Text: UMT5-XXL (the Wan 2.1 encoder; CLIP Loader type `wan`), 512 tokens, padded tokens zeroed after the text projection.
- VAE: Wan 2.1.
- Sampling: 8 flow-matching Euler steps, sigmas `1.0, 0.97998, 0.95449, 0.92094, 0.87478, 0.80728, 0.69916, 0.49799, 0`
  (the 1.5 distill grid with shift 7), timesteps rounded to the model dtype as in the official forward; guidance 1 (one pass
  per step); the DMD LoRA (loraA) for steps 1-4, the OPSD LoRA (loraB) for steps 5-8, applied at run time like the official
  `multi_lora_forward`, so they work on BF16 and INT8 weights.
- Video: 25 fps, 93 frames per segment; each next segment re-encodes the last 13 decoded frames as 4 condition latents and
  prepends the first segment's first latent as an identity reference at time position 10; 80 new frames per segment.
- Size: the official 480p / 720p aspect buckets nearest to the photo, or a custom size (multiple of 16).
- The 50-step base mode with text and audio guidance (official 4 / 4) is available with the 8-step switch off.

## Speed and memory

- INT8 ConvRot (default model): native ComfyUI int8 with ConvRot, block linears quantized, edge blocks kept BF16.
- Condition cache (default on): the reference and previous-frame tokens do not depend on the noisy frames, so their keys and
  values are computed once per LoRA phase and reused (exact). It switches itself off when it does not fit in free VRAM.
- Frame-chunked residual updates bound the temporary memory at large sizes (1280 x 1920 works on a 32 GB card).
- Torch compile is optional. Attention uses ComfyUI's backend (SageAttention with our launchers); `int8 (comfy kitchen)`
  runs the long full-context self-attention with comfy-kitchen's INT8 kernel.
- Premerged model (`LongCat_Video_Avatar_1.6_Premerged_DMD_OPSD_INT8_ConvRot.safetensors`): the DMD LoRA merged into the
  INT8 ConvRot weights, plus `opsd_phase.*` tensors that XOR the LoRA-targeted fields into the OPSD merge (bit-exact,
  built from both merged checkpoints). During the OPSD steps each of those layers flips the weight ComfyUI cast for the
  call and flips it back after the matmul (quantized `Linear._forward`), so the stored model is never written: it works
  with dynamic VRAM, offloading and torch.compile. No runtime LoRA work; distilled mode with Mix Alpha 1 only.
- Distilled steps 8 / 4 / 3 sample positions `0-7`, `0,2,4,6` or `0,3,6` of the official 8-step grid (DMD below Mix
  Steps). Shorten last segment samples the final segment only as long as the remaining audio needs.

## Tests

`dont_upload.../LongCat_Video_Avatar_20261117/scripts/`: `check_keys.py` (all 1608 official tensors load with matching
shapes), `whisper_parity.py` (cosine 1.000000 against the official HF fp32 features), `cpu_unit_test.py` (condition cache
and chunking equal the plain path), `vocal_sep/selftest_longcat_vocal_separator.py` (83-87 dB SNR against audio-separator).
