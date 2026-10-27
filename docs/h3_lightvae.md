# Optional H3 LightVAE decoder

`SEH3VideoDecoder` selects **Original video VAE** or an installed H3 LightVAE
decoder. It replaces final video decoding only. Image/video reference encoding
keeps the original H3 video VAE, and audio keeps its own VAE. The original
decoder is a one-click comparison; no graph rewiring is needed.

The FastH3 Omni LightVAE preset uses
`LightVAE/lightvae-lite-h3-decoder-int8-convrot.safetensors`, included in the
FastH3 Omni and MiniMax-H3 core bundles. The catalog also offers the original
FP32 Comfy decoder and encoder. Their safetensors headers were verified as F32;
there is no BF16 variant of these files at the audited source revision.

LightVAE is a block-pruned reconstruction alternative. It changes the decoded
frames, not the sampled latent. Compare fine detail and sharpness for your shot;
the preset does not promise universal quality or end-to-end speed gains.
The new variant enables exact Low VRAM mode and retains VSA, gallery references,
prompt/folder batches, seed/resolution and optional user LoRAs. It requires
ComfyUI's native LightVAE support (tested with v0.43.0).

`SEH3LongLipSync` also accepts an optional `decode_vae`; its original `video_vae`
still encodes references. The premium MiniMaxH3References extension wires the
decoder-only option for SwarmUI, including this long-form path.

Validated on physical GPU 0 (RTX 5090): real Chrome sidebar/Run generated a
1344x768, 124-frame, 24fps MP4 with 32kHz stereo audio. Native frames were
inspected against a same-seed original-decoder render. Audio listening was not
part of this preset validation.

Source: [LightVAE weights and model card](https://huggingface.co/lightx2v/LightVAE).
