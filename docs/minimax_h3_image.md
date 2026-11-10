# MiniMax H3 Image presets and adapters

MiniMax H3 Image is a 33B diffusion transformer of the MiniMax H3 family (H3 without its audio layers), distilled to 6 steps at CFG 1. One checkpoint generates images from text and edits up to 16 reference images; it works best around 16 MP. ComfyUI supports it natively (`MiniMaxH3ImageConditioning`, ComfyUI v0.45.0, #17655) with the MiniMax H3 Qwen3-VL 32B encoder (`CLIPLoader` type `minimax`) and the MiniMax H3 Image VAE. These adapters connect it to the SECourses gallery prompt and add MiniMax's own prompt enhancer. No ComfyUI core code is modified.

- `SEH3ImagePromptEnhance` (**MiniMax H3 Image Prompt Enhance (Gallery)**): prepares the gallery prompt (seeded `{a|b}` choices, `(word:1.2)` weights removed, `@image1` -> `<Picture 1>`) and, with the enhancer On, runs MiniMax's official prompt enhancer on a local Qwen3.5 9B through the native Generate Text node. The instructions are MiniMaxAI/MiniMax-H3-Image-PE's, unchanged, in `assets/minimax_h3_image_pe` (SHA-256 as in its `prompt/manifest.json`). Text to image uses its system and task prompt; edits use its edit task prompt with the message assembled as its README describes (each image's label and source ratio right before the image, then the instruction and `adaptive` ratio), passed as raw chat text so the tokenizer keeps that order. Sampling follows `request_params.json` (temperature 1, top-p 0.95, top-k 20); up to 8192 tokens as in the ComfyUI templates. Modes: official with reasoning (default), fast without reasoning, Off. The enhancer model is a lazy input, so Off never loads it. If no caption can be read from the answer, the prompt is used as written and the console says so. Outputs the prompt the model receives (shown by the preset's preview box) and the chosen aspect ratio inside the gallery pack.
- `SEH3ImageGalleryEncode` (**MiniMax H3 Image Gallery Encode**): calls the native `MiniMaxH3ImageConditioning` with the gallery images in card order (image 1 is the image edited). **Size** Auto (default): the enhancer's aspect ratio (Off: image 1's shape for edits, the width and height for text to image) at the megapixels of the width and height inputs, each side rounded to the nearest multiple of 32 as in MiniMax's size tables (16 MP: 1:1 4096x4096, 3:2 5024x3360, 16:9 5472x3072, 21:9 6272x2688); ratios are kept between 1:3 and 3:1. Exact: the width and height as given. `batch_size` repeats the latent (images per prompt).
- `web/edit_gallery.js`: gallery profile `h3_image_images` (images only, "n / 16 images" counter, prompt sent as typed). Image-only galleries (this profile, Fibo 2.0, MOSS Edit, FLUX.3, Qwen Image 2.1) now also hide the video-only "downscale videos" toggle.

## Presets

`Presets/MiniMax H3 Image - Text To Image 16 MP - 6 Steps - 261110.json` and `Presets/MiniMax H3 Image - Image Edit Up To 16 Images - 6 Steps - 261110.json`, also under `user/default/workflows/SECourses`. Both use `minimax_h3_image_pruned_int8_convrot` (the loader also takes the W4A8 and BF16 files), `qwen3vl_32b_minimax_h3_nvfp4_awq`, `MiniMaxH3/minimax_h3_image_vae_fp16`, `qwen3.5_9b_int8_convrot` for the enhancer, the official recipe (`SwarmKSampler`, 6 steps, CFG 1, Euler, simple, zeroed negative), 16 MP Auto size, the optional LoRA stack (off), the shared seed and images per prompt, and a preview of the prompt the model receives. Every file is in the downloader's MiniMax-H3 Core Bundle (Low VRAM Bundle: W4A8 and the W4A8 Qwen3.5).

| Size | RTX 5090, 32 GB | VRAM capped at 24 GB | 16 GB | 12 GB |
| --- | --- | --- | --- | --- |
| 1 MP (1024x1024) | 3.4 s | 4.6 s | 8.1 s | 8.6 s |
| 4 MP (2048x2048) | 5.7 s | 6.1 s | 8.7 s | 9.5 s |
| 16 MP (4096x4096) | 15.7 s | 17.1 s | 20.1 s | 19.5 s |

Seconds per image with the models loaded, including text encoding and decoding; the caps use ComfyUI's `--reserve-vram` on the same card (peaks 25.5, 18.0 and 13.1 GB), the rest streams from system RAM (96 GB here). The enhancer runs once per new prompt: about 60 s with reasoning (about 5,000 tokens at about 80 tokens/s), about 6 s in the fast mode.

## Validation, 2026-11-10

ComfyUI `989d93d0` (v0.45.0), frontend 1.58.5, physical GPU 0 (RTX 5090); remote probes on RTX A6000.

- 18 tests: `python -m pytest -q custom_nodes/SECoursesAudioTools/tests/test_minimax_h3_image.py -p no:cacheprovider` from the ComfyUI root (prompt labels and choices, MiniMax's size tables, answer parsing, the enhancer prompt files against the PE manifest, the edit message order, the fast mode, both nodes with the text generator mocked).
- Presets opened from the Workflows sidebar of ComfyUI started with `Windows_Run_ComfyUI_Default.bat` and run with Run: text to image with the default enhancer (3:2 chosen, 5024x3360); a one-image edit (restoration of a photo, 4672x3584); a four-image edit (room plus armchair, lamp and plant photos, 5024x3360); the fast enhancer with a two-prompt chain (both ran in order); W4A8 with the enhancer Off at 4 MP; the LoRA stack with an H3 LoRA at 0.2 (runs, no key warnings). The FLUX.3, MOSS Edit and Fibo 2.0 edit galleries still render and the FLUX.3 edit preset still runs.
- Encoder A/B (same seed, 4 MP): INT8 ConvRot and NVFP4 encoders gave equally clean images; the presets use NVFP4, already in the MiniMax-H3 bundles.
- Size A/B (same caption and seed): 16 MP shows clearly more real detail than 4 MP and 1 MP (fur, crumb, frost), so the presets default to 16 MP.
