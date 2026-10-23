# Fibo 2.0 presets and adapters

Bria Fibo 2.0 is an 8B flow-matching DiT that generates and edits images: text to image, instruction edits, masked edits and multi-reference edits with up to 4 images. It reads structured JSON captions through the Qwen3-VL 4B text encoder and works in the FLUX.2 VAE latent space. ComfyUI supports it natively (`CLIPLoader` type `fibo2`, `TextEncodeFibo2`). These adapters connect it to the SECourses gallery prompt and make the encoding and sampling match Bria's own pipeline. No ComfyUI core code is modified.

- `SEFibo2GalleryEncode` (**Fibo 2.0 Gallery Encode**) encodes with the native `TextEncodeFibo2`: the gallery prompt (JSON caption or plain text) and its images, plus the optional edit canvas. An empty gallery is text to image; images become the edit images and their reference latents, in order (`@image1` or `<image_1>` in the prompt; `@init` is the canvas image). Plain prompts get seeded `{a|b}` choices and `(word:1.2)` weights are removed (Qwen3-VL would read them as text); JSON is kept as written. **Negative prompt:** ComfyUI encodes it as written in Bria's chat template, like Bria's pipeline (#17285, 2026-10-22; older ComfyUI cleaned a JSON negative first, 163 instead of 181 tokens for the model card's `NEGATIVE_T2I`). A JSON negative is made compact first, so a pretty-printed copy encodes like the original; the node no longer wraps it itself, which doubled the template on current ComfyUI. Empty is the null caption Bria uses for edits. **Edit size** (default Auto): an edit keeps the first image's own size from 1 to 4 MP in multiples of 16 (a 2 MP photo is edited at 2 MP), larger images are scaled down to 4 MP and smaller ones go to Bria's trained ~1 MP size for their aspect ratio. "Bria default" edits at the trained ~1 MP size like the native node; "Always use the resolution inputs" center-crops a single image to the set resolution. Text to image always uses the resolution inputs.
- `SEFibo2Sampling` (**Fibo 2.0 Official Sigma Shift**): Bria's pipeline uses FLUX's resolution-dependent shift (0.5 at 256 image tokens to 1.15 at 4096, so 1.15 at 1024x1024 and about 2 at 1536x1536), while ComfyUI's model uses 1.15 at every size. The node applies it with the native `ModelSamplingFlux` and moves the caption's text-image conditioning to start at sigma 0.9 as in Bria's pipeline. "ComfyUI default" leaves the model unchanged.
- `SEFibo2EditCanvas` (**Fibo 2.0 Optional Edit Image + Mask**): an optional image for a masked edit, painted with ComfyUI's native Mask Editor ("Paint mask (native editor)" button). Like Bria's training, the masked area (mask above 50%) is filled with flat grey (128) at the image's own size before encoding. A painted mask edits this image alone, so gallery images are refused with a clear message. `(none - disabled)` turns the canvas off.
- `SEFibo2RestoreUnmasked` (**Fibo 2.0 Keep Unmasked Pixels**): after a masked edit, copies every pixel outside the mask back from the source (soft edges blend). Other generations pass through.
- `SEFibo2SwarmInputs` (**Fibo 2.0 SwarmUI Inputs**): turns the SwarmUI prompt images, Init Image and mask (base64) into the gallery pack, so the [Fibo2 SwarmUI extension](https://github.com/FurkanGozukara/SwarmUI_Premium_Extensions/tree/main/Fibo2) uses the same encoder.
- `web/edit_gallery.js`: the presets' gallery has the property `fibo2_images_only` (images only, prompt sent as typed, "1 image to edit (up to 4 recommended)" counter, video controls hidden; the MOSS Image Edit presets share this profile). `web/fibo2.js`: the edit canvas gets the Mask Editor and "Clear image + mask" buttons.

## Presets

`Presets/Fibo 2.0 * - 261022.json`, also under `user/default/workflows/SECourses`. All use `qwen3vl_4b_int8_convrot`, `Flux/flux2-vae`, the gallery encode, the official sigma shift, `SwarmKSampler` (Euler, simple), an optional LoRA stack (off by default), the optional masked-edit canvas and the shared seed and images-per-prompt controls. Every file is in the downloader's Fibo 2.0 Core Bundle; the loaders also accept the BF16 files.

| Preset | Recipe | RTX 5090, warm |
| --- | --- | --- |
| Dev - Advanced Text To Image | Dev INT8, official: 50 steps, CFG 5, Bria's JSON negative (`NEGATIVE_T2I`) | 17 s at 1024x1024, 35 s at 1664x1248 |
| Dev - Advanced Image Edit | Dev INT8, official: 50 steps, CFG 5, empty negative (null caption) | 35 s at 1 MP |
| Turbo - Image Edit 4 Steps | Edit Turbo INT8: 4 steps, CFG 1 | 2.5 s at 1 MP, 5 s at 2 MP |
| Turbo - Text To Image 4 Steps | Edit Turbo INT8: 4 steps, CFG 1 | 1.5 s at 1024x1024 |

Notes from the tests:

- Edit size: with the native auto size every edit runs at ~1 MP; a 2 MP photo lost detail and was over-edited. At its own size 2 MP edits were correct and sharp with both models, and 3 to 4 MP stayed coherent (Turbo about 11 s at 4 MP).
- Attention: the INT8 ConvRot files request Comfy Kitchen INT8 attention in 40 layers; on the RTX 5090 every attention backend gave equivalent images, so the presets keep the files' setting. BF16 with `--use-sage-attention` also rendered correctly.
- Dev with Comfy-Org's turbo LoRA (`fibo_2.0_turbo_lora_avg_rank_270_bf16`, strength 1, 4 steps, CFG 1) through the LoRA stack was comparable to Edit Turbo; it is an optional download.
- Prompts: JSON captions in Bria's VGL format give the most control; `Automatic_Prompt_Generate_Enhance_To_Video_And_Image_Models_LLM_Instructions/Fibo_2.0_Enchance_Prompt_Feed_For_LLMs.txt` makes an LLM write them. Fibo 2.0 is licensed CC BY-NC 4.0.

## Validation, 2026-10-22

ComfyUI `1a82e6d0` (0.42.0), frontend 1.57.1, Torch 2.13.0+cu130, physical GPU 0 (RTX 5090).

- 15 tests: `python -m pytest -q custom_nodes/SECoursesAudioTools/tests/test_fibo2.py -p no:cacheprovider` from the ComfyUI root. The JSON negative's tokens equal the Hugging Face tokenizer's for Bria's template, prompt labels and choices, edit sizes, the grey-out, the shift and text-image remap and the unmasked restore are covered.
- The four presets were opened from the Workflows sidebar of ComfyUI started with `Windows_Run_ComfyUI_Default.bat` and run with Run: Dev text to image 17.1 s warm (33.9 s cold), Turbo text to image 7.0 s including the model switch, Turbo edit 2.6 s, Dev edit 39.9 s including the switch back. A masked edit (photo in node 06, mask painted in the native Mask Editor, empty gallery) put a knitted hat inside the mask and kept the rest of the photo (2.2 s). Same-seed outputs of all four presets were pixel-identical to the SwarmUI Fibo2 extension's.
- Earlier API runs: up to 4 MP edits, two references, masked edits with the restore on and off, the LoRA stack with the turbo LoRA (no key warnings) and the BF16 files.
