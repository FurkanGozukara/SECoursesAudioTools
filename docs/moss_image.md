# MOSS-Image presets and adapters

MOSS-Image (OpenMOSS) is a 5B single-stream text-to-image DiT read by a Qwen3.5-9B text encoder through the FLUX.2 VAE. ComfyUI supports it natively (`CLIPLoader` type `moss_image`, `MOSSImageLayoutPrompt`, `CreateBoundingBoxes`). These adapters connect it to the SECourses gallery prompt and fix the details that otherwise change its output. No ComfyUI core code is modified.

- `SEMossImageGalleryPrompt` (**MOSS-Image Gallery Prompt**) reads the gallery prompt and rejects media cards with a clear message: the model does not use reference images. A complete JSON object is sent compactly and otherwise unchanged (official compiled layouts). A prompt-enhancer pixel draft (`width`, `height`, `bbox_coord_space: abs`, `bbox_format: xyxy`) is compiled with the official `compile_prompt.py` algorithm, scaled to the output size and checked; a draft without items becomes its caption. Plain prompts get seeded `{a|b}` choices, and `(word:1.2)` weights are removed because ComfyUI's Qwen3.5 tokenizer has weights disabled and would read them literally. Times and ratios such as `(10:30)` stay.
- `SEMossImageTextEncode` (**MOSS-Image Text Encode**) encodes JSON and plain text with the native `encode_from_tokens_scheduled`. Only real schedules (`[a:b:0.5]`, `[b:0.5]`) and alternation (`[a|b]`) go through Swarm's `SwarmClipTextEncodeAdvanced` with the sampler's step count; any other brackets stay literal.
- `SEMossImageSFTDeBand` (**MOSS-Image SFT De-Band (auto)**) applies [MATLOWAI's diff LoRA](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand) (layers 37-41 back to Base, optional 32-36 extra) only to the SFT model. SFT draws faint vertical color stripes on skies, fog, walls and backdrops; Base does not. The variant is recognized from the SHA-256 of `final_norm_video.weight`, which is identical in the INT8 and BF16 files. Base and unknown models pass through and the console logs `SFT de-band skipped`. A file whose name contains `sft` but holds the Base weights gets a warning: ComfyUI's official MOSS-Image bounding-box template (templates 0.11.89) downloads Base under the SFT name.
- `web/json_prompt_guard.js`: on queueing, the frontend expands `{a|b}` and strips `//` comments in every dynamic-prompt box, which corrupts JSON. The guard keeps a complete JSON object or array exactly as typed and, for a caption followed by a JSON table (FLUX.3), expands only the caption. `serializeValue` is an async chain, so the guard awaits the original.
- `web/moss_image.js`: the presets' gallery has the property `moss_text_only`, so its prompt box sends the text as typed, like the FLUX.3 profile. Without it the frontend picks `{a|b}` with `Math.random` before queueing; with it `SEMossImageGalleryPrompt` resolves choices with the shared seed in the main prompt, added prompts and folder prompts. `//` comments then reach the node as text, as in the FLUX.3 presets.

## Presets

`Presets/MOSS Image - * - 261017.json` (SFT) and `Presets/MOSS Image Base - * - 261018.json` (Base), also under `user/default/workflows/SECourses`. All use an INT8 ConvRot DiT, the INT8 Qwen3.5 encoder, `Flux/flux2-vae`, an explicit `ModelAttentionBackend` set to PyTorch, the de-band node, an optional LoRA stack and the shared gallery prompt/seed controls. The model loaders also accept the other variant and the BF16 files.

| Preset | Recipe | Warm 1024x1024, RTX 3090 |
| --- | --- | --- |
| Advanced Text To Image | Official: Euler/normal, 50 steps, CFG 3, shift 5, negative-branch skip of layer 22, official negative tags | 49.3 s |
| Fast 25 Steps | ER-SDE/simple, 25 steps, CFG 3, Temporal Score Rescaling 0.95, skip layers 22 and 24 from 10% to 70% | 24.1 s |
| Turbo 8 Steps | [lodestones 8-step DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) 0.7 plus de-band, Euler/simple, 8 steps, CFG 1 | 6.0 s |
| Advanced Layout Boxes | Official recipe plus the native box editor (Object, Character and Text boxes) feeding `MOSSImageLayoutPrompt` | 48.2 s |
| Base - Advanced Text To Image | Base model, official recipe (quality-first) | 50 s |
| Base - Advanced Layout Boxes | Base model, official recipe plus the box editor | 50 s |

Why these settings:

- Shift uses `ModelSamplingAuraFlow`, which keeps the model's timestep multiplier of 1.0. `ModelSamplingSD3` would set it to 1000. With shift 5 the `normal` scheduler reproduces the official sigma grid exactly, and same-seed initial noise matched the official Space.
- `SkipLayerGuidanceDiTSimple` with `single_layers` `22` skips that layer only in the negative branch, as the official pipeline does (24 and 26 also work). Official negative tags: `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`.
- Attention: the INT8 ConvRot files set `comfy_kitchen_int8` attention in all 42 layer configs. On the RTX 3090 that drifted from the official image and added a woven texture and muddier colors. PyTorch attention matched the official outputs. It was about 20% slower than the built-in setting, but INT8 stayed about twice as fast as BF16 (48 s versus 99 s at 50 steps, both with PyTorch attention). With the global `--use-sage-attention`, BF16 produced black (NaN) images on the 3090; the preset's explicit PyTorch attention avoids this.
- The Fast and Turbo recipes were picked from same-seed comparison sheets of photos, portraits and text posters. Large lettering held up in both; small text occasionally lost a letter (Fast) or Turbo added stray signage, so the 50-step preset remains the most faithful. The DMD LoRA is a community work in progress (`lora_latest`), so Turbo images differ more from the official model.
- Base presets: the official README's own Base example uses the same 50-step recipe. With the same seed, 80 steps reproduced the 50-step image on a storefront, a portrait and a text poster with no visible gain while taking about 60% longer (79 s versus 50 s), so the Base presets keep 50; raise the steps for experiments. Base rendered the sign, chalkboard and poster text correctly; the official README notes that SFT gives clearer text grouping and more organized layouts. The Base presets differ from the SFT ones only in the model file; the de-band node stays on and skips itself, so switching the loader to SFT keeps the fix. There is no Base Fast or Turbo preset: the 8-step LoRA was trained on SFT.
- Prompts may be up to about 4096 tokens. Pixel layouts and the official JSON schema are described in `Automatic_Prompt_Generate_Enhance_To_Video_And_Image_Models_LLM_Instructions/MOSS_Image_Enchance_Prompt_Feed_For_LLMs.txt`.

## Validation, 2026-10-17

ComfyUI `ce1e922d`, frontend 1.55.17, Torch 2.13.0+cu130. Physical GPU 1 (RTX 3090), pinned by UUID `GPU-d04c1512-d252-69ed-4858-a934e4430888`, because GPU 0 was busy with another session. Flags included SageAttention and Triton with the Comfy compiler disabled, plus `--reserve-vram 6`: this card also drives the desktop, and without the reserve Windows moved the VAE decode into shared memory, where it took minutes.

- All six presets ran through the API (the two Base presets on 2026-10-18, the de-band node logged its Base skip). Reloaded in headless Chrome, each kept its node IDs and produced exactly the API prompt of the build. The guard kept a compiled layout JSON byte-identical, kept a FLUX.3 trailing table while expanding its caption, and still expanded plain `{red|blue}`. The guard applies to every dynamic-prompt box; the shipped FLUX.3 Klein and Qwen Image 2.1 presets loaded without errors, and none of their default prompts ends in JSON, so it does not change them.
- API checks: a 1536x1024 pixel draft compiled and rendered its layout; the Base model skipped de-band with the log line; SFT BF16 rendered correctly; the LoRA stack applied; gallery media failed with the clear message.
- Ten regression tests (13 since 2026-10-18) cover the official draft compiler output (byte-identical), JSON pass-through, weights, schedules and the variant check. Run from the ComfyUI root: `python -B custom_nodes/SECoursesAudioTools/tests/test_moss_image.py`. Importing the package initializes CUDA, so pytest collection is not used.

## Fixes, 2026-10-18

- The six presets set `moss_text_only` on the gallery. Before, a fixed seed repeated the image but not the `{a|b}` choice in the main prompt box. In headless Chrome with frontend 1.55.18, the queued prompt of all MOSS presets now keeps `{red|blue}` as typed; with the property removed, six queues sent red or blue at random. JSON prompts are unchanged and the FLUX.3 presets behave as before.
- The four SFT presets recorded SECoursesAudioTools `1250cd8`, a commit without the MOSS nodes; all six now record the commit that has them.
- 13 regression tests, including the Base-weights-under-an-SFT-name warning.

Limitations: tested on an RTX 3090 only, so the attention findings may differ on RTX 50 series cards. MOSS-Image does not edit images or use references.

The SwarmUI presets use the [MossImage premium extension](https://github.com/FurkanGozukara/SwarmUI_Premium_Extensions/tree/main/MossImage), which builds the same recipe with native nodes; its README documents the Swarm validation.

Primary references: [OpenMOSS MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), [Comfy repack](https://huggingface.co/Comfy-Org/MOSS-Image), ComfyUI `comfy/text_encoders/moss_image.py` and `comfy_extras/nodes_moss_image.py`.
