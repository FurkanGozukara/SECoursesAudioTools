# Qwen Image 2.1 preset adapters

These three small nodes reuse native ComfyUI inference and the existing FoleyExtension gallery:

- `SEQwenImage21Canvas`: native `LoadImage` with a disabled sentinel, IMAGE and MASK outputs, upload/preview, and a visible native **Paint mask** button.
- `SEQwenImage21Prepare`: translates gallery `@imageN` references and separate `@init`, preserves RGBA uploads, calls `TextEncodeQwenImage21`, and chooses an empty or encoded latent. Mask mode supplies a native noise mask. Total images are capped at the model's documented 10, including init; nonexistent tokens and empty masks fail clearly.
- `SEQwenImage21Finish`: leaves ordinary/RGBA output unchanged; masked runs restore unmasked pixels from the resized source, with opaque alpha for an RGB source.

`web/qwen_image21.js` uses the native `Comfy.MaskEditor.OpenMaskEditor` command. The gallery's `qwen21_images_only` workflow property hides video-specific controls without changing H3 presets. Frontend buttons do not serialize widget values. No node widgets are reordered and no ComfyUI core code is modified. Qwen imports are deferred until execution so older ComfyUI installations can still load unrelated SECoursesAudioTools nodes.

Runtime dependencies are native ComfyUI, Pillow, NumPy and Torch, plus FoleyExtension for its validated gallery file resolver. The preset uses the downloader's existing filenames; it needs no local test files. Legacy Qwen Image/Edit models and their Turbo LoRAs are different architectures.

## Validation, 2026-09-20

ComfyUI `5ba116a`, frontend 1.53.6, Chrome 153.0.8010.52, Torch 2.13.0+cu130. Physical GPU 0 was pinned by UUID `GPU-21350b86-3099-1af0-58ef-f6432e60d9d0` (RTX 5090). The usual standalone flags included SageAttention and Triton, with the Comfy compiler disabled. Models: Qwen 2.1 INT8 ConvRot diffusion, INT8 ConvRot Qwen3-VL 8B and BF16 Qwen 2.1 VAE.

- 1024×1024 / 25-step text-to-image: cold full execution 28.15 s; warm Chrome sampler about 3.4 s, full run about 4 s.
- RGBA generation: ~5 s with alpha spanning 0–255, visually checked in Chrome over a checkerboard.
- One-reference edit ~7 s; two-reference composition ~20 s; img2img ~7 s; inpainting ~7 s, including polling overhead. These are per-prompt observations, not comparative benchmarks.
- A generated rectangular mask and a mask painted/saved with the actual native Chrome UI both ran successfully. The Chrome result had zero maximum RGB error over all 593,688 unmasked pixels. This guarantee applies after resizing to the working canvas, not to original pixels at a different size.
- Chrome gallery upload, reorder, remove, and graph serialization roundtrips preserved API inputs. Native Paint Mask, Save and Run were exercised, not just opened.
- Six regression tests cover init/token numbering, stale tokens, disabled input, alpha and exact unmasked/feathered compositing, and invalid mode inputs. Run from the ComfyUI environment with its root on `PYTHONPATH`: `python custom_nodes/SECoursesAudioTools/tests/test_qwen_image21.py`.

Reproduction: the shipped workflow's API prompt is embedded by SaveImage in the generated PNGs. On the development machine evidence is under `ComfyUI/output/Qwen_Image_2.1/validation`; disposable run scripts/logs are under the workspace's designated local test directory `Qwen21`. Neither is a runtime dependency.

SwarmUI upstream HEAD was independently checked as `eb39c7d103c245dba37fdc1a1389e4df32d82179`. It has old Qwen Image/Edit support, but lacks the Qwen 2.1 model registration, native encoder node routing, latent format and matching VAE selection. No core Swarm edits or unusable Swarm preset were added.

Primary references: [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-2.1), [Comfy repack](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), ComfyUI `comfy_extras/nodes_qwen.py`, official `image_qwen_image_2_1_t2i.json` and `image_qwen_image_2_1_image_edit.json` templates.
