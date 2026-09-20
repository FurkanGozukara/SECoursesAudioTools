# Qwen Image 2.1 preset adapters

These three small nodes reuse native ComfyUI inference and the existing FoleyExtension gallery:

- `SEQwenImage21Canvas`: native `LoadImage` with a disabled sentinel, IMAGE/MASK/denoise outputs, upload/preview, and a visible native **Paint mask** button followed immediately by **Denoise strength (img2img / mask)**. The preset wires this value through Prepare to KSampler; default remains 0.85.
- `SEQwenImage21Prepare`: translates gallery `@imageN` references and separate `@init`, preserves RGBA uploads, calls `TextEncodeQwenImage21`, and chooses an empty or encoded latent. Mask mode supplies a native noise mask. Total images are capped at the model's documented 10, including init; nonexistent tokens and empty masks fail clearly.
- `SEQwenImage21Finish`: leaves ordinary/RGBA output unchanged; masked runs restore unmasked pixels from the resized source, with opaque alpha for an RGB source.

`web/qwen_image21.js` uses the native `Comfy.MaskEditor.OpenMaskEditor` command. The gallery's `qwen21_images_only` workflow property hides video-specific controls without changing H3 presets. Upload/paint/clear buttons are excluded from both workflow and API serialization. Display places denoise after Paint mask while serialized values remain image, denoise; old upload placeholders reset to the new default. No ComfyUI core code is modified. Qwen imports are deferred until execution so older ComfyUI installations can still load unrelated SECoursesAudioTools nodes.

The UI calls `reference_resolution` **Reference / edit size (0 = original)**: each input is resized to approximately this value squared, preserving aspect ratio and rounding dimensions to 32. The output follows init/first reference; no-image generation uses canvas width/height. **Request transparent background** adds the official RGBA wording to the prompt; it is not deterministic background removal. The preset's expanded Quick Guide explains these controls and all three modes.

Inpainting uses a binary sampling mask for every painted pixel, then the original soft mask for the final composite. Feeding partial opacity to the sampler reintroduces source latents every step and suppresses editing; partial opacity now affects only the final blend. Use brush opacity 1 for a complete replacement and an explicit edit instruction such as “Replace the left man's head in @init with a warrior wearing a metal helmet.” Generate/reference edit ignores the mask and forces denoise 1.0; img2img edits the whole image at the selected strength.

Runtime dependencies are native ComfyUI, Pillow, NumPy and Torch, plus FoleyExtension for its validated gallery file resolver. The preset uses the downloader's existing filenames; it needs no local test files. Legacy Qwen Image/Edit models and their Turbo LoRAs are different architectures.

### Native mask editor help

Opening the native editor from the Qwen canvas adds explanations directly below its controls and widens the settings column from 220 to 320 CSS pixels (capped at 42% of the viewport on small screens). This is a scoped SECoursesAudioTools frontend extension, not a core frontend edit.

The preset's canvas property `mask_editor_brush` sets Thickness 50, Opacity 1 and Hardness 1 each time its editor opens and when Reset to Default is clicked. Users can change them during that editing session; switching tools does not reset them. Canvases without this property keep native behavior. Native last-used brush persistence is unchanged. Chrome checks verified initial/reopened/reset values of `[50, 1, 1]`, preserved manual `[17, 0.4, 0.2]` across tool switches, retained the native reset `[20, 1, 1]` when the property was absent, and preserved the property in workflow serialization.

- Brush shape/thickness change the saved stroke footprint. Brush Opacity changes saved coverage; in this workflow partial mask coverage blends generated pixels with the original. On the paint layer, it controls color opacity.
- Hardness controls edge feathering; lower values also slightly widen the brush. Step Size controls spacing between brush dabs, not sampler steps. Lower spacing makes smoother coverage; higher spacing can leave gaps. Settings affect new strokes, not existing pixels.
- Color Selector applies to the paint layer, which changes source-image pixels, not to mask coverage.
- Under Layers, Mask Opacity, Black/White/Negative blending and layer visibility checkboxes only change the preview. They do not alter the saved mask or exclude hidden paint from saving. Native Save uses the mask canvas alpha, not its CSS opacity or blend mode.

Chrome verification: moving Mask Opacity from 0.8 to 0.01 and 1, and selecting White/Negative/Black, left the mask alpha SHA-256 unchanged (`5a862c18b7cc7ac2839e88aba38920ae7d92900d560c2aa167add91373898809`). The expanded guide distinguishes Brush Opacity from the preview-only Mask Opacity control. The sidebar measured 320 CSS pixels with no horizontal overflow; switching to Paint Bucket and back preserved all ten hints without duplicates. Screenshots: `output/playwright/qwen21_mask_help_wide.png` and `qwen21_mask_help_layers.png`.

## Validation, 2026-09-20

ComfyUI `5ba116a`, frontend 1.53.6, Chrome 153.0.8010.52, Torch 2.13.0+cu130. Physical GPU 0 was pinned by UUID `GPU-21350b86-3099-1af0-58ef-f6432e60d9d0` (RTX 5090). The usual standalone flags included SageAttention and Triton, with the Comfy compiler disabled. Models: Qwen 2.1 INT8 ConvRot diffusion, INT8 ConvRot Qwen3-VL 8B and BF16 Qwen 2.1 VAE.

- 1024×1024 / 25-step text-to-image: cold full execution 28.15 s; warm Chrome sampler about 3.4 s, full run about 4 s.
- RGBA generation: ~5 s with alpha spanning 0–255, visually checked in Chrome over a checkerboard.
- One-reference edit ~7 s; two-reference composition ~20 s; img2img ~7 s; inpainting ~7 s, including polling overhead. These are per-prompt observations, not comparative benchmarks.
- A generated rectangular mask and a mask painted/saved with the actual native Chrome UI both ran successfully. The Chrome result had zero maximum RGB error over all 593,688 unmasked pixels. This guarantee applies after resizing to the working canvas, not to original pixels at a different size.
- Chrome gallery upload, reorder, remove, and graph serialization roundtrips preserved API inputs. Native Paint Mask, Save and Run were exercised, not just opened.
- Eight regression tests cover init/token numbering, stale tokens, disabled input, canvas strength, mode-dependent denoise, binary sampling versus soft compositing, alpha and exact unmasked/feathered compositing, and invalid mode inputs. Run from the ComfyUI environment with its root on `PYTHONPATH`: `python custom_nodes/SECoursesAudioTools/tests/test_qwen_image21.py`.

### Denoise UI and reported no-change reproduction

- Chrome 153/frontend 1.53.6: edited the visible strength to 0.37; four save/load cycles preserved the value and the Canvas → Prepare → KSampler links.
- The user's saved 1254×1254 mask had maximum opacity 178/255 (~70%). Their prompt “a warrior head” produced little change at 0.85, and still did at 1.0 with binary sampling. An explicit replacement instruction produced the helmet. Prompt specificity therefore mattered in this case; raising denoise alone was insufficient.
- With the original soft mask and explicit prompt, all 981,198 unpainted pixels at 1024×1024 had zero maximum RGB error. The remaining original face showed through the intended 30% final blend.
- Chrome's native Paint mask button, Clear, opacity 1, brush strokes and Save were exercised on a separate copy of the same image, followed by the actual Run button at strength 1.0. `output/Qwen_Image_2.1/validation/chrome_denoise_00001_.png` visibly replaces the left head with a steel helmet while preserving the surroundings. Browser screenshots are under `output/playwright/qwen21_*.png`.
- Changed that visible control to 0.25 and clicked Run again with the same seed, image, mask and prompt. The low-strength result retained the original face (`chrome_denoise_00002_.png`); mean masked RGB difference was 6.31/255 versus 49.97/255 at strength 1.0. Both runs preserved all 963,914 unpainted pixels exactly (maximum RGB error 0). These differences describe this case, not a general quality metric.

Reproduction: the shipped workflow's API prompt is embedded by SaveImage in the generated PNGs. On the development machine evidence is under `ComfyUI/output/Qwen_Image_2.1/validation`; disposable run scripts/logs are under the workspace's designated local test directory `Qwen21`. Neither is a runtime dependency.

SwarmUI upstream HEAD was independently checked as `eb39c7d103c245dba37fdc1a1389e4df32d82179`. It has old Qwen Image/Edit support, but lacks the Qwen 2.1 model registration, native encoder node routing, latent format and matching VAE selection. No core Swarm edits or unusable Swarm preset were added.

Primary references: [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-2.1), [Comfy repack](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), ComfyUI `comfy_extras/nodes_qwen.py`, official `image_qwen_image_2_1_t2i.json` and `image_qwen_image_2_1_image_edit.json` templates.
