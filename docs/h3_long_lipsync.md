# H3 Long Lip Sync

`SEH3LongLipSync` turns one complete driving audio file into one continuous 24-fps avatar using native ComfyUI H3 sampling. Use the [example workflow](../example_workflows/H3_Long_Lip_Sync.json), choose a portrait and audio file, then run. The default canvas is 832x1248, with the matching FL2VA/Ref2VA Turbo 4-step LoRA. The SOL and FirstBlockCache switches in the Turbo node are independent and default off.

## How continuation works

The controller encodes the driving audio once, samples short overlapping windows and holds both the previous video overlap and source audio latents with native zero-denoise masks. Each window also receives a clean audio guide. It retains the original overlap latents exactly instead of re-encoding the preceding window's decoded pixels. The incremental video decoder has bounded decode windows; the complete latent sequence and encoded audio remain in CPU memory.

The H3 video and audio grids meet every 51 video frames (24 fps / 40 audio ticks per second). Windows are 141, 192, 243, 294 or 345 frames, with 39 or 90 frames of overlap. The controller balances the fewest windows and aligns their starts on that shared grid. The default is at most 243 frames with 39 held frames. Output is trimmed to `ceil(audio_seconds * 24)` frames, preserving every input utterance and pause. There is no manual duration input or word-level cut.

The first-frame FL2VA anchor applies to the first window. Later windows inherit the held latents. Ref2VA identity references remain in conditioning. Sampling uses Euler/simple, CFG 1 and the connected step count; choose the corresponding Turbo LoRA, or a suitable quality step count without Turbo.

## Outputs and limitations

The node produces VIDEO, a file path and a JSON report, plus an MP4 preview. Files include the final H.264/AAC video, decoded video-only intermediate, normalized source PCM WAV and timings/overlap checks. The final soundtrack follows the complete driving audio at its original speed; normalization/resampling and AAC export mean it is not a bit-identical copy of the original file. No generated audio replaces the driving narration.

Latent retention avoids one source of drift. It does not guarantee indefinite identity, perfect lips or invisible joins. In the 15-clip FL2VA comparison (512x768, 704x1056 and 832x1248; 10/15/30/45/60 seconds), all files decoded fully but the 832x1248 45- and 60-second clips failed locked-camera review. A 60-second Ref2VA retry with a stronger fixed-camera prompt also changed framing. The other 13 passed sampled performance review, with a late framing/detail change noted at 704x1056/60 seconds. Short soft frames and motion changes can appear at joins. Treat long mode as experimental and review every result. Higher resolution and extra overlap cost time and memory. Audio duration grows CPU latent storage and initial audio encoding memory; this is not an unbounded streaming service.

Requires native H3 per-token denoise masks and the existing SECourses streaming decoder. Tested with ComfyUI `52f98af2`, Kitchen 0.2.37 and RTX 5090 on Windows. The separate second-pass H3 Face Inpainting path is not part of this controller; retain originals and run the project's CodeFormer mouth-restoration procedure afterward when required.

## SwarmUI

Update SECoursesAudioTools and SwarmUI_Premium_Extensions. Select a native H3 model, supply Init Audio and a reference portrait (or use a native H3 Video Model in Image To Video), and enable **MiniMax H3 Long Lip Sync** in Init Audio. Window and overlap controls appear below it. Select the matching LoRA and steps. **MiniMax H3 Optimizations** contains independent SOL and FirstBlockCache controls; its historical saved ID is retained. Turn the master off for the standard attention path. Long mode uses its own incremental H.264 output; the separate Video Face Inpainting option must be off.

The approach was informed by [comfyui-obvpm-timeline](https://github.com/chanon/comfyui-obvpm-timeline), inspected at `4a027a09cae607f9b36f8227a4f46ed05ccc8123`. That project offers interactive timeline continuation and bridge operations. Its bridge feature explicitly regenerates material to recover drift; it does not establish unlimited no-degradation generation. No code from that GPL project is copied here.

Swarm end-to-end checks produced a complete 832x1248/60-second Ref2VA clip (753.98 seconds API wall time) and a 512x768/30-second Image To Video clip (125.81 seconds). Both retained the full supplied audio and planned frames; the reference clip failed framing review as noted above. These integration times include transport/model preparation and are not attention-only benchmarks.
