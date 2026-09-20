# AvatarForever: image + audio to long avatar video

Implemented against [leeruibin/avatarforever](https://github.com/leeruibin/avatarforever)
commit `4dfc42b0e2dbbded4d148387d186219bd7601279`, checked 2026-09-20.
The linked release does not identify a separate “Pro” checkpoint or inference
pipeline. This integration covers its released audio-conditioned AR inference
controls. It is not an implementation of an unspecified separate product.

## Quick start

Load **AvatarForever Unified Image or Text + Audio To Long Video - 260920.json**.
Choose a portrait and an audio recording, describe the visible performance, and
run. The sampler saves an H.264 MP4 with the source soundtrack and a JSON report
under `output/video/AvatarForever`. Its own preview avoids a second Save Video
node and duplicate export. The two output sockets remain available to other nodes.

`duration_seconds = 0` uses all audio. `10`, `30`, `60`, or `300` selects up to that
many seconds; `audio_start_seconds` selects the starting point. A shorter recording
is never repeated or stretched. The optional silent lead-in is added after trimming.
Audio is re-encoded as AAC for the MP4, without voice generation or replacement.
Generation rounds up to LTX's 8n+1 frame geometry, then trims the export to the
selected audio duration. Video frame rounding can differ by less than one frame
for durations not divisible by the frame interval.

Turn `use_image` off for audio + text generation. Without an image, describe the
person and surroundings explicitly. With an image, `match_image_aspect` preserves
its shape at approximately the Width × Height pixel budget, on a 32-pixel grid.
Disable this for an exact canvas; the image is then center-cropped to fit.

The demo uses the existing `demo_singer_avatar.webp` and
`demon_singer_audio_18_sec.mp3` materials. Replace both with your own inputs.

## Existing local model recipe

| Component | Selected file |
|---|---|
| Model | `avatarforever-ltx-2.3-22b-INT8-ConvRot-HQ.safetensors` |
| Text encoder | `gemma_3_12B_it.safetensors` |
| Text projection | `ltx2/ltx-2.3_text_projection_bf16.safetensors` |
| Video VAE | `LTX23_video_vae_bf16.safetensors` |
| Audio VAE | `LTX23_audio_vae_bf16.safetensors` |

The generation models were already present in the shared SwarmUI model directory.
Generation itself needs no Whisper, Wav2Vec, face detector or LoRA. The preset's
optional mouth enhancement additionally uses `facerestore_models/codeformer.pth`
and `insightface/models/buffalo_l/det_10g.onnx` from existing model folders.
Install the node pack's `requirements.txt` through the normal installer. No model
is downloaded automatically; turn Mouth Enhancement off to run without these two
restoration files. Compatible native BF16/FP8/INT8 AvatarForever variants can also be
selected. The loader verifies the trained channel-conditioning tensors exist.
An ordinary LTX-2.3 checkpoint is not a substitute for AvatarForever's training.
Use matching LTX-2.3 encoders/VAEs; do not silently substitute LTX-2.5/Gemma 4.

The native Comfy loader logs three “unexpected” channel-condition tensors; this
adapter reads and uses those exact three tensors separately. This is expected
with this loader. Using the ordinary UNET loader alone would omit them.

## Controls and defaults

| Controls | Meaning / default |
|---|---|
| Width, Height, Match Image Aspect | 768×512 pixel budget; image aspect matching on |
| FPS | 25, the released recipe; changing it also changes the AV timeline |
| Seed | 42 in the preset, fixed; choose another for variation |
| Use Image, Image Strength | On, 1.0; strength affects first-frame latent conditioning |
| Image Compression | 0, disabled; optional native LTX image preprocessing |
| Audio Start, Duration, Lead In | 0, 0, 0; start at beginning, full recording, no lead-in |
| Sigmas | `1.0, 0.98125, 0.909375, 0.421875, 0.0` |
| Chunk Size | 4 video latent frames, approximately 1.28s after the first chunk |
| History Chunks | 1 previous chunk; -1 retains all and grows memory with duration |
| Sink First Chunk | On; retain the initial chunk alongside recent history |
| Relative Positions | On; compact the selected history's time positions |
| ForeverCache | Off, matching released CLI; approximate per-chunk feature reuse |
| Cache Device | Auto; GPU when the estimated budget fits, otherwise system RAM |
| Channel Condition, Mode | On, gated; learned identity condition on each new chunk |
| First Frame Prefix, Position | Off, prepend; also supports append |
| Resident Weights | On; native non-dynamic loading, useful for this large INT8 model |
| Tiled VAE | On; spatial tiling; temporal decoding is bounded even when this is off |
| Spatial Tile / Overlap | 512 / 64 pixels; overlap is constrained below tile size |
| Temporal Tile / Overlap | 128 / 32 frames; overlap is constrained below tile size |
| CRF / Encoding Preset | 17 / fast; export quality and compression speed |
| Mouth Enhancement | On in both unified presets; off preserves original decoded frames |
| Mouth Fidelity / Blend | 0.9 / 0.7; CodeFormer fidelity and feathered mouth blend are separate controls |
| Mouth Model / Detector | Existing CodeFormer checkpoint and SCRFD ONNX file, relative to their model categories |

CFG is 1 and the solver is Euler. The sigma list determines the number of steps.
There is no negative prompt or generic KSampler in this workflow. Leave the
released four-step schedule unchanged initially.

Mouth enhancement runs on decoded frames before their first MP4 export. It aligns
the largest detected face, restores 512-pixel face crops in FP32 batches of four,
and applies only a feathered mouth region. Frames without a face pass through.
It does not resynthesize speech or retime the lips, and can still alter teeth or
mouth details. CodeFormer uses the ComfyUI device and managed model offloading;
SCRFD uses CUDA on that same device when available. A CUDA provider failure is
reported instead of silently switching detection to CPU. The output JSON includes
the selected devices and restoration timings. Old API graphs that omit the new
optional inputs retain their original behavior.

Image strength and channel conditioning are different controls. Reducing image
strength while leaving channel conditioning enabled still supplies the portrait
as an identity condition throughout the video. To remove image influence, turn
Use Image off. Without an image, channel conditioning comes from the first
generated latent frame.

First Frame Prefix only activates when chunk 0 is absent from the selected
history. With Sink First Chunk on it normally has no effect. To experiment with
it, disable the sink and retain a finite history. Both prepend and append follow
the released dynamic-prefix layout. The extra audio prefix remains aligned.

ForeverCache reuses normalized history features after step 1 of each chunk. It
is not exact attention KV reuse and can change the result. GPU/CPU describes the
feature cache location, not where the model runs. The cache is discarded between
chunks. CPU cache can be slower due to transfers; auto is an estimate, not an OOM
guarantee. Start with defaults and lower the pixel budget if VRAM is insufficient.

## Long recordings and practical limits

There is no 10/30/60-second hard limit. Five-minute audio is accepted by the same
rolling-window algorithm. With defaults, transformer history and the cache stay
bounded; all CPU latents, source audio, encoded audio and output storage still
grow with duration. Video decoding streams temporal tiles into FFmpeg rather
than accumulating a full RGB movie. MP4 finalization happens after generation;
this is offline generation, not a live streaming-call service.

Start with a 10-second excerpt at about 512×768 or 768×512, with the mouth visible
and a steady camera. Check the result before committing a long recording. Long
generation is not a guarantee against identity drift, gesture artifacts, or
imperfect lip sync. Strong occlusion, large turns, multiple speakers and cuts
make the task harder. There is one global prompt and one identity image; this
workflow does not provide a timed script, multiple character routing, automatic
audio transcription, generated speech, resume-from-checkpoint, or a second
upscale/refine stage. These are not controls in the linked released CLI pipeline.

Ordinary LoRAs can be inserted after the model/text loaders in ComfyUI; SwarmUI
passes its normal LoRAs through. No LoRA is needed or enabled by default. A LoRA
must be architecture-compatible; compatibility with arbitrary LoRAs was not tested.

## SwarmUI

Install the **AvatarForever** folder from `SwarmUI_Premium_Extensions`, alongside
the updated SECoursesAudioTools pack in the Comfy backend. The installer manifest
includes this extension. The sampler's native video preview works with Swarm's
existing output handling; no extra transport node is needed. Restart both apps.

Select **AvatarForever Unified Image + Audio or Text + Audio - 260920** from
`Amazing_SwarmUI_Presets_v75.json`. Attach audio to the prompt (the first attachment
is used), or set Video Audio Input; optionally set a still Init Image. Prompt
audio takes precedence if both are present. The AvatarForever group exposes the
same controls and explicitly selects the existing encoder/VAEs. Main Model,
Prompt, Seed, Width, Height, Video FPS and LoRAs apply. Main Steps, CFG, negative
prompt, Video Frames, refiner and unrelated pipeline controls are bypassed.

Swarm uses the same Python sampler. The C# extension only registers parameters
and builds its graph, before the ordinary loader can choose/download fallbacks.
Turning AvatarForever Enabled off returns to the normal Swarm workflow.

## Initial validation on 2026-09-20 (speed measurements superseded below)

GPU 0: RTX 5090 32GB, native INT8 ConvRot checkpoint, existing Gemma 3 and VAEs.
No use of the second GPU. Tests ran on an isolated Comfy backend.

* Real 3-second image+audio smoke render, cache off.
* Real 10-second image+audio, 512×768, ForeverCache GPU: 250 saved frames;
  sampler 122.9s, DiT 102.82s.
* Real 4-second audio+text, CPU cache, no history/sink, appended dynamic prefix.
* Real 60-second image+audio at defaults, 512×768: 48 chunks, 1,500 saved frames;
  DiT 538.61s, sampler 578.6s. MP4 video and AAC durations both 60.000s.
* Final saved Comfy workflow reloaded and queued through the frontend: 2-second
  demo render at 640×640; native video preview confirmed without a Save Video node.
* Swarm extension compiled with zero warnings/errors. Its generated graph was
  rendered through the Swarm API, returning one MP4 with 50 frames and exactly
  2.000s video/audio; missing audio produces a clear input error.
* The preset was added to the live Swarm user library and the v75 export catalog
  (63 to 64 entries in each). Existing preset content was preserved. Swarm's
  original Settings.fds and Backends.fds hashes were unchanged.
* Extracted contact sheets checked for visible continuity, identity and movement.
  No listening, transcription, or phoneme-level lip-sync measurement performed.
* 15 CPU tests passed, including existing H3/last-frame tests. New checks cover
  10/30/60/300-second audio geometry, bounded history, trimming, sigma validation,
  and streamed decoder frame order/count across many temporal tile boundaries.
* Nine small-block comparisons against the upstream implementation passed with
  nonzero RoPE, full/current/middle slices, cache off/populate/reuse; maximum
  float32 absolute difference 5.96e-8. This verifies block math, not bit-identical
  whole-model generation. Native kernels, quantization, RNG layout, VAE tiling,
  encoder implementation and rounding can differ from upstream.

A full five-minute render and maximum-resolution stress test have not been run.
The initial 48-minute extrapolation for five minutes is superseded by the speed
fix and measurements below.

## Face detail and speed follow-up, 2026-09-20

**Fixed a model-ownership bug in the initial integration.** A bound forward method
was installed before `clone(disable_dynamic=True)`. Comfy rebuilds the underlying
model for that clone but shallow-copies object patches, leaving the bound method
pointing to the old CPU-backed model. The new model reported fully loaded while
execution repeatedly copied weights from the old instance. The sampler now calls
the forward function with its currently loaded model explicitly. This also ensures
that weight patches such as LoRAs are read from the correct loaded instance.

On physical GPU 0 (RTX 5090), a profiled steady step went from 7,139 pageable
host-to-device copies to 9. Median non-profiled steady steps dropped from 2.7301s
to 0.4440s (6.15× faster, cache off). Profiler overhead is excluded from those step
timings. The new model instance was asserted at the forward entry point in the
test harness. Decoded RGB frames were **exactly identical** before/after the fix
in both 4-second comparisons: 100/100 frames with cache off and 100/100 with cache
on. All 15 CPU regression tests still passed.

Complete sampler runs, including audio preparation, model residency setup, VAE
decode and export (text encoding occurs upstream), using the same portrait and
audio excerpt:

| Configuration | Output | DiT | Sampler total |
|---|---|---:|---:|
| INT8, default, cache off, CRF 17 | 10s, 512×768 | 16.82s | 33.34s |
| INT8, GPU ForeverCache, CRF 17 | 10s, 512×768 | 17.45s | 33.26s |
| INT8, sharper-face setting, cache off, CRF 12 | 10s, 768×1152 | 32.23s | 54.65s |
| INT8, default, cache off, CRF 17 | 60s, 512×768 | 87.32s | 126.32s |
| Original BF16 reference, cache off, CRF 17 | 4s, 512×768 | 33.12s | 89.13s |

The directly comparable one-minute run improved from 578.60s to 126.32s: **4.58×
faster overall**, or approximately 0.475× real time. Prompt execution including
upstream setup was 126.64s. Video and audio both remain 60.000s, with 1,500 frames.
The complete before/after one-minute MP4 files are byte-identical, including audio
and container data: SHA-256
`16727ec02bd34f846f14cce47b4804aee2ee53767b2b3160b218acfaec2c1971`.
This is still below real-time generation on the 5090; the H100 headline should
not be used as a promised target for this hardware/runtime.

The BF16 reference required approximately 11.3 GiB of model offloading on this
32 GiB card. Spot checks did not show a clear facial-detail advantage over INT8
at the same resolution. Its setup was cold, so use the DiT column when comparing
sampling cost. This short test is not a general proof of quantization equivalence.

Recommended changes for facial detail: keep INT8, the four-step schedule, image
strength 1, gated channel conditioning and the sink; use a 768×1152 portrait or
1152×768 landscape pixel budget and CRF 12. This increases generated pixel count
by 2.25×. The extracted higher-resolution frame shows more resolved beard/hair
and clothing detail, although no claim of matching all official examples is made.
CRF 12 reduces compression loss; it cannot create missing skin detail by itself.
A tighter input portrait can allocate more pixels to the face at the same render
cost, but changes the composition. Increase resolution if the original composition
must be retained. Do not increase steps blindly on this four-step distilled model.

ForeverCache did not deliver an end-to-end speed improvement in this 5090 test;
keep it off by default. A larger temporal tile with spatial tiling disabled was
also tested (256-frame tile, 8-frame overlap, CRF 12): 10 seconds took 42.08s, so
the original spatial/temporal tiling defaults remain unchanged. Hardware, kernels
and shape matter; the upstream H100 cache gains are not a universal prediction.

The selected video VAE (170 tensors), audio VAE/vocoder (1,329 tensors) and text
projection (4 tensors) were compared against the original AvatarForever checkpoint:
all are byte-identical. `gemma_3_12B_it.safetensors` is the existing mixed-precision
Gemma encoder, rather than an established full-precision reference; its perceptual
effect remains unmeasured. No new models were downloaded.

Further warm-session one-minute renders covered 768x1152, 832x1248, 896x1344,
960x1440 and 1024x1536. A same-latent GPU decode-blending experiment hit a memory
limitation, so the working decoder remains unchanged. Comparison videos and
development scripts are local test artifacts, not distributed dependencies.

The production CodeFormer mouth pass on a finished 896x1344, 60-second video
took 83.51 seconds including GPU detection, FP32 restoration, mouth compositing
and export, versus 126.60 seconds for the original CPU-detection recipe. This is
a 34% reduction in post-processing time, not total generation time. Both outputs
preserved all 1,500 frames and identical source audio packets. The unified
workflow instead enhances decoded frames before their first export. FP32 remains
the default: BF16 changed restoration details, and the tested ONNX export did not
improve speed. All 18 node-pack tests passed, along with SwarmUI generation,
multi-block VAE decoding and an off-toggle run with missing restoration weights.

The fix is shared by ComfyUI and SwarmUI through this sampler. Restart the Comfy
backend to activate it. The published H100 throughput is not a 5090 speed guarantee.

## Exact-output speed follow-up and 8 GB budget test, 2026-09-20

The adapter now computes only the timestep embeddings AvatarForever uses, selects
the native paired RoPE operation when both streams share positions, and reuses
identical cross-modal normalization results. CodeFormer overlaps CPU mouth
compositing with the next GPU batch, with one bounded worker per decoded block.
FP32 restoration, detection, batch size, fidelity, blend, model weights, sampler
schedule and VAE tile geometry are unchanged. Its new `processing_seconds` report
field measures elapsed mouth processing; the individual phase times overlap and
must not be added to obtain elapsed time.

Physical GPU 0, RTX 5090; same inputs and seed, INT8 ConvRot, cache off:

| Measurement | Before | After | Time reduction |
|---|---:|---:|---:|
| LTX denoising, 60s at 512x768 | 80.64s | 78.75s | 2.3% |
| Complete sampler, same minute with mouth enhancement | 184.91s | 172.39s | 6.8% |
| Warm CodeFormer processing, 192 frames at 896x1344, mean of two passes | 7.83s | 6.58s | 16.0% |

The complete original, optimized and constrained-memory minute-long MP4s are
byte-identical, SHA-256
`c1943dce7ab02e003ba7efefda4c85025b6eaf2b3f0fab696d8a527ce073b743`.
Latents and all 16 decoded/restored frame-block hashes also match exactly.
This establishes equality for these tests, not every possible model/backend.
The four-second cache-on and text-only/dynamic-prefix comparisons also retained
identical latents and frame hashes. Nineteen CPU tests and nine comparisons with
upstream block math passed. SwarmUI's preset generated both mouth on/off graphs
and a real two-second, 50-frame MP4 with a two-second audio track.

**An 8 GB memory budget was simulated, not a physical 8 GB GPU tested.** The
isolated backend used `--reserve-vram 25` on the 32 GB card, native non-dynamic
offloading (`--disable-dynamic-vram`), and a verified hard 7 GiB PyTorch allocation
limit. NVML also measured
allocations outside PyTorch, including CUDA face detection. A 512x768, 60-second
render with mouth enhancement succeeded: 1,500 saved frames, exactly 60.000s video
and audio, 466.26s sampler time, 6.86 GiB peak total GPU usage and 4.79 GiB peak
PyTorch sampler allocation. All 1,505 generated frames were enhanced before the
normal export trim. No quality setting or tile size was reduced, and no new
low-VRAM toggle was needed. The host had 96 GB RAM; worker resident RAM was sampled
at 48.4 GiB. This does not establish suitability for a machine with little system
RAM, or predict an actual 8 GB card's speed.

The reservation is a test setting for this 32 GB card, **not a setting to copy to
an 8 GB card**. Use `--disable-dynamic-vram` for the tested low-memory configuration;
the existing Lowest VRAM launcher includes it. In SwarmUI, add it to the ComfyUI
backend's ExtraArgs and restart that backend. A cold start with dynamic loading
enabled completed but peaked at 9.10 GiB, so it does **not** validate an 8 GB budget.
A reservation alone is not a hard memory limit, and AIMDO allocations are outside
the PyTorch allocator cap. Native model offloading provides the working path.
The saved unified preset's own demo inputs also passed a cold, two-second test
at 640x640 with `--disable-dynamic-vram`: monitoring began before startup and
covered text encoding, audio/image encoding, sampling, restoration and export.
Peak total GPU memory was 6.27 GiB; its 50-frame MP4 matched the normal-memory
run byte for byte.
Both existing unified presets use this shared sampler without changes to their
values. The test allocator cap exists only in the disposable test harness.

Detailed reports and comparison videos are local evidence under
`output/video/AvatarForever_SpeedVRAM`; no runtime or distributed preset depends
on the development folder. Baseline node-pack revision:
`ca7297ea2db19c6b27ac79f0f548b50b6f73aadf`.

## Implementation and attribution

Two new node classes; one model-forward adapter; no ComfyUI core changes or global
monkeypatches. The sampler passes its currently loaded model clone directly to
the forward adapter; it never binds a method to an earlier model instance. Native
loaders, quantized linear operations, Gemma encoding and VAEs are reused. FFmpeg
helpers are reused from this pack's H3 implementation without modifying it.

The trained first-frame channel projection/gate, zero-noise audio/history token
timesteps, current scalar sigma for prompt/cross-modal conditioning, sink/history
selection and chunk-local ForeverCache follow the linked source. Training-only
initialization choices are not exposed because the checkpoint already contains
the trained weights. Resident Weights addresses loading behavior; it is not a
claim to reproduce the upstream compiler/fast-inference infrastructure.

Conversion-build SHA-256 (before Swarm adds ModelSpec header metadata):
`598c9bad832640ac3d9d748770f025993884c9d505238b5e898d977e5600dadb`.
Source-derived portions are subject to the supplied
[LTX-2 Community License](AvatarForever-LICENSE.txt). Preserve that attribution
and check the upstream model terms when redistributing models or deploying them.
