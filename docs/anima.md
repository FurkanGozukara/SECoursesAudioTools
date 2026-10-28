# Anima gallery presets

The presets use native Anima loading and text conditioning. `SEAnimaGalleryPrompt` resolves seeded choices from the existing reference gallery; the normal advanced encoder handles weights and schedules. `SEAnimaGalleryGuider` selects plain CFG or the managed ElectronGirl reference adapter. Ordinary Easy Use LoRAs apply before it. `SEAnimaNoise` delegates noise and variation seeds to the existing Swarm sampler noise implementation.

Use Anima base v1.0, Qwen3 600M under Swarm's `qwen_3_600m.safetensors` filename and the shared Qwen-Image VAE. The core bundle prefers native INT8 ConvRot and keeps BF16 as an individual catalog alternative. Arithmetic is BF16 because Anima overflows in fp16. The adapter is loaded only on the reference path; use the first gallery image and describe the new pose, scene and framing.

The reference preset samples at 1024x1024, decodes, scales pixels 2x with Lanczos, re-encodes and runs 12 ordinary Anima detail steps at denoise 0.3. That produces a 2048x2048 final image. Direct 4 MP reference sampling showed grid artifacts in the author's native graph with both BF16 and INT8. Latent-space enlargement softened the line work; pixel-space refinement retained clean lines. Text-to-image also works directly at 2048x2048.

The gallery retains added prompts and .txt folder batches. Batch-size references repeat the same character latent; native guidance remains separate for positive, negative and reference-free branches. Fine accessories and strong painterly styles can drift. The reference adapter supports one character/style image; extra gallery cards do not supply separate character and style references.
