"""Thin preset adapters over the installed native H3 and SECourses nodes."""

import folder_paths
import nodes

DEFAULT_SCENE = (
    "A single adult presenter speaks directly to the viewer in a cinematic medium close-up. "
    "The face is clearly visible, with attentive eyes, realistic skin texture and a relaxed posture. "
    "Soft window light shapes the face; a warm, softly blurred studio adds depth. "
    "The camera holds steady at eye level with consistent focus and exposure."
)



REFERENCE_SCENE = (
    "The main visible performer addresses the viewer. Preserve the supplied performer's identity, "
    "facial proportions, hairstyle, clothing, setting, lighting and visual style. "
    "Keep the face and mouth clearly visible and the composition consistent with the source."
)



PERFORMANCE = (
    "The lips, jaw and cheeks articulate the supplied vocal performance with precise timing: "
    "clear consonant closures, rounded vowels and natural transitions between syllables. "
    "Small, purposeful head movements and subtle facial reactions follow the phrasing and emotion. "
    "The mouth settles naturally during pauses. Keep teeth, tongue, eyes and facial contours stable. "
    "Hands remain away from the face. The shot stays continuous, with no cuts, text overlays or extra speakers."
)



TURBO_LORAS = {
    (False, "4"): ("minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", 6.0),
    (False, "8"): ("minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", 6.0),
    (True, "4"): ("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 12.0),
    (True, "8"): ("minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", 12.0),
}



class SEH3LipSyncPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene": ("STRING", {"forceInput": True}),
            "references": ("SECOURSES_REF_PACK",),
            "performance": (["speech", "singing"], {"default": "speech"}),
            "exact_words": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional exact transcript or lyrics. Leave empty to follow the audio without inventing dialogue. Update this if you change the audio."}),
            "language": ("STRING", {"default": "English"}),
        }, "optional": {"first_frame": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "SECourses/MiniMax H3 Lip Synch"

    def build(self, scene, references, performance="speech", exact_words="", language="English", first_frame=None):
        scene = scene.strip()
        if "integrated_multimodal_description:" in scene or "subject_definitions:" in scene:
            return (scene,)
        ref_images = references.get("images") or []
        ref_videos = references.get("videos") or []
        ref_audios = references.get("audios") or []
        has_refs = bool(ref_images or ref_videos or ref_audios)
        has_first_frame = first_frame is not None or bool(references.get("init_image"))
        if (has_first_frame or ref_images or ref_videos) and (not scene or scene == DEFAULT_SCENE):
            scene = REFERENCE_SCENE
        if not scene:
            scene = DEFAULT_SCENE
        if performance == "singing":
            scene = scene.replace("presenter speaks", "singer performs").replace("addresses the viewer", "sings to the viewer")
        action = "sings" if performance == "singing" else "speaks"
        dialogue = f" (S1) {action} <d>[{language}] {exact_words.strip()}</d>." if exact_words.strip() else f" (S1) {action} the supplied vocal performance."
        description = f"[Shot 1] {scene}{dialogue} {PERFORMANCE}"
        sound = "The supplied source audio is the complete, authoritative soundtrack. Follow its words, pacing, breaths and silences; do not add dialogue or sound effects."
        music = "Retain only music already present in the source soundtrack."
        if has_refs:
            identities = []
            if ref_images:
                identities.append("<Subject 1> is the main performer in @image1; retain that performer's identity and appearance.")
            elif ref_videos:
                identities.append("<Subject 1> is the main performer in @video1; retain that performer's identity and appearance.")
            else:
                identities.append("<Subject 1> is the visible performer described below.")
            relations = ["<Subject 1> (appears in [Shot 1]): fully_preserved - identity and visual appearance; vocal timing follows the locked input audio."]
            if ref_images:
                relations.append("@image1: fully_preserved - the main performer's identity and visual design.")
            if ref_videos:
                relations.append("@video1: attribute_transfer - visual performance and camera context; vocal timing follows the locked input audio.")
            if ref_audios:
                relations.append("@audio1: weak_reference - vocal character only; the locked input soundtrack remains authoritative.")
            description = "One continuous shot, with the visual style of the supplied references. " + description.replace("[Shot 1] ", "[Shot 1] <Subject 1> is clearly framed as the main performer. ", 1)
            prompt = ("subject_definitions:\n" + " ".join(identities) + "\n\nsummary:\nA continuous, clearly framed vocal performance synchronized to the source audio."
                      + "\n\nretention_analysis:\n" + "\n".join(relations)
                      + "\n\ndetailed_description:\n" + description)
        else:
            prefix = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n" if has_first_frame else ""
            prompt = prefix + "integrated_multimodal_description: " + description
        prompt += "\n\noverall_soundscape: " + sound + "\n\nnon_diegetic_music: " + music
        return {"ui": {"text": [prompt]}, "result": (prompt,)}



class SEH3LipSyncTurbo:
    @classmethod
    def INPUT_TYPES(cls):
        # Missing optional Turbo weights must not block full-quality generation.
        loras = sorted(set(folder_paths.get_filename_list("loras") + [value[0] for value in TURBO_LORAS.values()]))
        return {"required": {
            "model": ("MODEL",), "video_vae": ("VAE",), "uses_ref2va": ("BOOLEAN", {"forceInput": True}),
            "enable_speed_lora": ("BOOLEAN", {"default": False, "label_on": "TURBO", "label_off": "FULL QUALITY"}),
            "turbo_steps": (["4", "8"], {"default": "4"}),
            "quality_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "enable_sol_attention": ("BOOLEAN", {"default": False, "tooltip": "Benchmark SOL against the current attention backend and use it when faster. Early layers/steps and the final step remain dense."}),
            "enable_first_block_cache": ("BOOLEAN", {"default": False, "tooltip": "Reuse similar intermediate steps. Changes the sampling trajectory; the final step always computes."}),
            "save_vram": ("BOOLEAN", {"default": True}),
            "exact_output": ("BOOLEAN", {"default": True, "label_on": "EXACT MEMORY SAVING", "label_off": "MAX MEMORY SAVING", "tooltip": "Exact chunks feedforward work; Max also groups attention heads and can change the result."}),
            "fast_vae_decode": ("BOOLEAN", {"default": True, "tooltip": "Use the existing H3 VAE tile batching optimization, independently of SOL and the speed LoRA."}),
        }, "optional": {
            "fl2va_4step_lora": (loras, {"default": TURBO_LORAS[(False, "4")][0], "tooltip": "Audio + text/image, 4-step Turbo. Default: LightX2V FL2V v1.2 768p. Sampling remains 4 steps and shifts 6 / 3."}),
            "fl2va_8step_lora": (loras, {"default": TURBO_LORAS[(False, "8")][0], "tooltip": "Audio + text/image, 8-step Turbo. Default: LightX2V FL2V v1.0 768p. Sampling remains 8 steps and shifts 6 / 3."}),
            "ref2va_4step_lora": (loras, {"default": TURBO_LORAS[(True, "4")][0], "tooltip": "Gallery references, 4-step Turbo. Default: LightX2V Ref2V v0.1. Sampling remains 4 steps and shifts 12 / 3."}),
            "ref2va_8step_lora": (loras, {"default": TURBO_LORAS[(True, "8")][0], "tooltip": "Gallery references, 8-step Turbo. Default: LightX2V Ref2V v1.0 768p. Sampling remains 8 steps and shifts 12 / 3."}),
        }}

    RETURN_TYPES = ("MODEL", "INT", "FLOAT", "FLOAT", "VAE")
    RETURN_NAMES = ("model", "steps", "video_shift", "audio_shift", "video_vae")
    FUNCTION = "apply"
    CATEGORY = "SECourses/MiniMax H3 Lip Synch"

    def apply(self, model, video_vae, uses_ref2va, enable_speed_lora=False, turbo_steps="4", quality_steps=50, lora_strength=1.0,
              enable_sol_attention=False, enable_first_block_cache=False, save_vram=True, exact_output=True, fast_vae_decode=True,
              fl2va_4step_lora="", fl2va_8step_lora="", ref2va_4step_lora="", ref2va_8step_lora=""):
        shift, steps, filename = 12.0, quality_steps, "no speed LoRA"
        if enable_speed_lora:
            filename, shift = TURBO_LORAS[(bool(uses_ref2va), turbo_steps)]
            choices = {(False, "4"): fl2va_4step_lora, (False, "8"): fl2va_8step_lora, (True, "4"): ref2va_4step_lora, (True, "8"): ref2va_8step_lora}
            filename = choices[(bool(uses_ref2va), turbo_steps)] or filename
            available = folder_paths.get_filename_list("loras")
            matches = [name for name in available if name == filename] or [name for name in available if name.replace("\\", "/").split("/")[-1] == filename]
            if not matches:
                raise FileNotFoundError(f"Turbo LoRA missing: {filename}. Select an installed LoRA, or disable speed LoRA. Default weights: lightx2v/Minimax-h3-Turbo.")
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, matches[0], lora_strength)[0]
            steps = int(turbo_steps)
        model = nodes.NODE_CLASS_MAPPINGS["MiniMaxH3SpeedOptimizer"]().apply(
            model=model, first_block_cache=enable_first_block_cache, fbc_threshold=0.08,
            fbc_start_percent=0.15, fbc_end_percent=0.95, fbc_max_consecutive=3,
            sparse_attention="auto" if enable_sol_attention else "disabled",
            sparse_dense_steps_pct=0.2, sparse_dense_layers=2, fbc_cache_device="cpu" if save_vram else "gpu",
            enable_speedup=enable_sol_attention or enable_first_block_cache,
            sparse_extra_tokens=256, sparse_dense_last_steps=1)[0]
        model = nodes.NODE_CLASS_MAPPINGS["MiniMaxH3LowVRAM"]().apply(model, save_vram, exact_output=exact_output)[0]
        video_vae = nodes.NODE_CLASS_MAPPINGS["MiniMaxH3VAESpeedup"]().apply(video_vae, tile_batch_size=1 if save_vram else 0, enable_speedup=fast_vae_decode)[0]
        report = f"{filename}\n{steps} steps | shifts {shift:g} / 3 | SOL {enable_sol_attention} | cache {enable_first_block_cache} | save VRAM {save_vram}"
        return {"ui": {"text": [report]}, "result": (model, steps, shift, 3.0, video_vae)}



NODE_CLASS_MAPPINGS = {"SEH3LipSyncPrompt": SEH3LipSyncPrompt, "SEH3LipSyncTurbo": SEH3LipSyncTurbo}
NODE_DISPLAY_NAME_MAPPINGS = {"SEH3LipSyncPrompt": "H3 Lip Synch - Automatic Prompt", "SEH3LipSyncTurbo": "H3 Lip Synch - Quality / Turbo"}
