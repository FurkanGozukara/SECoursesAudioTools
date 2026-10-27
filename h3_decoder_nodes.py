"""Select an H3 decoder without changing image/reference encoding or the audio VAE."""
import folder_paths
import nodes

ORIGINAL = "Original video VAE"
INT8 = "LightVAE/lightvae-lite-h3-decoder-int8-convrot.safetensors"


class SEH3VideoDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        names = [name for name in folder_paths.get_filename_list("vae") if "lightvae" in name.lower() and "h3-decoder" in name.lower()]
        normalized = {name.replace("\\", "/"): name for name in names}
        return {"required": {
            "video_vae": ("VAE", {"tooltip": "Original H3 video VAE. Image and reference encoders keep using this VAE."}),
            "decoder": ([ORIGINAL, *names], {"default": normalized.get(INT8, ORIGINAL),
                "tooltip": "LightVAE replaces only the final video decoder. Choose Original video VAE for a direct comparison. Audio is unchanged."}),
        }}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("decode_vae",)
    FUNCTION = "select"
    CATEGORY = "SECourses/MiniMax H3"

    def select(self, video_vae, decoder=ORIGINAL):
        if decoder == ORIGINAL:
            return (video_vae,)
        return nodes.VAELoader().load_vae(decoder)


NODE_CLASS_MAPPINGS = {"SEH3VideoDecoder": SEH3VideoDecoder}
NODE_DISPLAY_NAME_MAPPINGS = {"SEH3VideoDecoder": "H3 Optional LightVAE Video Decoder"}
