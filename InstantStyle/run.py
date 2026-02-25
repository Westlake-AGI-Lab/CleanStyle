# run.py
import time
import datetime
from pathlib import Path

import torch
from PIL import Image

from utils.config import load_config
from ip_adapter import IPAdapterXL
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

def main() -> None:
    cfg = load_config()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["base_model_path"],
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=False)

    h = cfg["hyper"]
    ip_model = IPAdapterXL(
        sd_pipe=pipe,
        image_encoder_path=cfg["image_encoder_path"],
        ip_ckpt=cfg["ip_ckpt"],
        device="cuda",
        target_blocks=cfg["target_blocks"],
        num_inference_steps=cfg["num_inference_steps"],
        attn_svd=h["attn_svd"],
        attn_cfg=h["attn_cfg"],
        attn_svd_k=h["attn_svd_k"],
        attn_svd_alpha=h["attn_svd_alpha"],
        attn_svd_gamma=h["attn_svd_gamma"],
        attn_svd_center=h["attn_svd_center"],
    )

    style_path = cfg["style_image"]
    style_img = Image.open(style_path).convert("RGB").resize((512, 512))

    t0 = time.time()
    images = ip_model.generate(
        pil_image=style_img,
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        scale=cfg["ipadapter_scale"],
        guidance_scale=cfg["guidance_scale"],
        num_samples=cfg["num_samples"],
        num_inference_steps=cfg["num_inference_steps"],
        seed=cfg["seed"],
        encoder_cfg=h["encoder_cfg"],
        encoder_svd=h["encoder_svd"],
    )
    print(f"[INFO] generation time: {time.time() - t0:.3f}s")

    save_name = f"example_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    images[0].save(save_name)
    print(f"[Res] saved: {save_name}")


if __name__ == "__main__":
    main()