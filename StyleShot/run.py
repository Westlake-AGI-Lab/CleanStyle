# run.py
import os
import time
import datetime
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from config import load_config
from ip_adapter import StyleShot


def main():
    cfg = load_config()
    h = cfg["hyper"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ip_ckpt = os.path.join(cfg["styleshot_model_path"], "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(
        cfg["styleshot_model_path"], "pretrained_weight/style_aware_encoder.bin"
    )

    pipe = StableDiffusionPipeline.from_pretrained(cfg["base_model_path"]).to(device, dtype=torch.float16)
    pipe.set_progress_bar_config(disable=False)

    styleshot = StyleShot(
        device=device,
        pipe=pipe,
        ip_ckpt=ip_ckpt,
        style_aware_encoder_ckpt=style_aware_encoder_path,
        transformer_patch=cfg["transformer_block_path"],
        num_inference_steps=cfg["num_inference_steps"],
        attn_svd=h["attn_svd"],
        attn_cfg=h["attn_cfg"],
        attn_svd_k=h["attn_svd_k"],
        attn_svd_alpha=h["attn_svd_alpha"],
        attn_svd_gamma=h["attn_svd_gamma"],
        attn_svd_center=h["attn_svd_center"],
    )

    style_img = Image.open(cfg["style_image"]).convert("RGB")

    t0 = time.time()
    generation = styleshot.generate(
        style_image=style_img,
        prompt=[[cfg["prompt"]]],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=h["guidance_scale"],
        encoder_svd=h["encoder_svd"],
        encoder_cfg=h["encoder_cfg"],
    )
    print(f"[INFO] generation time: {time.time() - t0:.3f}s")

    images = generation[0]
    save_name = f"example_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    images[0].save(save_name)
    print(f"[Res] saved: {save_name}")


if __name__ == "__main__":
    main()
