# config.py
import argparse
import yaml


REQUIRED_KEYS = [
    "base_model_path",
    "image_encoder_path",
    "ip_ckpt",
    "prompt",
    "style_image",
]


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for k in REQUIRED_KEYS:
        if k not in cfg:
            raise ValueError(f"Missing required key: {k}")

    cfg.setdefault("negative_prompt", "")
    cfg.setdefault("ipadapter_scale", 1.0)
    cfg.setdefault("guidance_scale", 5.0)
    cfg.setdefault("num_samples", 1)
    cfg.setdefault("num_inference_steps", 30)
    cfg.setdefault("seed", 42)
    cfg.setdefault("target_blocks", ["up_blocks.0.attentions.1"])
    cfg.setdefault("hyper", {})

    h = cfg["hyper"]
    h.setdefault("tag", "default")
    h.setdefault("encoder_svd", False)
    h.setdefault("encoder_cfg", False)
    h.setdefault("attn_svd", False)
    h.setdefault("attn_cfg", False)
    h.setdefault("attn_svd_k", 1)
    h.setdefault("attn_svd_alpha", 0.01)
    h.setdefault("attn_svd_beta", 1.0)
    h.setdefault("attn_svd_gamma", 20.0)
    h.setdefault("attn_svd_center", 0.3)

    return cfg
