# coding=utf-8
import os

import requests

import app.config
from app.tools import create_path
from app.tools import get_column_data
from app.tools import get_timestamp
from app.tools import save_base64_image


def get_loras(root_path: str = app.config.sd_lora_path) -> list[str]:
    loras = []
    for item in os.listdir(root_path):
        if item.startswith("."):
            continue
        child_path = f"{root_path}/{item}"
        if os.path.isdir(child_path):
            loras.extend(get_loras(child_path))
        else:
            loras.append(item.split(".")[0])
    return loras


def get_models(root_path: str = app.config.sd_model_path) -> list[str]:
    models = []
    for item in os.listdir(root_path):
        if item.startswith("."):
            continue
        child_path = f"{root_path}/{item}"
        if os.path.isdir(child_path):
            models.extend(get_models(child_path))
        else:
            models.append(item.split(".")[0])
    return models


def get_styles() -> list[str]:
    return get_column_data(app.config.sd_styles_file, "name")


def generate_image_by_sd(root_path: str, batch_id: str,
                         model: str, prompt: str, negative_prompt: str,
                         steps: int, cfg: float,
                         sampler_name: str, scheduler: str,
                         width: int, height: int,
                         styles: list[str] = [], n_iter: int = 1):
    try:
        output_path = f'{root_path}/{batch_id}'
        create_path(output_path)

        url = f'{app.config.api_sd_host}/sdapi/v1/txt2img'
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg,
            "width": width,
            "height": height,
            "styles": styles,
            "batch_size": 1,
            "n_iter": n_iter,
            "seed": -1,
            "override_settings": {
                "sd_model_checkpoint": model
            },
            "save_images": False,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        }
        response = requests.post(url, json=data)
        print(response.json()['info'])
        images = response.json()['images']
        files = []
        for image in images:
            path = f'{output_path}/{get_timestamp()}.png'
            save_base64_image(path, image)
            files.append(path)
        return files
    except Exception as e:
        print(f"generate_by_openai error: {e}")
        return None
