# coding=utf-8

import requests

import app.config
from app.tools import convert2rgb_image
from app.tools import create_path
from app.tools import delete_file
from app.tools import generate_random_id
from app.tools import get_all_file_path
from app.tools import get_base64_image
from app.tools import get_column_data
from app.tools import get_timestamp
from app.tools import save_base64_image
import shutil


def get_loras() -> list[str]:
    def get_file_name(path, item):
        return item.split(".")[0]

    return get_all_file_path(app.config.sd_lora_path, ["safetensors", "pt"], get_file_name)


def get_models() -> list[str]:
    def get_file_name(path, item):
        return item.split(".")[0]

    return get_all_file_path(app.config.sd_model_path, ["safetensors", "pt"], get_file_name)


def get_train_loras() -> list[str]:
    exit_loras = get_loras()

    def get_file_name(path, item):
        name = item.split(".")[0]
        if name in exit_loras:
            return name
        output_path = str(path).replace(f"{app.config.train_output_path}/", '')
        new_path = f'{app.config.sd_lora_path}/{output_path}'
        create_path(new_path)
        shutil.copy(f'{path}/{item}', f'{new_path}/{item}')
        return name

    return get_all_file_path(app.config.train_output_path, ["safetensors", "pt"], get_file_name)


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
            path = f'{output_path}/{get_timestamp()}_{generate_random_id(4)}.png'
            save_base64_image(path, image)
            files.append(path)
        return files
    except Exception as e:
        print(f"generate_image_by_sd error: {e}")
        return None


def controlnet_image_preprocessor(root_path: str, batch_id: str, preprocessor: str,
                                  input_images, processor_res=None, threshold_a=None, threshold_b=None):
    try:
        output_path = f'{root_path}/{batch_id}'
        create_path(output_path)

        url = f'{app.config.api_sd_host}/controlnet/detect'
        data = {
            "controlnet_module": preprocessor,
            "controlnet_input_images": input_images
        }
        if processor_res:
            data["controlnet_processor_res"] = processor_res
        if threshold_a:
            data["controlnet_threshold_a"] = threshold_a
        if threshold_b:
            data["controlnet_threshold_b"] = threshold_b
        response = requests.post(url, json=data)
        if not response.ok:
            raise Exception(response.text)
        print(response.json()['info'])
        images = response.json()['images']
        files = []
        for image in images:
            path = f'{output_path}/{get_timestamp()}_{generate_random_id(4)}.png'
            save_base64_image(path, image)
            files.append(path)
        return files
    except Exception as e:
        print(f"controlnet_image_preprocessor error: {e}")
        return None


def convert_image_line_art(root_path: str, batch_id: str, image_paths: list[str]):
    base64_images = [get_base64_image(convert2rgb_image(path)) for path in image_paths]
    black_image_paths = controlnet_image_preprocessor(root_path, batch_id, "softedge_anyline", base64_images,
                                                      threshold_a=2)
    base64_images2 = [get_base64_image(path) for path in black_image_paths]
    for path in black_image_paths:
        delete_file(path)
    return controlnet_image_preprocessor(root_path, batch_id, "invert", base64_images2)
