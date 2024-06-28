# coding=utf-8
import json

import gradio as gr

from app.sd_tools import get_models
from app.sd_tools import get_train_loras

sampling_method = [
    "DPM++ 2M",
    "DPM++ SDE",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Heun",
    "DPM++ 2S a",
    "DPM++ 3M SDE",
    "Euler a",
    "Euler",
    "LMS",
    "Heun",
    "DPM2",
    "DPM2 a",
    "DPM fast",
    "DPM adaptive",
    "Restart",
    "DDIM",
    "PLMS",
    "UniPC",
    "LCM",
]

schedule_type = [
    "Automatic",
    "Uniform",
    "Karras",
    "Exponential",
    "Polyexponential",
    "SGM Uniform",
]


def refresh_models(d):
    return gr.update(d, choices=get_models())


def refresh_loras(d):
    return gr.update(d, choices=get_train_loras())


def get_all_category() -> list[str]:
    with open('./data/category_prompts.json', 'r') as f:
        result = json.loads(f.read())
    return list(result.keys())


all_category = get_all_category()
