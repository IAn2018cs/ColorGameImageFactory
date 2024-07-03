# coding=utf-8
import os
import shutil

import gradio as gr

from app.tools import convert2svg_image
from app.tools import create_path
from app.tools import extract_line_by_gan
from app.tools import generate_random_id
from app.tools import get_timestamp
from app.tools import resolve_relative_path


def upload_file(file):
    root_path = resolve_relative_path(__file__, '../output')

    output_dir = f'{root_path}/gen_line_outputs'
    create_path(output_dir)

    source_dir = f'{root_path}/gen_line_inputs'
    create_path(output_dir)

    file_name_split = os.path.split(file)[-1].split('.')
    name_ext = file_name_split[-1]
    new_path = f'{source_dir}/{get_timestamp()}_{generate_random_id(4)}.{name_ext}'
    shutil.move(file, new_path)

    result = extract_line_by_gan(f'{source_dir}/', f'{output_dir}/')[0]
    svg_result = convert2svg_image(result, after_delete=True)
    print(svg_result)
    return gr.UploadButton(label="更换文件"), gr.Image(value=svg_result, type="filepath", visible=True)


def build_gan_extract_line_ui():
    with gr.TabItem("GAN 模型提取线稿", id=4):
        image = gr.Image(format="svg", visible=False, width=512, show_label=False)
        upload_button = gr.UploadButton(
            label="上传图片",
            file_types=['image']
        )
        upload_button.upload(upload_file, [upload_button], [upload_button, image])
