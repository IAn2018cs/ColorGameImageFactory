# coding=utf-8
import os
import shutil

import gradio as gr

from app.gen_tools import extract_line_by_gan
from app.tools import convert2svg_image
from app.tools import create_path
from app.tools import generate_random_id
from app.tools import get_timestamp
from app.tools import resolve_relative_path


def upload_file(files):
    try:
        root_path = resolve_relative_path(__file__, '../output')
        output_dir = f'{root_path}/gen_line_outputs'
        create_path(output_dir)

        source_dir = f'{root_path}/gen_line_inputs/{generate_random_id(12)}'
        create_path(source_dir)

        results = []
        for file in files:
            file_name_split = os.path.split(file)[-1].split('.')
            name_ext = file_name_split[-1]
            new_path = f'{source_dir}/{get_timestamp()}_{generate_random_id(4)}.{name_ext}'
            shutil.move(file, new_path)

        for path in extract_line_by_gan(f'{source_dir}/', f'{output_dir}/'):
            svg_result = convert2svg_image(path, after_delete=True)
            results.append(svg_result)
        return gr.UploadButton(label="更换文件"), results
    except Exception as e:
        raise gr.Error(f"{e}, 请重试")


def build_gan_extract_line_ui():
    with gr.TabItem("GAN 模型提取线稿", id=4):
        svg_gallery = gr.Gallery(
            show_label=False, format="svg",
            columns=4, rows=1, object_fit="contain")
        upload_button = gr.UploadButton(
            label="上传图片",
            file_types=['image'],
            file_count='multiple'
        )
        upload_button.upload(upload_file, [upload_button], [upload_button, svg_gallery])
