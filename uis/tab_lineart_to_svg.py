# coding=utf-8

import gradio as gr

from app.tools import convert2svg_image
from uis.tabs import TabId


def upload_file(files):
    try:
        results = []
        for file in files:
            svg_result = convert2svg_image(file, after_delete=True)
            results.append(svg_result)
        return gr.UploadButton(label="更换文件"), results
    except Exception as e:
        raise gr.Error(f"{e}, 请重试")


def build_line_art2svg_ui():
    with gr.TabItem("图片线稿转 SVG", id=TabId.LINE_ART2SVG.value):
        svg_gallery = gr.Gallery(
            show_label=False, format="svg",
            columns=4, rows=1, object_fit="contain")
        upload_button = gr.UploadButton(
            label="上传图片",
            file_types=['image'],
            file_count='multiple'
        )
        upload_button.upload(upload_file, [upload_button], [upload_button, svg_gallery])
