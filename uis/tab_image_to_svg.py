# coding=utf-8
import os
import shutil

import gradio as gr

from app.colorful_svg_v2 import color_quantization
from app.gen_tools import extract_line_by_gan
from app.tools import convert2svg_image
from app.tools import create_path
from app.tools import generate_random_id
from app.tools import resolve_relative_path
from uis.tabs import TabId


def convert2svg(root_path, image):
    gan_batch_id = generate_random_id(16)
    images_root_path = f'{root_path}/{gan_batch_id}/'
    create_path(images_root_path)

    shutil.copy(image, f"{images_root_path}{os.path.basename(image)}")
    line_result = extract_line_by_gan(images_root_path, images_root_path)
    result = []
    for path in line_result:
        new_path = convert2svg_image(path, after_delete=True)
        result.append(new_path)
    return result[0]


def generate_quantization_images(root_path, colorful_image, line_svg_image, num_colors, min_ares):
    quantization_batch_id = generate_random_id(16)
    output_dir = f'{root_path}/{quantization_batch_id}'
    create_path(output_dir)

    svg_img = color_quantization(colorful_image, line_svg_image, output_dir, num_colors, min_ares)
    return svg_img


def validate_image(image):
    if image is None:
        return False
    file_extension = os.path.splitext(image)[1].lower()
    if file_extension not in ['.png', '.jpg', '.jpeg', '.webp']:
        return False
    return True


def start_to_svg(color_image, num_colors, min_ares):
    try:
        if color_image is None:
            gr.Warning("请先上传图片")
            return gr.Image(), gr.Button(visible=False)
        if not validate_image(color_image):
            gr.Warning("只能上传以下格式：png、jpg、jpeg、webp")
            return gr.Image(), gr.Button(visible=False)
        print(color_image)
        root_path = resolve_relative_path(__file__, '../output')
        # 1. 通过 gan 模型提取线稿，转成 svg -> 线稿图 保存一个结果
        svg_image = convert2svg(root_path, color_image)

        # 2. 转成 彩色 svg
        quantization_svg_image = generate_quantization_images(root_path,
                                                              color_image, svg_image,
                                                              num_colors, min_ares)
        return (
            gr.Image(value=quantization_svg_image, type="filepath"),
            gr.Button(visible=True)
        )
    except Exception as e:
        raise gr.Error(f"发生错误：{e}，请重试")


def upload_file(file):
    print(file)
    return (gr.UploadButton(label="更换文件"),
            gr.Image(value=file, type="filepath", visible=True))


def build_image_to_svg_ui():
    with gr.TabItem("彩图转 SVG", id=TabId.IMAGE2SVG.value):
        with gr.Row():
            color_image = gr.Image(width=512, type='filepath', sources=["upload", "clipboard"], show_label=False)
            svg_image = gr.Image(format='svg', type='filepath', width=512, interactive=False, show_label=False)
        num_colors = gr.Slider(
            value=30,
            minimum=10,
            maximum=100,
            step=1,
            label="合并相似颜色阈值（值越大，颜色越少，越小，颜色越多）"
        )
        min_ares = gr.Slider(
            value=2000,
            minimum=0,
            maximum=10000,
            step=10,
            label="合并面积小的色块阈值（值越大，颜色越少，越小，颜色越多）"
        )
        send_to_coloring_game_btn = gr.Button("发送到填色游戏", visible=False)
        btn = gr.Button("开始转换", variant="primary")
        btn.click(
            fn=start_to_svg,
            inputs=[
                color_image, num_colors, min_ares
            ],
            outputs=[
                svg_image,
                send_to_coloring_game_btn
            ]
        )
    return send_to_coloring_game_btn, svg_image
