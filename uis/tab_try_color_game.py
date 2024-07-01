# coding=utf-8
import gradio as gr

from app.colorful_svg import read_svg_metadata
from app.colorful_svg import update_svg_colors

current_file_path = ""
current_colors_config = {}
zero_colors = {}


def upload_file(file, colors_dw):
    color_config = read_svg_metadata(file)
    if color_config is None:
        raise gr.Error("请导入颜色聚类后的 svg 文件")
    colors = list(color_config.keys())
    global current_colors_config, zero_colors, current_file_path
    zero_colors.clear()
    current_colors_config = color_config
    current_file_path = file
    for c in colors:
        zero_colors[c] = "#ffffff"
    new_path = update_svg_colors(file, zero_colors)
    print(new_path)
    return (gr.UploadButton(label="更换文件"), gr.Button(visible=True),
            gr.update(colors_dw, choices=colors, visible=True), gr.Image(value=new_path, type="filepath", visible=True))


def start_change_color(colors):
    if colors is None:
        colors = []
    global current_colors_config, zero_colors, current_file_path
    for color_id, old_color in zero_colors.items():
        hex_color = current_colors_config[color_id]
        if color_id in colors:
            zero_colors[color_id] = hex_color
        else:
            zero_colors[color_id] = "#ffffff"
    new_path = update_svg_colors(current_file_path, zero_colors)
    print(new_path)
    return gr.Image(value=new_path, type="filepath")


def build_try_color_game_ui():
    with gr.TabItem("尝试填色游戏", id=3):
        image = gr.Image(format="svg", visible=False, width=512)
        colors = gr.CheckboxGroup(
            visible=False,
            label="请选择要填色的色块号"
        )
        btn = gr.Button("开始填色", variant="primary", visible=False)
        btn.click(
            start_change_color,
            colors,
            image
        )
        upload_button = gr.UploadButton(
            label="上传 svg",
            file_types=['.svg']
        )
        upload_button.upload(upload_file, [upload_button, colors], [upload_button, btn, colors, image])
