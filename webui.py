import gradio as gr

import app.config
from app.tools import read_file_to_list_of_tuples
from uis.tab_batch2line_art import build_batch2line_art_ui
from uis.tab_batch_generate import build_batch_generate_ui
from uis.tab_generate_line_art import build_generate_line_art_ui
from uis.tab_try_color_game import build_try_color_game_ui


def build_webui():
    custom_css = """
.dark .thumbnail-item {
    background-color: white !important;
}
.dark .image-button {
    background-color: white !important;
}
"""
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as webui:
        gr.Markdown("# 填色游戏图片工厂")
        gr.Markdown(
            "## 通过 AI 生成相关提示词，再用 Stable Diffusion 批量生成填色游戏中的图片")
        with gr.Tabs(selected=2):
            build_batch_generate_ui()
            build_batch2line_art_ui()
            build_generate_line_art_ui()
            build_try_color_game_ui()

    auths = read_file_to_list_of_tuples(app.config.net_auth_file_path)
    webui.launch(show_api=False, server_name=app.config.net_host, server_port=app.config.net_port, auth=auths)


if __name__ == '__main__':
    build_webui()
