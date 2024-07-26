import gradio as gr

import app.config
from app.tools import read_file_to_list_of_tuples
from uis.tab_batch_generate_v2 import build_batch_generate_v2_ui
from uis.tab_gan_extract_line import build_gan_extract_line_ui
from uis.tab_generate_line_art import build_generate_line_art_ui
from uis.tab_generate_line_art_v2 import build_generate_line_art_v2_ui
from uis.tab_image_to_svg import build_image_to_svg_ui
from uis.tab_try_color_game import build_try_color_game_ui
from uis.tab_try_color_game import upload_file
from uis.tabs import TabId


def send_to_color_game(svg_file, colors_dw):
    if isinstance(svg_file, dict):
        svg_file = svg_file['image']['path']
    print(f"svg_file: {svg_file}")
    print(f"colors_dw: {colors_dw}")
    # 返回多个更新
    upload_button, reset_bt, image, colors = upload_file(svg_file, colors_dw)
    return (
        upload_button, reset_bt, image, colors,
        gr.Tabs(selected=TabId.TRY_COLOR_GAME.value)  # 切换到填色游戏 Tab
    )


def build_webui(default_tab: TabId):
    custom_css = """
.dark .thumbnail-item {
    background-color: white !important;
}
.dark .image-button {
    background-color: white !important;
}
.dark .image-container {
    background-color: white !important;
}
"""
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as webui:
        gr.Markdown("# 填色游戏图片工厂")
        gr.Markdown(
            "## 通过 AI 生成相关提示词，再用 Stable Diffusion 批量生成填色游戏中的图片")
        send_to_color_game_btn_list = []
        with gr.Tabs(selected=default_tab.value) as tabs:
            send_to_color_game_btn_list.append(build_batch_generate_v2_ui())
            build_generate_line_art_ui()
            send_to_color_game_btn_list.append(build_generate_line_art_v2_ui())
            upload_button, reset_bt, image, colors = build_try_color_game_ui()
            build_gan_extract_line_ui()
            send_to_color_game_btn_list.append(build_image_to_svg_ui())

        # 连接生成线稿的 Tab 和填色游戏的 Tab
        for item in send_to_color_game_btn_list:
            send_to_coloring_game_btn, selected_image = item
            send_to_coloring_game_btn.click(
                send_to_color_game,
                inputs=[selected_image, colors],
                outputs=[upload_button, reset_bt, image, colors, tabs]
            )

    auths = read_file_to_list_of_tuples(app.config.net_auth_file_path)
    webui.launch(show_api=False, server_name=app.config.net_host, server_port=app.config.net_port, auth=auths)


if __name__ == '__main__':
    build_webui(default_tab=TabId.GENERATE_LINE_ART_SVG)
