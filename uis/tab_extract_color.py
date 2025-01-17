# coding=utf-8
import gradio as gr

from app.tools import resolve_relative_path
from uis.tabs import TabId


def build_extract_color_ui():
    with gr.TabItem("SVG 区域标记示例", id=TabId.EXTRACT_COLOR.value):
        with open(resolve_relative_path(__file__, '../data/extract_color.html'), 'r') as f:
            gr.HTML(f.read())
