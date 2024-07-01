# coding=utf-8
import gradio as gr

from app.sd_tools import convert_image_line_art
from app.tools import generate_random_id
from app.tools import get_all_file_path
from app.tools import get_current_datatime
from app.tools import split_list_with_min_length
from app.tools import zip_dir


def start_convert_line_art(source_path, target_path):
    root_path = target_path
    batch_id = f"{get_current_datatime()}_{generate_random_id(4)}"

    files = get_all_file_path(source_path, ["png", "jpg", "jpeg", "webp"])

    line_art_result = []

    split_list = split_list_with_min_length(files, 5)
    for image_paths in split_list:
        split_result = convert_image_line_art(root_path, batch_id, image_paths, to_svg=True)
        line_art_result.extend(split_result)

    line_art_zip_file = zip_dir(f'{root_path}/{batch_id}', batch_id, root_path)

    return (line_art_result, gr.DownloadButton(value=line_art_zip_file, visible=True),
            gr.Textbox(value=f'{root_path}/{batch_id}', visible=True))


def build_batch2line_art_ui():
    with gr.TabItem("批量转线稿图", id=1):
        with gr.Row():
            source_path = gr.Textbox(
                placeholder="请填写服务器上的绝对路径（末尾不要带 / ）",
                label="待转换的图片文件夹路径",
            )
            target_path = gr.Textbox(
                placeholder="请填写服务器上的绝对路径（末尾不要带 / ）",
                label="转换后线稿图输出到哪里"
            )
        result_path = gr.Textbox(label="已经导出到下面路径中", interactive=False, visible=False)
        line_gallery = gr.Gallery(
            show_label=False, format="svg",
            columns=4, rows=1, object_fit="contain")
        download_line_button = gr.DownloadButton("下载所有图", visible=False)
        line_btn = gr.Button("开始批量转换", variant="primary")
        line_btn.click(
            fn=start_convert_line_art,
            inputs=[
                source_path, target_path
            ],
            outputs=[
                line_gallery,
                download_line_button,
                result_path
            ],
            scroll_to_output=True
        )
