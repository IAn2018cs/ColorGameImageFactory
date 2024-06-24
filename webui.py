import json

import gradio as gr

import app.config
from app.prompt_factory import create_sd_prompts
from app.sd_tools import convert_image_line_art
from app.sd_tools import generate_image_by_sd
from app.sd_tools import get_models
from app.sd_tools import get_styles
from app.sd_tools import get_train_loras
from app.tools import generate_random_id
from app.tools import get_all_file_path
from app.tools import get_current_datatime
from app.tools import read_file_to_list_of_tuples
from app.tools import resolve_relative_path
from app.tools import split_list_with_min_length
from app.tools import zip_dir

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


def start_gan(category, image_count, model, lora, weights, trigger, negative, styles, sampling, schedule, step, cfg):
    prompt_count = min(6, image_count)
    n_iter = max(int(image_count / prompt_count), 1)
    prompts = create_sd_prompts(category, prompt_count)

    result = []
    root_path = resolve_relative_path(__file__, './output')
    batch_id = generate_random_id(16)
    result.extend(
        generate_images(batch_id, cfg, lora, model, n_iter, negative, prompts, root_path, sampling, schedule, step,
                        styles, trigger, weights)
    )

    if image_count > prompt_count:
        last = image_count % prompt_count
        if last > 0:
            prompts = create_sd_prompts(category, last)
            result.extend(
                generate_images(batch_id, cfg, lora, model, 1, negative, prompts, root_path, sampling, schedule,
                                step,
                                styles, trigger, weights)
            )

    zip_file = zip_dir(f'{root_path}/{batch_id}', batch_id, root_path)

    line_art_result = []
    line_art_batch_id = generate_random_id(16)
    split_list = split_list_with_min_length(result, 5)
    for image_paths in split_list:
        split_result = convert_image_line_art(root_path, line_art_batch_id, image_paths, to_svg=True)
        line_art_result.extend(split_result)

    line_art_zip_file = zip_dir(f'{root_path}/{line_art_batch_id}', line_art_batch_id, root_path)

    return (result, line_art_result,
            gr.DownloadButton(value=zip_file, visible=True), gr.DownloadButton(value=line_art_zip_file, visible=True))


def generate_images(batch_id, cfg, lora, model, n_iter, negative, prompts, root_path, sampling, schedule, step,
                    styles, trigger, weights):
    lora = str(lora).strip()
    trigger = str(trigger).strip()
    result = []
    print(f"lora: {lora}")
    print(f"trigger prompt: {trigger}")

    for prompt in prompts:
        new_prompt = ""
        if lora != "" and lora != "None":
            new_prompt += f'<lora:{lora}:{weights}>, '
        if trigger != "" and trigger != "None":
            new_prompt += f'{trigger}, '
        new_prompt += prompt
        images = generate_image_by_sd(
            root_path, batch_id,
            model, new_prompt, negative, step, cfg, sampling, schedule, 1024, 1024, styles,
            n_iter
        )
        result.extend(images)
    return result


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


def build_webui():
    all_category = get_all_category()
    custom_css = """
.dark .thumbnail-item {
    background-color: white !important;
}
.dark .image-button {
    background-color: white !important;
}
"""
    with gr.Blocks(css=custom_css) as webui:
        gr.Markdown("# 填色游戏图片工厂")
        gr.Markdown(
            "## 通过 AI 生成相关提示词，再用 Stable Diffusion 批量生成填色游戏中的图片")
        with gr.Tab("批量生图"):
            category = gr.Dropdown(
                choices=all_category,
                value=all_category[0],
                multiselect=False,
                allow_custom_value=True,
                label="图片分类（比如 Food、Collections、Buildings 等，可以自定义）"
            )
            image_count = gr.Slider(
                value=2,
                minimum=1,
                maximum=100,
                step=1,
                label="生成图片的数量"
            )
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(
                        value=app.config.sd_default_model,
                        choices=get_models(),
                        multiselect=False,
                        label="Stable Diffusion checkpoint"
                    )
                    refresh_model_button = gr.Button("🔄", size="sm")
                    refresh_model_button.click(refresh_models, model, model)
                with gr.Row(equal_height=False):
                    with gr.Column():
                        lora = gr.Dropdown(
                            choices=get_train_loras(),
                            multiselect=False,
                            label="Lora"
                        )
                        refresh_lora_button = gr.Button("🔄", size="sm")
                        refresh_lora_button.click(refresh_loras, lora, lora)
                    weights = gr.Slider(
                        value=1,
                        minimum=0,
                        maximum=2,
                        step=0.05,
                        label="Lora weights"
                    )
            trigger = gr.Textbox(
                placeholder="Lora 的触发提示词（可以为空）",
                label="Trigger prompt",
            )
            negative = gr.Textbox(
                placeholder="反向提示词（可以为空）",
                value="text, watermark, negativeXL_D",
                label="Negative prompt"
            )
            styles = gr.Dropdown(
                choices=get_styles(),
                multiselect=True,
                label="Styles"
            )
            with gr.Row():
                sampling = gr.Dropdown(
                    choices=sampling_method,
                    value=sampling_method[0],
                    multiselect=False,
                    label="Sampling method"
                )
                schedule = gr.Dropdown(
                    choices=schedule_type,
                    value=schedule_type[0],
                    multiselect=False,
                    label="Schedule type"
                )
            step = gr.Slider(
                value=20,
                minimum=1,
                maximum=150,
                step=1,
                label="Sampling steps"
            )
            cfg = gr.Slider(
                value=7,
                minimum=1,
                maximum=30,
                step=0.5,
                label="CFG Scale"
            )
            with gr.Row():
                with gr.Column():
                    gallery = gr.Gallery(
                        label="原图", format="png",
                        columns=4, rows=1, object_fit="contain")
                    download_all_button = gr.DownloadButton("下载所有原图", visible=False)
                with gr.Column():
                    line_art_gallery = gr.Gallery(
                        label="线稿图", format="svg",
                        columns=4, rows=1, object_fit="contain")
                    download_line_art_button = gr.DownloadButton("下载所有线稿图", visible=False)

            btn = gr.Button("开始批量生成", variant="primary")
            btn.click(
                fn=start_gan,
                inputs=[
                    category, image_count,
                    model, lora, weights, trigger, negative, styles, sampling, schedule, step, cfg
                ],
                outputs=[
                    gallery,
                    line_art_gallery,
                    download_all_button,
                    download_line_art_button
                ],
                scroll_to_output=True
            )
        with gr.Tab("批量转线稿图"):
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

    auths = read_file_to_list_of_tuples(app.config.net_auth_file_path)
    webui.launch(show_api=False, server_name=app.config.net_host, server_port=app.config.net_port, auth=auths)


if __name__ == '__main__':
    build_webui()
