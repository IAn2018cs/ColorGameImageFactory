# coding=utf-8

import gradio as gr

import app.config
from app.prompt_factory import create_sd_prompts
from app.sd_tools import convert_image_line_art
from app.sd_tools import generate_image_by_sd
from app.sd_tools import get_models
from app.sd_tools import get_styles
from app.sd_tools import get_train_loras
from app.tools import generate_random_id
from app.tools import resolve_relative_path
from app.tools import split_list_with_min_length
from app.tools import zip_dir
from uis.tools import all_category
from uis.tools import refresh_loras
from uis.tools import refresh_models
from uis.tools import sampling_method
from uis.tools import schedule_type


def start_gan(category, image_count, model, lora, weights, trigger, negative, styles, sampling, schedule, step, cfg):
    prompt_count = min(6, image_count)
    n_iter = max(int(image_count / prompt_count), 1)
    prompts = create_sd_prompts(category, prompt_count)

    result = []
    root_path = resolve_relative_path(__file__, '../output')
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


def build_batch_generate_ui():
    with gr.TabItem("批量生图测试", id=0):
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
                    value=app.config.default_color_sd_model,
                    choices=get_models(),
                    multiselect=False,
                    label="Stable Diffusion checkpoint"
                )
                refresh_model_button = gr.Button("🔄", size="sm")
                refresh_model_button.click(refresh_models, model, model)
            with gr.Row(equal_height=False):
                with gr.Column():
                    lora = gr.Dropdown(
                        value=app.config.default_color_sd_lora,
                        choices=get_train_loras(),
                        multiselect=False,
                        label="Lora"
                    )
                    refresh_lora_button = gr.Button("🔄", size="sm")
                    refresh_lora_button.click(refresh_loras, lora, lora)
                weights = gr.Slider(
                    value=app.config.default_color_sd_lora_weight,
                    minimum=0,
                    maximum=2,
                    step=0.05,
                    label="Lora weights"
                )
        trigger = gr.Textbox(
            value=app.config.default_color_sd_prompt,
            placeholder="Lora 的触发提示词（可以为空）",
            label="Trigger prompt",
        )
        negative = gr.Textbox(
            placeholder="反向提示词（可以为空）",
            value=app.config.default_color_sd_negative,
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
                value=app.config.default_color_sd_sampling,
                multiselect=False,
                label="Sampling method"
            )
            schedule = gr.Dropdown(
                choices=schedule_type,
                value=app.config.default_color_sd_schedule,
                multiselect=False,
                label="Schedule type"
            )
        step = gr.Slider(
            value=app.config.default_color_sd_steps,
            minimum=1,
            maximum=150,
            step=1,
            label="Sampling steps"
        )
        cfg = gr.Slider(
            value=app.config.default_color_sd_cfg,
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
