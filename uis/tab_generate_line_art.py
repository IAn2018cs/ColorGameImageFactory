# coding=utf-8
import gradio as gr

import app.config
from app.prompt_factory import create_sd_prompts
from app.sd_tools import controlnet_image_preprocessor
from app.sd_tools import generate_image_by_sd
from app.tools import convert2svg_image
from app.tools import delete_file
from app.tools import generate_random_id
from app.tools import get_base64_image
from app.tools import resolve_relative_path
from app.tools import split_list_with_min_length
from app.tools import zip_dir
from uis.tools import all_category
from uis.tools import get_models
from uis.tools import get_train_loras
from uis.tools import refresh_loras
from uis.tools import refresh_models
from uis.tools import sampling_method
from uis.tools import schedule_type


def generate_line_art_images(root_path, batch_id, prompts, n_iter):
    result = []
    for prompt in prompts:
        lora = "Coloring_book_-_LineArt"
        weights = 0.7
        new_prompt = f"black and white, line art, coloring drawing of {prompt}, <lora:{lora}:{weights}>, white background, thick outlines"
        model = "crystalClearXL_ccxl"
        negative = "colorful, colors"
        step = 30
        cfg = 7
        sampling = "DPM++ 2M SDE"
        schedule = "Exponential"
        styles = ['SAI Line Art']
        images = generate_image_by_sd(
            root_path, batch_id,
            model, new_prompt, negative, step, cfg, sampling, schedule, 1024, 1024, styles,
            n_iter
        )
        result.extend([{'prompt': prompt, 'image': image} for image in images])
    return result


def generate_prompt_and_line_art(root_path, batch_id, category, image_count):
    prompt_count = min(6, image_count)
    n_iter = max(int(image_count / prompt_count), 1)
    prompts = create_sd_prompts(category, prompt_count)
    result = []
    result.extend(generate_line_art_images(root_path, batch_id, prompts, n_iter))
    if image_count > prompt_count:
        last = image_count % prompt_count
        if last > 0:
            prompts = create_sd_prompts(category, last)
            result.extend(generate_line_art_images(root_path, batch_id, prompts, 1))
    return result


def convert_line_art2black_images(root_path, batch_id, images):
    split_list = split_list_with_min_length(images, 5)
    result = []
    for image_infos in split_list:
        base64_images = []
        prompts = []
        for info in image_infos:
            base64_images.append(get_base64_image(info['image']))
            prompts.append(info['prompt'])
        black_image_paths = controlnet_image_preprocessor(root_path, batch_id, "lineart_anime_denoise", base64_images,
                                                          processor_res=512)
        for index, path in enumerate(black_image_paths):
            result.append({'prompt': prompts[index], 'image': get_base64_image(path)})
            delete_file(path)
    return result


def generate_colorful_images(root_path, batch_id, images,
                             model, lora, weights, trigger, negative, sampling, schedule, step, cfg):
    lora = str(lora).strip()
    trigger = str(trigger).strip()
    print(f"lora: {lora}")
    print(f"trigger prompt: {trigger}")

    result = []
    for image_info in images:
        prompt = image_info['prompt']
        base64_image = image_info['image']

        new_prompt = ""
        if lora != "" and lora != "None":
            new_prompt += f'<lora:{lora}:{weights}>, '
        if trigger != "" and trigger != "None":
            new_prompt += f'{trigger}, '
        new_prompt += prompt

        image = generate_image_by_sd(
            root_path, batch_id,
            model, new_prompt, negative, step, cfg, sampling, schedule,
            1024, 1024, ['SAI Comic Book'], 1,
            alwayson_scripts={
                "controlnet": {
                    "args": [
                        {
                            "input_image": base64_image,
                            "module": "none",
                            "model": "sai_xl_canny_128lora [19804483]",
                        }
                    ]
                }
            }
        )
        result.extend(image)
    return result


def invert_black_image2svg(root_path, batch_id, images):
    split_list = split_list_with_min_length(images, 5)
    result = []
    for image_infos in split_list:
        base64_images = [image['image'] for image in image_infos]
        line_image_paths = controlnet_image_preprocessor(root_path, batch_id, "invert", base64_images)
        for path in line_image_paths:
            new_path = convert2svg_image(path)
            result.append(new_path)
    return result


def start_gan_line_art(category, image_count, model, lora, weights, trigger, negative, sampling, schedule, step, cfg):
    root_path = resolve_relative_path(__file__, './output')
    # 0. 根据类型生成一批提示词 & 生成线稿图
    line_art_batch_id = generate_random_id(16)
    line_art_images = generate_prompt_and_line_art(root_path, line_art_batch_id, category, image_count)

    # 1. lineart_anime_denoise 预处理成 黑底白线图
    black_base64_images = convert_line_art2black_images(root_path, line_art_batch_id, line_art_images)

    # 2. 通过 ControlNet 生成上色图 -> 原图 保存一个结果
    colorful_batch_id = generate_random_id(16)
    colorful_images = generate_colorful_images(root_path, colorful_batch_id, black_base64_images,
                                               model, lora, weights, trigger, negative, sampling, schedule, step, cfg)
    colorful_zip_file = zip_dir(f'{root_path}/{colorful_batch_id}', colorful_batch_id, root_path)

    # 3. 上色图颜色聚类 -> 聚类图 保存一个结果


    # 4. 将第 1 步中的预处理图 invert 颜色反转，转成 svg 图 -> 线稿图 保存一个结果
    svg_batch_id = generate_random_id(16)
    svg_images = invert_black_image2svg(root_path, svg_batch_id, black_base64_images)
    svg_zip_file = zip_dir(f'{root_path}/{svg_batch_id}', svg_batch_id, root_path)

    return (
        colorful_images, svg_images,
        gr.DownloadButton(value=colorful_zip_file, visible=True),
        gr.DownloadButton(value=svg_zip_file, visible=True)
    )


def build_generate_line_art_ui():
    with gr.Tab("线稿图生图 + 上色模式"):
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
            fn=start_gan_line_art,
            inputs=[
                category, image_count,
                model, lora, weights, trigger, negative, sampling, schedule, step, cfg
            ],
            outputs=[
                gallery,
                line_art_gallery,
                download_all_button,
                download_line_art_button
            ],
            scroll_to_output=True
        )
