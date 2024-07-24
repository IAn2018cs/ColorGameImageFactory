# coding=utf-8
import gradio as gr

import app.config
from app.colorful_svg_v2 import color_quantization
from app.gen_tools import extract_line_by_gan
from app.prompt_factory import create_sd_prompts
from app.sd_tools import controlnet_image_preprocessor
from app.sd_tools import generate_image_by_sd
from app.tools import convert2svg_image
from app.tools import create_path
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


def generate_line_art_images(root_path, batch_id, prompts, n_iter,
                             line_model, line_lora, line_weight, line_trigger, line_negative, line_sampling,
                             line_schedule, line_step, line_cfg):
    lora = str(line_lora).strip()
    trigger = str(line_trigger).strip()
    print(f"line_lora: {lora}")
    print(f"line_trigger prompt: {trigger}")

    result = []
    for prompt in prompts:
        weights = line_weight

        new_prompt = ""
        if lora != "" and lora != "None":
            new_prompt += f'<lora:{lora}:{weights}>, '
        if trigger != "" and trigger != "None":
            new_prompt += f'{trigger}, '
        new_prompt += prompt

        model = line_model
        negative = line_negative
        step = line_step
        cfg = line_cfg
        sampling = line_sampling
        schedule = line_schedule
        styles = ['SAI Line Art']
        images = generate_image_by_sd(
            root_path, batch_id,
            model, new_prompt, negative, step, cfg, sampling, schedule, 1024, 1024, styles,
            n_iter
        )
        result.extend([{'prompt': prompt, 'image': image} for image in images])
    return result


def generate_prompt_and_line_art(root_path, batch_id, category, image_count,
                                 line_model, line_lora, line_weight, line_trigger, line_negative, line_sampling,
                                 line_schedule, line_step, line_cfg):
    prompt_count = min(6, image_count)
    n_iter = max(int(image_count / prompt_count), 1)
    prompts = create_sd_prompts(category, prompt_count)
    result = []
    result.extend(
        generate_line_art_images(root_path, batch_id, prompts, n_iter, line_model, line_lora, line_weight, line_trigger,
                                 line_negative, line_sampling, line_schedule, line_step, line_cfg))
    if image_count > prompt_count:
        last = image_count % prompt_count
        if last > 0:
            prompts = create_sd_prompts(category, last)
            result.extend(generate_line_art_images(root_path, batch_id, prompts, 1, line_model, line_lora, line_weight,
                                                   line_trigger, line_negative, line_sampling, line_schedule, line_step,
                                                   line_cfg))
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
                            "enabled": True,
                            "input_image": base64_image,
                            "model": "sai_xl_canny_128lora [19804483]",
                        }
                    ]
                }
            }
        )
        # 去掉最后一张 ControlNet 预处理图
        cn_im = image.pop()
        delete_file(cn_im)
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


def convert2svg(batch_id, images_root_path):
    root_path = resolve_relative_path(__file__, '../output')
    output_dir = f'{root_path}/{batch_id}/'
    line_result = extract_line_by_gan(images_root_path, output_dir)
    result = []
    for path in line_result:
        new_path = convert2svg_image(path, after_delete=True)
        result.append(new_path)
    return result


def generate_quantization_images(quantization_batch_id, colorful_images, line_svg_images, num_colors):
    root_path = resolve_relative_path(__file__, '../output')
    output_dir = f'{root_path}/{quantization_batch_id}'
    create_path(output_dir)

    svg_results = []
    for index, colorful_image in enumerate(colorful_images):
        line_svg = line_svg_images[index]
        svg_img = color_quantization(colorful_image, line_svg, output_dir, num_colors)
        svg_results.append(svg_img)
    return svg_results


def start_gan_line_art(category, image_count, num_colors,
                       line_model, line_lora, line_weight, line_trigger, line_negative, line_sampling, line_schedule,
                       line_step, line_cfg,
                       color_model, color_lora, color_weight, color_trigger, color_negative, color_sampling,
                       color_schedule, color_step, color_cfg):
    try:
        root_path = resolve_relative_path(__file__, '../output')
        # 0. 根据类型生成一批提示词 & 生成线稿图
        line_art_batch_id = generate_random_id(16)
        line_art_images = generate_prompt_and_line_art(root_path, line_art_batch_id, category, image_count, line_model,
                                                       line_lora, line_weight, line_trigger, line_negative,
                                                       line_sampling,
                                                       line_schedule, line_step, line_cfg)
        line_art_paths = [info['image'] for info in line_art_images]
        line_art_zip_file = zip_dir(f'{root_path}/{line_art_batch_id}', line_art_batch_id, root_path)

        # 1. lineart_anime_denoise 预处理成 黑底白线图
        black_base64_images = convert_line_art2black_images(root_path, line_art_batch_id, line_art_images)

        # 2. 通过 ControlNet 生成上色图 -> 原图 保存一个结果
        colorful_batch_id = generate_random_id(16)
        colorful_images = generate_colorful_images(root_path, colorful_batch_id, black_base64_images,
                                                   color_model, color_lora, color_weight, color_trigger, color_negative,
                                                   color_sampling, color_schedule, color_step, color_cfg)
        colorful_zip_file = zip_dir(f'{root_path}/{colorful_batch_id}', colorful_batch_id, root_path)

        # 3. 通过 gan 模型提取线稿，转成 svg -> 线稿图 保存一个结果
        gan_batch_id = generate_random_id(16)
        svg_images = convert2svg(gan_batch_id, f'{root_path}/{colorful_batch_id}/')
        svg_zip_file = zip_dir(f'{root_path}/{gan_batch_id}', gan_batch_id, root_path)

        # 4. 转成 彩色 svg
        quantization_batch_id = generate_random_id(16)
        quantization_svg_images = generate_quantization_images(quantization_batch_id,
                                                               colorful_images, svg_images,
                                                               num_colors)
        quantization_zip_file = zip_dir(f'{root_path}/{quantization_batch_id}', quantization_batch_id, root_path)

        return (
            line_art_paths, colorful_images, svg_images, quantization_svg_images,
            gr.DownloadButton(value=line_art_zip_file, visible=True),
            gr.DownloadButton(value=colorful_zip_file, visible=True),
            gr.DownloadButton(value=svg_zip_file, visible=True),
            gr.DownloadButton(value=quantization_zip_file, visible=True)
        )
    except Exception as e:
        raise gr.Error(f"发生错误：{e}，请重试")


def generate_gallery_load_js():
    return """
    function() {
        setTimeout(function() {
            const galleryElems = document.querySelectorAll('.gallery-item');
            galleryElems.forEach(function(elem) {
                if (!elem.querySelector('.send-to-coloring-game')) {
                    const btn = document.createElement('button');
                    btn.textContent = '发送到填色游戏';
                    btn.className = 'send-to-coloring-game';
                    btn.style.position = 'absolute';
                    btn.style.bottom = '10px';
                    btn.style.right = '10px';
                    btn.style.zIndex = '1000';
                    btn.onclick = function(e) {
                        e.stopPropagation();
                        const img = elem.querySelector('img');
                        if (img) {
                            gradioApp().querySelector('#send-to-coloring-game-hidden').click();
                            gradioApp().querySelector('#selected-image-for-coloring').value = img.src;
                        }
                    };
                    elem.style.position = 'relative';
                    elem.appendChild(btn);
                }
            });
        }, 100);
    }
    """


def add_buttons_to_gallery():
    pass


def build_generate_line_art_v2_ui():
    with gr.TabItem("线稿 + 上色 + 彩图SVG 模式", id=5):
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
        with gr.Accordion("线稿图 SD 配置"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        line_model = gr.Dropdown(
                            value=app.config.default_line_sd_model,
                            choices=get_models(),
                            multiselect=False,
                            label="生成线稿图 SD 模型"
                        )
                        line_refresh_model_button = gr.Button("🔄", size="sm")
                        line_refresh_model_button.click(refresh_models, line_model, line_model)
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            line_lora = gr.Dropdown(
                                value=app.config.default_line_sd_lora,
                                choices=get_train_loras(),
                                multiselect=False,
                                label="生成线稿图 SD Lora"
                            )
                            line_refresh_lora_button = gr.Button("🔄", size="sm")
                            line_refresh_lora_button.click(refresh_loras, line_lora, line_lora)
                line_weight = gr.Slider(
                    value=app.config.default_line_sd_lora_weight,
                    minimum=0,
                    maximum=2,
                    step=0.05,
                    label="生成线稿图 Lora weight"
                )

            with gr.Row():
                line_trigger = gr.Textbox(
                    placeholder="生成线稿图 Lora 的触发提示词（可以为空）",
                    value=app.config.default_line_sd_prompt,
                    label="生成线稿图 Trigger prompt",
                )
                line_negative = gr.Textbox(
                    placeholder="生成线稿图 反向提示词（可以为空）",
                    value=app.config.default_line_sd_negative,
                    label="生成线稿图 Negative prompt"
                )
            with gr.Row():
                line_sampling = gr.Dropdown(
                    choices=sampling_method,
                    value=app.config.default_line_sd_sampling,
                    multiselect=False,
                    label="生成线稿图 Sampling method"
                )
                line_schedule = gr.Dropdown(
                    choices=schedule_type,
                    value=app.config.default_line_sd_schedule,
                    multiselect=False,
                    label="生成线稿图 Schedule type"
                )
                line_step = gr.Slider(
                    value=app.config.default_line_sd_steps,
                    minimum=1,
                    maximum=150,
                    step=1,
                    label="生成线稿图 Sampling steps"
                )
                line_cfg = gr.Slider(
                    value=app.config.default_line_sd_cfg,
                    minimum=1,
                    maximum=30,
                    step=0.5,
                    label="生成线稿图 CFG Scale"
                )

        with gr.Accordion("线稿图上色 SD 配置"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        color_model = gr.Dropdown(
                            value=app.config.default_color_sd_model,
                            choices=get_models(),
                            multiselect=False,
                            label="线稿图上色 SD 模型"
                        )
                        color_refresh_model_button = gr.Button("🔄", size="sm")
                        color_refresh_model_button.click(refresh_models, color_model, color_model)
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            color_lora = gr.Dropdown(
                                value=app.config.default_color_sd_lora,
                                choices=get_train_loras(),
                                multiselect=False,
                                label="线稿图上色 SD Lora"
                            )
                            color_refresh_lora_button = gr.Button("🔄", size="sm")
                            color_refresh_lora_button.click(refresh_loras, color_lora, color_lora)
                color_weight = gr.Slider(
                    value=app.config.default_color_sd_lora_weight,
                    minimum=0,
                    maximum=2,
                    step=0.05,
                    label="线稿图上色 Lora weight"
                )

            with gr.Row():
                color_trigger = gr.Textbox(
                    value=app.config.default_color_sd_prompt,
                    placeholder="线稿图上色 Lora 的触发提示词（可以为空）",
                    label="线稿图上色 Trigger prompt",
                )
                color_negative = gr.Textbox(
                    placeholder="线稿图上色 反向提示词（可以为空）",
                    value=app.config.default_color_sd_negative,
                    label="线稿图上色 Negative prompt"
                )
            with gr.Row():
                color_sampling = gr.Dropdown(
                    choices=sampling_method,
                    value=app.config.default_color_sd_sampling,
                    multiselect=False,
                    label="线稿图上色 Sampling method"
                )
                color_schedule = gr.Dropdown(
                    choices=schedule_type,
                    value=app.config.default_color_sd_schedule,
                    multiselect=False,
                    label="线稿图上色 Schedule type"
                )
                color_step = gr.Slider(
                    value=app.config.default_color_sd_steps,
                    minimum=1,
                    maximum=150,
                    step=1,
                    label="线稿图上色 Sampling steps"
                )
                color_cfg = gr.Slider(
                    value=app.config.default_color_sd_cfg,
                    minimum=1,
                    maximum=30,
                    step=0.5,
                    label="线稿图上色 CFG Scale"
                )
        num_colors = gr.Slider(
            value=30,
            minimum=10,
            maximum=100,
            step=1,
            label="合并相似颜色阈值（值越大，颜色越少，越小，颜色越多）"
        )
        with gr.Row():
            with gr.Column():
                line_art_gallery = gr.Gallery(
                    label="生成的线稿图", format="png",
                    columns=2, rows=1, object_fit="contain")
                download_line_art_button = gr.DownloadButton("下载所有生成的线稿图", visible=False)
            with gr.Column():
                gallery = gr.Gallery(
                    label="上色图", format="png",
                    columns=2, rows=1, object_fit="contain")
                download_all_button = gr.DownloadButton("下载所有上色图", visible=False)
            with gr.Column():
                gan_line_gallery = gr.Gallery(
                    label="GAN 模型提取线稿图", format="svg",
                    columns=2, rows=1, object_fit="contain")
                download_gan_line_button = gr.DownloadButton("下载所有 GAN 提取的线稿图", visible=False)
            with gr.Column():
                color_art_svg_gallery = gr.Gallery(
                    label="最终 svg 图", format="svg",
                    columns=2, rows=1, object_fit="contain")
                color_art_svg_gallery.change(
                    fn=add_buttons_to_gallery,
                    inputs=[],
                    outputs=[],
                    js=generate_gallery_load_js()
                )
                # color_art_svg_gallery.attach_load_event(generate_gallery_load_js, None)
                download_color_art_button = gr.DownloadButton("下载所有 svg 彩图", visible=False)

        btn = gr.Button("开始批量生成", variant="primary")
        btn.click(
            fn=start_gan_line_art,
            inputs=[
                category, image_count, num_colors,
                line_model, line_lora, line_weight, line_trigger, line_negative, line_sampling, line_schedule,
                line_step, line_cfg,
                color_model, color_lora, color_weight, color_trigger, color_negative, color_sampling, color_schedule,
                color_step, color_cfg
            ],
            outputs=[
                line_art_gallery, gallery, gan_line_gallery, color_art_svg_gallery,
                download_line_art_button,
                download_all_button,
                download_gan_line_button,
                download_color_art_button
            ],
            scroll_to_output=True
        )
