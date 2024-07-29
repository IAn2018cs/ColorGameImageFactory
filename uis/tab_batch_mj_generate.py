# coding=utf-8

import gradio as gr

from app.colorful_svg_v2 import color_quantization
from app.gen_tools import extract_line_by_gan
from app.mj_bot import MJBot
from app.prompt_factory import create_mj_prompts
from app.tools import convert2svg_image
from app.tools import create_path
from app.tools import download_image_url
from app.tools import generate_random_id
from app.tools import resolve_relative_path
from app.tools import zip_dir
from uis.tabs import TabId
from uis.tools import all_category

mj_model_list = ['niji 6', 'niji 5', 'v 6', 'v 5.2']

mj_bot = MJBot()


def convert2svg(batch_id, images_root_path):
    root_path = resolve_relative_path(__file__, '../output')
    output_dir = f'{root_path}/{batch_id}/'
    line_result = extract_line_by_gan(images_root_path, output_dir)
    result = []
    for path in line_result:
        new_path = convert2svg_image(path, after_delete=True)
        result.append(new_path)
    return result


def generate_quantization_images(quantization_batch_id, colorful_images, line_svg_images, num_colors, min_ares):
    root_path = resolve_relative_path(__file__, '../output')
    output_dir = f'{root_path}/{quantization_batch_id}'
    create_path(output_dir)

    svg_results = []
    for index, colorful_image in enumerate(colorful_images):
        line_svg = line_svg_images[index]
        svg_img = color_quantization(colorful_image, line_svg, output_dir, num_colors, min_ares)
        svg_results.append(svg_img)
    return svg_results


def generate_image_by_mj(root_path: str, batch_id: str, prompt: str):
    output_path = f'{root_path}/{batch_id}'
    create_path(output_path)

    # files = [download_image_url(mj_bot.create_single_image_and_fetch(prompt), output_path)]
    files = []
    images = mj_bot.create_all_image_and_fetch(prompt)
    for image in images:
        files.append(download_image_url(image, output_path))
    return files


def start_gan(category, image_count, mj_model,
              num_colors, min_ares):
    try:
        prompts = create_mj_prompts(category, image_count)

        result = []
        root_path = resolve_relative_path(__file__, '../output')
        batch_id = generate_random_id(16)
        result.extend(
            generate_images(batch_id, prompts, root_path, mj_model)
        )
        zip_file = zip_dir(f'{root_path}/{batch_id}', batch_id, root_path)

        line_art_batch_id = generate_random_id(16)
        line_art_result = convert2svg(line_art_batch_id, f'{root_path}/{batch_id}')
        line_art_zip_file = zip_dir(f'{root_path}/{line_art_batch_id}', line_art_batch_id, root_path)

        quantization_batch_id = generate_random_id(16)
        quantization_svg_images = generate_quantization_images(quantization_batch_id,
                                                               result, line_art_result,
                                                               num_colors, min_ares)
        quantization_zip_file = zip_dir(f'{root_path}/{quantization_batch_id}', quantization_batch_id, root_path)

        return (result, line_art_result, quantization_svg_images,
                gr.DownloadButton(value=zip_file, visible=True),
                gr.DownloadButton(value=line_art_zip_file, visible=True),
                gr.DownloadButton(value=quantization_zip_file, visible=True))
    except Exception as e:
        raise gr.Error(f"发生错误：{e}，请重试")


def generate_images(batch_id, prompts, root_path, model):
    result = []
    print(f"model: {model}")
    sref = "https://media.discordapp.net/attachments/1262719142331551847/1267436192077451366/3.png?ex=66a8c770&is=66a775f0&hm=cedbf0e2b5c143dfb320d2a6271cad934b853eb8c07b8f6b35000279cae51600&=&format=webp&quality=lossless&width=1310&height=1310"
    for prompt in prompts:
        new_prompt = f"{prompt} --{model} --sref {sref}"
        images = generate_image_by_mj(
            root_path, batch_id, new_prompt
        )
        result.extend(images)
    return result


def build_batch_mj_generate_ui():
    with gr.TabItem("MJ 生图 + 彩图SVG 模式", id=TabId.BATCH_MJ_GENERATE.value):
        category = gr.Dropdown(
            choices=all_category,
            value=all_category[0],
            multiselect=False,
            allow_custom_value=True,
            label="图片分类（比如 Food、Collections、Buildings 等，可以自定义）"
        )
        image_count = gr.Slider(
            value=1,
            minimum=1,
            maximum=50,
            step=1,
            label="生成图片的数量，这里真实的图片数量会乘 4"
        )
        mj_model = gr.Radio(mj_model_list, value=mj_model_list[0], label="模型版本")

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
        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(
                    label="原图", format="png",
                    columns=3, rows=1, object_fit="contain")
                download_all_button = gr.DownloadButton("下载所有原图", visible=False)
            with gr.Column():
                line_art_gallery = gr.Gallery(
                    label="GAN 模型提取线稿图", format="svg",
                    columns=3, rows=1, object_fit="contain")
                download_line_art_button = gr.DownloadButton("下载所有线稿图", visible=False)
            with gr.Column():
                color_art_svg_gallery = gr.Gallery(
                    label="最终 svg 图", format="svg",
                    columns=3, rows=1, object_fit="contain")
                with gr.Row(visible=False) as send_row:
                    selected_image = gr.State(None)
                    send_to_coloring_game_btn = gr.Button("发送到填色游戏")

                def show_send_button(evt: gr.SelectData):
                    return gr.Row(visible=True), evt.value

                color_art_svg_gallery.select(show_send_button, None, [send_row, selected_image])

                download_color_art_button = gr.DownloadButton("下载所有 svg 彩图", visible=False)

        btn = gr.Button("开始批量生成", variant="primary")
        btn.click(
            fn=start_gan,
            inputs=[
                category, image_count,
                mj_model,
                num_colors, min_ares
            ],
            outputs=[
                gallery,
                line_art_gallery,
                color_art_svg_gallery,
                download_all_button,
                download_line_art_button,
                download_color_art_button
            ],
            scroll_to_output=True
        )
    return send_to_coloring_game_btn, selected_image
