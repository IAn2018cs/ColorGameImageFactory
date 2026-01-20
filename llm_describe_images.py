# coding=utf-8
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import natsort

from app.llm_tools import generate_by_openai
from app.tools import get_all_file_path, save_txt_to_file, get_image_data_url


def worker(file, destination, progress):
    print(f'{progress}%, start des {file}')
    prompt = des_image_prompt(file)
    if prompt:
        save_txt_to_file(destination, prompt)
    return prompt


def des_image_prompt(image_path: str):
    system = """Describe this image for image generation training. Write a single, natural English paragraph that captures:

- Main subject and its key features
- Colors, shapes, and visual style
- Background and composition
- Art style (e.g., cartoon, flat design, pixel art, realistic)

Requirements:
- 50-150 words, one continuous paragraph
- Start directly with the description (no "This image shows..." or "The image depicts...")
- Use descriptive adjectives for colors and details
- Keep it natural and flowing

Example output:
A cute orange cat sitting on a wooden table, with big round eyes and fluffy fur. The background is a cozy kitchen with warm yellow lighting. Simple flat illustration style with bold outlines and soft pastel colors."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_data_url(image_path)
                    }
                }
            ]
        }
    ]
    try:
        output = generate_by_openai('gemini-3-flash-preview', messages)
        return output
    except:
        return None


def generate_images_prompt_by_llm(res_path, save_prompts_json=True):
    all_files = get_all_file_path(res_path, ['png', 'jpg'])
    sorted_files = natsort.natsorted(all_files, reverse=False)
    all_count = len(sorted_files)
    futures = []
    prompts = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, file in enumerate(sorted_files):
            progress = round((index + 1) / all_count * 100, 2)
            path = Path(file)
            destination = path.with_suffix(".txt")
            if os.path.exists(destination):
                # 读取已存在的 prompt
                with open(destination, 'r', encoding='utf-8') as f:
                    prompts.append(f.read().strip())
                continue
            future = executor.submit(worker, file, destination, progress)
            futures.append(future)

        # 等待所有任务完成并收集结果
        for future in as_completed(futures):
            result = future.result()
            if result:
                prompts.append(result)

    # 保存所有 prompt 到 JSON 文件
    if save_prompts_json and prompts:
        save_prompts_to_json(res_path, prompts)

    return prompts


def save_prompts_to_json(res_path, prompts):
    """将所有 prompt 保存到 JSON 文件"""
    json_path = os.path.join(res_path, 'all_prompts.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(prompts)} prompts to {json_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    args = parser.parse_args()
    generate_images_prompt_by_llm(args.root_dir)

    # generate_images_prompt_by_llm('res/v2_train')
    # print(des_image_prompt('temp/demo_image/245.png'))
