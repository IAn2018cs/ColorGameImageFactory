# coding=utf-8
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import natsort

from app.llm_tools import generate_by_openai
from app.tools import get_all_file_path, save_txt_to_file, get_image_data_url


def worker(file, destination, progress):
    print(f'{progress}%, start des {file}')
    prompt = des_image_prompt(file)
    if prompt:
        save_txt_to_file(destination, prompt)


def des_image_prompt(image_path: str):
    system = """Generate a concise image description suitable for T5 text encoder training. Analyze the input image and create a description that:

1. Uses complete sentences or short clauses
2. Separates each sentence or clause with a period (.)
3. Describes the main subject, setting, key visual elements, colors, and style
4. Emphasizes distinct colors and clear lines
5. Incorporates relevant artistic terms
6. No more than 200 characters.

Your output should resemble this example:

a beautiful landscape with a waterfall cascading down a rocky cliff, surrounded by lush green trees and vibrant flowers. The sky is filled with fluffy white clouds, adding to the peaceful atmosphere of the scene."""
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
        output = generate_by_openai('gpt-4o', messages)
        return output
    except:
        return None


def generate_images_prompt_by_llm(res_path):
    all_files = get_all_file_path(res_path, ['png'])
    with ThreadPoolExecutor(max_workers=3) as executor:
        all_count = len(all_files)
        for index, file in enumerate(natsort.natsorted(all_files, reverse=False)):
            progress = round((index + 1) / all_count * 100, 2)
            path = Path(file)
            destination = path.with_suffix(".txt")
            if os.path.exists(destination):
                continue
            executor.submit(worker, file, destination, progress)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dir', type=str, required=True)
    # args = parser.parse_args()
    # generate_images_prompt_by_llm(args.root_dir)

    # generate_images_prompt_by_llm('res/v2_train')
    print(des_image_prompt('temp/demo_image/245.png'))
