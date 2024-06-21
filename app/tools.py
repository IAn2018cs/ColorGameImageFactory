# coding=utf-8
import base64
import csv
import os
import random
import shutil
import string
import time
from os import PathLike
from typing import AnyStr

from PIL import Image


def resolve_relative_path(file: PathLike[AnyStr], path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(file), path))


def get_column_data(csv_file_path, column_name):
    column_data = []
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            column_data.append(row[column_name])
    return column_data


def read_file_to_list_of_tuples(file_path: str):
    try:
        result = []

        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read each line in the file
            for line in file:
                # Strip any surrounding whitespace and split by comma
                items = line.strip().split(',')
                # Convert the split line into tuple and append to the result list
                result.append((items[0], items[1]))
        if len(result) == 0:
            return None
        return result
    except Exception as e:
        print(f"read_file_to_list_of_tuples error: {e}")
        return None


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_file(path):
    try:
        os.remove(path)
    except:
        pass


def get_timestamp() -> int:
    """
    获取毫秒时间戳
    :return: 毫秒
    """
    t = time.time()
    return int(round(t * 1000))


def generate_random_id(length=10):
    characters = string.ascii_letters + string.digits  # 包含所有字母（大写和小写）和数字
    return ''.join(random.choice(characters) for _ in range(length))  # 随机选择字符


def save_base64_image(path: str, b64):
    with open(path, 'wb') as f:
        f.write(base64.b64decode(b64))


def get_base64_image(path) -> str:
    with open(path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)  # base64编码
        return base64_data.decode('utf-8')


def is_rgb_image(path) -> bool:
    image = Image.open(path)
    bands = image.getbands()
    is_rgb = bands == ("R", "G", "B")
    image.close()
    return is_rgb


def convert2rgb_image(path) -> str:
    if is_rgb_image(path):
        return path
    dir_path = os.path.dirname(path)
    file_name_split = os.path.split(path)[-1].split('.')
    name = file_name_split[0]
    file_suffix = file_name_split[-1]
    new_file_path = f'{dir_path}/{name}_rgb.{file_suffix}'
    im = Image.open(path).convert('RGB')
    im.save(new_file_path)
    im.close()
    return new_file_path


def zip_dir(dir_path, out_name, output_path):
    zip_file = shutil.make_archive(out_name, format='zip', root_dir=dir_path)
    shutil.move(zip_file, output_path)
    return f'{output_path}/{out_name}.zip'


def split_list_with_min_length(original_list: list, min_length: int) -> list[list]:
    if min_length <= 0:
        raise "Error: Minimum length must be a positive number."

    length = len(original_list)

    # Calculate the number of sublists that can be created
    num_sublists = (length + min_length - 1) // min_length  # Use ceiling division

    # Initialize the starting index and result list
    start = 0
    result = []

    for i in range(num_sublists):
        # For the last sublist, include all remaining elements
        if i == num_sublists - 1:
            result.append(original_list[start:])
        else:
            # Append elements to each of the other sublists
            result.append(original_list[start:start + min_length])
            start += min_length

    return result
