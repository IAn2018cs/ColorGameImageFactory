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

import cv2
import numpy as np
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


def get_current_datatime():
    return time.strftime("%Y%m%d_%H_%M_%S", time.localtime())


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


def __get_path_name__(root_path, item):
    return f"{root_path}/{item}"


def get_all_file_path(root_path: str, suffix: list[str], process=__get_path_name__) -> list[str]:
    files = []
    for item in os.listdir(root_path):
        if item.startswith("."):
            continue
        child_path = f"{root_path}/{item}"
        if os.path.isdir(child_path):
            files.extend(get_all_file_path(child_path, suffix, process))
        else:
            file_suffix = item.split('.')[-1].lower()
            if file_suffix in suffix:
                files.append(process(root_path, item))
    return files


def enhance_line_drawing(input_path, output_path, blur_kernel=(3, 3), threshold_block_size=11, threshold_c=2,
                         dilate_kernel=(2, 2), dilate_iterations=1, erode_iterations=1,
                         use_canny=False, use_sharpening=False):
    """
    增强线稿图像的线条

    参数:
    input_path: 输入图像的路径
    output_path: 输出图像的保存路径
    blur_kernel: 高斯模糊的核大小,默认(3, 3)
    threshold_block_size: 自适应阈值的块大小,默认11
    threshold_c: 自适应阈值的常数,默认2
    dilate_kernel: 膨胀操作的核大小,默认(2, 2)
    dilate_iterations: 膨胀操作的迭代次数,默认1
    erode_iterations: 腐蚀操作的迭代次数,默认1
    use_canny: 是否使用Canny边缘检测，默认False
    use_sharpening: 是否使用图像锐化，默认False

    返回:
    None, 但会将处理后的图像保存到output_path
    """
    # 读取图像
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)

    if use_canny:
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 100, 200)
        thresh = edges
    else:
        # 使用自适应阈值处理
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, threshold_block_size, threshold_c)

    # 膨胀操作以加粗线条
    kernel = np.ones(dilate_kernel, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilate_iterations)

    # 腐蚀操作以细化线条
    eroded = cv2.erode(dilated, kernel, iterations=erode_iterations)

    if use_sharpening:
        # 使用 unsharp masking 进行图像锐化
        blurred = cv2.GaussianBlur(eroded, (5, 5), 0)
        sharpened = cv2.addWeighted(eroded, 1.5, blurred, -0.5, 0)
        result = sharpened
    else:
        result = eroded

    # 反转图像颜色（黑底白线变为白底黑线）
    result = cv2.bitwise_not(result)

    # 保存结果
    cv2.imwrite(output_path, result)

    print(f"处理完成,结果已保存至 {output_path}")


def convert2svg_image(path):
    dir_path = os.path.dirname(path)
    file_name_split = os.path.split(path)[-1].split('.')
    name = file_name_split[0]
    pbm_file_path = f'{dir_path}/{name}.pbm'
    im = Image.open(path)
    im.save(pbm_file_path)
    im.close()

    svg_output_path = f'{dir_path}/{name}.svg'
    os.system(f'potrace {pbm_file_path} -s -o {svg_output_path}')

    delete_file(pbm_file_path)
    delete_file(path)

    return svg_output_path


if __name__ == '__main__':
    convert2svg_image('../test/new_test.png')