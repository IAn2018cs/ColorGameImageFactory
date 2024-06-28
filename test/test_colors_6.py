# coding=utf-8

import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import svgwrite
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import KMeans
import json


def color_quantization(path, num_colors):
    image = cv2.imread(path)
    ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 原图
    blur_ori_image = cv2.GaussianBlur(ori_image, (5, 5), 0)

    pixels = blur_ori_image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    colors = kmeans.cluster_centers_

    new_pixels = colors[labels]
    new_image = new_pixels.reshape(ori_image.shape).astype(np.uint8)

    height, width = image.shape[:2]
    segmented_labels = labels.reshape(height, width)  # 颜色分区的编号图
    with Image.fromarray(new_image) as new_img:
        dir_path = os.path.dirname(path)
        file_name_split = os.path.split(path)[-1].split('.')
        name = file_name_split[0]
        new_file_path = f'{dir_path}/color_quantization_{name}.png'
        new_img.save(new_file_path)
    return new_file_path, segmented_labels, width, height


def separate_color_area(matrix):
    unique_values = np.unique(matrix)
    total_patches = 0
    labeled_matrix = np.zeros_like(matrix)
    for value in unique_values:
        binary_image = (matrix == value).astype(int)
        labels, num_features = label(binary_image)

        sizes = np.bincount(labels.ravel())
        labels *= (sizes > 40)[labels]
        # Ensure new labels don't overlap with previous labels
        labeled_matrix += (labels + (labeled_matrix.max() * (labels > 0)))
        total_patches += num_features
    return labeled_matrix


def calculate_area_sizes(labeled_matrix):
    unique, counts = np.unique(labeled_matrix, return_counts=True)
    return dict(zip(unique[1:], counts[1:]))  # 排除背景（标签为0）


def get_font_size(area, min_area, max_area, min_font_size=6, max_font_size=20):
    # 使用对数刻度来平滑字体大小的变化
    normalized_area = (np.log(area) - np.log(min_area)) / (np.log(max_area) - np.log(min_area))
    font_size = min_font_size + normalized_area * (max_font_size - min_font_size)
    return int(max(min_font_size, min(max_font_size, font_size)))


def create_svg_with_labels(image_path, centroids, color_labels, color_areas, area_sizes):
    # 获取图像尺寸
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 创建SVG绘图
    dwg = svgwrite.Drawing('labeled_image.svg', size=(width, height))

    # 添加原始图像作为背景
    dwg.add(dwg.image(image_path, size=(width, height)))

    # 计算字体大小范围
    min_area, max_area = min(area_sizes.values()), max(area_sizes.values())

    # 添加文本标签
    for centroid in centroids:
        y, x = centroid
        label = color_areas[int(y), int(x)]
        value = color_labels[int(y), int(x)]
        if label in area_sizes:
            area = area_sizes[label]
            font_size = get_font_size(area, min_area, max_area)

            # 创建文本元素
            text = dwg.text(str(value), insert=(x, y), fill='black',
                            font_size=font_size, text_anchor='middle',
                            dominant_baseline='central')
            dwg.add(text)

    # 保存SVG文件
    dwg.save()


def create_png_with_labels(image_path, centroids, color_labels, color_areas, area_sizes):
    # 计算字体大小范围
    min_area, max_area = min(area_sizes.values()), max(area_sizes.values())

    # 创建一个新的透明图层
    color_image = Image.open(image_path).convert("RGBA")
    line_layer = Image.new("RGBA", color_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(line_layer)

    for idx, centroid in enumerate(centroids):
        y, x = centroid  # centroid returns (row, column)
        c_label = color_areas[int(y), int(x)]
        value = color_labels[int(y), int(x)]
        if c_label in area_sizes:
            area = area_sizes[c_label]
            font_size = get_font_size(area, min_area, max_area)
            draw.text((x, y), str(value), align='center', fill=(0, 0, 0, 255), font=ImageFont.load_default(font_size))

    # 将文字图层合并到彩色图上
    result = Image.alpha_composite(color_image, line_layer)

    # 保存结果
    result.save('./label_image.png')


def parse_svg_size(root):
    # 从viewBox属性中获取尺寸
    viewbox = root.attrib.get('viewBox')
    if viewbox:
        _, _, width, height = map(float, viewbox.split())
        return width, height

    # 如果没有viewBox，从width和height属性获取
    width = root.attrib.get('width')
    height = root.attrib.get('height')

    if width and height:
        # 使用正则表达式提取数值部分
        width = float(re.findall(r"[\d.]+", width)[0])
        height = float(re.findall(r"[\d.]+", height)[0])
        return width, height


def add_labels_to_potrace_svg(svg_path, centroids, color_labels, color_areas, area_sizes, image_width, image_height):
    dir_path = os.path.dirname(svg_path)
    file_name_split = os.path.split(svg_path)[-1].split('.')
    name = file_name_split[0]
    new_file_path = f'{dir_path}/color_label_{name}.svg'

    # 解析SVG文件
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 获取SVG的宽度和高度
    svg_width, svg_height = parse_svg_size(root)

    # 计算字体大小范围
    min_area, max_area = min(area_sizes.values()), max(area_sizes.values())

    # 创建一个新的组元素来包含所有的文本标签
    text_group = ET.Element('g')
    text_group.set('id', 'labels')

    # 添加文本标签
    for centroid in centroids:
        y, x = centroid
        label = color_areas[int(y), int(x)]
        value = color_labels[int(y), int(x)]
        if label in area_sizes:
            area = area_sizes[label]
            font_size = get_font_size(area, min_area, max_area)

            # 创建文本元素
            text = ET.Element('text')
            text.set('x', str(x * svg_width / image_width))  # 调整坐标以匹配SVG尺寸
            text.set('y', str(y * svg_height / image_height))
            text.set('fill', 'black')
            text.set('font-size', str(font_size))
            text.set('text-anchor', 'middle')
            text.set('dominant-baseline', 'central')
            text.text = str(value)

            text_group.append(text)

    # 将文本组添加到SVG根元素
    root.append(text_group)

    # 保存修改后的SVG文件
    tree.write(new_file_path, xml_declaration=True, encoding='utf-8')
    return new_file_path


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def create_color_config(color_labels, color_areas, kmeans):
    unique_labels = np.unique(color_labels)
    color_config = {"colors": []}

    for label in unique_labels:
        mask = (color_labels == label)
        color = kmeans.cluster_centers_[label]
        hex_color = rgb_to_hex(color)

        color_config["colors"].append({
            "label": int(label),
            "color": hex_color
        })

    return color_config


if __name__ == '__main__':
    new_image, color_labels, width, height = color_quantization('data/00087-2910593776.png', 10)
    print(new_image)

    color_areas = separate_color_area(color_labels)
    area_sizes = calculate_area_sizes(color_areas)

    centroids = center_of_mass(color_areas, labels=color_areas, index=np.arange(1, color_areas.max() + 1))
    centroids = [i for i in centroids if not pd.isna(i[0])]
    print(len(centroids))

    # create_png_with_labels(new_image, centroids, color_labels, color_areas, area_sizes)
    # create_svg_with_labels('./new_test.png', centroids, color_labels, color_areas, area_sizes)
    add_labels_to_potrace_svg('./new_test.svg', centroids, color_labels, color_areas, area_sizes, width, height)

    # 创建颜色配置并保存为JSON文件
    color_config = create_color_config(color_labels, color_areas, kmeans)
    color_config_path = './color_config.json'
    with open(color_config_path, 'w') as f:
        json.dump(color_config, f, indent=2)
    print(f"Color configuration saved to {color_config_path}")