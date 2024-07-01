# coding=utf-8

import json
import os
import re
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import svgwrite
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import KMeans


def color_quantization(path, num_colors):
    dir_path = os.path.dirname(path)
    file_name_split = os.path.split(path)[-1].split('.')
    name = file_name_split[0]

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

    # 创建彩色 SVG
    color_config_path = f'{dir_path}/color_config_{name}.json'
    output_svg_path = f'{dir_path}/color_quantization_{name}.svg'
    create_color_svg(new_image, colors, width, height, output_svg_path, color_config_path)

    with Image.fromarray(new_image) as new_img:
        new_file_path = f'{dir_path}/color_quantization_{name}.png'
        new_img.save(new_file_path)
    return new_file_path, output_svg_path, color_config_path, segmented_labels, colors, width, height


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


def create_color_config(color_labels, colors):
    unique_labels = np.unique(color_labels)
    color_config = {"colors": []}

    for label in unique_labels:
        color = colors[label]
        hex_color = rgb_to_hex(color)

        color_config["colors"].append({
            "label": int(label),
            "color": hex_color
        })

    return color_config


def trace_boundary(mask):
    # 使用Marching Squares算法追踪边界
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    paths = []
    for contour in contours:
        path = "M"
        for point in contour:
            x, y = point[0]
            path += f"{x},{y} "
        path += "Z"
        paths.append(path)
    return paths


def create_mask(quantized_image, color, tolerance=4):
    # 使用容差范围来创建掩码
    lower_bound = np.array(color) - tolerance
    upper_bound = np.array(color) + tolerance
    mask = np.all((quantized_image >= lower_bound) & (quantized_image <= upper_bound), axis=2)
    return mask.astype(np.uint8) * 255


def create_color_svg(quantized_image, colors, width, height, output_path, config_path):
    temp_dir = "temp_potrace"
    os.makedirs(temp_dir, exist_ok=True)

    # 创建新的SVG文件
    dwg = svgwrite.Drawing(output_path, size=(width, height))
    color_config = {}

    # 添加背景矩形
    background_color = rgb_to_hex([0, 0, 0])
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=background_color))

    for i, color in enumerate(colors):
        # 创建掩码
        mask = create_mask(quantized_image, color)

        # 保存掩码图像用于调试
        temp_bmp = os.path.join(temp_dir, f"temp_{i}.png")
        cv2.imwrite(temp_bmp, mask)

        # 使用形态学操作清理掩码
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 追踪边界
        paths = trace_boundary(mask)

        # 添加路径到SVG
        for path in paths:
            path_id = str(i)
            rgb_color = rgb_to_hex(color)
            dwg.add(dwg.path(d=path, fill=rgb_color, id=path_id))
            color_config[path_id] = rgb_color

    # 保存主SVG文件
    dwg.save()
    shutil.rmtree(temp_dir)

    # 保存颜色配置到 JSON 文件
    with open(config_path, 'w') as f:
        json.dump(color_config, f, indent=2)


def edit_color_svg(quantized_image, colors, input_svg_path, output_svg_path):
    height, width = quantized_image.shape[:2]
    temp_dir = "temp_potrace"
    os.makedirs(temp_dir, exist_ok=True)

    # 解析现有的SVG文件
    tree = ET.parse(input_svg_path)
    root = tree.getroot()

    # 获取原SVG的宽度和高度
    svg_width, svg_height = parse_svg_size(root)

    # 计算缩放比例和偏移量
    scale = min(svg_width / width, svg_height / height)
    offset_x = (svg_width - width * scale) / 2
    offset_y = (svg_height - height * scale) / 2

    # 设置viewBox以确保整个图像都能显示
    root.set('viewBox', f"0 0 {svg_width} {svg_height}")

    # 创建一个组元素来包含所有新添加的路径
    group = ET.SubElement(root, 'g')
    # 添加变换以确保图像居中
    group.set('transform', f'translate({offset_x},{offset_y}) scale({scale})')

    for i, color in enumerate(colors):
        # 创建掩码
        mask = create_mask(quantized_image, color)

        # 保存掩码图像用于调试
        temp_bmp = os.path.join(temp_dir, f"temp_{i}.png")
        cv2.imwrite(temp_bmp, mask)

        # 使用形态学操作清理掩码
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 追踪边界
        paths = trace_boundary(mask)

        # 添加路径到SVG
        for path in paths:
            path_element = ET.SubElement(group, 'path')
            path_element.set('d', path)
            path_element.set('fill', rgb_to_hex(color))

        # 保存修改后的SVG文件
    tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)

    shutil.rmtree(temp_dir)


def scale_svg_path(path, scale_x, scale_y):
    parts = path.split()
    scaled_parts = []
    for i, part in enumerate(parts):
        if part.replace('-', '').replace('.', '').isdigit():
            x, y = map(float, part.split(','))
            scaled_parts.append(f"{x * scale_x:.2f},{y * scale_y:.2f}")
        else:
            scaled_parts.append(part)
    return ' '.join(scaled_parts)


def update_svg_colors(svg_path, updates):
    dir_path = os.path.dirname(svg_path)
    file_name_split = os.path.split(svg_path)[-1].split('.')
    name = file_name_split[0]
    new_file_path = f'{dir_path}/edit_temp_{name}.svg'

    # 更新 SVG 文件
    tree = ET.parse(svg_path)
    root = tree.getroot()

    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        path_id = path.get('id')
        if path_id in updates:
            path.set('fill', updates[path_id])

    tree.write(new_file_path)


if __name__ == '__main__':
    new_file_path, output_svg_path, color_config_path, color_labels, colors, width, height = color_quantization(
        'data/10_rgb.png', 40)
    print(new_file_path)
    print(output_svg_path)
    print(color_config_path)

    update_svg_colors(output_svg_path, {"1": "#000000"})

    color_areas = separate_color_area(color_labels)
    area_sizes = calculate_area_sizes(color_areas)

    centroids = center_of_mass(color_areas, labels=color_areas, index=np.arange(1, color_areas.max() + 1))
    centroids = [i for i in centroids if not pd.isna(i[0])]
    print(len(centroids))

    # create_png_with_labels(new_image, centroids, color_labels, color_areas, area_sizes)
    # create_svg_with_labels('./new_test.png', centroids, color_labels, color_areas, area_sizes)
    new_svg_path = add_labels_to_potrace_svg(output_svg_path, centroids, color_labels, color_areas, area_sizes, width,
                                             height)
