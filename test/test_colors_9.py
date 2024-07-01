# coding=utf-8

import json
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import pandas as pd
import math
import numpy as np
import svgwrite
from scipy.ndimage import label, center_of_mass
from PIL import Image
from sklearn.cluster import KMeans


def color_quantization(path, num_colors, add_label: bool = False):
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

    color_areas = separate_color_area(segmented_labels)
    centroids = center_of_mass(color_areas, labels=color_areas, index=np.arange(1, color_areas.max() + 1))
    centroids = [i for i in centroids if not pd.isna(i[0])]

    # 创建彩色 SVG
    color_config_path = f'{dir_path}/color_config_{name}.json'
    output_svg_path = f'{dir_path}/color_quantization_{name}.svg'
    create_color_svg(new_image, centroids, colors, segmented_labels, width, height, output_svg_path, color_config_path,
                     add_label)

    with Image.fromarray(new_image) as new_img:
        new_file_path = f'{dir_path}/color_quantization_{name}.png'
        new_img.save(new_file_path)
    return new_file_path, output_svg_path, color_config_path


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


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def create_mask(quantized_image, color, tolerance=4):
    # 使用容差范围来创建掩码
    lower_bound = np.array(color) - tolerance
    upper_bound = np.array(color) + tolerance
    mask = np.all((quantized_image >= lower_bound) & (quantized_image <= upper_bound), axis=2)
    return mask.astype(np.uint8) * 255


def create_color_svg(quantized_image, centroids, colors, labels, width, height, output_path, config_path, add_label):
    temp_dir = "temp_potrace"
    os.makedirs(temp_dir, exist_ok=True)

    # 创建新的SVG文件
    dwg = svgwrite.Drawing(output_path, size=(width, height))
    color_config = {}

    max_area = 0  # 用于归一化文字大小
    all_text_boxes = []  # 存储所有文本框的位置和大小

    # 添加背景矩形
    background_color = rgb_to_hex([0, 0, 0])
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=background_color))

    for i, centroid in enumerate(centroids):
        y, x = centroid
        color_label = labels[int(y), int(x)]
        print(color_label)
        color = colors[color_label]
        print(color)
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
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        color_hex = rgb_to_hex(color)
        color_config[i] = {'color': color_hex, 'areas': []}

        for contour in contours:
            path = "M"
            for point in contour:
                x, y = point[0]
                path += f"{x},{y} "
            path += "Z"

            path_id = f"{i}"
            dwg.add(dwg.path(d=path, fill=color_hex, id=path_id))

            if add_label:
                # 计算中心点和面积
                (cx, cy), area = calculate_centroid_and_area(contour)
                max_area = max(max_area, area)
                color_config[i]['areas'].append({'center': (cx, cy), 'area': area})

    # 添加文本
    if add_label:
        for i, color_info in color_config.items():
            for area_info in color_info['areas']:
                cx, cy = area_info['center']
                area = area_info['area']

                # 计算文字大小，范围在 8 到 24 之间
                font_size = max(10, min(26, int(16 * math.sqrt(area / max_area))))

                # 估计文本框的大小
                text_width = font_size * len(f"{i}") * 0.6  # 粗略估计
                text_height = font_size * 1.2

                # 检查是否与现有文本框重叠
                text_box = (cx - text_width / 2, cy - text_height / 2, text_width, text_height)
                if not any(rectangles_overlap(text_box, existing_box) for existing_box in all_text_boxes):
                    dwg.add(dwg.text(
                        f"{i}",
                        insert=(cx, cy),
                        fill='black',
                        font_size=font_size,
                        text_anchor="middle",
                        dominant_baseline="central"
                    ))
                    all_text_boxes.append(text_box)

    # 保存主SVG文件
    dwg.save()
    shutil.rmtree(temp_dir)

    # 简化 color_config 以便 JSON 序列化
    simplified_config = {str(k): v['color'] for k, v in color_config.items()}

    # 保存颜色配置到 JSON 文件
    with open(config_path, 'w') as f:
        json.dump(simplified_config, f, indent=2)


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


def calculate_centroid_and_area(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    area = cv2.contourArea(contour)
    return (cx, cy), area


def rectangles_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


if __name__ == '__main__':
    new_file_path, output_svg_path, color_config_path = color_quantization('data/10_rgb.png', 40, add_label=True)
    print(new_file_path)
    print(output_svg_path)
    print(color_config_path)

    # update_svg_colors(output_svg_path, {"1": "#000000"})
