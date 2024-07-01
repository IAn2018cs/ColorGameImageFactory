# coding=utf-8

import os

import cv2
import numpy as np
import svgwrite
from sklearn.cluster import KMeans


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

    return new_image, colors


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


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


def create_mask(quantized_image, color, tolerance=2):
    # 使用容差范围来创建掩码
    lower_bound = np.array(color) - tolerance
    upper_bound = np.array(color) + tolerance
    mask = np.all((quantized_image >= lower_bound) & (quantized_image <= upper_bound), axis=2)
    return mask.astype(np.uint8) * 255


def create_color_svg(quantized_image, colors, output_path):
    height, width = quantized_image.shape[:2]
    temp_dir = "temp_potrace"
    os.makedirs(temp_dir, exist_ok=True)

    # 创建新的SVG文件
    dwg = svgwrite.Drawing(output_path, size=(width, height))

    for i, color in enumerate(colors):
        # 创建掩码
        mask = create_mask(quantized_image, color)

        # 保存掩码图像用于调试
        temp_bmp = os.path.join(temp_dir, f"temp_{i}.png")
        cv2.imwrite(temp_bmp, mask)

        # 使用形态学操作清理掩码
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 追踪边界
        paths = trace_boundary(mask)

        # 添加路径到SVG
        for path in paths:
            dwg.add(dwg.path(d=path, fill=rgb_to_hex(color)))

    # 保存主SVG文件
    dwg.save()


# 主程序
image_path = 'data/00087-2910593776.png'
output_svg_path = 'output_color_quantized.svg'
n_colors = 40  # 你可以调整这个值

# 量化图像
quantized_image, colors = color_quantization(image_path, n_colors)

# 创建彩色 SVG
create_color_svg(quantized_image, colors, output_svg_path)

print(f"Color quantized SVG saved to {output_svg_path}")
