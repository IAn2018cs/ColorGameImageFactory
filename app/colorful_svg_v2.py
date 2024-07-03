# coding=utf-8
import json
import math
import os
import re
import xml.etree.ElementTree as ET

import vtracer


def convert_image2color_svg(img_path, output_path):
    input_path = img_path

    vtracer.convert_image_to_svg_py(
        input_path, output_path,
        colormode='color',  # ["color"] or "binary"
        hierarchical='stacked',  # ["stacked"] or "cutout"
        mode='spline',  # ["spline"] "polygon", or "none"
        filter_speckle=4,  # default: 4
        color_precision=6,  # default: 6
        layer_difference=16,  # default: 16
        corner_threshold=60,  # default: 60
        length_threshold=4.0,  # in [3.5, 10] default: 4.0
        max_iterations=10,  # default: 10
        splice_threshold=45,  # default: 45
        path_precision=8  # default: 8
    )


def color_quantization(img_path, line_svg_path, output_dir, threshold=30):
    file_name_split = os.path.split(img_path)[-1].split('.')
    name = file_name_split[0]
    svg_path = f'{output_dir}/{name}.svg'

    convert_image2color_svg(img_path, svg_path)

    # 解析SVG文件
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 创建颜色到路径索引的映射
    color_to_indices = {}

    # 遍历所有path元素
    for index, path in enumerate(root.findall('.//{http://www.w3.org/2000/svg}path')):
        fill_color = path.get('fill')
        if fill_color and not is_close_to_black(fill_color):
            if fill_color not in color_to_indices:
                color_to_indices[fill_color] = []
            color_to_indices[fill_color].append(index)

    # 合并相近的颜色
    merged_colors = merge_colors(color_to_indices.keys(), threshold)

    # 创建新的颜色配置
    new_color_to_indices = {}
    color_config = {}

    for color_index, (main_color, similar_colors) in enumerate(merged_colors.items()):
        color_config[str(color_index)] = main_color
        new_color_to_indices[main_color] = []
        for color in similar_colors:
            new_color_to_indices[main_color].extend(color_to_indices[color])
            for index in color_to_indices[color]:
                path = root.findall('.//{http://www.w3.org/2000/svg}path')[index]
                path.set('fill', main_color)
                path.set('id', str(color_index))

    # 将 color_config 添加到 SVG 中
    metadata = ET.Element("metadata")
    desc = ET.SubElement(metadata, "desc", id="color_config")
    desc.text = json.dumps(color_config)
    root.insert(0, metadata)  # 插入到第一个位置

    print(color_config)

    # 将修改后的SVG写回文件
    tree.write(svg_path, encoding='utf-8', xml_declaration=True)

    merge_svg_files(svg_path, line_svg_path, svg_path)

    return svg_path


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def color_distance(color1, color2):
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def is_close_to_black(color, threshold=20):
    r, g, b = hex_to_rgb(color)
    return all(c <= threshold for c in (r, g, b))


def merge_colors(colors, threshold):
    merged = {}
    for color in colors:
        if not merged:
            merged[color] = [color]
        else:
            for key in merged:
                if color_distance(color, key) < threshold:
                    merged[key].append(color)
                    break
            else:
                merged[color] = [color]
    return merged


def merge_svg_files(file1, file2, output_file):
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    # 解析两个 SVG 文件
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # 获取第一个 SVG 的宽度和高度
    width1 = root1.get('width')
    height1 = root1.get('height')

    # 如果宽度和高度包含单位，移除单位
    width1 = re.sub(r'[^0-9.]', '', width1)
    height1 = re.sub(r'[^0-9.]', '', height1)

    # 获取第二个 SVG 的宽度和高度
    width2 = root2.get('width')
    height2 = root2.get('height')

    # 如果宽度和高度包含单位，移除单位
    width2 = re.sub(r'[^0-9.]', '', width2)
    height2 = re.sub(r'[^0-9.]', '', height2)

    # 计算缩放比例
    scale_x = float(width1) / float(width2)
    scale_y = float(height1) / float(height2)

    # 创建一个 group 元素来包含第二个 SVG 的内容
    group = ET.Element('g')
    group.set('transform', f'scale({scale_x},{scale_y})')

    # 将第二个 SVG 的所有子元素移动到 group 中
    for child in root2:
        group.append(child)

    # 将 group 添加到第一个 SVG 中
    root1.append(group)

    # 保存合并后的 SVG
    tree1.write(output_file, encoding='utf-8', xml_declaration=True)
