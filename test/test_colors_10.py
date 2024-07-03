# coding=utf-8
import json
import xml.etree.ElementTree as ET
import math
import vtracer

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    return math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)

def is_close_to_black(color, threshold=30):
    r, g, b = hex_to_rgb(color)
    return all(c <= threshold for c in (r, g, b))

def merge_colors(colors, threshold=30):
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

def config_svg(path):
    # 解析SVG文件
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tree = ET.parse(path)
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
    merged_colors = merge_colors(color_to_indices.keys())

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
    tree.write('./data/modified_svg_file.svg', encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    input_path = "./data/10_rgb.png"
    output_path = "./data/10_rgb_vtracer.svg"

    vtracer.convert_image_to_svg_py(input_path,
                                    output_path,
                                    colormode='color',  # ["color"] or "binary"
                                    hierarchical='stacked',  # ["stacked"] or "cutout"
                                    mode='spline',  # ["spline"] "polygon", or "none"
                                    filter_speckle=4,  # default: 4
                                    color_precision=3,  # default: 6
                                    layer_difference=1,  # default: 16
                                    corner_threshold=60,  # default: 60
                                    length_threshold=4.0,  # in [3.5, 10] default: 4.0
                                    max_iterations=10,  # default: 10
                                    splice_threshold=45,  # default: 45
                                    path_precision=8  # default: 8
                                    )

    config_svg(output_path)
