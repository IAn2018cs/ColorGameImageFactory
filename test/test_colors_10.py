# coding=utf-8
from app.colorful_svg import read_svg_metadata
from app.colorful_svg import update_svg_colors
from app.colorful_svg_v2 import color_quantization


def clear_color(file, colors):
    zero_colors = {}
    for c in colors:
        zero_colors[c] = "#ffffff"
    new_path = update_svg_colors(file, zero_colors)


if __name__ == '__main__':
    input_path = "./data/10_rgb.png"

    svg_file = color_quantization(input_path, './data/1720011440260_BdUc_line.svg', './data', threshold=10)
    cs = read_svg_metadata(svg_file)

    clear_color(svg_file, cs)
