# coding=utf-8
from PIL import Image, ImageDraw, ImageFilter


def smooth_line_art(line_art, smooth_factor=1.5):
    # 放大图像
    large = line_art.resize((int(line_art.width * smooth_factor), int(line_art.height * smooth_factor)), Image.LANCZOS)

    # 应用高斯模糊
    smoothed = large.filter(ImageFilter.GaussianBlur(radius=smooth_factor))

    # 调整大小回原始尺寸
    return smoothed.resize(line_art.size, Image.LANCZOS)


def merge_images(color_image_path, line_art_path, output_path, threshold=100, smooth_factor=1.5):
    # 打开彩色图片和线稿图片
    color_image = Image.open(color_image_path).convert("RGBA")
    line_art = Image.open(line_art_path).convert("RGBA")

    # 确保两张图片大小一致
    if color_image.size != line_art.size:
        line_art = line_art.resize(color_image.size, Image.LANCZOS)

    # 平滑线稿
    smoothed_line_art = smooth_line_art(line_art, smooth_factor)

    # 创建一个新的透明图层
    line_layer = Image.new("RGBA", color_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(line_layer)

    # 遍历平滑后的线稿图的每个像素
    for x in range(smoothed_line_art.width):
        for y in range(smoothed_line_art.height):
            r, g, b, a = smoothed_line_art.getpixel((x, y))
            brightness = (r + g + b) / 3
            # 如果像素偏黑，则将其添加到新图层
            if brightness < threshold:
                draw.point((x, y), fill=(0, 0, 0, 255))

    # 将线条图层合并到彩色图上
    result = Image.alpha_composite(color_image, line_layer)

    # 保存结果
    result.save(output_path)


# 使用示例
merge_images("data/final_image_unified_lines.jpg", "data/test_org.png", "new_output.png",
             threshold=80, smooth_factor=2)
