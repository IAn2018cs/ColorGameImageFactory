# coding=utf-8
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


def color_quantization(image, num_colors):
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.predict(pixels)
    new_pixels = colors[labels]
    new_image = new_pixels.reshape(img_array.shape).astype(np.uint8)
    return Image.fromarray(new_image)


def remove_small_regions(image, min_size):
    img_array = np.array(image)
    for i in range(3):
        channel = img_array[:, :, i]
        labeled_array, num_features = ndimage.label(channel)
        if num_features > 0:
            sizes = ndimage.sum(channel, labeled_array, range(1, num_features + 1))
            mask_size = sizes < min_size
            if mask_size.any():
                remove_pixel = mask_size[labeled_array - 1]
                channel[remove_pixel] = ndimage.grey_dilation(channel, size=(3, 3))[remove_pixel]
        img_array[:, :, i] = channel
    return Image.fromarray(img_array)


def enhance_edges(image):
    denoised = image.filter(ImageFilter.MedianFilter(size=3))
    sharpened = denoised.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    enhancer = ImageEnhance.Contrast(sharpened)
    enhanced = enhancer.enhance(1.5)
    edge_enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return edge_enhanced


def unify_line_thickness(image, thickness=2):
    # 转换为灰度图
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 膨胀操作，使线条变粗
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 细化操作，使线条更均匀
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 创建掩码
    mask = eroded.astype(bool)

    # 应用掩码到原图
    result = np.array(image)
    result[mask] = [0, 0, 0]  # 将边缘设为黑色

    return Image.fromarray(result)


# 加载图片
image_path = "data/00018-3318816883.png"
original_image = Image.open(image_path)

# 设置要聚类的颜色数量
num_colors = 10

# 进行颜色量化
quantized_image = color_quantization(original_image, num_colors)

# 保存处理后的图片
quantized_image.save("final_image_unified_lines.jpg")

from app.sd_tools import convert_image_line_art

result = convert_image_line_art('../', '1', ["final_image_unified_lines.jpg"])[0]



# # 移除小区域
# cleaned_image = remove_small_regions(quantized_image, min_size=500)
#
# # 增强边缘和线条
# enhanced_image = enhance_edges(cleaned_image)
#
#
#
# # 统一线条粗细
# final_image = unify_line_thickness(enhanced_image, thickness=2)
#
# # 显示原图、颜色量化后的图片和最终处理后的图片
# plt.figure(figsize=(20, 5))
#
# plt.subplot(1, 4, 1)
# plt.imshow(original_image)
# plt.title("Original Image")
# plt.axis('off')
#
# plt.subplot(1, 4, 2)
# plt.imshow(quantized_image)
# plt.title(f"Quantized Image ({num_colors} colors)")
# plt.axis('off')
#
# plt.subplot(1, 4, 3)
# plt.imshow(enhanced_image)
# plt.title("Enhanced Image")
# plt.axis('off')
#
# plt.subplot(1, 4, 4)
# plt.imshow(final_image)
# plt.title("Final Image with Unified Lines")
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()



