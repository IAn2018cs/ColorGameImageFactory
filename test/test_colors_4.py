# coding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import KMeans


def color_quantization(image, num_colors):
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.predict(pixels)
    new_pixels = colors[labels]
    new_image = new_pixels.reshape(img_array.shape).astype(np.uint8)

    segmented_labels = labels.reshape(img_array.shape[:2])  # 颜色分区的编号图
    return Image.fromarray(new_image), segmented_labels


def separate_color_area(matrix):  ##区域分区的编号图，颜色分区个数<区域分区个数
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


# 加载图片
image_path = "data/00018-3318816883.png"
original_image = Image.open(image_path)

# 设置要聚类的颜色数量
num_colors = 10

# 进行颜色量化
quantized_image, segmented_labels = color_quantization(original_image, num_colors)

# 保存处理后的图片
quantized_image.save("final_image_unified_lines.jpg")

color_areas = separate_color_area(segmented_labels)
centroids = center_of_mass(color_areas, labels=color_areas, index=np.arange(1, color_areas.max() + 1))
centroids = [i for i in centroids if not pd.isna(i[0])]

median_filtered_image = Image.open("data/test_org.png")

plt.figure(figsize=(20, 20))

# plt.subplot(1, 1, 1)
# plt.imshow(median_filtered_image)
# plt.axis('off')

# Annotate each cluster with its corresponding number
# for centroid in centroids:
#     y, x = centroid  # centroid returns (row, column)
#     value = segmented_labels[int(y), int(x)]
#     plt.text(x, y, str(value), color='red', ha='center', va='center', fontsize=16)


# temp_image = np.stack(np.array(median_filtered_image)) + \
#              quantized_image * np.stack([(segmented_labels == 0) | (segmented_labels == 1) | (segmented_labels == 2) | (
#         segmented_labels == 3) | (segmented_labels == 4)] * 3, axis=-1)

# temp_image = np.stack(np.array(median_filtered_image)) + quantized_image

temp_image = np.stack(np.array(median_filtered_image)) + \
             quantized_image * np.stack([(segmented_labels == 0) | (segmented_labels == 1)] * 3, axis=-1)

plt.subplot(1, 1, 1)
plt.imshow(temp_image)
plt.axis('off')

plt.tight_layout()
plt.show()


