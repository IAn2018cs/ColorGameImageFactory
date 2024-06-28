# coding=utf-8
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# 假设我们已经有了LLM的颜色建议
color_suggestions = {
    'body': (255, 255, 255),  # 白色
    'ears_inside': (255, 192, 203),  # 粉色
    'carrot': (255, 165, 0)  # 橙色
}

# 读取图像
img = cv2.imread('data/line_test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建空白的彩色图层
color_layer = np.zeros_like(img)

# 对图像进行分割（这里使用简单的阈值分割作为示例）
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 为每个区域上色
for contour in contours:
    # 这里应该有更复杂的逻辑来决定每个区域应该使用哪种颜色
    color = color_suggestions['body']  # 默认使用身体颜色
    cv2.drawContours(color_layer, [contour], 0, color, -1)

# 混合原始线稿和彩色图层
result = cv2.addWeighted(img, 0.5, color_layer, 0.5, 0)

# 应用平滑滤镜
result = cv2.GaussianBlur(result, (5, 5), 0)

# 保存结果
cv2.imwrite('colored_rabbit.png', result)
