# coding=utf-8
import cv2
import numpy as np

# 读取图像
image = cv2.imread('data/00087-2910593776.png', 0)  # 以灰度模式读取

# 应用高斯模糊
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 二值化
_, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

# 形态学操作
kernel = np.ones((3,3), np.uint8)
morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 显示结果
cv2.imshow('Extracted Lines', morphology)
cv2.waitKey(0)
cv2.destroyAllWindows()