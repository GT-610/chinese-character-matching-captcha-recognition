import cv2
import numpy as np

def preprocess_image(image):
    """封装图像预处理操作"""
    # 二值化 + 反转
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed = cv2.bitwise_not(binary)

    # 中值滤波（使用3x3滤波器）
    binary = cv2.medianBlur(binary, 3)

    return processed