import cv2

def preprocess_image(image):
    """封装图像预处理操作"""
    # 二值化 + 反转
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inverted_binary = cv2.bitwise_not(binary)
    
    return inverted_binary