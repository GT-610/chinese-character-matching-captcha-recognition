import cv2
import numpy as np

def preprocess_image(image):
    """Encapsulate image preprocessing operations"""
    # Binarize and invert
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed = cv2.bitwise_not(binary)

    # Apply median blur with a 3x3 filter
    binary = cv2.medianBlur(binary, 3)

    return processed