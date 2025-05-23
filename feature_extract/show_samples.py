import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from skimage.feature import hog
from feature_extract.feature_analysis import extract_hog_features, preprocess_image

def show_features_visualization(img_path):
    """可视化单张图片的特征提取效果"""
    img = cv2.imread(img_path, 0)  # 灰度图
    if img is None:
        return

    processed_img = preprocess_image(img)
    features, hog_img = extract_hog_features(processed_img, visualize=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Processed Image (Before HOG)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hog_img, cmap='viridis')
    plt.title('HOG Features')
    plt.axis('off')
    plt.show()

def show_samples(dataset, num_samples=5):
    """
    随机展示num_samples个样本，展示验证码图片及其HOG特征图
    每一行一个样本，第一列为验证码图，第二列为HOG特征图
    """
    selected_samples = random.sample(dataset, num_samples)

    plt.figure(figsize=(8, 4 * num_samples))

    for i, sample in enumerate(selected_samples):
        print(f"Sample {i+1}: {sample['id']}")

        captcha_img = cv2.imread(sample['captcha_path'], 0)  # 灰度图

        # 展示验证码图片
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(captcha_img, cmap='gray')
        plt.title(f"CAPTCHA\nSample {i+1}: {sample['id']}")
        plt.axis('off')

        # 可视化特征
        features, hog_img = extract_hog_features(captcha_img, visualize=True)

        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(hog_img, cmap='viridis')
        plt.title(f"HOG Features\n(Sample {i+1})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()