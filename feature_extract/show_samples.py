import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from skimage.feature import hog  # 导入 hog 函数
from feature_extract.feature_analysis import extract_hog_features, preprocess_image  # 导入预处理函数

def show_features_visualization(img_path):
    """可视化单张图片的特征提取效果"""
    # 读取图片
    img = cv2.imread(img_path, 0)  # 灰度图
    if img is None:
        return
    
    # 调用预处理函数
    processed_img = preprocess_image(img)
    
    # 提取HOG特征并可视化
    features, hog_img = extract_hog_features(processed_img, visualize=True)

    # 显示原始图像、处理后的图像和HOG特征图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(processed_img, cmap='gray')  # 使用预处理后的图像
    plt.title('Processed Image (Before HOG)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(hog_img, cmap='viridis')
    plt.title('HOG Features')
    plt.axis('off')
    plt.show()

def show_samples(dataset, num_samples=5):
    """
    随机展示num_samples个样本，展示验证码图片和对应的4个单字图片
    并可视化特征分析结果
    """
    selected_samples = random.sample(dataset, num_samples)

    # 创建可视化窗口
    plt.figure(figsize=(12, 6 * num_samples))
    
    for i, sample in enumerate(selected_samples):
        print(f"Sample {i+1}: {sample['id']}")

        captcha_img = cv2.imread(sample['captcha_path'], 0)  # 读取为灰度图
        
        # 处理标签
        if isinstance(sample['label'], float):
            label_str = str(int(sample['label']))
            char_indices = [int(char) for char in label_str]
        else:
            char_indices = list(map(int, sample['label']))
        
        # 读取单字图片
        single_char_imgs = [cv2.imread(p, 0) for p in sample['single_char_paths']]
        matched_chars = [single_char_imgs[idx] for idx in char_indices]

        # 展示验证码图片
        total_cols = len(matched_chars) + 2  # 计算总列数（单字数 + 2（验证码+特征图））
        plt.subplot(num_samples, total_cols, i * total_cols + 1)
        plt.imshow(captcha_img, cmap='gray')
        plt.title(f"CAPTCHA\nSample {i+1}: {sample['id']}")
        plt.axis('off')

        # 展示匹配的单字图片
        for j, img in enumerate(matched_chars):
            plt.subplot(num_samples, total_cols, i * total_cols + j + 2)
            plt.imshow(img, cmap='gray')
            plt.title(f"Matched\nChar {j+1}: {char_indices[j]}")
            plt.axis('off')
            
        # 可视化特征（修改为调用统一函数）
        features, hog_img = extract_hog_features(captcha_img, visualize=True)  # 调用统一接口
        
        plt.subplot(num_samples, total_cols, i * total_cols + total_cols)
        plt.imshow(hog_img, cmap='viridis')
        plt.title(f"HOG Features\n(Sample {i+1})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()