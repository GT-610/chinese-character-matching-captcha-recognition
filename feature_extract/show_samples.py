import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from feature_extract.feature_analysis import extract_hog_features  # 导入特征提取函数

def show_features_visualization(img_path):
    """可视化单张图片的特征提取效果"""
    # 读取图片
    img = cv2.imread(img_path, 0)  # 灰度图
    if img is None:
        return
    
    # 提取HOG特征并可视化
    resized_img = cv2.resize(img, (64, 64))
    features, hog_img = hog(resized_img, 
                          orientations=9, 
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          visualize=True)
    
    # 显示原始图像和HOG特征图
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(resized_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
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

    for i, sample in enumerate(selected_samples):
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

        # 创建可视化窗口
        plt.figure(figsize=(12, 6))
        
        # 展示验证码图片
        plt.subplot(3, num_samples, i+1)
        plt.imshow(captcha_img, cmap='gray')
        plt.title(f"CAPTCHA\nSample {i+1}: {sample['id']}")
        plt.axis('off')

        # 展示匹配的单字图片
        for j, img in enumerate(matched_chars):
            plt.subplot(3, num_samples, num_samples + i * num_samples + j + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Matched\nChar {j+1}: {char_indices[j]}")
            plt.axis('off')
            
        # 可视化特征
        plt.subplot(3, num_samples, 2*num_samples + i + 1)
        features, hog_img = hog(cv2.resize(captcha_img, (64, 64)), 
                               orientations=9, 
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L2-Hys',
                               visualize=True)
        plt.imshow(hog_img, cmap='viridis')
        plt.title(f"HOG Features\n(Sample {i+1})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()