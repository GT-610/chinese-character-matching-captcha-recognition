import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def show_samples(dataset, num_samples=5):
    """
    随机展示num_samples个样本，展示验证码图片和对应的4个单字图片
    """
    selected_samples = random.sample(dataset, num_samples)

    # 创建一个大图，每行展示一个样本
    fig, axs = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))

    for i, sample in enumerate(selected_samples):
        captcha_img = cv2.imread(sample['captcha_path'], 0)  # 读取为灰度图
        char_indices = list(map(int, sample['label']))
        single_char_imgs = [cv2.imread(p, 0) for p in sample['single_char_paths']]
        matched_chars = [single_char_imgs[idx] for idx in char_indices]

        # 展示验证码图片（第一列）
        axs[i, 0].imshow(captcha_img, cmap='gray')
        axs[i, 0].set_title(f"Sample {i+1}: {sample['id']}")
        axs[i, 0].axis('off')

        # 展示匹配的单字图片（第二列到第五列）
        for j, img in enumerate(matched_chars):
            axs[i, j+1].imshow(img, cmap='gray')
            axs[i, j+1].set_title(f"Char {j+1}: {char_indices[j]}")
            axs[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()