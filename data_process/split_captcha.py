import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_process.image_preprocessing import preprocess_image  # 导入预处理函数

def split_captcha(image, num_splits=4):
    """
    将验证码图片分割成指定数量的部分，默认为4份。
    :param image: 验证码图片 (灰度图)
    :param num_splits: 分割份数
    :return: 分割后的图片列表
    """
    height, width = image.shape  # 获取图像高度和宽度
    
    split_width = width // num_splits
    splits = []
    for i in range(num_splits):
        start = i * split_width
        end = (i + 1) * split_width
        split_image = image[:, start:end]
        splits.append(split_image)
    return splits

def plot_split_results(dataset, num_samples=5):
    """
    绘制验证码分割结果。
    :param dataset: 数据集
    :param num_samples: 展示的样本数量
    """
    selected_samples = dataset[:num_samples]  # 取前几个样本

    plt.figure(figsize=(12, 3 * num_samples))

    for i, sample in enumerate(selected_samples):
        captcha_img = cv2.imread(sample['captcha_path'], 0)  # 读取为灰度图
        
        # 调用预处理函数
        processed_img = preprocess_image(captcha_img)
        
        # 分割预处理后的图片
        splits = split_captcha(processed_img)

        # 绘制原图
        plt.subplot(num_samples, 5, i * 5 + 1)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Processed\nSample {i + 1}")
        plt.axis('off')

        # 绘制分割后的四部分
        for j, split in enumerate(splits):
            plt.subplot(num_samples, 5, i * 5 + j + 2)
            plt.imshow(split, cmap='gray')
            plt.title(f"Split {j + 1}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()