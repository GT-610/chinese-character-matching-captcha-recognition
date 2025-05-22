import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.feature import hog
from tqdm import tqdm  # 新增进度条库导入

def preprocess_image(image):
    """封装图像预处理操作"""
    
    # 二值化 + 反转
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inverted_binary = cv2.bitwise_not(binary)
    
    # 调整大小
    resized_img = cv2.resize(inverted_binary, (64, 64))
    
    return resized_img

def extract_hog_features(image, visualize=False):
    """提取HOG特征"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 调用预处理函数
    processed_img = preprocess_image(image)
    
    if visualize:
        features, hog_img = hog(processed_img, 
                              orientations=16,  # 提升至16方向（适应汉字复杂结构）
                              pixels_per_cell=(32, 32),  # 放大单元尺寸（匹配汉字笔画粗细）
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True)
        return features, hog_img
    else:
        features = hog(processed_img, 
                      orientations=16,
                      pixels_per_cell=(32, 32),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=False)
        return features

def visualize_feature_distribution(features, labels, title="Feature Distribution"):
    """使用PCA和t-SNE可视化特征分布"""
    # 先使用PCA降维到50维
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(features)
    
    # 再使用t-SNE降维到2D
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(pca_result)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        plt.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], 
                   label=f'Char {label}', alpha=0.7)
    
    plt.title(title)
    plt.legend()
    plt.show()

def analyze_single_char_features(dataset):
    """分析单字图片的特征分布"""
    all_features = []
    all_labels = []
    
    # 添加带进度条的遍历
    for sample in tqdm(dataset, desc="分析单字特征"):
        # 修复特征-标签对应关系
        char_indices = list(map(int, sample['label']))
        
        # 遍历验证码中的4个字符索引（而非所有单字路径）
        for idx in char_indices:
            if idx >= len(sample['single_char_paths']):
                continue  # 防止索引越界
                
            path = sample['single_char_paths'][idx]
            img = cv2.imread(path, 0)  # 读取为灰度图
            if img is None:
                continue
                
            # 提取HOG特征
            features = extract_hog_features(img)
            all_features.append(features)
            all_labels.append(idx)  # 标签直接使用字符索引
            
    # 转换为numpy数组
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # 可视化特征分布
    # visualize_feature_distribution(all_features, all_labels, "Single Character Feature Distribution")
    
    return all_features, all_labels
