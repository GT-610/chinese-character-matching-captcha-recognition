import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.feature import hog
from tqdm import tqdm  # 新增进度条库导入

def extract_hog_features(image):
    """提取HOG特征"""
    # 确保输入是8位灰度图
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 调整图像尺寸到统一大小
    resized_img = cv2.resize(image, (64, 64))
    
    # 提取HOG特征
    features = hog(resized_img, 
                  orientations=9, 
                  pixels_per_cell=(8, 8),
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
