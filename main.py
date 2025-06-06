from data_process.load_dataset import load_dataset
from data_process.split_captcha import plot_split_results
from feature_extract.show_samples import show_samples, show_features_visualization
from feature_extract.feature_analysis import analyze_single_char_features
from models.knn_classifier import KNNCharClassifier, evaluate_accuracy

from experiments.knn import knn_experiment
from experiments.cnn import cnn_experiment
from experiments.siamese import siamese_experiment
from experiments.resnet_finetune import resnet_finetune_experiment

import random
import cv2
import os

if __name__ == "__main__":
    # 1. 读取整个数据集
    dataset = load_dataset(data_root='data')

    # 2. 显示分割后的样本
    # plot_split_results(dataset)

    # 3. 特征分析 - 可视化特征分布
    # print("Analyzing feature distribution...")
    # single_char_features, single_char_labels = analyze_single_char_features(dataset)

    # 4. 特征分析 - 可视化具体特征
    # print("Visualizing HOG features...")
    # sample = dataset[random.randint(0, len(dataset)-1)]
    # show_features_visualization(os.path.join(sample['captcha_path'], "0.jpg")) # 展示第一个分割字符的特征

    # 5. 随机展示样本及其特征分析
    # print("Showing sample visualizations...")
    # show_samples(dataset, num_samples=3)
    
    # 6. KNN分类实验
    print("\n运行KNN分类实验...")
    knn_experiment()

    # 7. CNN分类实验
    # print("\n运行CNN分类实验...")
    # cnn_experiment()

    # 8. Siamese网络实验
    # print("\n运行Siamese网络实验...")
    # siamese_experiment(force_retrain=False)  # 设置为True可强制重新训练

    # 9. ResNet微调实验
    # print("\n运行ResNet微调实验...")
    # resnet_finetune_experiment()
