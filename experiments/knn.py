import random
import cv2
from models.knn_classifier import KNNCharClassifier, evaluate_accuracy
from feature_extract.feature_analysis import analyze_single_char_features, extract_hog_features
from data_process.load_dataset import load_dataset
import os

def knn_experiment():
    # 获取所有单字特征和标签
    print("正在加载训练集...")
    train_features, train_labels = analyze_single_char_features(load_dataset(data_root='data', train=True))
    print("正在加载测试集...")
    test_features, test_labels = analyze_single_char_features(load_dataset(data_root='data', train=False))
    
    # 初始化并训练KNN分类器
    print(f"正在训练KNN分类器（k=5）...")
    knn = KNNCharClassifier(k=5)
    knn.train(train_features, train_labels)
    print("训练完成\n")
    
    # 评估准确率
    print("正在进行准确率评估...")
    accuracy = evaluate_accuracy(knn, test_features, test_labels)
    print(f"\nKNN分类器单字准确率: {accuracy['char_accuracy']:.2%}\n")
    print(f"KNN分类器验证码准确率: {accuracy['captcha_accuracy']:.2%}\n")