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
    
    # 随机测试几个验证码
    test_dataset = load_dataset(data_root='data', train=False)
    print("随机抽取3个验证码进行测试：")
    for i, sample in enumerate(random.sample(test_dataset, 3)):
        print(f"\n正在处理测试样本 {i+1}/3...")
        # 提取验证码中的4个字符特征
        char_features = []
        char_indices = list(map(int, sample['label']))
        for idx in char_indices:
            path = os.path.join(sample['captcha_path'], f"{idx}.jpg")
            img = cv2.imread(path, 0)
            char_features.append(extract_hog_features(img))
        pred = knn.predict_captcha(char_features)
        true = list(map(int, sample['label']))
        
        comparison = ['√' if p == t else '×' for p, t in zip(pred, true)]
        
        print(f"\n样本 {sample['id']}:")
        print(f"真实标签: {true}")
        print(f"预测结果: {pred}")
        print(f"对比结果: {comparison}")
        print(f"{'='*40}")