from data_process.load_dataset import load_dataset, debug_sample
from feature_extract.show_samples import show_samples, show_features_visualization
from feature_extract.feature_analysis import analyze_single_char_features, extract_hog_features
from models.knn_classifier import KNNCharClassifier, evaluate_accuracy


import random
import cv2

# 新增KNN训练和评估部分
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
    print(f"\nKNN分类器单字准确率: {accuracy:.2%}\n")
    
    # 随机测试几个验证码
    test_dataset = load_dataset(data_root='data', train=False)
    print("随机抽取3个验证码进行测试：")
    for i, sample in enumerate(random.sample(test_dataset, 3)):
        print(f"\n正在处理测试样本 {i+1}/3...")
        # 提取验证码中的4个字符特征
        char_features = []
        for j, path in enumerate(sample['single_char_paths'][:4]):
            img = cv2.imread(path, 0)
            char_features.append(extract_hog_features(img))
        
        # 预测和真实标签
        pred = knn.predict_captcha(char_features)
        true = list(map(int, sample['label']))
        print(f"\n样本 {sample['id']}:")
        print(f"真实标签: {true}")
        print(f"预测结果: {pred}\n{'='*40}")

if __name__ == "__main__":
    # 1. 读取整个数据集
    dataset = load_dataset(data_root='data')

    # 2. 特征分析 - 可视化特征分布
    # print("Analyzing feature distribution...")
    # single_char_features, single_char_labels = analyze_single_char_features(dataset)

    # 3. 特征分析 - 可视化具体特征
    print("Visualizing HOG features...")
    show_features_visualization(dataset[random.randint(0, 9000)]['single_char_paths'][0]) # 展示单个字符的特征提取效果

    # 4. 随机展示样本及其特征分析
    # print("Showing sample visualizations...")
    # show_samples(dataset, num_samples=3)
    
    # 5. KNN分类实验
    print("\n运行KNN分类实验...")
    knn_experiment()
