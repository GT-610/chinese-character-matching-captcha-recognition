import numpy as np
from collections import Counter
from feature_extract.feature_analysis import extract_hog_features

class KNNCharClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None  # 存储训练特征
        self.y_train = None  # 存储训练标签
    
    def train(self, features, labels):
        """训练KNN分类器（实际是存储特征数据）"""
        self.X_train = np.array(features)
        self.y_train = np.array(labels)
    
    def _predict_single(self, x):
        """预测单个样本"""
        # 计算欧氏距离
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # 获取最近的k个样本索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取最近k个样本的标签
        k_nearest_labels = self.y_train[k_indices]
        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict_captcha(self, single_char_features):
        """预测整个验证码（4个字符）"""
        if len(single_char_features) != 4:
            raise ValueError("需要提供4个字符的特征向量")
        return [self._predict_single(feature) for feature in single_char_features]

def evaluate_accuracy(classifier, test_features, test_labels):
    """评估分类器准确率"""
    from tqdm import tqdm  # 新增进度条库导入
    
    correct = 0
    print("评估进度：")
    for features, true_label in tqdm(zip(test_features, test_labels), total=len(test_labels)):
        pred = classifier._predict_single(features)
        if pred == true_label:
            correct += 1
    return correct / len(test_labels)
