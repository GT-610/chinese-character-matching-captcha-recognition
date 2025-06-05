import numpy as np
from collections import Counter
from feature_extract.feature_analysis import extract_hog_features

class KNNCharClassifier:
    def __init__(self, k=5):
        self.k = k
        self.class_weights = None  # 用于类别平衡
    
    def train(self, features, labels):
        """训练KNN分类器（实际是存储特征数据）"""
        self.X_train = np.array(features)
        self.y_train = np.array(labels)
        # 计算类别权重
        class_counts = Counter(labels)
        total = sum(class_counts.values())
        self.class_weights = {cls: total/count for cls, count in class_counts.items()}
    
    def _predict_single(self, x):
        """预测单个样本"""
        # 曼哈顿距离
        distances = np.sum(np.abs(self.X_train - x), axis=1)
        # 获取最近的k个样本索引
        k_indices = np.argsort(distances)[:self.k]
        # 加权投票
        weighted_votes = {}
        for idx in k_indices:
            label = self.y_train[idx]
            weight = 1 / (distances[idx] + 1e-5) * self.class_weights.get(label, 1)
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        return max(weighted_votes, key=weighted_votes.get)
    
    def predict_captcha(self, single_char_features):
        """预测整个验证码（4个字符）"""
        if len(single_char_features) != 4:
            raise ValueError("需要提供4个字符的特征向量")
        return [self._predict_single(feature) for feature in single_char_features]

def evaluate_captcha_accuracy(classifier, test_dataset):
    correct = 0
    total = len(test_dataset)
    print("验证码整体准确率评估进度：")
    for sample in tqdm(test_dataset, total=total):
        # 提取4个字符的特征
        char_features = [extract_hog_features(cv2.imread(path, 0)) for path in sample['single_char_paths'][:4]]
        # 预测验证码
        pred = classifier.predict_captcha(char_features)
        # 真实标签
        true = list(map(int, sample['label']))
        # 比较整个验证码是否完全一致
        if pred == true:
            correct += 1
    return correct / total

def evaluate_accuracy(classifier, test_features, test_labels):
    """评估分类器准确率"""
    from tqdm import tqdm
    
    correct = 0
    captcha_correct = 0 
    print("评估进度：")
    for features, true_label in tqdm(zip(test_features, test_labels), total=len(test_labels)):
        pred = classifier._predict_single(features)
        if pred == true_label:
            correct += 1
    return {
        'char_accuracy': correct / len(test_labels),
        'captcha_accuracy': captcha_correct / len(test_labels)
    }
