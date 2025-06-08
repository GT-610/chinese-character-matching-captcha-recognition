import numpy as np
from collections import Counter
from feature_extract.feature_analysis import extract_hog_features

class KNNCharClassifier:
    def __init__(self, k=5):
        self.k = k
        self.class_weights = None
        self.candidate_features = None

    def train(self, features, labels):
        self.X_train = np.array(features)
        self.y_train = np.array(labels)
        class_counts = Counter(labels)
        total = sum(class_counts.values())
        self.class_weights = {cls: total/count for cls, count in class_counts.items()}
        
        self.candidate_features = [self.X_train[self.y_train == i][0] for i in range(9)]

    def _predict_single(self, x, position=None):
        distances = np.sum(np.abs(self.X_train - x), axis=1)
        k_indices = np.argsort(distances)[:self.k]
        
        if position is not None:
            position_filter = (self.y_train % 4 == position)
            k_indices = [idx for idx in k_indices if position_filter[idx]]
        
        weighted_votes = {}
        for idx in k_indices:
            label = self.y_train[idx]
            weight = 1 / (distances[idx] + 1e-5) * self.class_weights.get(label, 1)
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        return max(weighted_votes, key=weighted_votes.get)

    def predict_captcha(self, img):
        """接收验证码路径，进行自动分割"""
        from data_process.split_captcha import split_captcha
        from data_process.image_preprocessing import preprocess_image
        import cv2
        
        # 加载并预处理验证码
        processed_img = preprocess_image(img)
        
        # 分割验证码
        splits = split_captcha(processed_img)
        
        # 对每个分割部分进行预测
        predictions = []
        for position, split_img in enumerate(splits):
            # 提取当前分割部分的特征
            split_feature = extract_hog_features(split_img)
            
            # 与候选字符比较（新增候选字符对比逻辑）
            candidate_distances = [np.sum(np.abs(split_feature - cf)) for cf in self.candidate_features]
            best_candidate = np.argmin(candidate_distances)
            
            # 使用KNN验证结果
            final_pred = self._predict_single(split_feature, position=position)
            
            predictions.append(final_pred)
        
        return predictions

def evaluate_captcha_accuracy(classifier, test_dataset):
    correct = 0
    total = len(test_dataset)
    print("验证码整体准确率评估进度：")
    for sample in tqdm(test_dataset, total=total):
        # 修改字符路径生成方式
        char_features = []
        char_indices = list(map(int, sample['label']))
        for idx in char_indices:
            # 根据captcha_path和字符索引生成路径
            path = os.path.join(sample['captcha_path'], f"{idx}.jpg")
            char_features.append(extract_hog_features(cv2.imread(path, 0)))
        # 预测验证码
        pred = classifier.predict_captcha(sample['captcha_path'])  # 修改传入参数为验证码路径
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
