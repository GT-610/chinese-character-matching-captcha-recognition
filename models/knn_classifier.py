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
        """Predict captcha by splitting image into characters"""
        from data_process.split_captcha import split_captcha
        from data_process.image_preprocessing import preprocess_image
        import cv2
        
        # Preprocess captcha image
        processed_img = preprocess_image(img)
        
        # Split into individual characters
        splits = split_captcha(processed_img)
        
        predictions = []
        for position, split_img in enumerate(splits):
            # Extract HOG features for current character
            split_feature = extract_hog_features(split_img)
            
            # Compare with candidate features
            candidate_distances = [np.sum(np.abs(split_feature - cf)) for cf in self.candidate_features]
            best_candidate = np.argmin(candidate_distances)
            
            # Use KNN for final prediction
            final_pred = self._predict_single(split_feature, position=position)
            
            predictions.append(final_pred)
        
        return predictions

def evaluate_captcha_accuracy(classifier, test_dataset):
    correct = 0
    total = len(test_dataset)
    print("Captcha accuracy evaluation progress:")
    for sample in tqdm(test_dataset, total=total):
        # Generate character paths based on index
        char_features = []
        char_indices = list(map(int, sample['label']))
        for idx in char_indices:
            # Construct path using captcha directory and index
            path = os.path.join(sample['captcha_path'], f"{idx}.jpg")
            char_features.append(extract_hog_features(cv2.imread(path, 0)))
        
        # Predict captcha characters
        pred = classifier.predict_captcha(sample['captcha_path'])
        true = list(map(int, sample['label']))
        
        # Check if entire captcha prediction is correct
        if pred == true:
            correct += 1
    return correct / total

def evaluate_accuracy(classifier, test_features, test_labels):
    """Evaluate classifier accuracy"""
    from tqdm import tqdm
    
    correct = 0
    captcha_correct = 0 
    print("Evaluation progress:")
    for features, true_label in tqdm(zip(test_features, test_labels), total=len(test_labels)):
        pred = classifier._predict_single(features)
        if pred == true_label:
            correct += 1
    return {
        'char_accuracy': correct / len(test_labels),
        'captcha_accuracy': captcha_correct / len(test_labels)
    }
