import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.feature import hog
from tqdm import tqdm

from data_process.image_preprocessing import preprocess_image

def extract_hog_features(image, visualize=False):
    """Extract HOG features"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Image preprocessing
    processed_img = preprocess_image(image)
    
    # Resize to standard dimension
    processed_img = cv2.resize(processed_img, (64, 64), interpolation=cv2.INTER_AREA)
    
    if visualize:
        features, hog_img = hog(processed_img, 
                              orientations=16,
                              pixels_per_cell=(32, 32),
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
    """Visualize feature distribution using PCA and t-SNE"""
    # Dimensionality reduction with PCA
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(features)
    
    # Further reduction with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(pca_result)
    
    # Visualization setup
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
    """Analyze feature distribution of single character images"""
    all_features = []
    all_labels = []
    
    # Process with progress bar
    for sample in tqdm(dataset, desc="Feature analysis"):
        char_indices = list(map(int, sample['label']))
        
        # Process each character in captcha
        for idx in char_indices:
            path = os.path.join(sample['captcha_path'], f"{idx}.jpg")
            img = cv2.imread(path, 0)  # Read as grayscale
            if img is None:
                continue
                
            # Extract HOG features
            features = extract_hog_features(img)
            all_features.append(features)
            all_labels.append(idx) 
    
    # Convert to numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels
