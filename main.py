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
    # Load the entire dataset
    dataset = load_dataset(data_root='data')

    # Display split results visualization
    plot_split_results(dataset)

    # Analyze feature distribution
    print("Analyzing feature distribution...")
    single_char_features, single_char_labels = analyze_single_char_features(dataset)

    # Visualize HOG features for a sample
    print("Visualizing HOG features...")
    sample = dataset[random.randint(0, len(dataset)-1)]
    show_features_visualization(os.path.join(sample['captcha_path'], "0.jpg"))  # Feature visualization for first split character

    # Show visualizations for multiple samples
    print("Showing sample visualizations...")
    show_samples(dataset, num_samples=3)
    
    # Run KNN classification experiment
    print("\nRunning KNN classification experiment...")
    knn_experiment()

    # Run CNN classification experiment
    print("\nRunning CNN classification experiment...")
    cnn_experiment()

    # Run Siamese network experiment
    print("\nRunning Siamese network experiment...")
    siamese_experiment(force_retrain=False)  # Set True for retraining

    # Run ResNet fine-tuning experiment
    print("\nRunning ResNet fine-tuning experiment...")
    resnet_finetune_experiment()