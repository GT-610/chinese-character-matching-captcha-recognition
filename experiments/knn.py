import random
import cv2
from models.knn_classifier import KNNCharClassifier, evaluate_accuracy
from feature_extract.feature_analysis import analyze_single_char_features, extract_hog_features
from data_process.load_dataset import load_dataset
import os

def knn_experiment():
    # Load training and test datasets
    print("Loading training dataset...")
    train_features, train_labels = analyze_single_char_features(load_dataset(data_root='data', train=True))
    print("Loading test dataset...")
    test_features, test_labels = analyze_single_char_features(load_dataset(data_root='data', train=False))
    
    # Initialize and train the KNN classifier
    print(f"Training KNN classifier (k=5)...")
    knn = KNNCharClassifier(k=5)
    knn.train(train_features, train_labels)
    print("Training complete\n")
    
    # Evaluate accuracy
    print("Evaluating accuracy...")
    accuracy = evaluate_accuracy(knn, test_features, test_labels)
    print(f"\nCharacter accuracy: {accuracy['char_accuracy']:.2%}\n")
    print(f"Captcha accuracy: {accuracy['captcha_accuracy']:.2%}\n")