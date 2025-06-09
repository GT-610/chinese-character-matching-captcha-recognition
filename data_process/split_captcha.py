import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_process.image_preprocessing import preprocess_image
import os

def split_captcha(image, num_splits=4):
    """
    Split the captcha image into a specified number of parts, defaulting to 4.
    :param image: Captcha image (grayscale)
    :param num_splits: Number of splits
    :return: List of split images
    """
    height, width = image.shape  # Get image height and width
    
    split_width = width // num_splits
    splits = []
    for i in range(num_splits):
        start = i * split_width
        end = (i + 1) * split_width
        split_image = image[:, start:end]
        splits.append(split_image)
    return splits

def plot_split_results(dataset, num_samples=5):
    """
    Plot the results of captcha splitting.
    :param dataset: Dataset
    :param num_samples: Number of samples to display
    """
    selected_samples = dataset[:num_samples]  # Select the first few samples

    plt.figure(figsize=(12, 3 * num_samples))

    for i, sample in enumerate(selected_samples):
        captcha_img = cv2.imread(os.path.join(sample['captcha_path'], f"{sample['id']}.jpg"), 0)
        
        processed_img = preprocess_image(captcha_img)

        # Split the preprocessed image
        splits = split_captcha(processed_img)

        # Plot the original image
        plt.subplot(num_samples, 5, i * 5 + 1)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Processed\nSample {i + 1}")
        plt.axis('off')

        # Plot the four split parts
        for j, split in enumerate(splits):
            plt.subplot(num_samples, 5, i * 5 + j + 2)
            plt.imshow(split, cmap='gray')
            plt.title(f"Split {j + 1}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()