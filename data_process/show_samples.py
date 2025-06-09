import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def show_samples(dataset, num_samples=5):
    """
    Randomly display `num_samples` samples, showing captcha images and their corresponding single character images.
    """
    selected_samples = random.sample(dataset, num_samples)

    # Create a figure to display samples in rows, each row contains one sample.
    fig, axs = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))

    for i, sample in enumerate(selected_samples):
        captcha_img = cv2.imread(sample['captcha_path'], 0)  # Read the captcha image as grayscale.
        char_indices = list(map(int, sample['label']))
        single_char_imgs = [cv2.imread(p, 0) for p in sample['single_char_paths']]
        matched_chars = [single_char_imgs[idx] for idx in char_indices]

        # Display the captcha image in the first column.
        axs[i, 0].imshow(captcha_img, cmap='gray')
        axs[i, 0].set_title(f"Sample {i+1}: {sample['id']}")
        axs[i, 0].axis('off')

        # Display the matched single character images in columns 2 to 5.
        for j, img in enumerate(matched_chars):
            axs[i, j+1].imshow(img, cmap='gray')
            axs[i, j+1].set_title(f"Char {j+1}: {char_indices[j]}")
            axs[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()