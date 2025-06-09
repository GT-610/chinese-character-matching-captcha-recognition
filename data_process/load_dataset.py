import os
import pandas as pd

def load_dataset(data_root='data', train=True):
    """
    Load the entire dataset and return a list containing information about all samples.
    Each sample is a dictionary with the following keys:
    - 'id': Sample ID (string)
    - 'captcha_path': Path to the captcha image
    - 'label': The true character sequence of the 4 characters in the captcha (string)
    """
    # Select the appropriate subdirectory and label file based on the 'train' parameter
    subset = 'train' if train else 'test'
    label_file = 'train_label.txt' if train else 'test_label.txt'

    dataset = []
    label_df = pd.read_csv(os.path.join(data_root, label_file), sep=' ', header=None, names=['id', 'label'])

    for _, row in label_df.iterrows():
        parts = row['id'].split(',', 1)
        if len(parts) != 2:
            continue  # Skip lines with incorrect format
            
        sample_id, label = parts
        sample_dir = os.path.join(data_root, subset, sample_id)
        captcha_path = os.path.join(sample_dir)

        dataset.append({
            'id': sample_id,
            'captcha_path': captcha_path,
            'label': label
        })

    return dataset