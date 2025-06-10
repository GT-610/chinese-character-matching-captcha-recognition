[简体中文](README-zh.md)

# Chinese Captcha Recognition

This project is a part of my Machine Learning Course Design, focusing on the recognition of Chinese captcha images. The goal is to develop a machine learning model capable of accurately identifying and transcribing Chinese characters from captcha images.

As the course is over, I decided to open-source the project, allowing others to learn from and potentially contribute to the project.

## Introduction

The project aims to solve the problem of recognizing Chinese captcha images, which are commonly used in web applications for security purposes. By using machine learning techniques, we aim to develop a robust system that can accurately transcribe Chinese characters from captcha images.

Dataset description is in `README.md` in datasets zip file.

## Course Design Task

Below is the original task requirement for my course.

### Data Description

The training set consists of 9000 samples, numbered 0000 ~ 9999, each sample corresponds to a folder, which includes: an image with the same name as the folder, which is the captcha image containing 4 Chinese characters; 9 single-character images, numbered 0 ~ 8, which include the 4 characters in the captcha image; `train_label.txt` corresponding to the sample number and label. The data format is image.

### Task Objectives

The captcha contains 4 Chinese characters and 9 single Chinese characters, requiring the selection of the 4 characters in the captcha in order from the 9 single characters to achieve character matching for the `test` dataset.

1. Read the data from the corresponding files.
2. Data preprocessing, feature analysis, feature representation design, visualization, etc..
3. Select different machine learning algorithms for character matching captcha recognition, compare and analyze the results of different models, and visualize them.
4. Analyze the differences between the results using accuracy as the evaluation metric based on the characteristics of different algorithms.

## Structure

- `data_process/`: Contains the data preprocessing scripts.
- `feature_extract`: Contains the feature extraction scripts
- `models/`: Stores the trained models and model architecture files.
- `experiments/`: Contains the methods used for solving the problem.
- `plot_figures/`: Contains the plotting scripts.
- `main.py`: The main script for running the project.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.10+ (I'm using Python 3.12)
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Seaborn

## Installation

1. Clone the repo.
2. It's recommanded to create a virtual environment before proceeding. Either `Conda` / `Mamba` or `Virtualenv` are ok.
3. Install the required packages. You can do this by running `pip install -r requirements.txt`.
4. Download the dataset from the Release page, and unzip `data` folder into the repo root directory.
5. Check and comment or uncomment the appropriate section in `main.py`.
6. Run `python main.py`.

## License
This project is licensed under the [MIT License](LICENSE).
