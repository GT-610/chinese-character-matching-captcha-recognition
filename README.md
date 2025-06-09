[简体中文](README-zh.md)

# Chinese Captcha Recognition

This project is a part of my Machine Learning Course Design, focusing on the recognition of Chinese captcha images. The goal is to develop a machine learning model capable of accurately identifying and transcribing Chinese characters from captcha images.

As the course is over, I decided to open-source the project, allowing others to learn from and potentially contribute to the project.

## Introduction

The project aims to solve the problem of recognizing Chinese captcha images, which are commonly used in web applications for security purposes. By using machine learning techniques, we aim to develop a robust system that can accurately transcribe Chinese characters from captcha images.

Dataset description is in `README.md` in datasets zip file.

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
2. It's recommanded to create a virtual environment before proceeding.
3. Install the required packages. You can do this by running `pip install -r requirements.txt`.
4. Check and comment or uncomment the appropriate section in `main.py`.
4. Run `python main.py`.

## License
This project is licensed under the [MIT License](LICENSE).