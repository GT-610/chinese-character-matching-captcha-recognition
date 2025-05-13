# Chinese Captcha Recognition

This project is a part of my Machine Learning Course Design, focusing on the recognition of Chinese captcha images. The goal is to develop a machine learning model capable of accurately identifying and transcribing Chinese characters from captcha images.

As the course is over, I decided to open-source the project, allowing others to learn from and potentially contribute to the project.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project aims to solve the problem of recognizing Chinese captcha images, which are commonly used in web applications for security purposes. By using machine learning techniques, we aim to develop a robust system that can accurately transcribe Chinese characters from captcha images.

## Structure

- `data/`: Contains the dataset used for training and testing.
- `models/`: Stores the trained models and model architecture files.
- `src/`: Includes the source code for data preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.10+ (I'm using Python 3.12)
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

## Installation

1. Clone the repo.
2. It's recommanded to create a virtual environment before proceeding.
3. Install the required packages. You can do this by running `pip install -r requirements.txt`.
4. Run `python main.py`.