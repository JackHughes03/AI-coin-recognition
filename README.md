# Coin Image Classification

An AI model that classifies different types of coins.

## Description

This project uses a Convolutional Neural Network (CNN) to classify images of various coins. The model is trained on a dataset from Kaggle containing images of different coins from around the world.

## Prerequisites

- Python 3.x
- Conda (recommended for TensorFlow installation)
- Required packages:
  - TensorFlow
  - OpenCV
  - NumPy
  - Pandas
  - scikit-learn
  - Pysimplegui

## How to Run

1. Install required packages using Conda (recommended for better TensorFlow compatibility)

2. Download the dataset from [Kaggle - Coin Images](https://www.kaggle.com/datasets/wanderdust/coin-images)

3. Set up the dataset:
   - Unzip the downloaded file
   - Navigate to `coins/data/`
   - Create a new folder called `dataset`
   - Move the `test` and `train` folders inside the `dataset` folder

4. Project setup:
   - Copy the `dataset` folder to the project's root directory
   - Copy `cat_to_name.json` to the project's root directory

5. Running the model:
   - Execute `main.py`
   - Choose 'train' out of the three options
   - Let it train the model

6. For testing custom images:
   - Modify line 61 in `main.py` with your image path
   - Run `main.py` and select the test option

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers
- 2 MaxPooling layers
- Dense layers for classification
- Trained on grayscale images (100x100 pixels)
- Highly experimental and way more depth and focus to come