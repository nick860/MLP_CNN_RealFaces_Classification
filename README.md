# MLP_CNN_RealFaces_Classification

Welcome to MLP_CNN_RealFaces_Classification! This repository contains Python scripts demonstrating the implementation and usage of various machine learning algorithms for real faces classification tasks.

## Overview

The repository consists of the following files:

- `main.py`: Main script for training and evaluating the MLP and CNN models.
- `helpers.py`: Helper functions for data preprocessing, visualization, and model evaluation.
- `cnn.py`: Script containing the architecture definition and training code for the Convolutional Neural Network (CNN) model.
- `skip.py`: Implementation of skip connections for improving the CNN architecture.
- `xg.py`: Script for training and evaluating the XGBoost model.

Additionally, you'll find a `README.md` file providing an overview of the repository and usage instructions.

## Files Description

### `main.py`

This file serves as the entry point for training and evaluating the MLP and CNN models. It includes functions for loading the dataset, preprocessing the data, training the models, and evaluating their performance.

Key functionalities include:

- Data loading and preprocessing.
- Training and evaluation of MLP and CNN models.
- Reporting performance metrics such as accuracy, precision, recall, and F1-score.

### `helpers.py`

The `helpers.py` file contains various helper functions used across different scripts. These functions assist in data preprocessing, visualization, and model evaluation, making it easier to work with machine learning algorithms.

### `cnn.py`

The `cnn.py` script contains the definition of the Convolutional Neural Network (CNN) architecture for face classification tasks. It includes functions for building the CNN model, training the model using backpropagation, and evaluating the model's performance on test data.

### `skip.py`

The `skip.py` file implements skip connections for improving the performance of the CNN architecture. Skip connections, also known as residual connections, help alleviate the vanishing gradient problem during training by allowing gradients to flow directly through the network.

### `xg.py`

The `xg.py` script is responsible for training and evaluating the XGBoost model for face classification. XGBoost is an ensemble learning method that uses decision trees as base learners and is known for its high performance and efficiency.

## Usage

To utilize the functionalities provided in these files, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies: NumPy, pandas, scikit-learn, TensorFlow (for CNN), XGBoost.
3. Execute the `main.py` script to train and evaluate the MLP and CNN models.
4. Optionally, you can use the `xg.py` script to train and evaluate the XGBoost model independently.

