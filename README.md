# MLP_CNN_RealFaces_Classification

Welcome to MLP_CNN_RealFaces_Classification! This repository contains Python scripts demonstrating the implementation and usage of various machine learning algorithms. From ridge regression to decision trees, you'll find practical examples and explanations to understand how these algorithms work and how to use them in your projects.

## Overview

The repository consists of the following files:

- `ridge_regression.py`: Implementation of Ridge Regression for linear regression tasks.
- `logistic_regression.py`: Implementation of Logistic Regression for binary classification tasks.
- `stochastic_gradient_descent.py`: Implementation of Stochastic Gradient Descent for training logistic regression models with PyTorch.
- `decision_trees.py`: Implementation of Decision Trees for classification tasks using scikit-learn.
- `helpers.py`: Helper functions for data preprocessing, visualization, and model evaluation.
- `skip.py`: Implementation of skip connections for ResNet18.

Additionally, you'll find a `README.md` file providing an overview of the repository and usage instructions.

## Files Description

### `ridge_regression.py`

This file contains the implementation of the `Ridge_Regression` class for performing ridge regression. Ridge regression is a linear regression technique that mitigates multicollinearity among the predictor variables by imposing a penalty on the size of the coefficients.

Key functionalities include:

- `Ridge_Regression` Class: Implements ridge regression using the closed-form solution or matrix inversion method. The `fit` method fits the model to training data, while the `predict` method predicts outputs for new data.

### `logistic_regression.py`

Here, you'll find the `Logistic_Regression` class for logistic regression tasks. Logistic regression is a binary classification algorithm that estimates probabilities for the binary outcome based on one or more predictor variables.

This class is implemented using PyTorch, a popular deep learning framework, and includes methods for training the model (forward) and making predictions (predict).

### `stochastic_gradient_descent.py`

The `Stochastic_gradient_descent` function in this file demonstrates the usage of Stochastic Gradient Descent (SGD) for training logistic regression models. SGD is an optimization algorithm commonly used in machine learning for minimizing the loss function. It updates the model's parameters iteratively by computing the gradient of the loss with respect to the parameters on a small subset of the training data.

This script includes options for regularization, learning rate decay, and handling multi-class classification.

### `decision_trees.py`

This file provides functions for training and visualizing decision tree classifiers using scikit-learn's `DecisionTreeClassifier`. Decision trees are non-parametric supervised learning models used for classification and regression tasks. They learn simple decision rules inferred from the data features to predict the target variable's value.

Key functionalities include training decision trees with different depths and visualizing decision boundaries to understand how the model makes predictions.

### `helpers.py`

The `helpers.py` file contains various helper functions used across different scripts. These functions assist in data preprocessing, visualization, and model evaluation, making it easier to work with machine learning algorithms.

### `skip.py`

The `skip.py` file implements skip connections for ResNet18, a popular convolutional neural network architecture. Skip connections, also known as residual connections, allow the gradient to flow more directly through the network during training, mitigating the vanishing gradient problem. ResNet18 does not have skip connections built-in by default, so this script provides a way to incorporate them into the architecture.

## Usage

To utilize the functionalities provided in these files, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies: NumPy, pandas, scikit-learn, PyTorch, and matplotlib.
3. Execute the desired Python script (e.g., `python ridge_regression.py`) to run the algorithm and observe the results.

## Dependencies

- NumPy: A fundamental package for scientific computing with Python.
- pandas: A powerful data analysis and manipulation library.
- scikit-learn: A machine learning library for Python that provides simple and efficient tools for data mining and data analysis.
- PyTorch: An open-source machine learning library that provides a flexible deep learning framework.
- matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python.
