# Deep Learning for Restaurant Reviews

## Overview
 We were given a dataset of restaurant reviews. Our objective is to create Recurrent Neural Network (RNN) models to forecast the Liked values based on Review, and then determine the most optimal RNN model to recommend to a company.

## Files Included
- Restaurant_Reviews.py: Python script containing the code for loading the dataset, building CNN models, training, and evaluation.
- Restaurant_Reviews.ipynb: Google Colab Notebook containing the detailed code implementation and explanations.
- requirements.txt: Text file listing the required Python packages and their versions.
-LICENSE.txt: Text file containing the license information for the project.


## Installation
To run this project, ensure you have Python installed on your system. You can install the required dependencies using the `requirements.txt` file.

## Usage
1. Begin by preprocessing the restaurant reviews dataset. Tokenize the text, remove stop words, and convert the text data into a numerical format using techniques like TF-IDF or word embeddings.
2. Implement various machine learning models to classify the sentiment of restaurant reviews. Consider using traditional algorithms like Naive Bayes or SVM, or explore more advanced options such as RNNs or BERT-based models.
3. Train the models on a portion of the preprocessed dataset and evaluate their performance on a separate validation or test set. Monitor key performance metrics like accuracy, precision, recall, and F1-score during training. Employ techniques like cross-validation to ensure model robustness.
4. Optimize the models by tuning their hyperparameters. Experiment with different combinations of hyperparameters using methods like grid search or random search to find the best configuration.

## Data Source
The dataset used for this project is Restaurant_Reviews.tsv sourced from [Kaggle](https://www.kaggle.com/datasets/hj5992/restaurantreviews).
