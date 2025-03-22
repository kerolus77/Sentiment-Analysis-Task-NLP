# Sentiment Analysis on IMDB Reviews Using LSTM

## Introduction

This project focuses on building a sentiment analysis model to classify IMDB movie reviews as either positive or negative. The model leverages Long Short-Term Memory (LSTM) networks and is implemented using TensorFlow and Keras.

## Dataset Description

The IMDB dataset, available through Keras, is used for this project. It contains 50,000 movie reviews, evenly divided into training and testing sets. Each review is labeled as either positive (1) or negative (0). To simplify processing, only the 10,000 most frequently occurring words are included in the vocabulary.

## Model Design

The architecture of the model is as follows:

- **Embedding Layer**: Maps word indices to dense vectors of size 32.
- **Dropout Layer (Rate: 0.2)**: Helps mitigate overfitting after the embedding layer.
- **LSTM Layer (32 Units)**: Handles sequential data with a recurrent dropout rate of 0.2.
- **Dropout Layer (Rate: 0.2)**: Adds further regularization to reduce overfitting.
- **Dense Output Layer**: A single neuron with a sigmoid activation function for binary classification.

## Key Features

- Padding sequences to a fixed length of 500 tokens for uniform input size.
- Dropout layers to improve generalization and reduce overfitting.
- Early stopping to halt training when validation performance stops improving.
- Visualization of training and validation metrics, including accuracy and loss.

## Prerequisites

To run the project, ensure the following dependencies are installed:

- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

## Steps to Execute

1. Open the `sentiment_analysis_task.ipynb` file in Jupyter Notebook.
2. Execute all cells in the notebook sequentially.
3. The IMDB dataset will be automatically downloaded, and the model will be trained and evaluated.
4. Training results, including accuracy and loss plots, will be displayed.

## Performance Summary

The model demonstrates strong performance in classifying sentiments, aided by regularization techniques:

- **Early Stopping**: Monitors validation loss and halts training when no further improvement is observed.
- **Dropout Layers**: Prevent overfitting by reducing reliance on specific neurons.
- **Evaluation Metrics**: Includes accuracy and loss curves for both training and validation phases.

## Potential Enhancements

- Experiment with alternative architectures, such as GRU or Bidirectional LSTM.
- Incorporate attention mechanisms to improve focus on key parts of the text.
- Explore transfer learning using pre-trained language models like BERT or GPT.
