# MNIST Digit Classifier with CNN - Gradio Interface

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras and is deployed with a Gradio interface, allowing users to draw a digit on a canvas and get real-time predictions.

## Project Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9). In this project, we use a Convolutional Neural Network (CNN) model to train on the MNIST dataset and then deploy the trained model with a Gradio app where users can draw digits to be predicted.

Key Steps:
1. **Preprocessing** the MNIST dataset.
2. **Building and training** a CNN model using Keras/TensorFlow.
3. **Creating an interactive Gradio interface** where users can draw digits to get predictions.

## Features

- **Convolutional Neural Network (CNN)**: A deep learning model for classifying handwritten digits.
- **Gradio Interface**: An easy-to-use web interface where you can draw digits and get predictions.
- **Real-time Prediction**: Predict the digit instantly by drawing on the Gradio sketchpad.

## Technologies Used

- **Python 3.x**: Programming language for model development.
- **TensorFlow**: Deep learning framework for building and training the CNN.
- **Keras**: High-level neural network API for TensorFlow.
- **OpenCV**: Image processing library for preprocessing input images.
- **Gradio**: Library to quickly build and share machine learning demos.
- **NumPy**: Numerical computing library used for data manipulation.

## Installation

You can run this project on Google Colab or install it locally. Follow the instructions below to get started:

### 1. Install dependencies

To install the required dependencies locally, run:

```bash
pip install tensorflow opencv-python numpy gradio
