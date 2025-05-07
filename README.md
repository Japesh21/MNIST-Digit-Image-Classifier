MNIST Digit Classifier with CNN - Gradio Interface
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras and is deployed with a Gradio interface, allowing users to draw a digit on a canvas and get real-time predictions.

Project Overview
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9). In this project, we use a Convolutional Neural Network (CNN) model to train on the MNIST dataset and then deploy the trained model with a Gradio app where users can draw digits to be predicted.

Key Steps:

Preprocessing the MNIST dataset.

Building and training a CNN model using Keras/TensorFlow.

Creating an interactive Gradio interface where users can draw digits to get predictions.

Features
Convolutional Neural Network (CNN): A deep learning model for classifying handwritten digits.

Gradio Interface: An easy-to-use web interface where you can draw digits and get predictions.

Real-time Prediction: Predict the digit instantly by drawing on the Gradio sketchpad.

Technologies Used
Python 3.x: Programming language for model development.

TensorFlow: Deep learning framework for building and training the CNN.

Keras: High-level neural network API for TensorFlow.

OpenCV: Image processing library for preprocessing input images.

Gradio: Library to quickly build and share machine learning demos.

NumPy: Numerical computing library used for data manipulation.

Installation
You can run this project on Google Colab or install it locally. Follow the instructions below to get started:

1. Install dependencies
To install the required dependencies locally, run:

bash
Copy
Edit
pip install tensorflow opencv-python numpy gradio
2. Clone this repository
bash
Copy
Edit
git clone https://github.com/your-username/mnist-cnn-gradio.git
cd mnist-cnn-gradio
3. Run the Code
Run the provided Python script to start the Gradio app.

bash
Copy
Edit
python app.py
Or, open the notebook directly on Google Colab and run it step by step.

4. Open the Gradio Interface
After running the code, the Gradio interface will automatically launch in your browser where you can draw digits and see the predictions.

Model Details
The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:

Input Layer: 28x28x1 grayscale image.

Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation.

MaxPool Layer 1: 2x2 pool size.

Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation.

MaxPool Layer 2: 2x2 pool size.

Flatten Layer: Flatten the 2D output to 1D for the Dense layer.

Dense Layer: 128 neurons with ReLU activation.

Output Layer: 10 neurons (for digits 0–9) with softmax activation.

Model Training
The model is trained on the MNIST dataset with 5 epochs and uses the following:

Optimizer: Adam

Loss Function: Sparse categorical crossentropy

Metrics: Accuracy

Running the Gradio App
To interact with the model:

Draw a Digit: Use the sketchpad to draw a digit (0-9).

Click "Predict": After drawing, click the "Predict" button to see the model's predicted digit.

File Structure
app.py: The main Python script that runs the model and Gradio interface.

model.py: Contains the model-building and training code.

requirements.txt: List of required Python packages.

README.md: Documentation for the project.

Example Outputs
When you draw a digit and click "Predict," the output will display something like:

yaml
Copy
Edit
Predicted Digit: 7
Visualizations
Prediction Results: The predicted digit is displayed in the Gradio interface.

Model Evaluation
After training, the model is evaluated on the MNIST test dataset to determine its accuracy. The test accuracy should be around 98.5%.

Preprocessing Details
The input image is preprocessed in the following steps:

Convert to Grayscale: If the image is in RGB format, it's converted to grayscale.

Invert Colors: The image is inverted (white digits on black background).

Resize: The image is resized to fit the input shape of the model (28x28 pixels).

Normalization: The pixel values are scaled to a range of 0–1 for better model performance.

Padding: If needed, the image is padded to ensure it's centered.

Contributing
Feel free to fork this repository, open issues, or submit pull requests. Contributions are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
MNIST Dataset: Thanks to Yann LeCun, Corinna Cortes, and CJ Burges for the MNIST dataset.

TensorFlow and Keras: Thanks to the TensorFlow team for providing excellent deep learning tools.

Gradio: Thanks to the Gradio team for making it easy to create machine learning demos.
