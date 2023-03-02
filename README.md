# Handwriting-Detection-using-Deep-Learing-with-Neural-Network
Handwriting Detection using Deep Learing with Neural Network, tensorflow, keras and jupyter notebook



This project is a demonstration of using deep learning techniques to detect handwriting in images, using the Keras and TensorFlow libraries in Python. The goal is to create a neural network model that can accurately classify images of handwritten digits.

## Prerequisites
Before running this project, you will need the following software installed:

* Python 3
* Keras
* TensorFlow
* NumPy
* Matplotlib
* Jupyter Notebook

You can install these packages using pip, the Python package manager:
```
pip install keras tensorflow numpy matplotlib jupyter
```

## Dataset
The dataset used in this project is the MNIST dataset, which consists of 70,000 images of handwritten digits, with 60,000 images in the training set and 10,000 images in the test set. Each image is 28x28 pixels in size.

You can download the dataset from the official [MNIST](https://yann.lecun.com/exdb/mnist/) website, or you can use the following code to download and load the dataset into your Python environment:

```
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

```

## Model Architecture
The neural network model used in this project consists of a sequence of layers:

 * Input layer: This layer receives the input image, which is a 28x28 pixel grayscale image.
 * Flatten layer: This layer converts the 2D image data into a 1D array, which can be processed by the following layers.
 * Dense layer: This is a fully connected layer with 128 units and ReLU activation.
 * Output layer: This layer has 10 units, one for each possible digit (0-9), and uses softmax activation to output a probability distribution over the possible digits.
The model is trained using the Adam optimizer and categorical cross-entropy loss. The training process is run for 5 epochs with a batch size of 32.

## Results
After training the model on the MNIST dataset, we achieved an accuracy of 98.11% on the test set. This means that the model correctly classified 9,811 out of 10,000 test images.

## Conclusion
In this project, we demonstrated how to use deep learning techniques to detect handwriting in images using Keras and TensorFlow. By training a neural network model on the MNIST dataset, we were able to achieve a high level of accuracy in detecting handwritten digits. This technology has many practical applications, such as recognizing handwritten text in documents or verifying signatures on legal documents.
<hr>

## Contributor : [Ankit Nainwal](https://github.com/nano-bot01)


### Other Models 
 * [Fake News Prediction System](https://github.com/nano-bot01/Fake-News-Prediction-System-)
 * [Heart Disease Prediction System based on Logistic Regression](https://github.com/nano-bot01/Heart-Disease-Prediction-System-using-Logistic-Regression)

## Please ⭐⭐⭐⭐⭐ 

