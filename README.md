# digit-recognition-neural-network
Handwritten digit recognition using a neural network built with PyTorch.
Handwritten Digit Recognition using Neural Networks

This project implements a fully connected neural network to classify handwritten digits (0–9) using the MNIST dataset.

The model is implemented using PyTorch and demonstrates the full deep learning pipeline including:

- Data preprocessing
- Model architecture design
- Training and validation
- Performance evaluation

Architecture

Input Layer: 784 (flattened 28×28 image)  
Hidden Layer 1: 512 neurons (ReLU)  
Hidden Layer 2: 256 neurons (ReLU)  
Output Layer: 10 neurons (digit classes)

Technologies

Python  
PyTorch  
NumPy  
Matplotlib  
Scikit-learn

Training

Loss Function: CrossEntropyLoss  
Optimizer: Stochastic Gradient Descent (SGD)  
Batch Size: 128  
Epochs: 15

Dataset

MNIST handwritten digit dataset
70,000 grayscale images (28×28)

Results

The model achieves high classification accuracy on the test dataset after training.

Running the Project
