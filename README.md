# Multilayer-Perceptron

The project focused on implementing a Multilayer Perceptron (MLP), a type of artificial neural network, from scratch. The goal is to classify breast cancer diagnoses based on a dataset of breast mass characteristics, distinguishing between malignant and benign cases.

## Project Overview

1. Introduction to Multilayer Perceptron (MLP)

Multilayer Perceptron: A feedforward neural network with one or more hidden layers between the input and output layers. Each neuron in a layer is connected to every neuron in the next layer. This type of network is particularly useful for binary classification tasks, such as predicting whether a tumor is malignant or benign.
Perceptron: The basic unit of the MLP, which consists of inputs, weights, a bias, an activation function, and an output. The perceptron computes a weighted sum of its inputs, adds a bias, and then applies an activation function to determine the output.

2. Objectives

Implementation: You are expected to implement the MLP from scratch, without using any libraries that handle neural networks. The project emphasizes understanding the algorithms involved in the training process, including feedforward propagation, backpropagation, and gradient descent.
Mathematical Foundations: Familiarity with linear algebra and derivatives is crucial, as these are used extensively in the training algorithms.

3. Dataset

The dataset provided is a CSV file with 32 columns, where the target label is the diagnosis (either 'M' for malignant or 'B' for benign). The other columns represent various features of the cell nuclei.
Data Preprocessing: The data is raw and will need to be cleaned and preprocessed before training. You’ll need to split the dataset into training and validation sets, and possibly normalize or standardize the features.

4. Implementation Requirements

Your MLP should contain at least two hidden layers.
You must implement the softmax function for the output layer to obtain a probabilistic distribution, which is crucial for binary classification.
The implementation should be modular, allowing for flexibility in the number of layers, activation functions, and other parameters.
During training, you’ll need to visualize the learning process by plotting learning curves for both the loss and accuracy over epochs.
You are required to submit three programs:
Data Splitting Program: To split the dataset into training and validation sets.
Training Program: To train the MLP using backpropagation and gradient descent, saving the model at the end.
Prediction Program: To load the trained model, perform predictions on a given dataset, and evaluate the performance using binary cross-entropy.

5. Bonus Part

If the mandatory part is completed perfectly, you can implement additional features like:
Advanced optimization techniques (e.g., Adam, RMSprop).
Multiple learning curves on the same graph.
Early stopping to prevent overfitting.
Evaluating the model with different metrics.

## Steps to Solve the Project


- Understand the Dataset:

Start by loading and inspecting the dataset.
Perform exploratory data analysis (EDA) to understand the distribution of features and identify any anomalies.

- Preprocess the Data:

Clean the data by handling missing values, if any.
Normalize or standardize the features to ensure the training process is stable.

- Design the MLP Architecture:

Define the network structure with the required number of layers and neurons.
Choose appropriate activation functions (e.g., ReLU, Sigmoid).
Implement the forward pass, where you compute the output of the network given an input.

- Implement Backpropagation and Gradient Descent:

Compute the gradients of the loss function with respect to the network’s parameters.
Update the parameters using gradient descent to minimize the loss function.

- Training and Validation:

Train the network on the training set and validate it on the validation set.
Track the training progress by plotting the loss and accuracy over epochs.

- Evaluate and Fine-tune:

Evaluate the model’s performance on the validation set.
If necessary, adjust the network architecture or hyperparameters and retrain.

- Implement the Bonus Features (optional):

Add more sophisticated optimization techniques, learning curve visualizations, or early stopping mechanisms.

- Submit the Project:

Ensure that your code is well-organized and meets the submission requirements.
Include clear documentation and explanations, especially for the training phase and the algorithms used.
This structured approach should help you successfully complete the project. If you need more detailed guidance on any specific part, feel free to ask!