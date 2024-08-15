# Multilayer-Perceptron

The project focused on implementing a Multilayer Perceptron (MLP), a type of artificial neural network, from scratch. The goal is to classify breast cancer diagnoses based on a dataset of breast mass characteristics, distinguishing between malignant and benign cases.

## MLP Architecture

### Optimization Algorithms

- **Gradient Descent**: An optimization algorithm used to minimize the cost function.

- **SGD (Stochastic Gradient Descent)**: A variant of gradient descent that updates model parameters using a single or a few training examples at each iteration.

- **Adam Optimizer**: An optimization algorithm that combines the advantages of both AdaGrad and RMSProp.

- **AdaGrad** adjusts the learning rate based on historical gradients, tracks the sum of squared gradients, and may suffer from a diminishing learning rate over time.

- **RMSProp** adjusts the learning rate using a moving average, employs an exponentially decaying average of squared gradients, and prevents the learning rate from becoming too small.

### Neural Network Processes

- **Feedforward**: The process where input data passes through the network layers to produce an output.

- **Backpropagation**: A supervised learning algorithm used for training neural networks by calculating the gradient of the loss function.

### Activation Functions

- **ReLU (Rectified Linear Unit)**: Outputs the input directly if it is positive; otherwise, it outputs zero.

- **Sigmoid**: Maps input values to a range between 0 and 1.

- **Softmax**: Converts raw output scores into probabilities that sum to one.

- **Tanh (Hyperbolic Tangent)**: maps input values to a range between -1 and 1, is symmetric around the origin, and introduces non-linearity into the model, allowing it to learn complex patterns.

- **Common Activation Functions**: ReLU, Sigmoid, and Tanh.

### Neural Network Components

- **Dense**: A fully connected layer where each neuron is connected to every neuron in the previous layer.

### Evaluation Metrics

- **Loss**: A measure of how well the neural network's predictions match the actual target values.

- **Accuracy**: A metric used to evaluate the performance of a classification model.

### Summary

These concepts are fundamental to understanding and working with neural networks and machine learning models.