from abc import ABC, abstractmethod
from .loss import CategoricalCrossEntropy
import numpy as np

class Activation(ABC):
	inputs: np.ndarray
	output: np.ndarray
	dinputs: np.ndarray

	@abstractmethod
	def forward(self, inputs):
		pass

	@abstractmethod
	def backward(self, dvalues):
		pass

class ReLU(Activation):
	""" Rectified Linear Unit activation function  
		- Forward pass: output = max(0, input)
		- Backward pass: dinputs = dvalues.copy() if input > 0 else
	"""
	def forward(self, inputs):
		self.inputs = inputs
		self.output = np.maximum(0, inputs)
	def backward(self, dvalues):
		self.dinputs = dvalues.copy()
		self.dinputs[self.inputs <= 0] = 0

class Sigmoid(Activation):
	"""
	 	 Sigmoid activation function
		- Forward pass: output = 1 / (1 + exp(-inputs))
		- Backward pass: dinputs = dvalues * (1 - output) * output
	"""
	def forward(self, inputs):
		self.inputs = inputs
		self.output = 1/(1 + np.exp(-inputs))
	def backward(self, dvalues):
		self.dinputs = dvalues * (1 - self.output) * self.output
	
class Softmax(Activation):
	"""
		Softmax activation function
		- Forward pass: output = exp(inputs - max(inputs)) / sum(exp(inputs - max(inputs)))
		- Backward pass: dinputs = dvalues.copy()
	"""
	def forward(self, inputs):
		# we take all the inputs (matrix of vectors e.g. [[1, 2, 3], [2, 1, 3], [3, 1, 2]])
		# for each vector we subtract the maximum value ex: [[1 - 3, 2 - 3, 3 - 3], [2 - 3, 1 - 3, 3 - 3], [3 - 3, 1 - 3, 2 - 3]]
		# then pass the result in exponential. So no negative value and exponential
			
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		# Problem: The exponential grows much too fast, so we normalize the values between 0 - 1. Dividing that by the sum of all the values
		# exponential.
		self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

	def backward(self, dvalues):
		self.dinputs

class Softmax_CategoricalCrossEntropy(Activation):
	"""
		Softmax activation function
		- Forward pass: output = exp(inputs - max(inputs)) / sum(exp(inputs - max(inputs)))
		- Backward pass: dinputs = dvalues.copy()
	"""
	def __init__(self):
		self.activation = Softmax()
		self.loss = CategoricalCrossEntropy()
	def forward(self, inputs, y_true):
		self.activation.forward(inputs)
		self.output = self.activation.output
		print( self.loss.calculate(self.output, y_true))
		return self.loss.calculate(self.output, y_true)

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)
		self.dinputs = dvalues.copy()
		self.dinputs[range(samples), y_true] -= 1
		self.dinputs = self.dinputs / samples

def activation(activation:str)->Activation:
	""" Activation function factory"""
	activation = activation.lower()
	if (activation == 'relu'):
		return ReLU()
	elif (activation == 'sigmoid'):
		return Sigmoid()
	elif activation == "softmax":
		return Softmax()
	elif activation == "softmax_categoricalcrossentropy":
		return Softmax_CategoricalCrossEntropy()
	raise ValueError(f"ActivationError: Unknown activation type :'{activation}'")

__all__ = ['Activation', 'ReLU', 'Sigmoid', 'Softmax', 'Softmax_CategoricalCrossEntropy', 'activation']