from abc import ABC, abstractmethod
import numpy as np
from .layer import Layer

class Loss(ABC):
	""" Base class for all loss functions """
	dinputs: np.ndarray

	@abstractmethod
	def forward(self, y_pred, y_true):
		pass
	
	def backward(self, dvalues, y_true):
		pass

	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses) # calcul la moyenne de la loss
		return data_loss
	
	def regularization_loss(self, layer: Layer):
		loss = 0

		if (layer.weight_regularizer_l1):
			loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
		if (layer.bias_regularizer_l1):
			loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
		if (layer.weight_regularizer_l1):
			loss += layer.weight_regularizer_l1 * np.sum(np.square(layer.weights))
		if (layer.bias_regularizer_l2):
			loss += layer.bias_regularizer_l2 * np.sum(np.square(layer.biases))

class CategoricalCrossEntropy(Loss):
	""" Categorical Crossentropy loss 
		- Forward pass: -sum(y_true * log(y_pred))
		- Backward pass: y_pred - y_true

		Note: y_true is a one-hot encoded vector

		Example:

		y_pred = [[0.1, 0.7, 0.2], [0.9, 0.1, 0.0]]
		y_true = [[0, 1, 0], [1, 0, 0]]

		loss = -sum(y_true * log(y_pred))
		loss = -sum([[0, 1, 0] * log([0.1, 0.7, 0.2]), [1, 0, 0] * log([0.9, 0.1, 0.0])])
		loss = -sum([0, 0.35667494, 0] + [0.10536052, 0, 0])
		loss = -sum([0.10536052, 0.35667494])
		loss = -0.46203546

		Note: The loss is averaged over all samples
	"""
	def forward(self, y_pred, y_true: np.ndarray):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1: # quand on reçoit un vecteur de label : [0, 2, 0, 1] -> la classe 0, 2, 0, 1 sont vrai.
			# donc on cherche directement a la case des predictions les valeurs
			correct_confidences = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2: # One hot encoding, on reçoit une matrice : [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
			# donc on a juste a multiplier chaque vecteur par la reponse : [1, 0, 0] * 0 = 0, [0, 0, 1] * 2 = 2.... 
			#(meme resultat que pour la ligne au dessus)
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
		
		negative_log_likelihoods = -np.log(correct_confidences)
		# ici on met au -logarithme (naturel !! base E) les resultat. Ca permet de pouvoir revenir au resultat en 
		# mettant en exponentiel le logarithme. (pratique pour la backpropagation et l'optimisation)
		return negative_log_likelihoods
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		labels = len(dvalues[0])

		if (len(y_true.shape) == 1):
			y_true = np.eye(labels)[y_true]

		self.dinputs = -y_true / dvalues
		self.dinputs = self.dinputs / samples
	
class BinaryCrossEntropy(Loss):
	""" Binary Crossentropy loss 
		- Forward pass: -sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
		- Backward pass: -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / outputs

		Note: y_true is a binary vector

		It quantifies the dissimilarity between probability distributions, 
		aiding model training by penalizing inaccurate predictions.

		log loss is a measure of error, where the likelihood of the true class is evaluated.
	"""
	def forward(self, y_pred, y_true):
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis=-1)

		return sample_losses
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		outputs = len(dvalues[0])
		clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
		self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
		self.dinputs = self.dinputs / samples

def loss(loss:str) -> Loss:
	""" Loss function factory """
	loss = loss.lower()
	if (loss == 'categoricalcrossentropy'):
		return CategoricalCrossEntropy()
	elif (loss == 'binarycrossentropy'):
		return BinaryCrossEntropy()
	raise ValueError(f"LossError: Unknown loss type :'{loss}'")

__all__ = ['Loss', 'CategoricalCrossEntropy', 'BinaryCrossEntropy', 'loss']