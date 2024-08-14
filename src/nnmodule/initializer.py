import numpy as np

class Initializer:
	def __call__(self, shape: tuple) -> np.ndarray:
		pass

class Zero(Initializer):
	""" Zero initializer """
	def __init__(self):
		pass

	def __call__(self, shape: tuple) -> np.ndarray:
		return np.zeros(shape)

class HeUniform(Initializer):
	""" He Uniform initializer 
		He initialization is designed to keep 
		the scale of the gradients roughly the same in all layers.
	"""
	def __init__(self, n_inputs=2, **kwargs):
		self.n_inputs = n_inputs

	def __call__(self, shape: tuple) -> np.ndarray:
		std = np.sqrt(6.0 / self.n_inputs)
		return np.random.uniform(-std, std, shape)

def initializer(initializer_type: str, *args, **kwargs) -> Initializer:
	initializer_type = initializer_type.lower()
	if args and isinstance(args[-1], dict):
		kwargs.update(args[-1])
		args = args[:-1]

	if (initializer_type == "randomnormal"):
		return RandomNormal(*args, **kwargs)
	if (initializer_type == "zero"):
		return Zero()
	if (initializer_type == "henormal"):
		return HeNormal(*args, **kwargs)
	if (initializer_type == "heuniform"):
		return HeUniform(*args, **kwargs)
	if (initializer_type == "xavier"):
		return Xavier(*args, **kwargs)
	if (initializer_type == "lecun"):
		return LeCun(*args, **kwargs)
	raise ValueError(f"InitializerError: Unknown initializer type :'{initializer_type}'")

__all__ = ['Initializer', 'Zero', "HeUniform", 'initializer']