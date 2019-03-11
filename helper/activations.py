import numpy as np

sigmoid_activation = lambda x: 1.0 / (1 + np.exp(-x))
sigmoid_derivation = lambda x: x * (1 - x)

tanh_activation = lambda x: np.tanh(x)
tanh_derivation = lambda x: 1 - x*x
