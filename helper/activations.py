import numpy as np

sigmoid_activation = lambda x: 1.0 / (1 + np.exp(-x))
sigmoid_derivation = lambda x: x * (1 - x)