"""
Builds a simple convolutional nerual network for MNIST classification with keras.
"""

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_model():
    input_shape = (28, 28, 1) #channels last
    num_classes = 10

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu', input_shape=input_shape, name="dense_1"))
    model.add(Dense(64, activation='relu', name="dense_2"))
    model.add(Dense(32, activation='relu', name="dense_3"))
    model.add(Dense(num_classes, activation='softmax', name="output"))

    return model
