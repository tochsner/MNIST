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
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name="conv_1"))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32 , activation='relu', name="dense"))
    model.add(Dense(num_classes, activation='softmax', name="output"))

    return model
