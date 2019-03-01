"""
Builds a simple convolutional nerual network for MNIST classification with keras.
"""

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate

def build_model():
    input_shape = (816, 1) #channels last
    num_classes = 10

    input_layer = Input(input_shape)
    flatten_layer = Flatten()(input_layer)
    dense_layer = Dense(128, activation='relu')(flatten_layer)
    dense_layer = Dense(64, activation='relu')(dense_layer)
    dense_layer = Dense(32, activation='relu')(dense_layer)
    pretrained_output_layer = Dense(num_classes, activation='relu', trainable=False, name="output")(dense_layer)
    output_layer = Concatenate()([pretrained_output_layer, dense_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def build_full_model():
    input_shape = (784, 1) #channels last
    num_classes = 10

    input_layer = Input(input_shape)
    flatten_layer = Flatten()(input_layer)
    dense_layer = Dense(128, activation='relu', name="dense_1")(flatten_layer)
    dense_layer = Dense(64, activation='relu', name="dense_2")(dense_layer)
    dense_layer = Dense(32, activation='relu', name="dense_3")(dense_layer)
    pretrained_output_layer = Dense(num_classes, activation='relu', trainable=False, name="output")(dense_layer)
    output_layer = Concatenate()([pretrained_output_layer, dense_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
