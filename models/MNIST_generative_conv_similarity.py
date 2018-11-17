"""
Builds a simple convolutional neural network for MNIST classification, pretrained
with a quadruplet cross-digit encoder.
"""

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Concatenate, Average


def build_model(embedding_dimensions, pretrained_model_path):
    input_shape = (28, 28, 1)  # channels last
    num_classes = 10

    input_layer = Input(shape=input_shape)
    conv = Conv2D(64, (3, 3), activation="relu", trainable=False)(input_layer)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2, 2))(conv)
    conv = Conv2D(64, (3, 3), activation="relu", trainable=False)(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2, 2))(conv)
    conv = Conv2D(64, (3, 3), activation="relu", trainable=False)(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2, 2))(conv)
    dense = Flatten()(conv)
    dense = Dense(400, activation='relu', trainable=False )(dense)

    encoder_output_layer = Dense(embedding_dimensions, activation='sigmoid', trainable=False)(dense)
    
    decoder_dense = Dense(400, activation='relu', trainable=False)(encoder_output_layer)
    decoder_dense = Dropout(0.25)(decoder_dense)
    decoder_dense = Dense(600, activation='relu', trainable=False)(decoder_dense)
    decoder_output_layer = Dense(784, activation='sigmoid', trainable=False)(decoder_dense)

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    classification_output_layer = Dense(num_classes, activation="sigmoid")(encoder_output_layer)

    pretrained_model = Model(inputs=input_layer, outputs=output_layer)
    pretrained_model.load_weights(pretrained_model_path)

    model = Model(inputs=input_layer, outputs=classification_output_layer)

    return model
