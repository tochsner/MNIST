"""
Trains a simple NN on MNIST classification, using keras.
"""

from data.MNIST import *
from models.MNIST_basic_conv_model import *
from helper.hyperparameter import *


def train_model():
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_keras(data)

    hp = Hyperparameter()
    hp.epochs = 10
    #hp.optimizer = "adam"

    model = build_model()

    model.compile(loss=hp.loss,
                  optimizer=hp.optimizer,
                  metrics=hp.metrics)

    model.fit(x_train, y_train,
              batch_size=hp.batch_size,
              epochs=hp.epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = Model.evaluate(x_test, y_test, verbose=0)

    model.save_weights("saved_models/MNIST")

    return score


train_model()
