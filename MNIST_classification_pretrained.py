"""
Trains a simple NN on MNIST classification, using keras. Uses a model pretrained with
a quadruplet cross-digit encoder.
"""

from data.MNIST import *
from models.MNIST_generative_conv_similarity import *
from helper.hyperparameter import *


def train_model(pretrained_model_path, embedding_dimensions):
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_keras(data)

    hp = Hyperparameter()
    hp.optimizer = "adam"
    hp.epochs = 4

    model = build_model(embedding_dimensions, pretrained_model_path)

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


train_model("C:/Users/tobia/Documents/Programmieren/AI/Few-Shot-Learning/saved_models/MNIST Generative Mine", 400)
