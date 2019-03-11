"""
Trains a simple NN on MNIST classification, using keras.
"""

from data.fashion_MNIST import *
from models.MNIST_basic_dense_model import *
from helper.hyperparameter import *
from helper.transformation import Transformation


def get_random_orthonormal_matrix(n):
    a = np.random.random(size=(n, n))
    q, r = np.linalg.qr(a)
    return q


def train_model():
    hp = Hyperparameter()

    trans = Transformation(10, hp.batch_size)

    data = load_data()
    data = prepare_data_for_keras(data)

    (x_train, y_train), (x_test, y_test) = trans.transform_training_data(data)
    
    hp.epochs = 30
    hp.optimizer = "adam"

    model = build_model()

    model.compile(loss=hp.loss,
                  optimizer=hp.optimizer,
                  metrics=[trans.accuracy_metric])

    model.fit(x_train, y_train,
              batch_size=hp.batch_size,
              epochs=hp.epochs,
              verbose=2,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    model.save_weights("saved_models/fashion_MNIST")

    return score


train_model()
