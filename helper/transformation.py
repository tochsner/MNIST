import keras.backend as K
import tensorflow as tf
import numpy as np

class Transformation:
    def __init__(self, n, batch_size):
        self.A = self.get_random_orthonormal_matrix(n)
        self.A_t = self.get_random_orthonormal_matrix(n).T

        self.A_t_batch = tf.convert_to_tensor(np.repeat(self.A_t[np.newaxis,:,:], batch_size, axis=0), dtype=tf.float32)

    def get_random_orthonormal_matrix(self, n):
        a = np.random.random(size=(n, n))
        q, r = np.linalg.qr(a)
        return q

    def transform_training_data(self, data):
        (x_train, y_train), (x_test, y_test) = data

        for i in range(y_train.shape[0]):
            y_train[i] = self.A.dot(y_train[i])

        for i in range(y_test.shape[0]):
            y_test[i] = self.A.dot(y_test[i])

        return (x_train, y_train), (x_test, y_test)        

    def accuracy_metric(self, y_true, y_pred):        
       
        return K.cast(K.equal(K.argmax(K.batch_dot(self.A_t_batch, y_pred), axis=-1),
                        K.argmax(y_true, axis=-1)),
                        K.floatx())

     