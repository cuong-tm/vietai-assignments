"""
This file is for binary classification using TensorFlow 2
Author: Kien Huynh, Thanh Le
"""
import tensorflow as tf
import numpy as np 
tf.random.set_seed(2018)
assert tf.__version__.startswith("2."), "Make sure that you installed Tensorflow 2.x"

from logistic_np import *
from util_tf import *


np.random.seed(2018)
tf.random.set_seed(2018)


class LogisticClassifierTF2(BaseTF2Model):
    def feed_forward(self, X):
        # [TODO 1.15]: Define feed forward for logistic classifier
        # It's similar with `feed_forward` implementation in numpy, 
        # but use tensorflow's operators
        # HINT: `tf.sigmoid` is one of many built-in operators in tensorflow
        
        predictions = tf.sigmoid(tf.linalg.matmul(X, self.w))
        # 1.0 / (1.0 + tf.exp(-tf.linalg.matmul(X, self.w)))
        print(predictions)
        return predictions


if __name__ == "__main__":
    # Define hyper-parameters and train-related parameters
    num_epoch = 1000
    learning_rate = 0.01
    # Some meta parameters
    epochs_to_draw = 100
    all_loss = []
    plt.ion()

    train_x, train_y, test_x, test_y = load_vehicle_data()
    w_shape = (train_x.shape[1], 1)
    model = LogisticClassifierTF2(w_shape)
    # [TODO 1.16] Init BinaryCrossentropy from tf.losses 
    # Because we got the classes'probabilities, `from_logits` need to set to False

    loss_fn = tf.losses.BinaryCrossentropy(from_logits=False)

    # [TODO 1.16] Create an SGD optimizer using `tf.optimizers`
    # HINT: you can try other tensorflow's optimizers and compares their convergent time
    # optimizer = tf.optimizers.Adam(0.001)
    optimizer = tf.keras.optimizers.SGD(0.001)
    # print(model)

    # Start training
    for e in range(num_epoch):
        train_loss = train_one_step(model, train_x, train_y, optimizer, loss_fn)
        all_loss.append(train_loss)
        if (e % epochs_to_draw == epochs_to_draw-1):
            plot_loss(all_loss)
            plt.show()
            plt.pause(0.1)     
            print("Epoch %d: loss is %.5f" % (e+1, train_loss))
        
    y_hat = model(test_x).numpy()
    test(y_hat, test_y)

