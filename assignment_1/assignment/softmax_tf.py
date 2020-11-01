"""
This file is for multiclass classification using TensorFlow 2
Author: Kien Huynh, Thanh Le
"""

import matplotlib.pyplot as plt
from softmax_np import load_and_process_fmnist_data, plot_loss, draw_weight, test
from util_tf import *
import tensorflow as tf 
import numpy as np


np.random.seed(2018)
tf.random.set_seed(2018)

def stop_train(all_val_loss):
    stop = False
    _num = 0
    if (len(all_val_loss)>100):
        for i in range(len(all_val_loss)-1,100,-1):
            if all_val_loss[i] > all_val_loss[i-1]:
                _num += 1
            if all_val_loss[i] < all_val_loss[i-1]:
                _num = 0
            if _num >= 10:
                return True
    return stop


class SoftmaxRegressionTF2(BaseTF2Model):
    def feed_forward(self, inputs):
        # [TODO 2.9]: Define forward for Softmax Regression here
        # This function will return predictions, which are classes'probabilities
        predictions = tf.nn.softmax(tf.linalg.matmul(inputs, self.w))
        # print(predictions)
        return predictions


if __name__ == "__main__":
    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    epochs_to_draw = 10
    learning_rate = 0.01

    # Load data & initialize model
    train_x, train_y, val_x, val_y, test_x, test_y = load_and_process_fmnist_data()
    w_shape = (train_x.shape[1],10)
    model = SoftmaxRegressionTF2(w_shape)

    # [TODO 2.9] Create an SGD optimizer and CategoricalCrossentropy loss
    optimizer = tf.optimizers.Adam(0.001)
    # optimizer =tf.keras.optimizers.SGD(0.01)

    loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
    
    # start training 
    all_train_loss = []
    all_val_loss = []
    plt.ion()

    for e in range(num_epoch):
        train_loss = train_one_step(model, train_x, train_y, optimizer, loss_fn)
        all_train_loss.append(train_loss)

        val_loss = loss_fn(val_y, model(val_x))
        all_val_loss.append(val_loss.numpy())
        
        if (e % epochs_to_draw == epochs_to_draw-1):
            plot_loss(all_train_loss, all_val_loss)
            draw_weight(model.w.numpy())
            plt.show()
            plt.pause(0.01)
            print("Epoch %d: train loss: %.5f || val loss: %.5f" % (e+1, train_loss, val_loss))
        
        # [TODO 2.10] Define your own stopping condition here
        if stop_train(all_val_loss):
            break
        # if len(all_val_loss) >100:
        #     _num=0
        #     for i in range(len(all_val_loss)-1,100,-1):
        #         if all_val_loss[i] > all_val_loss[i-1]:
        #             _num +=1
        #         if _num >=10:
        #             break
        
    print('all_val_loss',all_val_loss)
    
    y_hat = model(test_x)
    test(y_hat, test_y)
