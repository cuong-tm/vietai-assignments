"""
This files contain the base structure when defining and training a Tensorflow 2 model
Author: Thanh Le
"""
import tensorflow as tf
import numpy as np
from typing import Tuple


class BaseTF2Model(tf.Module):
    """
    Base class for training Tensorflow model
    """
    def __init__(self, w_shape: Tuple[int]):
        super().__init__()
        # [TODO 1.11] Initialize `self.w` as tensorflow Variable
        # HINT: You can pass initialize values similar with numpy implementation
        self.w = tf.Variable(np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape) , dtype=tf.float64)
        # tf.random.normal(w_shape,0, np.sqrt(2./np.sum(w_shape)))

    def feed_forward(self, inputs):
        """
        For each type of model, there will be a different feed_forward implementation
        e.g: 
            - Softmax regression : inputs, logits (dot product), softmax
            - Logistic regression: inputs, logits, sigmoid
        """
        raise NotImplementedError

    def __call__(self, inputs):
        """
        model(inputs) will call this function
        """
        return self.feed_forward(inputs)


def train_one_step(
    model: BaseTF2Model, 
    train_x: np.ndarray,
    train_y: np.ndarray,
    optimizer: tf.optimizers.Optimizer, 
    loss_fn: tf.losses.Loss):
    """
    Perform one step gradient update
    Later this function can be reused when training SoftmaxRegressionTF2 or LogisticRegressionTF2
    """
    with tf.GradientTape() as tape:
        # [TODO 1.12] Calculate model predictions and loss inside tf.GradientTape context
        # HINT: Operations on trainable variables that executed inside tf.GradientTape context
        # will be recorded automatically for autograd
        # print(train_x.shape)
        # print('model',model)
        predictions = model(train_x)
        # print(predictions.shape)
        # print(train_y.shape)
        train_loss = loss_fn(train_y,predictions)
            
    # [TODO 1.13] Compute gradient of loss w.r.t model's parameters
    # HINT : since BaseTF2Classifier extends tf.Module, you can access all
    #         model's parameters using `model.trainable_variables`
    gradients = tape.gradient(train_loss,model.trainable_variables)
    # [TODO 1.14] Perform one step weight update using optimizers.apply_gradients
    # HINT: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    # `train_loss`, `predictions` currently is a EegerTensor,
    #  you can access its numpy data by using
    return train_loss.numpy()
