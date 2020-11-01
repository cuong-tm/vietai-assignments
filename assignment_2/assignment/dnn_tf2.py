"""dnn_tf2.py
Deep neural network implementation using tensorflow 2
Author: Thanh Le, Duy Le
"""
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dnn_np import DataLoader, Config, test
from util import *

np.random.seed(2020)
tf.random.set_seed(2020)


class NeuralNet(tf.Module):
    def __init__(self, input_dim, n_classes, list_hidden_dim: Set[int] = (128, 128)):
        super().__init__()
        # TODO: Create Input layer from tf.keras.layers.Input that accept data points with input_dim features
        input_layer = tf.keras.Input(shape=(input_dim,))
        hidden_layers = []  # Stack of fully connected layer, or Dense layer from tf.keras.layers.Dense
        for hidden_dim in list_hidden_dim:
            # TODO: Append a Dense layer that has hidden_dim units to hidden_layers, with activation='relu'
            layer=tf.keras.layers.Dense(hidden_dim, activation='relu')
            # You can also try adding activity regularizers
            hidden_layers.append(layer)
            pass

        # TODO: Create output layer that map hidden representation to a distribution of probability over classes
        # HINT: It should be a Dense layer with softmax activation
        output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')
        layers = [input_layer] + hidden_layers + [output_layer]
        self.layers = tf.keras.Sequential(layers)
        self.n_classes = n_classes

    def __call__(self, X):
        return self.layers(X)


# We will re-use this function from assignment 1
def train_one_step(
    model: tf.Module, 
    train_x: np.ndarray,
    train_y: np.ndarray,
    optimizer: tf.optimizers.Optimizer, 
    loss_fn: tf.losses.Loss):
    """Perform one step gradient update
    """
    with tf.GradientTape() as tape:
        predictions = model(train_x)
        train_loss = loss_fn(train_y, predictions)

    # NOTE: This time we use `model.trainabale_variables` to access 
    # all trainable parameters (at every layers)
    # Calculate gradients of loss w.r.t models' parameters
    gradients = tape.gradient(train_loss, model.trainable_variables) 
    # Then perform one step weight update using optimizers.apply_gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss.numpy()


def trainer(
    model: tf.Module, 
    train_set: DataLoader,
    config: Config, 
    optimizer: tf.optimizers.Optimizer, 
    loss_fn: tf.losses.Loss,
    validate_set: DataLoader=None):
    """Function that handles training a tensorflow 2 model
    """
    all_loss = []

    plt.ion()

    for epoch in range(config.num_epoch):
        epoch_train_losses = []
        for minibatch_x, minibatch_y in train_set.batch(config.batch_size):
            train_loss = train_one_step(model, minibatch_x, minibatch_y, optimizer, loss_fn)
            epoch_train_losses.append(train_loss)

        avg_epoch_train_loss = np.mean(epoch_train_losses)
        all_loss.append(avg_epoch_train_loss)
        print_msg = f"Epoch #{epoch + 1} Train loss: {avg_epoch_train_loss}"

        if validate_set is not None:
            all_val_loss = []
            for minibatch_x_val, minibatch_y_val in validate_set.batch(config.batch_size):
                val_loss = loss_fn(minibatch_y_val, model(minibatch_x_val))
                all_val_loss.append(val_loss.numpy())
            print_msg += f" Val loss {np.mean(all_val_loss)}"

        print(print_msg)

        if config.visualize and epoch % config.epochs_to_draw == config.epochs_to_draw - 1:
            s = model(train_set.inputs[0::3])[-1].numpy()
            s = s.reshape([1, s.shape[0]])
            visualize_point(train_set.inputs[0::3], train_set.labels[0::3], s)
            plot_loss(all_loss, 2)
            plt.show()
            plt.pause(0.01)

    if config.experiment_name is not None:
        # TODO: Serialize your trained model to file named {experiment_name}.h5
        model.save('{0}.h5'.format(config.experiment_name))
        # HINT: https://www.tensorflow.org/guide/keras/save_and_serialize
        pass
    return model


def bat_classification_tf2():
    train_x, train_y, test_x, test_y = get_bat_data()
    train_x, _, test_x = normalize(train_x, train_x, test_x)    

    test_y  = test_y.flatten()
    train_y = train_y.flatten()
    num_class = (np.unique(train_y)).shape[0]

    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)

    # Define hyper-parameters and train-related parameters
    cfg = Config(
        num_epoch=20,
        learning_rate=0.001,
        num_train=train_x.shape[0],
        experiment_name='bat_clf'
    )

    dataloader = DataLoader(train_x, train_y)
    print(train_x.shape,num_class)
    # TODO: Initalize a NeuralNet with hidden dims of [128, 64, 64]
    model = NeuralNet(input_dim=train_x.shape[1],n_classes=num_class,list_hidden_dim=[128, 64, 64])
    # Initialize suitable optimizer and loss_fn
    optimizer = tf.keras.optimizers.Adam()
    # HINT: Since we didn't convert label to one-hot, loss function should be SparseCategoricalCrossentropy
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    #Train model using trainer, and save it
    model = trainer(model, dataloader, cfg, optimizer, loss_fn)
    # TODO: Load and evaluate your trained model on test set
    _model = tf.keras.models.load_model("{0}.h5".format(cfg.experiment_name))
    results = _model.evaluate(test_x, test_y, batch_size=128)

def fashion_mnist_classification_tf2():
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data(1)
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    

    num_class = (np.unique(train_y)).shape[0]

    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x)
    val_x = add_one(val_x)
    test_x = add_one(test_x)

    # Define hyper-parameters and train-related parameters
    cfg = Config(
        num_epoch=30,
        learning_rate=0.001,
        batch_size=200,
        num_train=train_x.shape,
        visualize=False,
        experiment_name = 'mnist_clf'
    )

    train_loader = DataLoader(train_x, train_y)
    val_loader = DataLoader(val_x, val_y)
    print(1)
    # print(train_x.shape[1]*train_x.shape[2)
    # TODO: Initalize a NeuralNet with hidden dims of [256, 256, 64]
    model = NeuralNet(n_classes=num_class,input_dim=28*28, list_hidden_dim=[256, 256, 64])
    print(2)
    optimizer = tf.keras.optimizers.Adam()
    # HINT: Since we didn't convert label to one-hot, loss function should be SparseCategoricalCrossentropy
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # Train model using trainer, and save it
    model = trainer(model, train_loader, cfg, optimizer, loss_fn, validate_set=val_loader)
    print(3)
    # TODO: Load and evaluate your trained model on test set
    _model = tf.keras.models.load_model("{0}.h5".format(cfg.experiment_name))
    results = _model.evaluate(test_x, test_y, batch_size=128)

if __name__ == "__main__":
    # bat_classification_tf2()
    fashion_mnist_classification_tf2()
    pass