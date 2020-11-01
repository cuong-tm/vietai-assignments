"""
This file provides functionalities for unit testing
Author: Le Duy
"""
import argparse
import sys

import joblib
import numpy as np
from activation_np import *
from util import *

seed = 2020

def testcase_check(your_arr, test_arr, testname, print_all, print_ind=None):
    eps = 0.00001
    print(np.sum((your_arr-test_arr)**2))
    if (type(your_arr) != type(test_arr)):
        print("Testing %s: Failed. Your arr should be %s but it is %s instead." % (testname, type(test_arr), type(your_arr)))
        return False
    
    if (your_arr.shape != test_arr.shape):
        print("Testing %s: Failed. Your arr should have a shape of %s but its shape is %s instead." % (testname, test_arr.shape, your_arr.shape))
        return False

    if (np.sum((your_arr-test_arr)**2) < eps):
        print("Testing %s: Passed." % testname)
    else:
        print("Testing %s: Failed." % testname)
        if (print_all): 
            print("Your array is")
            print(your_arr)
            print("\nWhile it should be")
            print(test_arr)
        else:
            print("The first few rows of your array are")
            print(your_arr[print_ind, 0])
            print("\nWhile they should be")
            print(test_arr[print_ind, 0])
        return False
    print("----------------------------------------")
    return True


def activation_unit_test():
    """
    activation_unit_test
    Test all functions in the activation np assignment
    """
    np.random.seed(seed)
    testcase = joblib.load('./data/activation_unittest.joblib')
    matrix = np.random.rand(5, 5)
    vector = np.arange(-5,5,0.5)[np.newaxis]


    sigmoid_arr = np.array([sigmoid(x) for x in vector[0]])
    assert testcase_check(
        sigmoid_arr,
        testcase['sigmoid'],
        "sigmoid function",
        True
    ), "Fail test case"

    sigmoid_grad_arr = np.array([sigmoid_grad(x) for x in vector[0]])
    assert testcase_check(
        sigmoid_grad_arr,
        testcase['sigmoid_grad'],
        "sigmoid_grad function",
        True
    ), "Fail test case"

    reLU_arr = np.array([reLU(x) for x in vector[0]])
    assert testcase_check(
        reLU_arr,
        testcase['reLU'],
        "reLU function",
        True
    ), "Fail test case"

    reLU_grad_arr = np.array([reLU_grad(x) for x in vector[0]])
    assert testcase_check(
        reLU_grad_arr,
        testcase['reLU_grad'],
        "reLU_grad function",
        True
    ), "Fail test case"

    tanh_arr = np.array([tanh(x) for x in vector[0]])
    assert testcase_check(
        tanh_arr,
        testcase['tanh'],
        "tanh function",
        True
    ), "Fail test case"

    tanh_grad_arr = np.array([tanh_grad(x) for x in vector[0]])
    assert testcase_check(
        tanh_grad_arr,
        testcase['tanh_grad'],
        "tanh_grad function",
        True
    ), "Fail test case"

    softmax_arr_vector = softmax(vector)
    assert testcase_check(
        softmax_arr_vector,
        testcase['softmax_vector'],
        "softmax function with vector",
        True
    ), "Fail test case"

    softmax_arr_matrix = softmax(matrix)
    assert testcase_check(
        softmax_arr_matrix,
        testcase['softmax_matrix'],
        "softmax function with matrix",
        True
    ), "Fail test case"

    softmax_minus_max_arr_vector = softmax_minus_max(vector)
    assert testcase_check(
        softmax_minus_max_arr_vector,
        testcase['softmax_minus_max_vector'],
        "softmax_minus_max function with vector",
        True
    ), "Fail test case"

    softmax_minus_max_arr_matrix = softmax_minus_max(matrix)
    assert testcase_check(
        softmax_minus_max_arr_matrix,
        testcase['softmax_minus_max_matrix'],
        "softmax_minus_max function with matrix",
        True
    ), "Fail test case"


def dnn_np_unit_test(todo: float, test_all: bool = False):
    test_case = joblib.load('data/dnn_np_unittest.joblib')

    if todo == 1.2 or test_all:
        np.random.seed(seed)
        print('Running forward and backward Layer unit test ...')
        activation_list = ['sigmoid', 'relu', 'tanh', 'softmax']
        for activation in activation_list:
            print(f"Testing {activation}:",end=' ')
            layer = Layer((60, 100), activation)
            if activation == 'softmax':
                assert unit_test_layer(layer) < 1e-2, "Fail test case"
            else:
                assert unit_test_layer(layer) < 1e-4, "Fail test case"
            print('-'*96)

    if todo == 1.3 or test_all:
        np.random.seed(seed)
        print('Running compute_loss unit test ...')
        y = np.random.rand(2,5)
        nn = NeuralNet()
        loss = nn.compute_loss(y[1],y[0])
        assert testcase_check(
            loss,
            test_case['compute_loss'],
            "compute_loss function",
            True
        ), "Fail test case"

    if todo == 1.4 or test_all:
        np.random.seed(seed)
        print('Running regularization unit test ...')
        activation_list = ['sigmoid', 'relu', 'tanh', 'softmax']
        # for activation in activation_list:
        for i in range(len(activation_list)):
            print(f"Testing {activation_list[i]}:",end=' ')
            layer = Layer((3, 5), activation_list[i])
            x = np.random.rand(3,5)
            z = layer.forward(x.T)
            assert testcase_check(
                z,
                test_case['regularization'][i],
                "regularization function with " + activation_list[i],
                True
            ), "Fail test case"

    if todo == 1.5 or test_all:
        np.random.seed(seed)
        print('Running backpropagation unit test ...')
        x_set = np.random.rand(10,5)
        y_set =  np.random.randint(2, size=10)
        y_set = create_one_hot(y_set, 2)

        net = NeuralNet()
        net.add_linear_layer((x_set.shape[1], 4), 'relu')
        net.add_linear_layer((4, 2), 'softmax')

        # one epoch
        all_x = net.forward(x_set)
        grads = net.backward(y_set, all_x)
        for i in range(len(grads)):
            assert testcase_check(
                grads[i],
                test_case['backpropagation'][i],
                "backpropagation layer" + str(i+1),
                True
            ), "Fail test case"

    if todo == 1.6 or test_all:
        np.random.seed(seed)
        print('Running minibatch_train unit test ...')
        x_set = np.random.rand(100,5)
        y_set =  np.random.randint(2, size=100)
        # Define hyper-parameters and train-related parameters
        cfg = Config(num_epoch=9, learning_rate=0.01, num_train=x_set.shape[0])

        net = NeuralNet()
        net.add_linear_layer((x_set.shape[1], 4), 'relu')
        net.add_linear_layer((4, 2), 'softmax')

        minibatch_train(net, x_set, y_set, cfg)

        s = net.forward(x_set)[-1]

        assert testcase_check(
            s,
            test_case['minibatch_train'],
            "minibatch_train",
            True
        ), "Fail test case"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform unitest on assignment')
    parser.add_argument('choice', nargs='?', type=float, help='TODO need check', default=-1)
    parser.add_argument('sol', nargs='?', type=str, help='', default='')
    parser.add_argument('all', nargs='?', type=str, help='', default='')
    args = parser.parse_args()
    choice = args.choice
    sol = args.sol
    all = args.all

    np.random.seed(seed)

    if (choice == -1):
        choice = input('Please enter todo session for unit test todo')
        if(sys.version_info[0] == 3):
            choice = float(choice)

    if len(all) > 0:
        all = True
    else:
        all = False

    np.set_printoptions(precision=3, edgeitems=2)
    if choice == 1.1:
        if sol.lower().startswith('sol'):
            from activation_np_sol import *
        else:
            from activation_np import *

        print("Running activation unit test ...")
        activation_unit_test()
    elif (choice >= 1.2) and (choice <= 1.6):
        if sol.lower().startswith('sol'):
            from dnn_np_sol import *
        else:
            from dnn_np import *
        dnn_np_unit_test(choice, all)
