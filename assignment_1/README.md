[![vietAi](img/vietAi.jpg)](http://vietai.org/)
----------------------
Assignment 1
----------------------

Welcome to the first Assignment of VietAi foundation course.

In this Assignment, we will try to create a simple machine learning pipeline with
**Logistic Regression** & **Softmax Regression**

`Deadline` : 23:00:00:24:06:2020

## Tutorial and Hint

We can get the tutorial of this assignment at [Tutorial doc](assignment-1/doc/Assignment_1Logistic_Regression_Softmax_Regression.pdf)
. And we need done all TODO had show in doc to complete this assignment.

## Set up tool

The core library add-in file [Requirements](requirements.txt). To install run:

```bash
    python -m pip install --upgrade pip && pip3 install -r requirements.txt
```

About data, we need download [data](https://storage.googleapis.com/vietai/assignment1-data.zip)
 flow link or use script:

```bash
    wget https://storage.googleapis.com/vietai/assignment1-data.zip
```

Then extract and add file to directory like as:

    data
    ├── fashion-mnist
    │   ├── t10k-images-idx3-ubyte.gz
    │   ├── t10k-labels-idx1-ubyte.gz
    │   ├── train-images-idx3-ubyte.gz
    │   └── train-labels-idx1-ubyte.gz
    ├── logistic_unittest.npy
    ├── read_the_doc
    ├── softmax_unittest.npy
    └── vehicles.dat

## Evaluate assignment

We can self check by run the [unit test](assignment-1/unit_test.py)

```bash
    cd assignment-1
    python unit_test.py
```

Then press `0` or `1`, respectively `Logistic` or `Softmax` to perform the assessment

## Submit assignments

We only need commit and push this repo

## Reference

- https://numpy.org/doc/1.18/
- https://www.tensorflow.org/guide/
- https://machinelearningcoban.com/