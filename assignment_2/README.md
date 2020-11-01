[![vietAi](img/vietAi.jpg)](http://vietai.org/)
----------------------
Assignment 2: Deep Learning - Neural Network 
----------------------

Welcome to the second assignment of the VietAI Foundation course.

In this assignment, we will be looking at classification problems with neural networks. 
You will implement a **multilayer perceptron** (MLP) from scratch using numpy and tensorflow 2.

**Deadline**: 23:00:00 05/07/2020

**Document handout**: [pdf](assignment/doc/assignment_2.pdf) 

## Setup environment
Follow these steps to setup your local environment using [conda](https://www.anaconda.com/products/individual)
1. Create a `python 3.7` environment:
    ```bash
    conda create -n assignment2 python=3.7 -y
    ```
2. Activare your environment:
    ```bash
    conda activate assignment2 
    ```
3. On this project root folder, run:
    ```bash
    pip install -r requirements.txt
    ```

**WARNING**
We found that the current version of tensorflow (2.2) will cause troubles (freezing) on some machine, so we suggest using the cpu-only version that was listed on `requirements.txt`

## For testing on Colab
You can follow this [Colab link](https://colab.research.google.com/gist/daemon-Lee/aaf6f50ed1109cb19bad1d75ee92d587/colab_running.ipynb) to check your implementation if you have any troubles when running them offline

## Submit assignments
Remember to push your lastest commit on this repo.
