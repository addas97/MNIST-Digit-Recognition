#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def main():
    # Load the dataset
    num_classes = 10 # For the MNIST dataset, there are 10 classes, corresponding to the digits 0 through 9.
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev (validation set)
    dev_split_index = int(9 * len(X_train) / 10) 
    X_dev = X_train[dev_split_index:] # The development set, also known as the validation set, is used to evaluate the model during training and fine-tuning. 
                                      # It helps in selecting the best model and tuning hyperparameters.
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index] # The training set is the subset of the dataset used to fit the model. It is the data that the model learns from by adjusting its parameters.
    y_train = y_train[:dev_split_index] 

    # Shuffle training data
    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split training and dev datasets into batches to tune model (batch size, lr, momentum, hidden layers)
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification - 1 layer with 128 
    model = nn.Sequential(
              nn.Linear(784, 128), # Input Layer: The input data has 784 features (28x28 pixels for the MNIST dataset).
                                   # First Hidden Layer: This layer is a linear transformation that maps the 784 input features to 128 hidden features.
              nn.ReLU(),           # Activiation funtion via ReLU
              #nn.LeakyReLU(0.01), # Activiation funtion via leaky ReLU - allow for small negative vectors
              nn.Linear(128, 10),  # Output Layer: The final linear transformation maps the 10-dimensional hidden representation to the 10 output classes (one for each digit from 0 to 9).
            )
    lr=0.1
    momentum=0 # Accelerate the convergence of the gradient descent algorithm and prevents it from getting stuck in local minima.
               # Stores past parameter updates to determine velocity, then updates parameters directly via velocity instead of gradient
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle.
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
