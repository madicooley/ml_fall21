import numpy as np
import matplotlib.pyplot as plt

import utils.utils as utils
from neural_networks.neural_network import NeuralNetwork


def problem2(train, test, epochs):
    layers = [2, 2, 1]
    activation = 'sigmoid' 
    
    nn = NeuralNetwork(layers, activation, train)


def main():
    np_train_bank = utils.readin_dat_np('data/bank-note/', 'train.csv', neg_lab=True)
    np_test_bank = utils.readin_dat_np('data/bank-note/', 'test.csv',   neg_lab=True)
    
    epochs = 100
    print('\n-------------------------------Problem 2 output:')
    problem2(np_train_bank, np_test_bank, epochs)
    

if __name__=="__main__":
    main()