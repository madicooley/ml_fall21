import numpy as np
import matplotlib.pyplot as plt

import utils.utils as utils
from neural_networks.neural_network import NeuralNetwork, TensorflowNeuralNetwork


def problem2(train, test, epochs):
    layers = [2, 2, 1]
    activation = 'sigmoid' 
    depth=2
    
    def learning_rate_sched_a(gamma, a, t):
        return gamma/(1+(gamma/a)*t)
    
    for width in [5, 10, 25, 50, 100]:
        layers = [4+1]
        for hl in range(depth):
            layers.append(width+1)
        layers.append(1)                
        print('\nActivation: ', activation, 'Depth: ', depth, 'Width: ', width)
        
        nn = NeuralNetwork(layers, activation, train, test, epochs=epochs, lr_sched=learning_rate_sched_a)
        nn.sgd()
        nn.final_prediction()

def problem2c(train, test, epochs):
    layers = [2, 2, 1]
    activation = 'sigmoid' 
    depth=2
    
    def learning_rate_sched_a(gamma, a, t):
        return gamma/(1+(gamma/a)*t)
    
    for width in [5, 10, 25, 50, 100]:
        layers = [4+1]
        for hl in range(depth):
            layers.append(width+1)
        layers.append(1)                
        print('\nActivation: ', activation, 'Depth: ', depth, 'Width: ', width)
        
        nn = NeuralNetwork(layers, activation, train, test, 
                           epochs=epochs, 
                           lr_sched=learning_rate_sched_a,
                           zero_init=True)
        nn.sgd()
        nn.final_prediction()

def problem2e(train, test, epochs):
    for activation in ['tanh', 'RELU']:
        for depth in [3, 5, 9]:
            for width in [5, 10, 25, 50, 100]:
                layers = [4+1]
                for hl in range(depth):
                    layers.append(width+1)
                layers.append(1)                
                print('\nActivation: ', activation, 'Depth: ', depth, 'Width: ', width)
                
                tf_nn = TensorflowNeuralNetwork(activation, layers, train, test)
                tf_nn.train()
                tf_nn.final_prediction()

def main():
    np_train_bank = utils.readin_dat_np('data/bank-note/', 'train.csv', neg_lab=True)
    np_test_bank = utils.readin_dat_np('data/bank-note/', 'test.csv',   neg_lab=True)
    
    epochs = 5000
    print('\n-------------------------------Problem 2 output:')
    problem2(np_train_bank, np_test_bank, epochs)
    
    print('\n-------------------------------Problem 2c output:')
    problem2c(np_train_bank, np_test_bank, epochs)
    
    print('\n-------------------------------Problem 2e output:')
    problem2e(np_train_bank, np_test_bank, epochs)
    

if __name__=="__main__":
    main()