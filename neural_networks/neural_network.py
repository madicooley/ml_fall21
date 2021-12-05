
import numpy as np
import tensorflow as tf

import neural_networks.neural_network_utils as utils

class NeuralNetwork(object):
    def __init__(self, layers, activation, train_dat):
        '''
        layers : nn architecture specifications. 
                 widths do not include bias term for each layer
        
        '''
        self.weights = []
        self.layers = layers
        
        if activation == 'sigmoid':
            self.activation = utils.sigmoid_activation 
        
        self.n = train_dat.shape[0]+1
        self.m = train_dat.shape[1]

        self.train_dat = np.append(np.ones((self.n-1, 1)), train_dat, axis=1)
    
        X_train = self.train_dat[:, 0:-1]
        y_train = self.train_dat[:, -1]
        
        self.X_train = X_train
        self.y_train = y_train
        
        
        ### NOTE - for testing
        #self.init_network()
        W1 = np.array([[-1, 1], [-2, 2], [-3, 3]])
        W2 = np.array([[-1, 1], [-2, 2], [-3, 3]])
        W3 = np.array([-1, 2, -1.5]) 
        self.weights.append(W1)
        self.weights.append(W2)
        self.weights.append(W3)
        
        self.predict(np.array([1, 1, 1]))
        ###
        
    def init_network(self):
        rows = self.m
        for layer in self.layers:
            W = np.zeros((rows,  layer+1))
            self.weights.append(W)
            rows = layer+1 
    
    def z(self, x, W):
        zpart = self.activation(x.dot(W)) 
        return np.append(1, zpart)
    
    def predict(self, x):
        for W in self.weights:
            x = self.z(x, W)
        y = np.sign(x)
        return y

    def backpropogation(self):
        pass
    
    def sgd(self):
        pass
    
    
class TensorflowNeuralNetwork(object):
    def __init__(self, layers, activation, train_dat):
        pass
    