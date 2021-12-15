
import numpy as np
import tensorflow as tf
import timeit, math
from tqdm import notebook, tqdm
tf.compat.v1.disable_eager_execution()

import neural_networks.neural_network_utils as utils


class NeuralNetwork(object):
    def __init__(self, layers, activation, train_dat, test_dat, epochs=100, lr_sched=None, zero_init=False):
        '''
        layers : nn architecture specifications. 
                 widths do not include bias term for each layer
        
        '''
        self.zero_init=zero_init
        self.weights = []
        self.layers = layers
        
        self.a = 0.5
        self.gamma = 0.1
        print('a=', self.a, 'gamma=', self.gamma)
        self.lr_sched = lr_sched
        
        if activation == 'sigmoid':
            self.activation = utils.sigmoid_activation 
            self.activation_der = utils.sigmoid_derivative
        
        self.n = train_dat.shape[0]+1
        self.m = train_dat.shape[1]

        # add bias term to input
        self.train_dat = np.append(np.ones((self.n-1, 1)), train_dat, axis=1)
        self.test_dat  = np.append(np.ones((test_dat.shape[0], 1)), test_dat, axis=1)
            
        self.X_train = self.train_dat[:, 0:-1]
        self.y_train = self.train_dat[:, -1]
        
        self.X_test = self.test_dat[:, 0:-1]
        self.y_test = self.test_dat[:, -1]
        
        self.w = np.zeros(self.m)
        self.epochs = epochs
        self.init_network()
        
        ### NOTE - for testing
        # W1 = np.array([[-1, 1], [-2, 2], [-3, 3]])
        # W2 = np.array([[-1, 1], [-2, 2], [-3, 3]])
        # W3 = np.array([-1, 2, -1.5]) 
        # self.weights.append(W1)
        # self.weights.append(W2)
        # self.weights.append(W3)
        # self.backpropogation(np.array([1, 1, 1]), 1)        
        ###
        
    def init_network(self):
        rows = self.m
        for layer in self.layers[1:-1]:
            if self.zero_init:
                W = np.zeros((rows,  layer))
            else:
                W = np.random.normal(size=(rows, layer))
            self.weights.append(W)
            rows = layer+1               # plus one for bias
        W = np.zeros((rows,  1))  
        self.weights.append(W)
    
    def z(self, x, W):
        xW = x.dot(W)
        zpart = self.activation(xW) 
        return np.append(1, zpart)
    
    def forwardpropagation(self, x):
        Z = []
        for W in self.weights[0:-1]:
            x = self.z(x, W)
            Z.append(x)
        y_hat = x.dot(self.weights[-1])
        # y_hat = self.z(x, self.weights[-1])[-1]
        return y_hat, Z
    
    def predict_all(self, X):
        y_hats = []
        for x in X:
            y_hat, _ = self.forwardpropagation(x)
            # if y_hat < 0.5:
                # y_hat = 0
            # else: 
                # y_hat = 1
            y_hats.append(y_hat)
        return y_hats
    
    def squared_loss(self, X, y):
        y_hats = np.array(self.predict_all(X)).reshape(y.shape)
        diff = y - y_hats
        error = np.linalg.norm(diff, 2) / np.linalg.norm(y, 2)   
        return error
    
    def final_prediction(self):
        train_loss = self.squared_loss(self.X_train, self.y_train)
        test_loss  = self.squared_loss(self.X_test,  self.y_test)
        print('\nFinal Train Loss: ', train_loss, 'Final Test Loss: ', test_loss)
    
    def backpropogation(self, x, y_star):
        '''
            Returns D a list of derivatives for each weight matrix. 
            Derivatives listed from first layer derivatives to last layer in D.
        '''
        y, Z = self.forwardpropagation(x)
        dL_dy = y - y_star
        D = []
        ##### last layer derivatives
        dL_dw3 = dL_dy * Z[1]
        
        ##### middle layer derivatives
        dy_dz2 = self.weights[2][1:]
        z1 = Z[0]
        z1w2 = z1.dot(self.weights[1])
        dsigma_dz1w2 = self.activation_der(z1w2)

        dL_dw2 = []
        w2_flat = self.weights[1].flatten()
        rng = dy_dz2.shape[0]
        for z1_i in z1:
            for i in range(rng):
                dy_dz2_i = dy_dz2[i]
                dsig_dz1w2_i = dsigma_dz1w2[i]
                dL_dw2.append(dL_dy * dy_dz2_i * dsig_dz1w2_i * z1_i)
        dL_dw2 = np.array(dL_dw2).reshape(self.weights[1].shape)
        
        ##### first layer derivatives
        dz2_dz1 = (dsigma_dz1w2 * self.weights[1][1:]).flatten()
        xw1 = x.dot(self.weights[0])
        dsigma_dxw1 = self.activation_der(xw1)
        dL_dw1 = []
        w1_flat = self.weights[0].flatten()
    
        dz1_dw1 = []   
        for sig in dsigma_dxw1:
            dz1_dw1.append(sig * x) 
        dz1_dw1 = np.array(dz1_dw1).flatten()  
        
        j=0
        for dz1_w1_i in dz1_dw1:
            dL_dw1_i = 0 
            for i in range(len(dy_dz2)):
                dL_dw1_i += dy_dz2[i] * dz2_dz1[j] * dz1_w1_i
                j+=1
            if j>=len(dz2_dz1):
                j=0
            dL_dw1.append(dL_dy * dL_dw1_i)
        dL_dw1 = np.array(dL_dw1).reshape(self.weights[0].shape)
        
        D.append(dL_dw1)
        D.append(dL_dw2)
        D.append(dL_dw3)
        return D
    
    def sgd(self):
        for epoch in range(self.epochs):     
            if epoch % 5 == 0:
                train_loss = self.squared_loss(self.X_train, self.y_train)
                test_loss = self.squared_loss(self.X_test, self.y_test)
                print('Epoch: ', epoch, '/', self.epochs, ' Train Loss: ', train_loss, 'Test Loss: ', test_loss)
            
            self.gamma_t = self.lr_sched(self.gamma, self.a, epoch)
            np.random.shuffle(self.train_dat)
            np.random.shuffle(self.train_dat)
            self.X_train = self.train_dat[:, 0:-1]
            self.y_train = self.train_dat[:, -1]
            
            for i in range(self.n-1):
                x = self.X_train[i, :]
                y = self.y_train[i]
                
                j = 0
                dL_dW = self.backpropogation(x, y)
                for dL_dW_i in dL_dW:
                    wj_shape = self.weights[j].shape
                    g_dwi = (self.gamma_t * dL_dW_i).reshape(wj_shape)
                    self.weights[j] = self.weights[j] - g_dwi
                    j+=1

    
class TensorflowNeuralNetwork(object):
    def __init__(self, activation, layers, train_dat, test_dat):     
        self.layers = layers
        self.activation = activation 
        
        self.n = train_dat.shape[0]+1
        self.m = train_dat.shape[1]

        # add bias term to input
        self.train_dat = np.append(np.ones((self.n-1, 1)), train_dat, axis=1)
        self.test_dat = np.append(np.ones((len(test_dat), 1)), test_dat, axis=1)
    
        X_train = self.train_dat[:, 0:-1]
        y_train = self.train_dat[:, -1].reshape((self.n-1, 1))
        
        X_test = self.test_dat[:, 0:-1]
        y_test = self.test_dat[:, -1].reshape((len(test_dat), 1))
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False)) 
        
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        
        self.weights, self.biases = self.initialize_NN(self.layers)  
        
        self.y_pred = self.forward_pass(self.x_tf)
    
        self.loss = tf.reduce_mean(tf.square(self.y_tf - self.y_pred))        
        self.train_op = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)
        
        # starter_learning_rate = 0.1
        # self.train_op = tf.compat.v1.train.GradientDescentOptimizer(starter_learning_rate).minimize(self.loss)
        
        self.loss_log = []
        
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)
    
    def zero_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        return tf.Variable(tf.zeros([in_dim, out_dim], dtype=tf.float32),
                           dtype=tf.float32)
        
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.zero_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)   
        return weights, biases
    
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            if self.activation == 'RELU':
                H = tf.nn.relu(tf.add(tf.matmul(H, W), b))  
            elif self.activation == 'tanh':
                H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))  
            elif self.activation == 'sigmoid':
                H = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))  
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    def predict_y(self, x):
        tf_dict = {self.x_tf : x}
        y_pred = self.sess.run(self.y_pred, tf_dict)
        return y_pred
    
    def final_prediction(self):
        y_pred_train = self.predict_y(self.X_train)
        error_train = np.linalg.norm(self.y_train - y_pred_train, 2) / np.linalg.norm(self.y_train, 2)        
        err_title = 'Relative L2 error y_train: %e' % (error_train)
        print(err_title)
        
        y_pred_test = self.predict_y(self.X_test)
        error_test = np.linalg.norm(self.y_test - y_pred_test, 2) / np.linalg.norm(self.y_test, 2)        
        err_title = 'Relative L2 error y_test: %e' % (error_test)
        print(err_title)
    
    def train(self, nIter=5000, save_preds=False, frequency=1000):
        start_time = timeit.default_timer()
        for it in notebook.tqdm(range(nIter)):
            tf_dict = {self.x_tf : self.X_train,
                       self.y_tf : self.y_train}
            self.sess.run(self.train_op, tf_dict)
            
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                y_pred = self.predict_y(self.X_train)
                error = np.linalg.norm(self.y_train - y_pred, 2) / np.linalg.norm(self.y_train, 2)

                self.loss_log.append(loss_value)
                start_time = timeit.default_timer()
                
                # print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
        
    