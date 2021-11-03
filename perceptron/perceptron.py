

import numpy as np


class Perceptron():
    def __init__(self, train_dat, test_dat, r, epochs):
        self.train_dat = train_dat
        self.test_dat = test_dat
        
        train_bias = np.ones((self.train_dat.shape[0], 1))
        self.train_dat = np.append(train_bias, self.train_dat, axis=1)
        
        test_bias = np.ones((self.test_dat.shape[0], 1))
        self.test_dat = np.append(test_bias, self.test_dat, axis=1)
        
        self.shuffle_data()
        
        self.n = self.X_train.shape[0]    # num training examples
        self.m = self.X_train.shape[1]   # num features
        self.w = np.zeros(self.m)
        
        self.r = r
        self.epochs = epochs

    def shuffle_data(self):
        np.random.shuffle(self.train_dat)
        np.random.shuffle(self.test_dat)
    
        self.X_train = self.train_dat[:, 0:-1]
        self.y_train = self.train_dat[:, -1]
        
        self.X_test = self.test_dat[:, 0:-1]
        self.y_test = self.test_dat[:, -1]
    
    def pred_error(self, X, y_true):
        y_pred = self.predict(X)
        correct = np.sum(y_pred == y_true)
        return (X.shape[0] - correct) / X.shape[0] 
        
    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T))
    
    def train(self):
        for epoch in range(self.epochs):
            # self.shuffle_data()
            i = 0
            for x in self.X_train:
                y_pred = np.sign(np.dot(self.w.T, x))
                if y_pred != self.y_train[i]:
                    self.w = self.w + self.r * (self.y_train[i] * x)
                i += 1
        return self.w
    
   
class VotedPerceptron(Perceptron):
    def __init__(self, train_dat, test_dat, r, epochs):
        super().__init__(train_dat, test_dat, r, epochs)
        self.wm = [] 
        self.C = [] 
    
    def predict(self, X):        
        sum = 0
        i=0
        for ci in self.C:
            sum += ci * np.sign(np.dot(self.wm[i].T, X.T))
            i+=1
        return np.sign(sum)
        
    def train(self):
        for epoch in range(self.epochs):
            self.shuffle_data()
            Cm = 1
            i = 0
            for x in self.X_train:
                if self.y_train[i] * np.dot(self.w.T, x) <= 0:
                    self.C.append(Cm)
                    self.w = self.w + self.r * (self.y_train[i] * x)
                    self.wm.append(self.w)
                    Cm = 1
                else: 
                    Cm += 1
                i+=1
        return self.wm, self.C
        
    
class AveragePerceptron(Perceptron):
    def __init__(self, train_dat, test_dat, r, epochs):
        super().__init__(train_dat, test_dat, r, epochs)
        self.a = [0] * self.m 
        
    def predict(self, X):
        return np.sign(np.dot(self.a.T, X.T))
    
    def train(self):
        for epoch in range(self.epochs):
            self.shuffle_data()
            i = 0
            for x in self.X_train:
                if self.y_train[i] * np.sign(np.dot(self.w.T, x)) <= 0:
                    self.w = self.w + self.r * (self.y_train[i] * x)
                self.a = self.a + self.w
                i+=1
        return self.a / (self.n * self.epochs)
       
    
    
    
    
    
    