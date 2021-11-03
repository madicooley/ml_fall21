
import numpy as np
import copy

import utils.utils as utils
from perceptron.perceptron import Perceptron, VotedPerceptron, AveragePerceptron

def save_weights_csv(W, fname):
    # saves the voted perceptron weights to csv
    W_list = W
    
    # for w in W:
    #     W_list.append(list(w.round(4)))
    
    W_list = np.array(W_list)
    np.savetxt(fname, W_list, delimiter=',')
    return W_list

def problem2a(train_dat, test_dat, r, epochs):
    perceptron = Perceptron(train_dat, test_dat, r, epochs)
    w = perceptron.train()
    train_err = perceptron.pred_error(perceptron.X_train, perceptron.y_train)
    test_err = perceptron.pred_error(perceptron.X_test, perceptron.y_test)
    
    print('\nFinal weight vector=', w)
    print('Training error = ', train_err)
    print('Testing error = ', test_err)
    

def problem2b(train_dat, test_dat, r, epochs):
    perceptron = VotedPerceptron(train_dat, test_dat, r, epochs)
    W, C = perceptron.train()
    train_err = perceptron.pred_error(perceptron.X_train, perceptron.y_train)
    test_err = perceptron.pred_error(perceptron.X_test, perceptron.y_test)
    
    fname = 'voted_perceptron_weights.csv'
    save_weights_csv(W, fname)
    
    print('\nFinal weight vector length =', len(W), 'output to file: ', fname)
    
    print('First 10 weight vectors: ')
    count=0
    for w in W:
        print(w)
        count+=1
        if count>=10:
            break
    
    print('C: ', C)
    print('Training error = ', train_err)
    print('Testing error = ', test_err)


def problem2c(train_dat, test_dat, r, epochs):
    perceptron = AveragePerceptron(train_dat, test_dat, r, epochs)
    w = perceptron.train()
    train_err = perceptron.pred_error(perceptron.X_train, perceptron.y_train)
    test_err = perceptron.pred_error(perceptron.X_test, perceptron.y_test)
    
    print('\nFinal weight vector=', w)
    print('Training error = ', train_err)
    print('Testing error = ', test_err)
    

def main():
    
    np_train_bank = utils.readin_dat_np('data/bank-note/', 'train.csv', neg_lab=True)
    np_test_bank = utils.readin_dat_np('data/bank-note/', 'test.csv', neg_lab=True)
    
    r = 0.1
    epochs = 10
    
    print('\n-------------------------------Problem 2a output: ')
    problem2a(np_train_bank, np_test_bank, r, epochs)
    
    print('\n-------------------------------Problem 2b output: ')
    problem2b(copy.deepcopy(np_train_bank), 
               copy.deepcopy(np_test_bank), r, epochs)
    
    print('\n-------------------------------Problem 2c output: ')
    problem2c(copy.deepcopy(np_train_bank), 
                copy.deepcopy(np_test_bank), r, epochs)
    

if __name__=="__main__":
    main()