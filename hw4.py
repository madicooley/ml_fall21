
import numpy as np
import matplotlib.pyplot as plt

import utils.utils as utils
from svm.svm import primal_svm, dual_svm


def problem2a(train, test, T, Cs):
    def learning_rate_sched_a(gamma, a, t):
        return gamma/(1+(gamma/a)*t)
    
    final_train_errs=[]
    final_test_errs=[]
    Ws = []
    a = 0.05
    for c in Cs:
        print('\n\n--------------------C: ', c)
        train_errs, test_errs, w = primal_svm(train, test, c, T, a=a, lr_sched=learning_rate_sched_a)
        plt.plot(list(range(0, T)), train_errs, label='Train error')
        plt.plot(list(range(0, T)), test_errs, label='Test error')
        plt.ylabel('Prediction Error')
        plt.xlabel('Iteration')
        plt.show()
        
        Ws.append(w)
        final_train_errs.append(train_errs[-1])
        final_test_errs.append(test_errs[-1])
    return Ws, final_train_errs, final_test_errs

def problem2b(train, test, T, Cs):
    def learning_rate_sched(gamma, t):
        return gamma/(1+gamma*t)

    final_train_errs=[]
    final_test_errs=[]
    Ws = []
    for c in Cs:
        print('\nC: ', c)
        train_errs, test_errs, w = primal_svm(train, test, c, T, lr_sched=learning_rate_sched)
        plt.plot(list(range(0, T)), train_errs, label='Train error')
        plt.plot(list(range(0, T)), test_errs, label='Test error')
        plt.ylabel('Prediction Error')
        plt.xlabel('Iteration')
        plt.show()
        
        Ws.append(w)
        final_train_errs.append(train_errs[-1])
        final_test_errs.append(test_errs[-1])
    return Ws, final_train_errs, final_test_errs


def problem3a(train, test, T, Cs):
    final_train_errs=[]
    final_test_errs=[]
    Ws = []
    bs = []
    SVS = []
    for c in Cs:
        print('\nC: ', c)
        train_err, test_err, w, b, support_vectors = dual_svm(train, test, c, T, kernel="linear")
        final_train_errs.append(train_err)
        final_test_errs.append(test_err)
        Ws.append(w)
        bs.append(b)
        SVS.append(support_vectors)
    return Ws, bs, final_train_errs, final_test_errs, SVS

def problem3b(train, test, T, Cs):
    gammas = [0.1, 0.5, 1, 5, 100]
    final_train_errs=[]
    final_test_errs=[]
    Ws = []
    bs = []
    SVS = []
    for gamma in gammas:
        for c in Cs:
            print('\nC: ', c, ' gamma=', gamma)
            numovlp = 0
            train_err, test_err, w, b, support_vectors = dual_svm(train, test, c, T, kernel="gaussian", gamma=gamma)
            final_train_errs.append(train_err)
            final_test_errs.append(test_err)
            Ws.append(w)
            bs.append(b)
            try:
                overlap = [v for v in SVS[-1] if v in support_vectors]
                numovlp=len(overlap)
            except:
                pass
            SVS.append(support_vectors)
            print('Num overlap between gamma and prev=', numovlp)
            
    return Ws, bs, final_train_errs, final_test_errs, SVS

def main():
    np_train_bank = utils.readin_dat_np('data/bank-note/', 'train.csv', neg_lab=True)
    np_test_bank = utils.readin_dat_np('data/bank-note/', 'test.csv', neg_lab=True)
    
    epochs = 100
    Cs = [100/873, 500/873, 700/873]
    
    # testing
    # print('Testing')
    # test = np.array([[0.5, -1, 0.3, 1], 
                     # [-1, -2, -2, -1], 
                     # [1.5, 0.2, -2.5, 1]])
    # problem2a(test, test, 3, [0.2])
    
    print('\n-------------------------------Problem 2a output:')
    Ws1, final_train_errs1, final_test_errs1 = problem2a(np_train_bank, np_test_bank, epochs, Cs)
    print(Ws1)
    
    print('\n-------------------------------Problem 2b output:')
    Ws2, final_train_errs2, final_test_errs2 = problem2b(np_train_bank, np_test_bank, epochs, Cs)
    
    print('\nW differences: ')
    for i in range(len(Ws1)):
        print('diff: ', Ws1[i] - Ws2[i])
        
    print('\nTrain err differences: ')
    for i in range(len(Ws1)):
        print('diff: ', final_train_errs1[i] - final_train_errs2[i])
    
    print('\nTest err differences: ')
    for i in range(len(Ws1)):
        print('diff: ', final_test_errs1[i] - final_test_errs2[i])
    
    print('\n-------------------------------Problem 3a output:')
    Ws, bs, train_errs, test_errs, SVS = problem3a(np_train_bank, np_test_bank, epochs, Cs)
    print('Train errors=', train_errs, ' Test errors=', test_errs)
    
    print('\n-------------------------------Problem 3b output:')
    Ws, bs, train_errs, test_errs, SVS = problem3b(np_train_bank, np_test_bank, epochs, [500/873])
    print('Train errors=', train_errs, ' Test errors=', test_errs)
    

if __name__=="__main__":
    main()
