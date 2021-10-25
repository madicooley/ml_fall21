'''

'''

import sys, copy, statistics
import numpy as np
import pandas as pd
from os.path import dirname,realpath

import utils.utils as utils
import utils.tree as tree
from decision_tree.decision_tree import ID3
from ensemble_learning.ada_boost import ada_boost_alg
from ensemble_learning.bagged_tree import bagged_tree_alg
from ensemble_learning.random_forest import random_forest_alg
import ensemble_learning.ensemble_utils as ens_utils
from linear_regression.optimization import gradient_descent, stochastic_gradient_descent


def tests(dataset, attr_selection, most_common=False, most_label=False):
    '''
        1. show data subsets S
        2. attributes A
        3. info gain calculation
        4. how you split the data
    '''
    attr_names=None
    df_train=None
    df_test=None
    
    if dataset=='boolean':
        attr_names = {0 : 'x_1', 1 : 'x_2', 2 : 'x_3', 3 : 'x_4', 4 : 'y'}
        df_train = utils.readin_dat_pd('data/tests/', 'bool_train.csv', columns=attr_names)
    elif dataset=='tennis_full':
        attr_names = {0 : 'outlook', 1 : 'temperature', 2 : 'humidity', 3 : 'wind', 4 : 'y'}
        df_train = utils.readin_dat_pd('data/tests/', 'tennis_train.csv', columns=attr_names)
    elif dataset=='tennis_missing':
        attr_names = {0 : 'outlook', 1 : 'temperature', 2 : 'humidity', 3 : 'wind', 4 : 'y'}
        df_train = utils.readin_dat_pd('data/tests/', 'tennis_train_missing.csv', columns=attr_names)
    df_test = df_train
    
    attributes = utils.preprocess_pd(df_train, most_common=most_common, most_label=most_label)
    utils.pos_neg_labels(df_train)
    
    #d_tree = tree.Tree()
    max_depth = 1
    T = 10
    #m = len(df_train)
    #Dt = [1]*m 
    
    #ID3(df_train,
        #model=d_tree,
        #attributes=attributes, 
        #attr_selection=attr_selection, 
        #max_depth=max_depth,
        #model_type=tree.Tree)
    
    ##train_error = utils.get_pred_error(df_train, d_tree)
    #d_tree.print_tree()
    
    # ada_boost_alg(df_train, 
    #               df_test,
    #               T, 
    #               attributes, 
    #               attr_selection=attr_selection, 
    #               max_depth=max_depth)
                  
    bagged_tree_alg(df_train, 
                    df_test,
                    T, 
                    attributes, 
                    attr_selection=attr_selection, 
                    max_depth=max_depth)
    
    # random_forest_split=2
    # random_forest_alg(df_train, 
    #                 df_test,
    #                 T, 
    #                 attributes, 
    #                 random_forest_split,
    #                 attr_selection=attr_selection, 
    #                 max_depth=max_depth, 
    #                 plot=True)
    
    
def problem2A(df_train, df_test, attributes, attr_selection, T=2):
    
    max_depth = 1
    ada_boost_alg(df_train, 
                  df_test,
                  T, 
                  attributes, 
                  attr_selection=attr_selection, 
                  max_depth=max_depth)


def problem2B(df_train, df_test, attributes, attr_selection, T=2):
    bagged_tree_alg(df_train, 
                    df_test,
                    T, 
                    attributes, 
                    attr_selection=attr_selection,
                    max_depth=2,
                    plot=True)


def problem2C(df_train, df_test, attributes, attr_selection, T=2):
    return   # NOTE

    bagged_trees = []
    for i in range(100):
        # sample 1000 examples uniformly w.out replacement
        samp = ens_utils.get_bootstrap_sample(df_train, 1000, replace=False)
        
        # run bagged tree alg based on samp for 500 trees
        first_tree = bagged_tree_alg(samp, 
                                    df_test,
                                    T, 
                                    attributes, 
                                    attr_selection=attr_selection)
        bagged_trees.append(first_tree)
        
    # for each test example, compute pred on each tree
    # TODO 
    avg_test_preds = []
    for ind, x in df_test.iterrows():
        x_preds = []
        for tree in bagged_trees:
            y_hat = None 
            x_preds.append(y_hat)
            
        avg_test_preds.append(sum(x_preds)/len(x_preds))
            
        

def problem2D(df_train, df_test, attributes, attr_selection, T=2):
    random_forest_splits = [2, 4, 6]
    
    for random_forest_split in random_forest_splits:
        random_forest_alg(df_train, 
                        df_test,
                        T, 
                        attributes, 
                        6,
                        attr_selection=attr_selection, 
                        plot=True)
        

def problem3(df_train, df_test, attr_selection):
    pass


def problem4A(X_train, y_train, X_test, y_test):
    '''
    - The task is to predict the real-valued SLUMP of the concrete, with 7 features.
    
    To test SLUMP, we use 7 features (which are the first 7 columns)
    Cement, Slag, Fly ash, Water, SP, Coarse Aggr, Fine Aggr

    The output is the last column
    '''
    
    best_r = None
    best_cost = np.inf 
    R_vals = [0.1, 0.01, 0.001, 0.0001, 0.00001]  
    for r in R_vals:
        print('r: ', r)
        final_test_cost = gradient_descent(X_train, y_train, 
                                            X_test, y_test, r)
        
        if final_test_cost < best_cost:
            best_cost = final_test_cost
            best_r = r
    
    #best_r = 0.00001
    print('Best r: ', best_r)
    final_test_cost = gradient_descent(X_train, y_train, 
                                       X_test, y_test, 
                                       best_r, plot=True)


def problem4B(X_train, y_train, X_test, y_test):

    best_r = None
    best_cost = np.inf 
    R_vals = [0.1, 0.01, 0.001, 0.0001, 0.00001]    
    for r in R_vals:
        print('r: ', r)
        final_test_cost = stochastic_gradient_descent(X_train, y_train, 
                                                      X_test, y_test, r)
        
        if final_test_cost < best_cost:
            best_cost = final_test_cost
            best_r = r
    
    #best_r = 0.001
    print('Best r: ', best_r)
    final_test_cost = stochastic_gradient_descent(X_train, y_train, 
                                                  X_test, y_test, 
                                                  best_r, plot=True)
    

def main():
    attr_selection = ['info_gain', 'maj_error', 'gini_ind']
    
    ## Tests
    #tests('boolean', 'info_gain') 
    # tests('tennis_full', 'info_gain') 
    #tests('boolean', 'maj_error')
    #tests('boolean', 'gini_ind')
    # return
    
    ##################### 
    
    ## to initially process the bank data
    # df_train_bank = utils.readin_dat_pd('data/bank/', 'train.csv')
    # df_test_bank = utils.readin_dat_pd('data/bank/', 'test.csv')
    
    # print('preprocessing bank train data')
    # attributes = utils.preprocess_pd(df_train_bank, 
    #                                 numeric_features=True)
    
    # utils.preprocess_pd(df_test_bank, numeric_features=True)
    # utils.pos_neg_labels(df_train_bank)
    # utils.pos_neg_labels(df_test_bank)
    # print('finished preprocessing bank')
    
    # df_train_bank.to_csv('data/bank/train_processed.csv', header=False, index=False)
    # df_test_bank.to_csv('data/bank/test_processed.csv', header=False, index=False)
    # return
    
    df_train_bank = utils.readin_dat_pd('data/bank/', 'train_processed.csv')
    df_test_bank = utils.readin_dat_pd('data/bank/', 'test_processed.csv')
    attributes = utils.preprocess_pd(df_train_bank)
    
    ##################### 
    
    df_train_concrete = utils.readin_dat_pd('data/concrete/', 'train.csv')
    df_test_concrete = utils.readin_dat_pd('data/concrete/', 'test.csv')
    
    X_train = df_train_concrete
    y_train = df_train_concrete['y']
    del X_train['y']
    
    X_test = df_test_concrete
    y_test = df_test_concrete['y']
    del X_test['y']
        
    ##################### 
    
    # prob 2: 
    print('\n-----------------------------Problem 2A. Output: ')
    problem2A(df_train_bank, df_test_bank, attributes,
              'info_gain', T=5)
    
    print('\n-----------------------------Problem 2B. Output: ')
    problem2B(df_train_bank, df_test_bank, attributes,
            'info_gain', T=10)
    
    print('\n-----------------------------Problem 2C. Output: ')
    problem2C(df_train_bank, df_test_bank, attributes,
              'info_gain', T=50)
    
    print('\n-----------------------------Problem 2D. Output: ')
    problem2D(df_train_bank, df_test_bank, attributes,
              'info_gain', T=500)
    
    ## prob 4: 
    print('\n-----------------------------Problem 4A. Output: ')
    problem4A(X_train, y_train, X_test, y_test)
    
    print('\n-----------------------------Problem 4B. Output: ')
    problem4B(X_train, y_train, X_test, y_test)


if __name__=="__main__":
    main()
    
