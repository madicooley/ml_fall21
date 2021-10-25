'''

'''

import sys, copy, statistics
import numpy as np
import pandas as pd

from os.path import dirname,realpath
#sys.path.insert(0, dirname(realpath(__file__))[:-13])

import utils.utils as utils
import utils.tree as tree
from decision_tree.decision_tree import ID3


def tests(dataset, attr_selection, most_common=False, most_label=False):
    '''
        1. show data subsets S
        2. attributes A
        3. info gain calculation
        4. how you split the data
    '''
    global d_tree
    d_tree = tree.Tree()
    attr_names=None
    df_train=None
    df_test=None
    
    if dataset=='boolean':
        attr_names = {0 : 'x_1', 1 : 'x_2', 2 : 'x_3', 3 : 'x_4', 4 : 'y'}
        df_train = utils.readin_dat_pd('decision_tree/data/tests/', 'bool_train.csv', columns=attr_names)
    elif dataset=='tennis_full':
        attr_names = {0 : 'outlook', 1 : 'temperature', 2 : 'humidity', 3 : 'wind', 4 : 'y'}
        df_train = utils.readin_dat_pd('decision_tree/data/tests/', 'tennis_train.csv', columns=attr_names)
    elif dataset=='tennis_missing':
        attr_names = {0 : 'outlook', 1 : 'temperature', 2 : 'humidity', 3 : 'wind', 4 : 'y'}
        df_train = utils.readin_dat_pd('decision_tree/data/tests/', 'tennis_train_missing.csv', columns=attr_names)
    
    attributes = utils.preprocess_pd(df_train, most_common=most_common, most_label=most_label)
    
    ID3(df_train, attributes, attr_selection, 5)
    
    #train_error = utils.get_pred_error(df_train, d_tree)
    d_tree.print_tree()


def run_exps(df_train, df_test, attr_selection, depth_iters, numeric_features=False, unknown_missing=False, most_common=True):
    attributes = utils.preprocess_pd(df_train, numeric_features=numeric_features, 
                               unknown_missing=unknown_missing, most_common=most_common)
    
    utils.preprocess_pd(df_test, numeric_features=numeric_features, 
                  unknown_missing=unknown_missing, most_common=most_common)
    
    print('Max Depth & & Info Gain & & Maj Error & & Gini Ind \\\\')
    print(' & train & test & train & test & train & test \\\\')
    for max_depth in range(1, depth_iters+1):
        print(max_depth, ' & ', end='')
        for attr_slct in attr_selection:
            #global d_tree
            d_tree = tree.Tree()

            ID3(d_tree, df_train, copy.deepcopy(attributes), attr_slct, max_depth)
            #run_ID3(d_tree, df_train, copy.deepcopy(attributes), attr_slct, max_depth)
            
            train_error = utils.get_pred_error(df_train, d_tree)
            test_error = utils.get_pred_error(df_test, d_tree)
            
            print(round(train_error,3), ' & ', round(test_error, 3), ' & ', end='')
        print(' \\\\') 
    
    
def main():
    attr_selection = ['info_gain', 'maj_error', 'gini_ind']
    
    #tests('boolean', 'info_gain') 
    #tests('boolean', 'maj_error')
    #tests('boolean', 'gini_ind')
    
    #tests('tennis_full', 'info_gain')
    #tests('tennis_full', 'maj_error')
    #tests('tennis_full', 'gini_ind')
    
    #tests('tennis_missing', 'info_gain', most_common=True)
    #tests('tennis_missing', 'info_gain', most_label=True)
    #tests('tennis_missing', 'info_gain')  
    
    df_train_car = utils.readin_dat_pd('decision_tree/data/car/', 'train.csv')
    df_test_car = utils.readin_dat_pd('decision_tree/data/car/', 'test.csv')
    
    df_train_bank_a = utils.readin_dat_pd('decision_tree/data/bank/', 'train.csv')
    df_test_bank_a = utils.readin_dat_pd('decision_tree/data/bank/', 'test.csv')
    
    df_train_bank_b = utils.readin_dat_pd('decision_tree/data/bank/', 'train.csv')
    df_test_bank_b = utils.readin_dat_pd('decision_tree/data/bank/', 'test.csv')
    
    ####### Decision Tree Practice - problem 2
    print('\n-----------------------------Problem 2b. Output: ')
    run_exps(df_train_car, df_test_car, attr_selection, 6)
    
    ####### Decision Tree Practice - problem 3
    print('\n\n-----------------------------Problem 3a. Output: ')
    run_exps(df_train_bank_a, df_test_bank_a, attr_selection, 16, numeric_features=True)
    
    print('\n\n-----------------------------Problem 3b. Output: ')
    run_exps(df_train_bank_a, df_test_bank_a, attr_selection, 16, 
             numeric_features=True, unknown_missing=True, most_common=True)


if __name__=="__main__":
    main()
    
