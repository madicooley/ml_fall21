'''


'''
import matplotlib.pyplot as plt
import math, copy, time
import numpy as np

import utils.utils as utils 
import utils.tree as tree
import ensemble_learning.ensemble_utils as ens_utils
from decision_tree.decision_tree import ID3
import utils.plot as plotter



def random_forest_alg(df_train, df_test, T, attributes, 
                      random_forest_split,
                      attr_selection='info_gain', 
                      max_depth=np.inf, 
                      plot=False):
    '''
    
    '''    
    m = len(df_train)
    Hts = np.array([0]*m)
    Hts_test = np.array([0]*m)
    
    train_errors = []
    test_errors = []
    
    start = time.time()
    end = start
    for i in range(T):
        print(i, '/', T, ' time: ', end-start)
        
        model = tree.Tree()
        ID3(copy.deepcopy(df_train), 
            model=model, 
            max_depth=max_depth,
            attr_selection = attr_selection,
            attributes=copy.deepcopy(attributes),
            random_forest_split=random_forest_split)
        
        hts = [model.make_prediction(x) for ind, x in df_train.iterrows()]
        hts_test = [model.make_prediction(x) for ind, x in df_test.iterrows()]
        
        Hts = np.sum(np.concatenate((hts, Hts)).reshape((2, m)), axis=0)
        Hts_test = np.sum(np.concatenate((hts_test, Hts_test)).reshape((2, m)), axis=0)
        
        y_train_pred = np.sign(Hts / (i+1))
        y_test_pred = np.sign(Hts_test / (i+1))
        
        train_pred_err = ens_utils.prediction_error(y_train_pred, df_train['y'])
        test_pred_err = ens_utils.prediction_error(y_test_pred, df_test['y'])
        
        print('train error: ', train_pred_err)
        print('test error: ',  test_pred_err)
        
        train_errors.append(train_pred_err)
        test_errors.append(test_pred_err)
        
        end=time.time()       
    
    if plot:        
        plot_fname = "Total Train/Test Errors by T (RandomForest), subset size="+str(random_forest_split)
        
        plt.plot(list(range(0, len(train_errors))), train_errors, label='Train error')
        plt.plot(list(range(0, len(test_errors))), test_errors, label='Test error')
        plt.ylabel('Cost')
        plt.xlabel('Tree Number')
        plt.title(plot_fname)
        plt.show()
        
        
        
