

import math, copy, time
import numpy as np

import utils.utils as utils 
import utils.tree as tree
import ensemble_learning.ensemble_utils as ens_utils
from decision_tree.decision_tree import ID3
import utils.plot as plotter




def ada_boost_alg(df_train, df_test, T, attributes, 
                  attr_selection='info_gain', 
                  max_depth=np.inf, 
                  plot=False):
    '''
    just use entropy to compute feature splits
    
    args:
        learner: learning algorithm function (e.g ID3(..., ...))
    '''    
    m = df_train.shape[0] # num training samp
    n = df_train.shape[1] # num cols
    
    Dt = np.array([1/m]*m)
    alphas = []
    models = []
    Dts = []
    Hts_train = []
    #Hts_test = []
    
    #weighted_train_error = []
    #train_errors = []
    #test_errors = []
    
    #tree_train_err = []  
    #tree_test_err = []
        
    #Hts_alpha_sum = np.array([0]*m)
    
    start = time.time()
    end = start
    for i in range(T):
        print('\n\n', i, '/', T, ' time: ', end-start)
        
        model = tree.Tree()
        #samp = ens_utils.get_bootstrap_sample(df_train, m)
        #chosen_idx = np.random.choice(m, replace=False, size=int(m*0.80))
        #samp = df_train.iloc[chosen_idx]
        #Dt_samp = Dt[chosen_idx]
        
        ID3( copy.deepcopy(df_train), 
            Dt=copy.deepcopy(Dt), 
            model=model, 
            max_depth=max_depth,
            attr_selection = attr_selection,
            attributes=copy.deepcopy(attributes),
            weighted_samples=True)
        
        hts_train = [model.make_prediction(x) for ind, x in df_train.iterrows()]
        #hts_test  = [model.make_prediction(x) for ind, x in df_test.iterrows()]
        Hts_train.append(hts_train)
        #Hts_test.append(hts_test)
        
        et = ens_utils.weighted_error(df_train, Dt, hts_train)
        alpha = ens_utils.compute_vote(et)
        Dt = ens_utils.update_weights(df_train, Dt, hts_train, alpha)   
        
        alphas.append(alpha)
        models.append(model)
        Dts.append(Dt)
                
        y_pred = ens_utils.compute_final_hypothesis(Hts_train, alphas)
        train_pred_err = ens_utils.prediction_error(y_pred, df_train['y'])        
        curr_tree_pred_err = ens_utils.prediction_error(np.sign(hts_train), df_train['y'])  
        
        print('alpha', alpha)
        print('curr tree pred err: ', curr_tree_pred_err)
        print('train error: ', train_pred_err, 'weighted train err: ', et)

        #weighted_train_error.append(et)
        end=time.time()       
    
    if plot:
        #plotter.plot(train_errors, test_errors, "Total Train/Test Errors by T (AdaBoost)")
        #plotter.plot(tree_train_err, tree_test_err, "Tree-wise Train/Test Errors (AdaBoost)")
        pass
    
        
