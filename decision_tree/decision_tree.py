'''

'''

import sys, copy, statistics
import numpy as np
import pandas as pd

from os.path import dirname,realpath
sys.path.insert(0, dirname(realpath(__file__))[:-13])

import utils.utils as utils
import utils.tree as tree


def get_error(df, tree):    
    n = df.shape[0]
    miss_predictions = 0
    i=0
    for index, x in df.iterrows():
        y_hat = tree.make_prediction(x)
        if y_hat is None:
            print('ERROR')
            
        if y_hat != df['y'][i]:
            miss_predictions+=1
        i+=1
    
    return miss_predictions/n
        

def compute_fractional_feat(S, attr, values, nan_row_ind):
    values = {x for x in values if x==x}
    label = S['y'][nan_row_ind]    
    S_len = len(S)-1
    
    sum_v = 0
    for attr_value in values:
        Sv = S[S[attr]==attr_value]
        Sv_len = len(Sv)
        Sv_prime = Sv_len + (Sv_len/S_len)
        
        num_pos = len(Sv[Sv['y'] == '+']) 
        num_neg = len(Sv[Sv['y'] == '-']) 
        
        if label == '+':
            c_neg = num_neg
            c_pos = num_pos + (Sv_len/S_len)
        elif label == '-':
            c_neg = num_neg + (Sv_len/S_len)
            c_pos = num_pos
        
        if c_pos != 0:
            pos = -((c_pos/Sv_prime)*np.log2((c_pos/Sv_prime)))
        else:
            pos = 0
            
        if c_neg != 0:
            neg = - ((c_neg/Sv_prime)*np.log2((c_neg/Sv_prime)))
        else:
            neg = 0
            
        frac_entrpy = pos + neg
        
        sum_v += (Sv_prime/S_len) * frac_entrpy
    return sum_v
    
  
def max_gain_attr(S, attrs, attr_selection):    
    S_measure = get_attr_measure(S['y'].tolist(), attr_selection)
    S_len = len(S)
    
    max_gain = 0
    split_attr_i = 0
    i = 0
    
    for A in attrs:
        for attr, values in A.items():
            nan_vals = False
            nan_row_inds = list(S.loc[pd.isna(S[attr]), :].index)
            if len(nan_row_inds) > 0:
                nan_vals=True
            
            sum_v = 0
            if not nan_vals:
                for attr_value in values:  
                    Sv = S[S[attr]==attr_value]
                    Sv_len = len(Sv)
                    
                    if Sv_len > 0:
                        # when a sample has label
                        Sv_measure = get_attr_measure(Sv['y'].tolist(), attr_selection)
                        sum_v += (Sv_len/S_len)*Sv_measure
            else:
                sum_v = compute_fractional_feat(S, attr, values, nan_row_inds[0]) # NOTE know theres only one for now
                
        gain = S_measure - sum_v
        #print('\n$Gain(S, A={}) = H(S)-\\sum_{{v\in vals(A)}}\\frac{{|S_v|}}{{|S|}} = {} - {}={}$'.format(attr, 
                                                                                                          #round(S_measure, 3), 
                                                                                                          #round(sum_v, 3), 
                                                                                                          #round(gain, 3)))
        if gain>max_gain:
            max_gain=gain
            split_attr_i=i
        i+=1
    
    attr = list(attrs[split_attr_i].keys())[0]    
    attr_vals = attrs[split_attr_i][attr]
    del attrs[split_attr_i]
    
    #print('\nChosen splitting $A \\text{=}', end='')
    #print('{}$'.format(attr))
    return attr, list(attr_vals) 


def get_attr_measure(labels, attr_selection):
    if attr_selection=='info_gain':
        return information_gain(labels)
    elif attr_selection=='maj_error':
        return majority_error(labels)
    elif attr_selection=='gini_ind':
        return gini_index(labels)
    

def get_majority_label(label):
    unique = sorted(set(label))
    freq = [label.count(x) for x in unique]
    max_value = max(freq)
    ind = freq.index(max_value)
    return label[ind]


def entropy(labels):
    '''
    Entropy(S) = H(S) = -p_+ * log(p_+) - p_- * log(p_-)
    
    • The proportion of positive examples is p_+
    • The proportion of negative examples is p_-
    '''
    entrpy = 0
    n = len(labels)
    values, counts = np.unique(sorted(labels), return_counts=True)
    
    for c in counts:
        entrpy -= (c/n)*np.log2((c/n))

    return entrpy
    

def information_gain(labels):
    '''
    S       : data subset
    attrs   : list of attributes
    
    Information gain of an attribute A is the expected reduction 
    in entropy (expected increase of purity) caused by partitioning 
    on this attribute
    
    Entropy of partitioning the data is calculated by weighing the 
    entropy of each partition by its size relative to the original set
    
    choose feature w. maximum information gain
    '''
    return entropy(labels)
    

def majority_error(labels):
    '''
    choose feature w. maximum information gain
    '''    
    values, counts = np.unique(sorted(labels), return_counts=True)
    S_len = len(labels)
    ME_S = (S_len-max(counts)) / S_len
    
    return ME_S


def gini_index(labels):
    '''
    choose feature w. maximum information gain
    '''
    values, counts = np.unique(sorted(labels), return_counts=True)
    S_len = len(labels)
    
    GI_S = 1
    for c in counts:
        GI_S -= (c/S_len)**2
    
    return GI_S


def ID3(S, attrs, attr_selection, max_depth, edge_attr=None):
    global d_tree
    
    #print('\n____________________________________________________________________\nS: ')
    #print(S)
    #print('\nA: ', end='')
    #for a in attrs:
        #print(' ', list(a.keys())[0], end='')
    #print()
    
    labels = S['y'].tolist()
    if max_depth <= 0:
        attrs = []

    if len(set(labels)) == 1: 
        # return leaf node w. label
        leaf = tree.Node(leaf_label=labels[0])
        return leaf
    elif len(attrs) <= 1:
        # return leaf node w. most common label
        leaf = tree.Node(leaf_label=get_majority_label(labels))
        return leaf
    else:
        # find attribute that best splits S
        attr, A_values = max_gain_attr(S, attrs, attr_selection)
        A_values = {x for x in A_values if x==x}
        
        root_node = tree.Node(splitting_attr=attr)
        root_index = d_tree.add_node(node=root_node) # create root node for tree
        
        for attr_value in A_values:
            # add new tree branch corresponding to A=v    
            Sv = S[(S[attr]==attr_value) | (S[attr].isna())]
            
            if len(Sv) < 1: 
                # add leaf node w. most common value of label in S 
                leaf_node = tree.Node(leaf_label=get_majority_label(labels))
                leaf_index = d_tree.add_node(node=leaf_node)
                d_tree.add_edge(leaf_index, root_index, edge_attr=attr_value)
            else: 
                # below this branch add the subtree ID3(Sv, attrs-{A}, labels)
                node = ID3(Sv, copy.deepcopy(attrs), attr_selection, max_depth-1, edge_attr=attr_value)
                subtree_index = d_tree.add_node(node=node)
                d_tree.add_edge(subtree_index, root_index, edge_attr=attr_value)
                
        return root_node

    

##############################################################################################

def preprocess_pd(df, numeric_features=False, most_common=False, most_label=False, unknown_missing=False):      
    attributes = []
    col_names = list(df.columns)
    
    if col_names[-1] != 'y':
        df.rename( {col_names[-1] : 'y'})
        col_names = list(df.columns)
        
    # convert numeric features to bool using threshold
    if numeric_features:        
        for col in col_names:
            if df[col].dtype == 'int64':
                median = df[col].median()
                for index, val in df[col].items():
                    if val<= median:
                        df.iloc[index, col]='less'
                    else:
                        df.iloc[index, col]='more'
        
    i=0
    for col in col_names:
        if col != 'y':
            attributes.append({col : set(df[col].unique()) })
        i+=1
    
    # fill in missing sample feature values
    if unknown_missing or most_common or most_label:
        for col in col_names:
            missing = list(df.loc[pd.isna(df[col]), :].index)
            
            if unknown_missing:  # append unknown indices to missing list
                unknwn = df.index[df[col]=='unknown'].tolist()
                missing = missing + unknwn
            
            for miss in missing: 
                val_counts = None
                
                # fill missing feature values w. most common val in that column
                if most_common:
                    val_counts = [[val, count] for val, count in df[col].value_counts().items() if val!='unknown']
                    
                # fill missing feature vals. w. most common among same labels
                elif most_label:    
                    missing_label = df['y'][miss]                
                    df_sub = df[df['y']==missing_label][col]
                    val_counts = [[val, count] for val, count in df_sub.value_counts().items() if val!='unknown']
                    
                fill_value = val_counts[0][0]
                df[col][miss] = fill_value
            
    return attributes


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
    
    attributes = preprocess_pd(df_train, most_common=most_common, most_label=most_label)
    
    ID3(df_train, attributes, attr_selection, 5)
    
    #train_error = get_error(df_train, d_tree)
    d_tree.print_tree()


def run_exps(df_train, df_test, attr_selection, depth_iters, numeric_features=False, unknown_missing=False, most_common=True):
    attributes = preprocess_pd(df_train, numeric_features=numeric_features, 
                               unknown_missing=unknown_missing, most_common=most_common)
    
    preprocess_pd(df_test, numeric_features=numeric_features, 
                  unknown_missing=unknown_missing, most_common=most_common)
    
    print('Max Depth & & Info Gain & & Maj Error & & Gini Ind \\\\')
    print(' & train & test & train & test & train & test \\\\')
    for max_depth in range(1, depth_iters+1):
        print(max_depth, ' & ', end='')
        for attr_slct in attr_selection:
            global d_tree
            d_tree = tree.Tree()

            ID3(df_train, copy.deepcopy(attributes), attr_slct, max_depth)
            
            train_error = get_error(df_train, d_tree)
            test_error = get_error(df_test, d_tree)
            
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
    
    
