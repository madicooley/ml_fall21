'''

'''

import sys, copy, statistics, random
import numpy as np
import pandas as pd

from os.path import dirname,realpath
sys.path.insert(0, dirname(realpath(__file__))[:-13])

import utils.utils as utils
import utils.tree as tree


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


def max_gain_attr(S, Dt, attrs, attr_selection, random_forest_split, weighted=False): 
    '''
    
    '''
    #Y = np.array(S['y']) * Dt
    Y = S['y'].tolist()
    S_measure = get_attr_measure(Y, attr_selection, weights=Dt, weighted=weighted) 
    S_len = sum(Dt)
    #S_len = len(S)
    
    max_gain = 0
    split_attr_i = 0
    i = 0
    
    attributes = copy.deepcopy(attrs) 
    if random_forest_split is not None:
        weighted=False
        n_attrs = len(attributes)
        
        if n_attrs > random_forest_split:
            inds = random.sample(attributes, random_forest_split)
            attributes = inds
        
        # remove non-chosen feature from attributes
        while True:
            rem=False
            i=0
            for a in attrs:
                if a not in attributes:
                    rem=True
                    break
                i+=1
            if rem:
                del attrs[i]
            else:
                break
                
    i=0
    for A in attributes:
        for attr, values in A.items():
            nan_vals = False
            nan_row_inds = list(S.loc[pd.isna(S[attr]), :].index)
            if len(nan_row_inds) > 0:
                nan_vals=True
            
            sum_v = 0
            if not nan_vals:
                for attr_value in values:  
                    Sv = S[S[attr]==attr_value]
                    #Dtv = Dt[S[attr]==attr_value]
                    Dtv = [1]*len(Sv)              #WARNING
                    Sv_len = sum(Dtv)
                    #Sv_len = len(Sv)
                    
                    if Sv_len > 0:
                        # when a sample has label
                        Sv_measure = get_attr_measure(Sv['y'].tolist(), 
                                                      attr_selection,
                                                      weights=Dtv,
                                                      weighted=weighted)
                        
                        sum_v += (Sv_len/S_len)*Sv_measure
            else:
                sum_v = compute_fractional_feat(S, attr, values, nan_row_inds[0]) 
                # NOTE know theres only one for now
        
        
        gain = S_measure - sum_v
        
        if gain>max_gain:
            max_gain=gain
            split_attr_i=i
        i+=1
    
    attr = list(attrs[split_attr_i].keys())[0]    
    attr_vals = attrs[split_attr_i][attr]
    del attrs[split_attr_i]
    return attr, list(attr_vals) 


def get_attr_measure(labels, attr_selection, weights=None, weighted=False):
    if attr_selection=='info_gain':
        return information_gain(labels, weights, weighted=weighted)
    elif attr_selection=='maj_error':
        return majority_error(labels)
    elif attr_selection=='gini_ind':
        return gini_index(labels)
    
    
def get_weighted_label(label, Dt):
    '''
        sum of weighted predictions
    '''
    unique = set(label)   
    maj_label = None
    max_lab_weight = 0
    for u_lab in unique:
        i=0
        lab_weight = 0
        for y_true in label:
            if y_true == u_lab:
                lab_weight+=Dt[i]
            i+=1
        
        if lab_weight > max_lab_weight:
            max_lab_weight = lab_weight
            maj_label = u_lab
    
    return maj_label
    #return (np.array(labels).dot(np.array(Dt)))/len(labels)
    
    

def get_majority_label(label):
    '''
    args:
        Dt - weights of each sample 
    '''
    unique = set(label)        
    freq = [label.count(x) for x in unique]
    max_value = max(freq)
    ind = freq.index(max_value)
    return label[ind]
    

def entropy(labels, weights=None, weighted=False):
    '''
    Entropy(S) = H(S) = -p_+ * log(p_+) - p_- * log(p_-)
    
    • The proportion of positive examples is p_+
    • The proportion of negative examples is p_-
    '''    
    entrpy = 0
    n = len(labels)
    #values, counts = np.unique(sorted(labels), return_counts=True)
    values, counts = np.unique(labels, return_counts=True)
    
    if not weighted:
        for c in counts:
            entrpy -= (c/n)*np.log2((c/n))
    else:
        for lab_val in values: # - label / + label
            p = 0
            i=0
            for lab in labels:
                if lab==lab_val:
                    p += weights[i]
                i+=1
            entrpy -= p * np.log2(p)
    
    return entrpy
    

def information_gain(labels, weights=None, weighted=False):
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
    return entropy(labels, weights, weighted=weighted)
    

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


def ID3(S, **kwargs):
    '''
    returns:
        leaf node - recursively constructs decision tree of max_depth
        
    args:
        S: set of examples
        Dt: weights of corresponding training examples
        model: the decision tree object
        attrs: set of remaining attributes for samples S
        attr_selection: gain measure for attribute attribute selection 
        max_depth: maximum depth of decision tree
        edge_attr: attribute associated w. the parent node to child node 
        random_forest_split: size of attribute subset to randomly sample for rand forest
    '''        
    edge_attr, random_forest_split, Dt = None, None, None
    weighted_samples = False
        
    model               = kwargs.get('model')
    attrs               = kwargs.get('attributes')
    attr_selection      = kwargs.get('attr_selection')
    max_depth           = kwargs.get('max_depth')
    edge_attr           = kwargs.get('edge_attr') 
    random_forest_split = kwargs.get('random_forest_split')
    Dt                  = kwargs.get('Dt')
    weighted_samples    = kwargs.get('weighted_samples')
    
            
    if Dt is None:
        Dt = np.array([1]*len(S))
        
    labels = S['y'].tolist()
    if max_depth <= 0:
        attrs = []

    if len(set(labels)) == 1: 
        # return leaf node w. label
        leaf = tree.Node(leaf_label=labels[0])
        return leaf
    elif len(attrs) <= 1:
        label = None
        if weighted_samples:
            label = get_weighted_label(labels, Dt)
        else:
            # return leaf node w. most common label
            label = get_majority_label(labels)
            
        leaf = tree.Node(leaf_label=label)
        return leaf
    else:
        # find attribute that best splits S
        attr, A_values = max_gain_attr(S, Dt, attrs, attr_selection, random_forest_split)
        A_values = {x for x in A_values if x==x}
        
        root_node = tree.Node(splitting_attr=attr)
        root_index = model.add_node(node=root_node) # create root node for tree
        
        for attr_value in A_values:
            # add new tree branch corresponding to A=v    
            inds = ((S[attr]==attr_value) | (S[attr].isna())).tolist()            
            Sv = S[inds]
            Dtv = Dt[inds]
                        
            if len(Sv) < 1: 
                label=None
                if weighted_samples:
                    label = get_weighted_label(labels, Dt)
                else:
                    # return leaf node w. most common label
                    label = get_majority_label(labels)
                    
                # add leaf node w. most common value of label in S 
                leaf_node = tree.Node(leaf_label=label)
                leaf_index = model.add_node(node=leaf_node)
                model.add_edge(leaf_index, root_index, edge_attr=attr_value)
            else: 
                # below this branch add the subtree ID3(Sv, attrs-{A}, labels)
                node = ID3(Sv, 
                           Dt=Dtv,
                           model=model,
                           attributes=copy.deepcopy(attrs), 
                           attr_selection=attr_selection, 
                           max_depth=max_depth-1, 
                           edge_attr=attr_value,
                           random_forest_split=random_forest_split,
                           weighted_samples=weighted_samples)
                
                subtree_index = model.add_node(node=node)
                model.add_edge(subtree_index, root_index, edge_attr=attr_value)
                
        return root_node


