'''

- we have 6 car attributes
- the label is the evaluation of the car
- all the attributes are categorical
- the training data consist of 1,000 examples
- the test data comprise 728 examples

'''

import sys, copy, statistics
import numpy as np

from os.path import dirname,realpath
sys.path.insert(0, dirname(realpath(__file__))[:-13])

import utils.utils as utils


class Tree():
    def __init__(self):
        self.adjacency_list = []
    
    def add_node(self, node=None):
        if node.index is None:
            self.adjacency_list.append(node)
            node.index = len(self.adjacency_list)-1
        return node.index
    
    def add_edge(self, child_index, parent_index, edge_attr=None):
        edge = Edge(child_index, attr_value=edge_attr)
        self.adjacency_list[parent_index].children.append(edge)
    
    def make_prediction(self, x):
        y_hat = None
        i = 0
        
        while True:
            node = self.adjacency_list[i]
            
            if len(node.children) <= 0:
                y_hat = node.leaf_label
                break
                
            x_attr_value = x[node.splitting_attr]            
            for child in node.children:
                if child.attr_value == x_attr_value:
                    i = child.child_node
        return y_hat
                
    
    def print_tree(self):
        print('\n--------------------------------------tree:')
        i=0
        for node in self.adjacency_list:
            print(i, ': ', end='')
            node.print_node()
            print()
            i+=1
        print('--------------------------------------')


class Node():
    def __init__(self, splitting_attr=None, leaf_label=None):
        self.index = None
        self.splitting_attr = splitting_attr        # splitting attr value 
        self.leaf_label = leaf_label                # leaf_label
        self.children = []
    
    def print_node(self):
        print('___________________')
        print('node index= ', self.index)
        #print('parent node = ', self.parent)
        print('splitting atrr=', self.splitting_attr)
        print('leaf label=', self.leaf_label)
        print('children: [', end='')
        for child in self.children:
            print('(', child.child_node, child.attr_value, ') , ', end='')
        print(' ]\n______________________\n')
        
        
class Edge():
    def __init__(self, child_node, attr_value=None):
        self.child_node = child_node
        self.attr_value = attr_value


def get_error(X, Y, tree):
    miss_predictions = 0
    i=0
    for x in X:
        y_hat = tree.make_prediction(x)
        if y_hat is None:
            print('ERROR')
            
        if y_hat != Y[i]:
            miss_predictions+=1
        i+=1
    
    return miss_predictions/len(X)
        
 
def get_attr_instances(S, labels, v, key):
    '''
    key : attribute index
    v : attribute value
    
    returns list of labels of samples with attr=v
    '''
    S_inds = [i for i in range(len(S)) if S[i][key]==v]
    Sv = [S[x] for x in S_inds]
    labels_S = [labels[x] for x in S_inds]
    
    return Sv, labels_S
    

def max_gain_attr(S, labels, attrs, attr_selection):
    S_measure = get_attr_measure(labels, attr_selection)
    S_len = len(S)
    
    max_gain = 0
    split_attr_i = 0
    i = 0
    
    for A in attrs:
        for key, values in A.items():
            sum_v = 0
            for v in values:                
                Sv, labels_Sv = get_attr_instances(S, labels, v, key)
                Sv_len = len(Sv)
                
                if Sv_len > 0:
                    # when a sample has label
                    Sv_measure = get_attr_measure(labels_Sv, attr_selection)
                    sum_v += (Sv_len/S_len)*Sv_measure
                else:
                    # when a label is missing
                    pass
                
        gain = S_measure - sum_v
        
        if gain>max_gain:
            max_gain=gain
            split_attr_i=i
        i+=1
    
    attr = list(attrs[split_attr_i].keys())[0]    
    attr_vals = attrs[split_attr_i][attr]
    del attrs[split_attr_i]
    
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
        entrpy -= (c/n)*np.log((c/n))

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


def ID3(max_depth, S, attrs, labels, attr_selection, edge_attr=None, unknown_as_missing=False):
    global tree, attributes
            
    if max_depth <= 0:
        attrs = []

    if len(set(labels)) == 1: 
        # return leaf node w. label
        leaf = Node(leaf_label=labels[0])
        return leaf
    elif len(attrs) < 1:
        # return leaf node w. most common label
        leaf = Node(leaf_label=get_majority_label(labels))
        return leaf
    else:
        # find attribute that best splits S
        A, A_values = max_gain_attr(S, labels, attrs, attr_selection)
        root_node = Node(splitting_attr=A)
        root_index = tree.add_node(node=root_node) # create root node for tree
        
        for v in A_values:
            # add new tree branch corresponding to A=v            
            Sv, label_v = get_attr_instances(S, labels, v, A)
            
            if len(Sv) < 1: 
                # add leaf node w. most common value of label in S 
                leaf_node = Node(leaf_label=get_majority_label(labels))
                leaf_index = tree.add_node(node=leaf_node)
                tree.add_edge(leaf_index, root_index, edge_attr=v)
            else: 
                # below this branch add the subtree ID3(Sv, attrs-{A}, labels)
                node = ID3(max_depth-1, Sv, copy.deepcopy(attrs), label_v, attr_selection, edge_attr=v)
                subtree_index = tree.add_node(node=node)
                tree.add_edge(subtree_index, root_index, edge_attr=v)
        
        return root_node


def problem_two():
    '''
        Decision Tree Practice - problem 2
    '''
    attr_selection = ['info_gain', 'maj_error', 'gini_ind']
    
    #x_train, y_train = utils.readin_dat('decision_tree/data/car/', 'train.csv')
    x_train, y_train = utils.readin_dat('decision_tree/data/car/', 'shapes_dat.csv')
    x_test, y_test = utils.readin_dat('decision_tree/data/car/', 'test.csv')
    
    global attributes
    #attributes = [{0 : set()}, # 'buying'
                  #{1 : set()}, # 'maint' 
                  #{2 : set()}, # 'doors'
                  #{3 : set()}, # 'persons'
                  #{4 : set()}, # 'lug_boot'
                  #{5 : set()}] # 'safety'
                  
    #attributes = [{i : set()} for i in range(1,7)]
    attributes = [{0 : set()}, {1 : set()}]
    
    for x in x_train:
        i=0
        for attr in x:
            k = list(attributes[i].keys())[0]
            attributes[i][k].add(attr)
            i+=1
    print(attributes, '\n')
    
    for max_depth in range(1, 7):
        for attr_slct in attr_selection:
            global tree
            tree = Tree()
            
            ID3(max_depth, x_train, copy.deepcopy(attributes), y_train, attr_slct)
            #tree.print_tree()
            
            train_error = get_error(x_train, y_train, tree)
            print('training error: ', train_error, attr_slct, max_depth)
    

def problem_three():
    '''
        Decision Tree Practice - problem 3
    '''
    attr_selection = ['info_gain', 'maj_error', 'gini_ind']
    
    x_train_a, y_train = utils.readin_dat('decision_tree/data/bank/', 'train.csv')
    x_test_a, y_test = utils.readin_dat('decision_tree/data/bank/', 'test.csv')
    
    x_train_b = copy.deepcopy(x_train_a)
    x_test_b = copy.deepcopy(x_test_a)
    
    global attributes
    attributes = []
    
    all_attribute_vals = []
    for i in range(16):
        all_attribute_vals.append({i : []})
        
    for x in x_train_a:
        i=0
        for attr in x:
            k = list(all_attribute_vals[i].keys())[0]
            if attr.isnumeric():
                attr = int(attr)
            elif len(attr.split('-')) > 1:
                if attr.split('-')[1].isnumeric():
                    attr = int(attr)                
            all_attribute_vals[i][k].append(attr)
            i+=1
    
    ## preprocess input data for part a
    i=0
    for attr in all_attribute_vals:
        l = list(attr.values())[0]
        if isinstance(l[0], int):
            sorted_values = sorted(l)
            med = statistics.median(sorted_values)
            attributes.append({i : {"less", "greater"}})
            
            for j in range(len(x_train_a)):
                if int(x_train_a[j][i]) <= med:
                    x_train_a[j][i] = "less"
                else:
                    x_train_a[j][i] = "greater"
                    
            for j in range(len(x_test_a)):
                if int(x_test_a[j][i]) <= med:
                    x_test_a[j][i] = "less"
                else:
                    x_test_a[j][i] = "greater"
        else:
            attributes.append({i : set(l)})
        i+=1
    
    ############ 3.a
    print('Max Depth & & Measure 1 & & Measure 2 & & Measure 3 //')
    print(' & train & test & train & test & train & test //')
    for max_depth in range(1, 17):
        print(max_depth, ' & ', end='')
        for attr_slct in attr_selection:
            global tree
            tree = Tree()
            ID3(max_depth, x_train_a, copy.deepcopy(attributes), y_train, attr_slct)
            
            train_error = get_error(x_train_a, y_train, tree)
            test_error = get_error(x_test_a, y_test, tree)
            
            print(train_error, ' & ', test_error, ' & ', end='')
        print(' //') 
            
            
    ############ 3.b
    ## preprocess input data for part b
    i=0
    for attr in all_attribute_vals:
        l = list(attr.values())[0]
        if isinstance(l[0], int):
            sorted_values = sorted(l)
            med = statistics.median(sorted_values)
            attributes.append({i : {"less", "greater"}})
            
            for j in range(len(x_train_b)):
                if int(x_train_b[j][i]) <= med:
                    x_train_b[j][i] = "less"
                else:
                    x_train_b[j][i] = "greater"
                    
            for j in range(len(x_test_b)):
                if int(x_test_b[j][i]) <= med:
                    x_test_b[j][i] = "less"
                else:
                    x_test_b[j][i] = "greater"
        else:
            attributes.append({i : set(l)})
        i+=1       
            
            
    print('Max Depth & & Measure 1 & & Measure 2 & & Measure 3 //')
    print(' & train & test & train & test & train & test //')
    for max_depth in range(1, 17):
        print(max_depth, end='')
        for attr_slct in attr_selection:
            global tree
            tree = Tree()
            ID3(max_depth, x_train_b, copy.deepcopy(attributes), y_train, attr_slct, unknown_as_missing=True)
            
            train_error = get_error(x_train_b, y_train, tree)
            test_error = get_error(x_test_b, y_test, tree)
            
            print(train_error, test_error, end='')
        print(' //') 
            

def main():
    #problem_two()
    problem_three()


if __name__=="__main__":
    main()
    
    
    
    
    
    
