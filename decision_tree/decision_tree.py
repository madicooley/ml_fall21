'''

- we have 6 car attributes
- the label is the evaluation of the car
- all the attributes are categorical
- the training data consist of 1,000 examples
- the test data comprise 728 examples

'''

import sys, copy

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


def get_best_attribute(attrs):
    i = 0                                  
    attr = list(attrs[i].keys())[0]    
    attr_vals = attrs[i][attr]
    
    del attrs[i]
    return attr, list(attr_vals) 

def get_majority_label(label):
    unique = sorted(set(label))
    freq = [label.count(x) for x in unique]
    max_value = max(freq)
    ind = freq.index(max_value)
    return label[ind]
 
def information_gain():
    pass

def majority_error():
    pass

def gini_index():
    pass


def ID3(max_depth, S, attrs, label, edge_attr=None):
    global tree, attributes

    if len(set(label)) == 1:                    # return leaf node w. label
        leaf = Node(leaf_label=label[0])
        return leaf
    
    elif len(attrs) < 1:                        # return leaf node w. most common label
        leaf = Node(leaf_label=get_majority_label(label))
        return leaf
    
    else:
        # find attribute that best splits S
        A, A_values = get_best_attribute(attrs)
        root_node = Node(splitting_attr=A)
        root_index = tree.add_node(node=root_node)          # create root node for tree
        
        for v in A_values:                                  # add new tree branch corresponding to A=v            
            # Sv be subset of examples in S w. A=v
            inds = [i for i in range(len(S)) if S[i][A]==v]            
            Sv = [S[x] for x in inds]
            label_v = [label[x] for x in inds]
            
            if len(Sv) < 1:                     # add leaf node w. most common value of label in S 
                leaf_node = Node(leaf_label=get_majority_label(label))
                leaf_index = tree.add_node(node=leaf_node)
                tree.add_edge(leaf_index, root_index, edge_attr=v)
            else:                               # below this branch add the subtree ID3(Sv, attrs-{A}, label)
                node = ID3(max_depth, Sv, copy.deepcopy(attrs), label_v, edge_attr=v)
                subtree_index = tree.add_node(node=node)
                tree.add_edge(subtree_index, root_index, edge_attr=v)
        
        return root_node


def problem_two():
    '''
        Decision Tree Practice - problem 2
    '''
    
    #x_train, y_train = utils.readin_dat('decision_tree/data/car/', 'train.csv')
    x_train, y_train = utils.readin_dat('decision_tree/data/car/', 'shapes_dat.csv')
    x_test, y_test = utils.readin_dat('decision_tree/data/car/', 'test.csv')
    
    global attributes
    attributes = [{0 : set()}, # 'buying'
                  {1 : set()}, # 'maint' 
                  {2 : set()}, # 'doors'
                  {3 : set()}, # 'persons'
                  {4 : set()}, # 'lug_boot'
                  {5 : set()}] # 'safety'
                  
    attributes = [{0 : set()}, {1 : set()}]
    
    for x in x_train:
        i=0
        for attr in x:
            k = list(attributes[i].keys())[0]
            attributes[i][k].add(attr)
            i+=1
    print(attributes, '\n')
    
    global tree
    tree = Tree()
    
    max_depth = 1
    ID3(max_depth, x_train, copy.deepcopy(attributes), y_train)
    
    tree.print_tree()
    
    

def problem_three():
    pass


def main():
    problem_two()
    #problem_three()


if __name__=="__main__":
    main()
    
    
    
    
    
    
