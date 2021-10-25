'''
    Decision tree datastructure
'''


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
