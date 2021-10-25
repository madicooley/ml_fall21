
import numpy as np


def delta_mse(X, w, y, n, m):
    '''
    
    args:
        X - [n x m]
        w - [m x 1]
        y - [n x 1]
    '''
    #ws = []
    #for j in range(m):
        #sm = 0
        #for i in range(n):
            #sm += ( y[i] - np.dot(w, X[i, :]) ) * X[i, j]
        #ws.append(-1*sm)
    
    #print(ws)
    # ( [n x 1] - ([m x 1] * [m x n].T ) ) * [n x m]
    
    y_pred = np.dot(X, w)
    return -1 * np.dot( (y - y_pred ),  X)
    
    
def mean_squared_error(X, w, y):
    #cost = 0
    #for i in range(len(X)):
        #cost += (y[i] - np.dot(w, X[i]))**2
    #return 0.5*cost

    return 0.5*np.dot((y - np.dot(w, X.T)).T, (y - np.dot(w, X.T)))
        
    
