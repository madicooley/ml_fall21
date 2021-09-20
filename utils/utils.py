


def readin_dat(directory, fname):
    '''
    
    '''
    x = []
    y = []
    
    with open ( directory+fname , 'r') as f:
        for line in f:
            samp = line.strip().split(',')
            x.append(samp[0:-1])
            y.append(samp[-1])
        
    return x, y 
    
