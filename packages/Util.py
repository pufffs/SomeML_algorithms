import numpy as np
import math
from .Regularizers import Regularizer

def GD(w0, optimizer, opts=dict()):
    w = w0
    dim = w0.size
    n_samples = optimizer.n
    
    eta0 = opts.get('eta0', 0.1)
    ite = opts.get('iter', 50)
    bs = opts.get('batchsize', 10)
    lrmode = opts.get('learning rate', 'None')
    
    trajectory = np.zeros((ite+1, dim))
    trajectory[0] = w0
    gradsum = 0
    for i in range(ite):
        index = np.random.choice(n_samples, bs)
        grad = optimizer.gradient(w, index)
        if lrmode == "Adagrad":
            gradsum += np.sum(np.square(grad))
            eta = eta0 / np.sqrt(gradsum)
        elif lrmode == "Annealing":
            eta = eta0 / np.power(i+1, 0.8)
        else:
            eta = eta0
            
        w = w - eta * grad
        trajectory[i+1] = w
    return trajectory

def CrossVal(x_full, y_full, folds, optimizer, closed_f=False, opts=dict()):
    """
    x_full, y_full: Full dataset.
    
    folds: the number of folds.
    
    optimizer: the function/feature space.
    
    opts: some parameters required by the optimizer.
    """
    
    try:
        x, y = np.split(x_full, folds), np.split(y_full, folds) #split the full dataset into folds, needed to be divided.
    except:
        raise
    error = 0 #total CV error.
    for i in range(folds):
        x_ = x[:]
        y_ = y[:]
        x_val = x_[i] 
        y_val = y_[i]
        x_.pop(i)
        y_.pop(i)
        
        x_train = np.concatenate(x_) if folds !=1 else x_val
        y_train = np.concatenate(y_) if folds !=1 else y_val
        
        
        if closed_f:
            obj = optimizer(x_train, y_train, opts.get('reg', 0), opts.get('features', 'linear'))
            error += obj.test_loss(x_val, y_val)
        else:
            obj = optimizer(x_train, y_train)
            w0 = np.random.randn(x_full.shape[1])
            w_hat = GD(w0, obj)[-1]
            error += obj.test_loss(x_val, y_val, w_hat)
    return error / folds
        
        
        
        
        
        
        
        
        
        
        
        