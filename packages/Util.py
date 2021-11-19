"""
pf
"""
import numpy as np

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

class CV(object):
    def __init__(self, x_full, y_full, folds, optimizer):
        self.X = x_full
        self.Y = y_full
        self.folds = folds
        self.optimizer = optimizer

    def CrossVal(self, closed_f=False, opts=dict()):
        """
        x_full, y_full: Full dataset.
        
        folds: the number of folds.
        
        optimizer: the function/feature space.
        
        opts: some parameters required by the optimizer.
        """

        x, y = np.array_split(self.X, self.folds), np.array_split(self.Y, self.folds) 
        error = 0 #total CV error.
        for i in range(self.folds):
            x_ = x.copy()
            y_ = y.copy()
            x_val = x_[i] 
            y_val = y_[i]
            x_.pop(i)
            y_.pop(i)
            
            x_train = np.concatenate(x_) if self.folds !=1 else x_val
            y_train = np.concatenate(y_) if self.folds !=1 else y_val
            
            if closed_f:
                obj = self.optimizer(x_train, y_train, opts.get('reg', 0), opts.get('features', 'linear'))
                error += obj.test_loss(x_val, y_val)
            else:
                obj = self.optimizer(x_train, y_train)
                w0 = np.random.randn(self.X.shape[1])
                w_hat = GD(w0, obj)[-1]
                error += obj.test_loss(x_val, y_val, w_hat)
        return error / self.folds
        
    def Tree_CV(self, opts={}):
        
        error_type = opts.get("error_type", "RSS")
        alpha = opts.get("alpha", 0.)
        max_depth = opts.get("max_depth", 5)
        mn = opts.get("mn", 5)
        
        x, y = np.array_split(self.X, self.folds), np.array_split(self.Y, self.folds) 
        error = 0 #total CV error.
        for i in range(self.folds):
            x_ = x.copy()
            y_ = y.copy()
            x_val = x_[i] 
            y_val = y_[i]
            x_.pop(i)
            y_.pop(i)
            
            x_train = np.concatenate(x_) if self.folds !=1 else x_val
            y_train = np.concatenate(y_) if self.folds !=1 else y_val   
            
            obj = self.optimizer(max_depth,mn, alpha)
            obj.fit(x_train, y_train)
            pred = obj.predict(x_val)
        
            if error_type == "RSS":
                error += np.sum(np.square(pred-y_val))
            elif error_type == "Misclassification":
                error += 1. - ( (pred==y_val).sum() / y_val.size )
        return error/self.folds
        
        
        
        
        
        
        
        
        
        
        