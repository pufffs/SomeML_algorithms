"""
pf
"""
import numpy as np
from .Trees import ClassifierTree

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
    
class Bootstrap(object):
    def __init__(self, X, Y, optimizer=ClassifierTree):
        self.X = X
        self.Y = Y
        self.optimizer = optimizer
        self.n = X.shape[0]
        self.forests = None
    
    def get_boot(self, B=100):
        boot = []
        for i in range(B):
            ind = np.random.choice(np.arange(self.n), self.n, replace=True)
            obs = self.X[ind]
            response = self.Y[ind]
            mat = (response, obs)
            boot.append(mat)
        return boot
    
    def bagging_fit(self, B=100):
        self.forests = []
        boot_data = self.get_boot(B)
        for mat in boot_data:
            y, x = mat
            obj = self.optimizer()
            obj.fit(x, y)
            self.forests.append(obj)
    
    def tree_predict(self, Xtest, infer_type="Consensus"):
        models = self.forests
        
        if infer_type == "Probability": #if probability, end up earlier
            ag = np.array([model.prob_predict(Xtest) for model in models])
            ag = np.mean(ag, axis=0) # probability over B bootstrap samples
            return np.argmax(ag, axis=1) # choose the class with the highest average probability
        ag = np.array([model.predict(Xtest) for model in models])
        preds = []
        if infer_type == "Average": # if regression tree, directly return the average without calling the rest of the codes.
            return np.mean(ag, axis=0)
        clss = np.unique(self.Y)
        for pred in ag.T:
            counts = np.array([np.sum(pred==i) for i in clss])
            cls_p = np.argmax(counts)
            preds.append(cls_p)
        return np.array(preds)
        
    def OOB_error(self, B=100, error_type="Misclassification"): #Out-of-bag error used to estimate the model error,                                          
        bh = {i:[] for i in np.arange(self.n)}              #similar to 2-fold cross-validation.
        for i in range(B):
            ind = np.random.choice(np.arange(self.n), self.n, replace=True)
            obs = self.X[ind]
            response = self.Y[ind]
            obj = self.optimizer()
            obj.fit(obs, response)
            for j in np.arange(self.n):
                if j not in ind:
                    bh[j].append(obj)
                    
        Pred = []    
        if error_type == "Misclassification":
            for I in np.arange(self.n):
                data = np.atleast_2d(self.X[I])
                ag = np.array([model.predict(data) for model in bh[I]])
                cls = np.unique(self.Y)
                counts = np.array([np.sum(ag==i) for i in cls])
                pp = np.argmax(counts)
                Pred.append(pp)
            return np.mean(np.array(Pred) != self.Y)
        
        elif error_type == "MSE":
            for I in np.arange(self.n):
                data = self.X[I]
                if bh[I]:
                    ag = np.array([ model.predict(data) for model in bh[I]])
                    Pred.append(np.mean(ag))
            return np.mean(np.square(np.array(Pred)-self.Y))
        else:
            return 0.
        
        
        
        
        
        
        
        
        
        