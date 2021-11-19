"""
pf
"""
import numpy as np

class Nodes(object):
    def __init__(self, pred, depth):
        self.feature_col = None
        self.cut = None
        self.left = None
        self.right = None
        self.prediction = pred
        self.depth = depth
    
    def inf(self):
        """
        used to give some brief information about the tree, including the predicted value, which feature 
        selected to cut and the corresponding cutpoint on that feature.
        """
        return [self.prediction, self.feature_col, self.cut]

class DecisionTree(object):
    def __init__(self, max_dep=5, min_node=5, alpha=0):
        self.tree = None
        self.max_depth = max_dep
        self._mn = min_node
        self.alpha = alpha #alpha here is the penalty for growing number of tree nodes, helps to avoid overfitting.
        self.node_number = 1 # the number of nodes of a tree, initially it is one.
    
    def fit(self, X, Y):
        X = X[:, np.newaxis] if len(X.shape)==1 else X
        self.dim = X.shape[1] # fit method creates two unchangable attributes: dimension and classes.
        self.tree = self.build_tree(X, Y) # learn to grow the tree
    
    def node_split(self, X, Y):
        pass
    
    def build_tree(self, X, Y):
        pass
    
    def predict(self, Xtest):
        pass
    
class ClassifierTree(DecisionTree):
    def __init__(self, max_dep, min_node, alpha):
        super().__init__(max_dep, min_node, alpha)
        
    def fit(self, X, Y):
        self.classes = np.unique(Y)
        super().fit(X, Y)
    
    def node_split(self, X, Y):
        best_feature = None
        best_cut = None
        classes, counts = np.unique(Y, return_counts=True)
        num_obs = Y.size
        dim = self.dim
        best_gini = 1. - np.sum( (counts/num_obs)**2)
        parent_gini = best_gini
        
        for col in range(0, dim):
            order = np.argsort(X[:,col])
            #temp_Y = self._Y[order]
            temp_X = X[:,col][order]
            
            for i in range(1, num_obs):
                left_order = order[:i]
                left_Y = Y[left_order]
                cls, cts = np.unique(left_Y, return_counts=True)
                left_gini = 1 - np.sum( (cts/i)**2 )
                
                right_order = order[i:]
                right_Y = Y[right_order]
                cls, cts = np.unique(right_Y, return_counts=True)
                right_gini = 1 - np.sum( (cts/(num_obs-i))**2 )
                
                current_gini = (i * left_gini + (num_obs-i) * right_gini) / num_obs
                
                if temp_X[i] == temp_X[i-1]:
                    continue
                if (current_gini + self.alpha) < best_gini:
                    best_gini = current_gini
                    best_feature = col
                    best_cut = (temp_X[i] + temp_X[i-1]) / 2 
        if parent_gini != best_gini: 
            self.node_number += 1 #number of nodes == number of splits + 1
        return best_feature, best_cut
    
    def build_tree(self, X, Y, depth=0):
        """
        debug caution! Cannot use np.unique() like above here. if there is only one class, it will always classify to 0
        """
        counts = np.array([np.sum(Y==i) for i in self.classes])
        predict_class = np.argmax(counts) 
        node = Nodes(predict_class, depth)
        
        if depth <= self.max_depth and Y.size >= self._mn: #stopping if depth exceeds or datapoints fewer than minimum node.
            col, cut = self.node_split(X, Y)
            if col and cut:
                node.feature_col, node.cut = col, cut

                left_index = X[:,col] < cut
                left_X, left_Y = X[left_index], Y[left_index]
                
                right_index = X[:,col] >= cut
                right_X, right_Y = X[right_index], Y[right_index]

                node.left = self.build_tree(left_X, left_Y, depth+1)
                node.right = self.build_tree(right_X, right_Y, depth+1)
        return node
    
    def predict(self, Xtest):
        predictions = []
        for row in Xtest:
            node = self.tree
            while node.left:
                if row[node.feature_col] < node.cut:
                    node = node.left
                else: #be cautious here, need to use else to include that node.left is None.
                    node = node.right
            predictions.append(node.prediction)
        return np.array(predictions)

class RegressionTree(DecisionTree):
    def __init__(self, max_dep, min_node, alpha):
        super().__init__(max_dep, min_node, alpha)
    
    def node_split(self, X, Y):
        best_feature = None
        best_cut = None
        y_mean = np.mean(Y)
        num_obs = Y.size
        dim = self.dim
        best_rss = np.sum(np.square(Y - y_mean))
        parent_rss = best_rss
        
        for col in range(0, dim):
            order = np.argsort(X[:,col])
            #temp_Y = self._Y[order]
            temp_X = X[:,col][order]
            
            for i in range(1, num_obs):
                left_order = order[:i]
                left_Y = Y[left_order]
                left_ymean = np.mean(left_Y)
                left_rss = np.sum(np.square(left_Y - left_ymean))
                
                right_order = order[i:]
                right_Y = Y[right_order]
                right_ymean = np.mean(right_Y)
                right_rss = np.sum(np.square(right_Y - right_ymean))
                
                current_rss = left_rss + right_rss
                
                if temp_X[i] == temp_X[i-1]:
                    continue
                if (np.log(current_rss) + self.alpha) < np.log(best_rss): #to better include alpha, use log to compare.
                    best_rss = current_rss
                    best_feature = col
                    best_cut = (temp_X[i] + temp_X[i-1]) / 2 
        if parent_rss != best_rss: 
            self.node_number += 1 #number of nodes == number of splits + 1
        return best_feature, best_cut
    
    def build_tree(self, X, Y, depth=0):
        
        predict_value = np.mean(Y)
        node = Nodes(predict_value, depth)
        
        if depth <= self.max_depth and Y.size >= self._mn: #stopping if depth exceeds or number of datapoints fewer than 5.
            col, cut = self.node_split(X, Y)
            if col and cut:
                node.feature_col, node.cut = col, cut

                left_index = X[:,col] < cut
                left_X, left_Y = X[left_index], Y[left_index]
                
                right_index = X[:,col] >= cut
                right_X, right_Y = X[right_index], Y[right_index]

                node.left = self.build_tree(left_X, left_Y, depth+1)
                node.right = self.build_tree(right_X, right_Y, depth+1)
        return node
    
    def predict(self, Xtest):
        predictions = []
        for row in Xtest:
            node = self.tree
            while node.left:
                if row[node.feature_col] < node.cut:
                    node = node.left
                else: #be cautious here, need to use else to include that node.left is None.
                    node = node.right
            predictions.append(node.prediction)
        return np.array(predictions)