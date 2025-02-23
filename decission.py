import numpy as np




class Node:
    def __init__(self, gini, predict_class, num_sample, num_sample_class):
        self.threshold = None
        self.num_sample_class = num_sample_class
        self.num_sample = num_sample
        self.value = predict_class
        self.gini = gini
        self.feature_index = None
        self.right = None
        self.left = None




# Decision Tree Class
class d_tree:
    def __init__(self, max_depth=3):
        self.max_d = max_depth
        self.tree = None
        
    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))
    
    def grow_tree(self, x, y, depth=0):
        num_sample_class = {c: np.sum(y == c) for c in np.unique(y)}
        predicted_class = max(num_sample_class, key=num_sample_class.get)
        
        node = Node(
            gini=self._gini_impurity(y),
            num_sample=len(y),
            num_sample_class=num_sample_class,
            predict_class=predicted_class,
        )
        
        if depth < self.max_d:
            index, thr = self.best_split(x, y)
            if index is not None:
                left_index = x[:, index] < thr
                x_left, y_left = x[left_index], y[left_index]
                x_right, y_right = x[~left_index], y[~left_index]
                
                if len(y_left) > 0 and len(y_right) > 0:  # Ensure valid splits
                    node.left = self.grow_tree(x_left, y_left, depth + 1)
                    node.right = self.grow_tree(x_right, y_right, depth + 1)
                    node.threshold = thr
                    node.feature_index = index
        return node
    
    def best_split(self, x, y):
        m, n = x.shape
        if m <= 1:
            return None, None
        
        best_gini = self._gini_impurity(y)
        best_index, best_thr = None, None
        no_sample_class = {c: np.sum(y == c) for c in np.unique(y)}
        
        for idx in range(n):
            sorted_pairs = sorted(zip(x[:, idx], y))
            thr, classes = zip(*sorted_pairs)
            
            n_left = {c: 0 for c in np.unique(y)}
            n_right = no_sample_class.copy()
            
            for k in range(1, m):
                c = classes[k - 1]
                n_left[c] += 1
                n_right[c] -= 1
                
                gini_left = 1.0 - sum((n_left[j] / k) ** 2 for j in np.unique(y))
                gini_right = 1.0 - sum((n_right[j] / (m - k)) ** 2 for j in np.unique(y))
                gini = (k * gini_left + (m - k) * gini_right) / m
                
                if k < m - 1 and thr[k] == thr[k + 1]:  # Skip duplicate thresholds
                    continue
                
                if gini < best_gini:
                    best_gini = gini
                    best_thr = (thr[k] + thr[k - 1]) / 2
                    best_index = idx   
        
        return best_index, best_thr
    
    def fit(self, x, y):
        self.tree = self.grow_tree(x, y)
        
    def predict(self, test):
        return np.array([self._predict(self.tree, i) for i in test])
    
    def _predict(self, node, i):
        if node.left is None or node.right is None:
            return node.value
        if i[node.feature_index] < node.threshold:
            return self._predict(node.left, i)
        else:
            return self._predict(node.right, i)
    
    def predict_proba(self, test):
        return np.array([self._predict_proba(self.tree, i) for i in test])
    
    def _predict_proba(self, node, i):
        if node.left is None or node.right is None:
            total_samples = sum(node.num_sample_class.values())
            prob = [node.num_sample_class.get(j, 0) / total_samples for j in range(len(node.num_sample_class))]
            return prob
        if i[node.feature_index] < node.threshold:
            return self._predict_proba(node.left, i)
        else:
            return self._predict_proba(node.right, i)



