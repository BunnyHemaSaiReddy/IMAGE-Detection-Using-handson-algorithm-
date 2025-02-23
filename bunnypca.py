# import numpy as np


# class PCa:
#     def __init__(self, n_components):
#         self.n_comp = n_components  
#         self.X_ = None 
#         self.components = None 
#     def fit(self, X):

#         self.X_ = np.mean(X, axis=0)
#         X_centered = X - self.X_

       
#         cov = np.cov(X_centered, rowvar=False)

     
#         e_values, e_vectors = np.linalg.eig(cov)

#         sorted_idx = np.argsort(e_values)[::-1]
#         e_values = e_values[sorted_idx]
#         e_vectors = e_vectors[:, sorted_idx]

       

#         # Select the top n_components
#         self.components = -e_vectors[:, :self.n_comp].T

#     def transform(self, X):
#         # Center the data using the stored mean
#         X_centered = X - self.X_
#         # Project the data onto principal components
#         return np.dot(X_centered, self.components.T)

import numpy as np

class PCa:
    def __init__(self, n_components):
        self.n_comp = n_components  
        self.mean = None  
        self.components = None  

    def fit(self, X):
       
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

       
        self.components = -Vt[:self.n_comp]  

    def transform(self, X):
      
        X_centered = X - self.mean
      
        return np.dot(X_centered, self.components.T) 