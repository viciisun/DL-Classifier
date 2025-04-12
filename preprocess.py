import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self, method='standard', n_components=None):
        """
        Initialize the data preprocessor
        
        Args:
            method (str): Preprocessing method to use
                - 'standard': Standard scaling (zero mean, unit variance)
                - 'minmax': Min-max scaling (0-1 range)
                - 'pca': Principal Component Analysis
            n_components (int): Number of components for PCA
        """
        self.method = method
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        
    def fit(self, X):
        """Fit the preprocessor on training data"""
        if self.method == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
        elif self.method == 'pca':
            if self.n_components is None:
                self.n_components = min(X.shape[0], X.shape[1])
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X)
            
    def transform(self, X):
        """Transform data using the fitted preprocessor"""
        if self.method == 'standard' or self.method == 'minmax':
            return self.scaler.transform(X)
        elif self.method == 'pca':
            return self.pca.transform(X)
        return X
        
    def fit_transform(self, X):
        """Fit and transform data in one step"""
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        """Inverse transform the data"""
        if self.method == 'standard' or self.method == 'minmax':
            return self.scaler.inverse_transform(X)
        elif self.method == 'pca':
            return self.pca.inverse_transform(X)
        return X
        
    def get_explained_variance_ratio(self):
        """Get explained variance ratio for PCA"""
        if self.method == 'pca':
            return self.pca.explained_variance_ratio_
        return None 