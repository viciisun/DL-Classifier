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
        
    @property
    def mean_(self):
        """Get mean values used in standard scaling"""
        if self.method == 'standard' and self.scaler is not None:
            return self.scaler.mean_
        return None
        
    @property
    def std_(self):
        """Get standard deviation values used in standard scaling"""
        if self.method == 'standard' and self.scaler is not None:
            return self.scaler.scale_
        return None
        
    @property
    def min_(self):
        """Get minimum values used in min-max scaling"""
        if self.method == 'minmax' and self.scaler is not None:
            return self.scaler.min_
        return None
        
    @property
    def scale_(self):
        """Get scale values used in min-max scaling"""
        if self.method == 'minmax' and self.scaler is not None:
            return self.scaler.scale_
        return None
    
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
            if self.scaler is None:
                raise ValueError("Preprocessor must be fitted before transform")
            return self.scaler.transform(X)
        elif self.method == 'pca':
            if self.pca is None:
                raise ValueError("Preprocessor must be fitted before transform")
            return self.pca.transform(X)
        return X
        
    def fit_transform(self, X):
        """Fit and transform data in one step"""
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        """Inverse transform the data"""
        if self.method == 'standard' or self.method == 'minmax':
            if self.scaler is None:
                raise ValueError("Preprocessor must be fitted before inverse_transform")
            return self.scaler.inverse_transform(X)
        elif self.method == 'pca':
            if self.pca is None:
                raise ValueError("Preprocessor must be fitted before inverse_transform")
            return self.pca.inverse_transform(X)
        return X
        
    def get_explained_variance_ratio(self):
        """Get explained variance ratio for PCA"""
        if self.method == 'pca' and self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None
        
    def get_params(self):
        """Get preprocessing parameters"""
        params = {}
        if self.method == 'standard':
            if self.scaler is not None:
                params['mean'] = self.mean_.tolist() if self.mean_ is not None else None
                params['std'] = self.std_.tolist() if self.std_ is not None else None
        elif self.method == 'minmax':
            if self.scaler is not None:
                params['min'] = self.min_.tolist() if self.min_ is not None else None
                params['scale'] = self.scale_.tolist() if self.scale_ is not None else None
        return params 