# pca.py
#
# A principal component analysis implementation
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from matrix import Matrix
from tensor import Tensor
from baseml import MLBase, Regression, Classification
from scaling import Scaling

class PCA(Regression, Classification):
    """
    Principal Component Analysis (PCA) class built upon the Matrix class.
    It performs data centering/standardization, computes the covariance matrix,
    calculates eigenvalues/eigenvectors, projects data onto principal components,
    and computes the variance explained.
    """
    
    __attr__ = "MML.PCA"
    
    def __init__(self, n_components=None, method="centralize", eigen_method="selfimpl"):
        """
        Initializes the PCA class.
        
        Args:
            n_components (int): Number of principal components to keep. If None, all components are kept.
            method (str): Method for scaling the data ('centralize', 'normalize',or 'minmax'). Default is 'centralize'.
            eigen_method(str): Method for eigenvalues/eigenvectors calculation ('selfimpl', 'kernel'). Default is 'selfimpl'
        
        Attributes:
            n_components (int): Number of principal components to retain.
            eigen_method(str): Method for eigenvalues/eigenvectors calculation.
            scaling (Scaling): An instance of Scaling class used for centering and standardizing the data.
            cov_matrix (Matrix): Matrix containing the covariance matrix of the training matrix.
            components (Matrix): Matrix containing the principal components.
            eigenvalues (Matrix): Eigenvalues corresponding to each principal component.
            eigenvectors (Matrix): Eigenvectors corresponding to each principal component.
            explained_variance_ratio (Matrix): The proportion of variance explained by each principal component.
        
        Raises:
            ValueError: If an unsupported scaling method is provided. Choose 'centralize' or 'minmax'.
        
        """
        self.n_components = n_components
        self.eigen_method = eigen_method
        self.scaling = Scaling(method=method)
        # Principal components
        self.cov_matrix = None
        self.components = None
        self.eigenvalues = None
        self.eigenvectors = None
        # Statistical results
        # Implemented by Nathmath Huang
        self.explained_variance_ratio = None
        
    def fit(self, X: Matrix, sample_cov:bool = True):
        """
        Fits the PCA model to the input data.
        
        Args:
            X (Matrix): The matrix of features.
            sample_cov (bool): Whether to compute the sample covariance. Default is True.
        
        Returns:
            self: The fitted instance of the class, allowing method chaining.
        
        Raises:
            TypeError: If `X` is not a Matrix or Tensor type due to eigen computation requirements.
        
        """
        # Not a matrix
        if isinstance(X, Matrix) == False:
            if isinstance(X, Tensor) == True:
                raise TypeError("PCA class only accept `Matrix` type due to eigen computation. Convert your Tensor to Matrix by using X = Matrix(X.data)")
            else:
                raise TypeError(f"PCA class only accept `Matrix` instead of {type(X)}")
                
        # Data centering/standardization
        self.scaling.fit(X = X, axis = 0)
        X_centered = self.scaling.transform(X).to(backend=X._backend, dtype=X.dtype)

        # Covariance matrix computation
        n_samples = X.shape[0]
        X_centered_T = X_centered.transpose()
        self.cov_matrix = (X_centered_T @ X_centered) / (n_samples - 1 if sample_cov == True else 0)
        
        # Eigenvalue/eigenvector calculation (use my implementation in Matrix)
        eigenvalues, eigenvectors = self.cov_matrix.eigen(method=self.eigen_method)
        # Note:
        # The Eigen values and vectors here MIGHT be in complex form.
        # So, call .to_rational() to forcefully transform into rational numbers.
        # Moreover, ensures all of the tensors are on the same device.
        eigenvalues = eigenvalues.to_rational().copy().to(backend=X._backend, dtype=X.dtype, device=X.device)
        eigenvectors = eigenvectors.to_rational().copy().to(backend=X._backend, dtype=X.dtype, device=X.device)
        
        # Sort them in descending order
        if eigenvalues._is_numpy:
            sorted_idx = np.argsort(eigenvalues.data)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]
        else:
            sorted_idx = torch.argsort(eigenvalues.data, descending=True)
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]
            
        # Save the eigen values and eigen vectors
        self.eigenvalues = eigenvalues.copy(dtype=X.dtype, device=X.device)
        self.eigenvalues = eigenvectors.copy(dtype=X.dtype, device=X.device)      
            
        # Select the top n_components if specified
        if self.n_components is not None:
            components_eigenvalues = eigenvalues[:self.n_components]
            components_eigenvectors = eigenvectors[:, :self.n_components]
        self.components = components_eigenvectors.copy(dtype=X.dtype, device=X.device) 

        # Variance explained
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio = (components_eigenvalues / total_variance).copy(dtype=X.dtype, device=X.device) 
        return self

    def predict(self, X: Matrix):
        """
        Projects the input matrix `X` onto the principal components learned during fit.
        
        Args:
            X (Matrix): The matrix of features for prediction.
        
        Returns:
            projected (Matrix): The transformed data in the new coordinate system defined by the principal components.
        
        Raises:
            TypeError: If `X` is not a Matrix type due to eigen computation requirements.
            TypeError: If the model has not been fitted (`components` is None).
            ValueError: If the number of features in `X` does not match the number of columns in the covariance matrix.
        
        """
        
        # Not a matrix
        if isinstance(X, Matrix) == False:
            if isinstance(X, Tensor) == True:
                raise TypeError("PCA class only accept `Matrix` type due to eigen computation. Convert your Tensor to Matrix by using X = Matrix(X.data)")
            else:
                raise TypeError(f"PCA class only accept `Matrix` instead of {type(X)}")
                
        # Not fitted
        if self.components is None:
            raise TypeError("You must call `fit` before calling this `predict`")
            
        # Dimension mismatch
        if X.shape[1] != self.cov_matrix.shape[1]:
            raise ValueError(f"You must give a matrix `X` having exactly the same number of features of {self.cov_matrix.shape[1]}")
                
        # Project data onto principal components
        X_centered = self.scaling.transform(X).to(backend=self.components._backend, dtype=self.components.dtype, device=self.components.device)
        projected = X_centered @ self.components
        return projected

    def fit_predict(self, X: Matrix):
        """
        Fits the PCA model to the input data and then predicts the transformed data.
        
        Args:
            X (Matrix): The matrix of features for both fitting and prediction.
        
        Returns:
            projected (Matrix): The transformed data in the new coordinate system defined by the principal components.
        
        Raises:
            TypeError: If `X` is not a Matrix type due to eigen computation requirements.
        
        """
        
        # Not a matrix
        if isinstance(X, Matrix) == False:
            if isinstance(X, Tensor) == True:
                raise TypeError("PCA class only accept `Matrix` type due to eigen computation. Convert your Tensor to Matrix by using X = Matrix(X.data)")
            else:
                raise TypeError(f"PCA class only accept `Matrix` instead of {type(X)}")
                
        self.fit(X)
        return self.predict(X)

    def inversely(self, Y: Matrix, scale_back: bool = True):
        """
        Projects the transformed matrix `Y` onto the approximated original matrix X.
        
        Args:
            Y (Matrix): The matrix of features that are transformed.
        
        Returns:
            projected (Matrix): The inversely transformed data.
        
        Raises:
            TypeError: If `Y` is not a Matrix type due to eigen computation requirements.
            TypeError: If the model has not been fitted (`components` is None).
            ValueError: If the number of features in `Y` does not match the number of columns in the component matrix.
        
        """
        
        # Not a matrix
        if isinstance(Y, Matrix) == False:
            if isinstance(Y, Tensor) == True:
                raise TypeError("PCA class only accept `Matrix` type due to eigen computation. Convert your Tensor to Matrix by using Y = Matrix(Y.data)")
            else:
                raise TypeError(f"PCA class only accept `Matrix` instead of {type(Y)}")
                
        # Not fitted
        if self.components is None:
            raise TypeError("You must call `fit` before calling this `predict`")
            
        # Dimension mismatch
        if Y.shape[1] != self.components.shape[1]:
            raise ValueError(f"You must give a matrix `X` having exactly the same number of features of {self.cov_matrix.shape[1]}")
                
        # Project data back from principal components
        projected = Y @ self.components.transpose()
        if scale_back == True:
            projected = self.scaling.inverse_transform(projected).to(backend=self.components._backend, dtype=self.components.dtype, device=self.components.device)
        
        return projected

    def variance_explained(self):
        """
        Returns the explained variance ratio for each principal component.
        
        Returns:
            explained_variance_ratio (Matrix): The proportion of variance in `X` that is explained by each principal component.
        
        """
        # Note this is a Matrix
        return self.explained_variance_ratio 
    
    
# Test cases
if __name__ == "__main__":
    # A simple dataset for PCA demonstration
    data = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])
    X = Matrix(torch.tensor(data), backend="torch")
    X = X.to("torch", device = "cuda")
    
    # Instantiate PCA to retain 1 component without standardization (only centering)
    pca = PCA(n_components=1)
    pca.fit(X)
    
    print("Eigenvalues:")
    print(pca.eigenvalues)
    print("\nPrincipal Components (each column is a component):")
    print(pca.components)
    
    X_projected = pca.predict(X)
    print("\nProjected Data onto Principal Components:")
    print(X_projected)
    
    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio)
    
    pca.inversely(X_projected)
    
