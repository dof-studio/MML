# ensemble.py
#
# Base Ensemble Models in Machine Learning
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from objtyp import Object
from tensor import Tensor
from matrix import Matrix

from baseml import MLBase, Regression, Classification


# Base Class for Ensemble Models
class Ensemble(Regression, Classification):
    
    __attr__ = "MML.Ensemble"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None):
        
        super().__init__()
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        self.feature_names = feature_names

    def fit(self):
        raise NotImplementedError("Fit is NOT implemented in the base class.")
        
    def predict(self):
        raise NotImplementedError("Predict is NOT implemented in the base class.")    
    
    def __repr__(self):
        return "Ensemble(Abstract Class)."


# Base Class for Bagging Models
class Bagging(Ensemble):
    
    __attr__ = "MML.Bagging"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None):
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        super().__init__(feature_names = feature_names)
        # Assigned in the base class - Ensemble
        
    def _sample_bootstrapping(self, X: Matrix | Tensor, y: Matrix | Tensor, M: int, k: int | float | None, 
                              replace: bool = True, shuffle: bool = True,
                            *, random_state: int | None = None, container: type = list) -> list | tuple | dict:
        """
        Generate M bootstrapped datasets containing k entries from the original data.
        k must be smaller than or equal to the 1st dim of data, or error will be raised.
        
        Parameters:
            -----------
            X: Matrix | Tensor, the input Feature Matrix or Tensor to be bootstrapped.
            y: Matrix | Tensor, the input Target Matrix or Tensor to be bootstrapped.
            M: int, Number of bootstrapped datasets to generate, the length of the container.
            k: int | float | None, Number of entries for each subset, or None for k = X.shape[0].
            replace: bool, When choosing samples, if using sampling WITH REPLACEMET. Default True.
            shuffle: bool, If to shuffle the indices of rows, or make them in an increasing manner.
            Optional:
                random_state: int | None, the random seed set to numpy backend to do the sampling.
                container: type, the container type to contain the output data, can be list, tuple, or dict.
            
        Returns:
            -----------
            Container of tuple of subset (X, y, row_indices), like:
                [
                    (X_0, y_0, row_indices),
                    (X_1, y_1, row_indices),
                    ...
                ]
                where X_i, y_i are Matrix | Tensor subset of data,
                and  row_indices is also a Matrix | Tensor that is flatten.
        """
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False or isinstance(y, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Integer validity Check.
        if M < 0 or k < 0 or k > X.shape[0]:
            raise ValueError("Input integer set (M, k) are not valid. Make sure you understand them!")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input feature `X` must have the same size of shape[0] with the input target `y`.")
        
        # Container Type Check.
        if container not in (list, tuple, dict):
            raise ValueError("The container type must be one of `list`, `tuple`, `dict`.")
        
        # Number of entries reset.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if k is None:
            k = n_samples
        if isinstance(k, float):
            k = int(round(n_samples * k))
            
        # Random State
        if random_state is not None:
            np.random.seed(random_state)
        
        # The created, bootstrapped datasets.
        bootstrapped_datasets = container() if container in (list, dict) else []
        
        for i in range(M):
            indices = np.random.choice(n_samples, size=k, replace=replace)
            if shuffle == False:
                indices.sort()
            X_i = X[indices]
            y_i = y[indices]
            if container in (list, tuple):
                bootstrapped_datasets.append((X_i, y_i, type(X)(indices, backend=X._backend, dtype=int, device=X.device)))
            else:
                bootstrapped_datasets[i] = (X_i, y_i, type(X)(indices, backend=X._backend, dtype=int, device=X.device))
            np.random.seed(np.random.randint(0, int(2**31 - 1)))
                
        return bootstrapped_datasets if container in (list, dict) else tuple(bootstrapped_datasets)
    
    def _feature_random_select(self, M: int, N: int, q: int | float, replace: bool = False, *,
                               random_state: int | None = None, container: type = list) -> list | tuple | dict:
        """
        Generate M dataset feature indices by randomly selecting q features from N features
        
        Parameters:
        M: int, Number of datasets to generate.
        N: int, Number of features contained in the dataset.
        q: int, Number of features to select in each dataset, or
           float, proportion of features to select in each dataset.
        replace: bool, When choosing features, if using sampling WITH REPLACEMET. Default False.
            Optional:
                random_state: int | None, the random seed set to numpy backend to do the sampling.
                container: type, the exterior container type to contain the output data, can be list, tuple, or dict.
            
        
        Returns:
            -----------
            Container of np.array of feature indices, like:
                [
                    np.array([0,1,2,3,10]), # Must be 1 dimensional and increasing manner
                    np.array([1,2,4,8,9])   # Must be 1 dimensional and increasing manner
                ]
                where the internal data type is a np.array of selected feature indices.
                # Note, it is a np.array NOT Matrix or Tensor either
                
            Note: why using feature indices?
                  answer: to be compatible to most of MML's ensemble-abled models, like CART or LinearRegression.
        """
        
        # Integer validity Check.
        if M < 0 or N < 0 or q < 0 or q > N:
            raise ValueError("Input integer set (M, N, q) are not valid. Make sure you understand them!")
        
        # Container Type Check.
        if container not in (list, tuple, dict):
            raise ValueError("The container type must be one of `list`, `tuple`, `dict`.")
            
        # Number of entries reset.
        if q is None:
            q = N
            
        # If q is float, convert to N
        if isinstance(q, float):
            q = min(int(round(N * q)), N)
            
        # Random State.
        if random_state is not None:
            np.random.seed(random_state)
            
        # The selected feature-index container.
        feature_indices = container() if container in (list, dict) else []
        
        for i in range(M):
            indices = np.random.choice(N, size=q, replace=replace)
            indices.sort()
            if container in (list, tuple):
                feature_indices.append(indices)
            else:
                feature_indices[i] = (indices)
            np.random.seed(np.random.randint(0, int(2**31 - 1)))
        
        return feature_indices if container in (list, dict) else tuple(feature_indices)
    
    def __repr__(self):
        return "Bagging(Abstract Class)."
    
    
# Base Class for Boosting Models
class Boosting(Ensemble):
    
    __attr__ = "MML.Boosting"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None):
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        super().__init__(feature_names = feature_names)
        # Assigned in the base class - Ensemble
    
    def __repr__(self):
        return "Boosting(Abstract Class)."
    
    
# Some Basic Tests
if __name__ == "__main__":
    
    bag = Bagging()
    typeclass = Matrix
    
    m10 = typeclass([
        [10, 12, 15],
        [7.1, 4, 20],
        [25, 14, 19],
        [42, 2821, 0],
        [17, 4, 1216],
        [20, 13, 727]], backend = "torch")
    
    n10 = typeclass([
        [1.2],
        [2.4],
        [7.1],
        [0.9],
        [0.2],
        [2.2]], backend = "torch")
    
    # Bootstrapping
    samples = bag._sample_bootstrapping(m10, n10, 3, k = 4, random_state=42)
    
    # Feature selecting
    features = bag._feature_random_select(4, 3, q = 2, random_state=None)
    
    # See what is going on (in practice, DO NOT do it)
    print(
        samples[0][0][:,features[0]]
        )
