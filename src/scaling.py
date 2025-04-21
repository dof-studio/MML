# scaling.py
#
# A base class for ML algorithms
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None
    
from matrix import Matrix
from tensor import Tensor

# Implementation of Data Scaler
class Scaling:
    """
    Scale class that fits on a Matrix and can perform either centralization (subtracting the mean)
    or min-max scaling (scaling features to the [0, 1] range), with the ability to reverse the operation.
    """
    
    __attr__ = "MML.Scaling"
    
    def __init__(self, method="centralize", *, robust_p = 0.25):
        '''
        Args:
            `method` can be:
                "centralize": only subtract the mean
                "normalize": subtract the mean and standardize the variance to 1
                "minmax": keep the data with in the range of [0, 1]
                "robust": compute median and interquartile range to reduce the effect of outliers.
            `robust_p` the lower percentile [0,1] of the percentile estimate. 0.25 means 25% and 75%
        '''
        self.method = method
        self.params = {}
        
        # Method specific parameters
        self.robust_p = robust_p if robust_p < 0.5 else 1 - robust_p

    def fit(self, X: Matrix | Tensor, axis = 0):
        """
        Fits the scaling parameters to the data.
    
        Args:
            X (Matrix | Tensor): The input matrix or tensor for fitting.
            axis (int): Axis along which to compute the mean and standard deviation. Default is 0.
    
        Returns:
            self: The fitted instance of the class, allowing method chaining.
    
        Raises:
            ValueError: If an unsupported scaling method is provided.
    
        """
        type_X = type(X)
        if self.method == "centralize":
            # Just demean the data to 0 mean
            if X._is_numpy:
                mean_val = np.mean(X.data, axis=axis)
            else:
                mean_val = torch.mean(X.data, dim=axis)
            self.params['mean'] = type_X(mean_val, backend=X._backend, device=X.device, dtype=X.dtype)
        
        elif self.method == "normalize":
            # Normalize the data with 0 mean and std of 1
            if X._is_numpy:
                mean_val = np.mean(X.data, axis=axis)
                stdev_val = np.std(X.data, axis=axis)
            else:
                mean_val = torch.mean(X.data, dim=axis)
                stdev_val = torch.std(X.data, dim=axis)
            self.params['mean'] = type_X(mean_val, backend=X._backend, device=X.device, dtype=X.dtype)
            self.params['std'] = type_X(stdev_val, backend=X._backend, device=X.device, dtype=X.dtype)
        
        elif self.method == "minmax":
            # Minmax to make data in a range of [0,1]
            if X._is_numpy:
                min_val = np.min(X.data, axis=axis)
                max_val = np.max(X.data, axis=axis)
            else:
                min_val = torch.min(X.data, dim=axis).values
                max_val = torch.max(X.data, dim=axis).values
            self.params['min'] = type_X(min_val, backend=X._backend, device=X.device, dtype=X.dtype)
            self.params['max'] = type_X(max_val, backend=X._backend, device=X.device, dtype=X.dtype)
        
        elif self.method == "robust":
            # Compute median and interquartile range to reduce the effect of outliers.
            if X._is_numpy:
                median_val = np.median(X.data, axis=axis)
                q1 = np.percentile(X.data, int(self.robust_p * 100), axis=axis)
                q3 = np.percentile(X.data, 100 - int(self.robust_p * 100), axis=axis)
                iqr_val = q3 - q1
            else:
                median_val = torch.median(X.data, dim=axis).values
                q1 = torch.quantile(X.data, self.robust_p, dim=axis)
                q3 = torch.quantile(X.data, 1 - self.robust_p, dim=axis)
                iqr_val = q3 - q1
            self.params['p'] = self.robust_p
            self.params['median'] = type_X(median_val, backend=X._backend, device=X.device, dtype=X.dtype)
            self.params['iqr'] = type_X(iqr_val, backend=X._backend, device=X.device, dtype=X.dtype)
        else:
            raise ValueError("Unsupported scaling method. Choose 'centralize', 'normalize', 'minmax', or 'robust'.")
        return self

    def transform(self, X: Matrix | Tensor):
        """
        Transforms the input matrix using the fitted parameters.
        
        Args:
            X (Matrix | Tensor): The input matrix for transformation.
        
        Returns:
            Matrix | Tensor: The transformed matrix or tensor.
        
        Raises:
            InterruptedError: If no scaling parameters have been fitted yet.
            ValueError: If an unsupported scaling method is provided.
        
        """
        if len(self.params) == 0:
            raise InterruptedError("You should call `fit` before doing any transformation")
        if self.method == "centralize":
            return (X - self.params['mean'])
        elif self.method == "normalize":
            return (X - self.params['mean']) / self.params['std']
        elif self.method == "minmax":
            range_matrix = self.params['max'] - self.params['min']
            return (X - self.params['min']) / range_matrix
        elif self.method == "robust":
            return (X - self.params['median']) / self.params['iqr']
        else:
            raise ValueError("Unsupported scaling method. Choose 'centralize', 'normalize', 'minmax', or 'robust'.")

    def inverse_transform(self, X: Matrix | Tensor):
        """
        Inverses the transformation applied during fitting.
        
        Args:
            X (Matrix | Tensor): The transformed matrix for inversion.
        
        Returns:
            Matrix | Tensor: The original matrix or tensor before scaling.
        
        Raises:
            InterruptedError: If no scaling parameters have been fitted yet.
            ValueError: If an unsupported scaling method is provided. 
        
        """
        if len(self.params) == 0:
            raise InterruptedError("You should call `fit` before doing any transformation")
        if self.method == "centralize":
            # Inverse centralization: add the mean back.
            return X + self.params['mean']
        if self.method == "normalize":
            # Inverse centralization: multiply the std and add the mean back.
            return X * self.params['std'] + self.params['mean']
        elif self.method == "minmax":
            # Inverse minmax scaling: X*(max - min) + min
            range_matrix = self.params['max'] - self.params['min']
            return X * range_matrix + self.params['min']
        elif self.method == "robust":
            # Inverse robust scaling: X*(iqr) + median
            return X * self.params['iqr'] + self.params['median']
        else:
            raise ValueError("Unsupported scaling method. Choose 'centralize', 'normalize', 'minmax', or 'robust'.")


# Test cases
if __name__ == "__main__":
    # Create a simple dataset using numpy.
    data = np.array([[1, 2],
                     [3, 4],
                     [5, 6],
                     [7, 8],
                     [9, 10],
                     [11, 101]])
    targets = np.array([1, 0, 1, 0, 1, 1])
    
    # Instantiate Matrix objects.
    X = Matrix(data, backend="numpy")
    y = Matrix(targets, backend="numpy")
    
    # If uses torch, using torch to speed up.
    if torch is not None:
        X = X.to("torch", device = "cpu")
        X = X.astype(torch.float32)

    # Test the Scale class with centralization.
    scaler_central = Scaling(method="centralize").fit(X)
    X_centered = scaler_central.transform(X)
    X_reconstructed = scaler_central.inverse_transform(X_centered)
    print("\nCentralization Scaling:\n")
    print("Original X:\n", X)
    print("Centered X:\n", X_centered)
    print("Reconstructed X:\n", X_reconstructed, "\n\n")
    
    # Test the Scale class with normalization
    scaler_central = Scaling(method="normalize").fit(X)
    X_centered = scaler_central.transform(X)
    X_reconstructed = scaler_central.inverse_transform(X_centered)
    print("\nNormalization Scaling:\n")
    print("Original X:\n", X)
    print("Centered X:\n", X_centered)
    print("Reconstructed X:\n", X_reconstructed, "\n\n")
    
    # Test the Scale class with min-max scaling.
    scaler_minmax = Scaling(method="minmax").fit(X)
    X_scaled = scaler_minmax.transform(X)
    X_inversed = scaler_minmax.inverse_transform(X_scaled)
    print("\nMinMax Scaling:\n")
    print("Original X:\n", X.data)
    print("Scaled X:\n", X_scaled.data)
    print("Inversed X:\n", X_inversed.data, "\n\n")
    
    # Test the Scale class with robust scaling.
    scaler_minmax = Scaling(method="robust").fit(X)
    X_scaled = scaler_minmax.transform(X)
    X_inversed = scaler_minmax.inverse_transform(X_scaled)
    print("\nRobust Scaling:\n")
    print("Original X:\n", X.data)
    print("Scaled X:\n", X_scaled.data)
    print("Inversed X:\n", X_inversed.data, "\n\n")
    