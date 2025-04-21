# lm.py
#
# Ordinary Linear Regressions
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from objtyp import Object
from matrix import Matrix
from tensor import Tensor

from baseml import Regression, Classification

from linear import ClosedFormSingleValueRegression
from linear import GradientDescendSingleValueRegression


# Ordinary Single Value Linear Regressoin Model (Collection wrapper)
class OrdinaryLinearRegression(Regression):
    
    __attr__ = "MML.OrdinaryLinearRegression"
    
    def __init__(self, family: str = "gaussian",
                 fit_intercept: bool = True, 
                 use: str = "Closed",
                 cov: Matrix | Tensor | None = None,
                 *,
                 tol: float = 1e-8,
                 lr: float = 1e-2,
                 max_iter: int = 10000,
                 batch_size: int | None = None,
                 shuffle: bool = True,
                 l1: float = 0.0,
                 l2: float = 0.0,
                 random_state: int | None = None,
                 **kwargs) -> None:
        """
        Initialize a Gradient Descend Single Value Linear Regression model.
        The family must be `gaussian` or raised with a value error.
         
        Parameters:
            family: str, the estimation family. Must be `gaussian`. Or, consider Gradient Descend Implementations.
            fit_intercept: bool, if True, the intercept is learned during fitting. Default to True.
            use: str, can be {"Closed" or "GD"}, which helps to determine whether to use closed form or gradient method.
            cov: Matrix | Tensor | None
                 Error-term covariance matrix Σ (shape m×m, m = n_samples), or just the variance.
                 Provide for GLS; leave None for OLS.
            Optional: Only used when choosing GD, or ignored.
                tol: float, tolerence for convergence when optimizing.
                lr: float, learning rate. Default 1e-2.
                max_iter: int, maximum number of passes over the data. Default 10000.
                batch_size: int | None, minibatch size (None = full-batch GD).
                shuffle : bool, whether to reshuffle data at each epoch
                l1: float, optional Lasso penalty λ * sign(β) (0 = plain OLS).
                l2: float, optional Ridge penalty λ * ‖beta‖² (0 = plain OLS).
                random_state: int | None, random seed, can be None.
        """
        
        # Check if use is "Closed" ot "GD"
        if use not in ("Closed", "GD"):
            raise ValueError("Parameter `use` must be `Closed` or `GD`.")
        self.use = use
        
        # Special Record: fit_intercept
        self.fit_intercept = fit_intercept
        
        # Help set up the kernel class
        if use == "Closed":
            self.regressor = ClosedFormSingleValueRegression(
                family, fit_intercept, cov,
                tol = tol, lr = lr, max_iter = max_iter, batch_size = batch_size,
                shuffle = shuffle, l1 = l1, l2 = l2, random_state = random_state, **kwargs)
        else:
            self.regressor = GradientDescendSingleValueRegression(
                family, fit_intercept, cov,
                tol = tol, lr = lr, max_iter = max_iter, batch_size = batch_size,
                shuffle = shuffle, l1 = l1, l2 = l2, random_state = random_state, **kwargs)

    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs):
        """
        Fit the linear regression model on training data.
        
        Parameters:
            X: Matrix or Tensor, the input features, must be a 2D array-like.
            y: Matrix or Tensor, the target values, must also be a 2D array-like (shape [-1,1]).
        
        Returns:
            -------
            self
        """
        return self.regressor.fit(X, y, **kwargs)

    def predict(self, X: Matrix | Tensor, **kwargs) -> Matrix | Tensor:
        """
        Predict the target values using the linear model for given input features.
        
        Parameters:
            X: Matrix or Tensor, The input feature data.

        Returns:
            Matrix or Tensor: The predicted output in 2D array ([n_samples, 1]).
        
        Raises:
            ValueError: If model is not fitted. Call `fit()` before using predict method.
        """
        return self.regressor.predict(X, **kwargs)

    def summary(self) -> str:
        """
        Return a formatted string summary table with:
          - N            : number of samples
          - k            : number of predictors
          - outputs      : number of outputs
          - Intercept    : the fitted intercept
          - Betas        : list of fitted coefficients
          - AIC          : Akaike Information Criterion
          - BIC          : Bayesian Information Criterion
          - R²           : coefficient of determination
          - Adj R²       : adjusted R²
    
        All floating‐point numbers are rounded or shown in scientific notation
        with up to 6 significant digits.
        
        Returns:
            -------
            str: the returned summary table.
        """
        return self.regressor.summary()

    def __repr__(self):
        try:
            self.regressor._check_is_fit()
            return f"OrdinaryLinearRegression Wrapper, (N = {self.regressor.original_X.shape[0]}, k = {self.regressor.original_X.shape[1]}, {'with intercept' if self.regressor.fit_intercept == True else 'without intercept'})."
        except:
            return f"OrdinaryLinearRegression Wrapper, (Not fitted, {'with intercept' if self.regressor.fit_intercept == True else 'without intercept'})."


# Alias for Ordinary Single Value Linear Regression
LM  = OrdinaryLinearRegression
LR  = OrdinaryLinearRegression
OLS = OrdinaryLinearRegression
GLS = OrdinaryLinearRegression


if __name__ == "__main__":
    
    backend = "numpy"
    
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2821)

    # Convert data to the required format (assuming Matrix and Tensor are numpy arrays for simplicity)
    X_train = Matrix(X_train, backend=backend)
    y_train = Matrix(y_train, backend=backend).reshape([-1, 1])
    X_test = Matrix(X_test, backend=backend)
    y_test = Matrix(y_test, backend=backend).reshape([-1, 1])

    # Initialize the model with some parameters
    model = OrdinaryLinearRegression(
        use = "GD",
        fit_intercept=True,
        cov=None,
        tol=1e-8,
        lr=0.1,
        max_iter=50000,
        batch_size=None,
        shuffle=True,
        l1=0.0,
        l2=0.0001
    )

    # Train the model with training data
    model.fit(X_train, y_train)

    # Predict using test data and check if it returns a 2D array-like object of shape [-1, 1]
    predictions = model.predict(X_test)
    
    # Evaluate mse
    mse_error = mse(y_test.flatten().data, predictions.flatten().data)
    print(f"RMSE Error: {mse_error ** 0.5}")
    
    # Print the summary
    print(model.summary())
    
    