# lm.py
#
# Ordinary Linear Regressions
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, Dict, List, Tuple

from .objtyp import Object
from .matrix import Matrix
from .tensor import Tensor

from .baseml import Regression, Classification

from .linear import ClosedFormSingleValueRegression
from .linear import GradientDescendSingleValueRegression


# Ordinary Single Value Linear Regressoin Model (Collection wrapper)
class OrdinaryLinearRegression(Regression):
    
    __attr__ = "MML.OrdinaryLinearRegression"
    
    def __init__(self, 
                 task: str = "regression",
                 family: str = "gaussian",
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
            task: str, the task of doing the regression, can be `regression` most of time, or `classification` in `logistic` mode.
            family: str, the estimation family. Can be {"gaussian", "logistic", "gamma", "inverse_gaussian", "poisson"}.
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
                task, family, fit_intercept, cov,
                tol = tol, lr = lr, max_iter = max_iter, batch_size = batch_size,
                shuffle = shuffle, l1 = l1, l2 = l2, random_state = random_state, **kwargs)

    def fit_easy(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs):
        """
        Fit the linear regression model on training data easily.
        
        Parameters:
            X: Matrix or Tensor, the input features, must be a 2D array-like.
            y: Matrix or Tensor, the target values, must also be a 2D array-like (shape [-1,1]).
        
        Returns:
            -------
            self
        """
        return self.regressor.fit(X, y, **kwargs)
    
    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor,
            *,
            verbosity: int | None = None,
            evalset: Dict[str, Tuple[Matrix | Tensor, Matrix | Tensor]] | None = None,
            evalmetrics: List[str] | str | None = None,
            early_stop: int | None = None,
            early_stop_logic: str = "some",
            continue_to_train: bool | None = None,
            val_per_iter: int = 50,
            **kwargs):
        """
        Fit the linear regression model on training data with perhaps evaluation sets and early stopping.
        You may want to evaluate datasets while training. If so, please do the following things:
            1. set `verbosity` = 1 to print the evaluation
            2. set the `evalset` to a dict of tuples of your dataset that is going to be evaluated
            3. set the `evalmetrics` either to a string of metrics or a list of strings
        You may want the algorithm to decide to stop training automatically. If so, please do things above, plus:
            1. set `early_stop` to a number of batches, like 1 or 2, which acts like: 
                if the metrics for all/some/any/most of the evaluation sets do not decrease anymore, 
                the training process will be terminated and return
            2. set `early_stop_logic` to determine the way of processing non-decreasing datasets/metrics
            3. If you hope to continue to train again, call this `fit` again with `continue_to_train` set to True
        
        Parameters:
            X: Matrix or Tensor, the input features, must be a 2D array-like.
            y: Matrix or Tensor, the target values, must also be a 2D array-like (shape [-1,1]).
            Optional:
                verbosity: int | None, if >= 1 and having `evalset`, then will report metrics each batch.
                evalset: Dict[name : Tuple[X, y],
                              ...], | None, if provided, it may be used as evaluation set. XGBoost style.
                evalmetrics: list of str | str | None, metrics used to do the evaluation. Will be printed.
                early_stop: int | None, if non-None, then if metrics NOT gained for `early_stop` times, the forest will stop training.
                early_stop_logic: str, the logic when deciding on multiple metrics, can be {"any", "some", "most", "all"}.
                continue_to_train: bool | None, if non-None and True, the machine will try to restore the place it was and continue
                                   to train new estimators until a new stopping criterion meets or until reaches the max number of allowed estimators.
                val_per_iter: int, the number of iterations to do, before next evaluation happens, recommend 20, 50.
                
        Returns:
            -------
            self
        """
        return self.regressor.fit(X, y, verbosity=verbosity, 
                                  evalset=evalset, evalmetrics=evalmetrics,
                                  early_stop=early_stop, early_stop_logic=early_stop_logic,
                                  continue_to_train=continue_to_train, val_per_iter=val_per_iter, **kwargs)

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
    
    from scaling import Scaling
    from metrices import RegressionMetrics as regm
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2821)

    # Convert data to the required format (assuming Matrix and Tensor are numpy arrays for simplicity)
    X_train = Matrix(X_train, backend=backend)
    y_train = Matrix(y_train, backend=backend).reshape([-1, 1])
    X_test = Matrix(X_test, backend=backend)
    y_test = Matrix(y_test, backend=backend).reshape([-1, 1])
    
    # Train a scaler
    scaler = Scaling("robust").fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    # Initialize the model with some parameters
    model = OrdinaryLinearRegression(
        task = "regression",
        use = "GD",
        fit_intercept=True,
        cov=None,
        tol=1e-8,
        lr=0.02,
        max_iter=10000,
        batch_size=100,
        shuffle=True,
        l1=0.0,
        l2=0.0001
    )

    # Train the model with training data
    model.fit(X_train, y_train,
              verbosity = 1, evalset = {"Test": (X_test, y_test)}, evalmetrics = ["rmse", "mse", "r2"], 
              val_per_iter=100, early_stop = 1, early_stop_logic = "most")

    # Predict using test data and check if it returns a 2D array-like object of shape [-1, 1]
    predictions = model.predict(X_test)
    
    # Evaluate mse
    rmse_error = regm(y_test, predictions, "rmse").compute()
    print(f"RMSE Error: {rmse_error}")
    
    # Print the summary
    print(model.summary())
    
    