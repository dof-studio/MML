# tree_wrapper.py
#
# Tree-API-like wrapped models in Machine Learning
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from copy import deepcopy

from dump import save, load

from objtyp import Object
from tensor import Tensor
from matrix import Matrix

from tree import BaseTree
from lm import LR


# Linear Regression (Ordinary) Tree Wrapper
class LRTW(LR, BaseTree):
    
    __attr__ = "MML.MRTW"
    
    def __init__(self, 
                 task: str = 'regression', 
                 tree_id: int = None, 
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
                 floattype: type = float,
                 feature_names: Matrix | Tensor | list | tuple | None = None,
                 **kwargs):
        """
        Initialize a CART tree that can be used for regression or classification.

        Parameters:
            task: str, 'regression' or 'classification'. (placeholder).
            tree_id: int, the identifiation number for this tree. If None, then 0.
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
                floattype: type, type of float used in here (for compatibility use).
                feature_names: Matrix | Tensor, A 1D Matrix or Tensor of string names for features.
                **kwargs: other key word arguments, reserved for compatibility use.
        """
        # Initialize the base classes
        LR.__init__(self, task=task, family=family, fit_intercept=fit_intercept, use=use,
                    cov = cov, tol = tol, lr = lr, max_iter = max_iter, batch_size = batch_size,
                    shuffle = shuffle, l1 = l1, l2 = l2, random_state = random_state, **kwargs)
        BaseTree.__init__(self, feature_names = feature_names, **kwargs)        
        
        # Record the task
        self.task = task.lower()
        if self.task not in ("regression", "classification"):
            raise ValueError("Unsupported task. Choose 'regression' or 'classification'.")
        
        # KWargs Accepted
        self.kwargs = kwargs
        
        # #####################################################################
        #
        # Tree wrapper members
        
        # Unsliced Original X and y
        self.original_X = None
        self.original_y = None
        
        # Sliced X
        self.wrapper_sliced_X = None
        
        # Task and Loss Check
        self.tree_id = tree_id if tree_id is not None else 0
            
        # Special Inverse Feature Index Map
        # Why?
        # Since I have to slice th
        # Key: Sliced Feature Index (Ignore Intercept if Any)
        # Values: Original Feature Index in the input X Matrix
        self.inverse_feature_index_map = {}     
        
    def _create_inverse_feature_index_map(self, used_feature_idx: list) -> None:
        """
        Creates a mapping from sliced column indices to original feature indices that are unsliced.
        
        Args:
            used_feature_idx: list, original feature indices that are used
        
        Returns:
            None
        """
        
        self.inverse_feature_index_map = {}
        # Key: Sliced Feature Index (Ignore Intercept if Any)
        # Values: Original Feature Index in the input X Matrix
        
        idx = 0
        if self.fit_intercept == True:
            idx += 1 # Starting from 1, since 0 is always for intercept
            
        for used_idx in used_feature_idx:
            self.inverse_feature_index_map[idx] = used_idx
            idx += 1
        return
        
    def _create_feature_importance(self, fitted_: bool = False) -> None:
        """
        Creates a feature importance dict in tree style.
        
        Args:
            fitted_: bool, a sign indicates the programmer to notice if this model is fitted or not.
        
        Returns:
            None
        
        """
        # If not fitted or internally not fitted
        if fitted_ == False:
            raise RuntimeError("Trying to create feature importance without fitting. Call fit() first.")
        self.regressor._check_is_fit()
        
        # Initialize the feature importance dict
        self.feature_importance = {} # Key: Original Feature Index
                                     # Value: Total Importance measured by coef * mean(feature) / mean(y)
        for i in range(len(self.feature_index_map)):
            self.feature_importance[i] = 0.0
        
        # Get the betas from the regressor (without intercept)
        betas = self.regressor.betas.copy() # in (n, 1) shape
        betas = betas.flatten()             # in (n, )  shape  
        y_mean = self.original_y.mean()
        
        for i, new_key in enumerate(self.inverse_feature_index_map.keys()):
            beta_i = betas.to_list()[i]
            original_key = self.inverse_feature_index_map[new_key]
            data_X_i = self.original_X[:,original_key]
            data_X_i_mean = data_X_i.mean()
            self.feature_importance[original_key] = (beta_i * data_X_i_mean / y_mean).abs()
            
        return None
    
    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor, use_features_idx: tuple | list | Matrix | Tensor | None = None, **kwargs):
        """
        Fit the Linear Regression Tree Wrapper to the data.
        
        Parameters:
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y: Matrix | Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            use_features_idx, Matrix| Tensor | tuple | list of indices or None (all features)
            
            Complex fitting supportted but implicitly.
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
            self
        """
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False or isinstance(y, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
                           
        # Record the unsliced original X and y
        self.original_X = X.copy()
        self.original_y = y.copy()
            
        # Store total number of training samples to check against min_samples_split.
        total_samples = X.shape[0]
        total_features = X.shape[1]
        
        # If feature names were provided, create a map for feature names
        self._create_feature_index_map(total_features)
        
        # Set features used (will be converted to a list).
        self.use_features_idx = use_features_idx
        if self.use_features_idx is None:
            # Use all
            self.use_features_idx = list(range(X.shape[1]))
        else:
            if isinstance(self.use_features_idx, Object):
                self.use_features_idx = self.use_features_idx.flatten().to_list()
            elif isinstance(self.use_features_idx, tuple):
                self.use_features_idx = list(self.use_features_idx)
            elif isinstance(self.use_features_idx, np.ndarray):
                self.use_features_idx = self.use_features_idx.tolist()
        
        # Create the inverse feature index map
        self._create_inverse_feature_index_map(self.use_features_idx)
        
        # Slice the feature matrix and fit by lm
        self.wrapper_sliced_X = X[:,self.use_features_idx]
        
        # Fit by linear model (internally)
        LR.fit(self, self.wrapper_sliced_X, y, **kwargs)
        
        # Create feature importance mapping
        self._create_feature_importance(fitted_ = True)
            
        return self
    
    def predict(self, X: Matrix | Tensor, **kwargs) -> Matrix | Tensor:
        """
        Predict target values for samples in X.
        
        Returns:
            Matrix | Tensor, output of predictions.
        """
        # Check if fitted or not
        self.regressor._check_is_fit()
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if X.shape[1] != self.original_X.shape[1]:
            raise ValueError(f"Input feature `X` must have the same number of columns as the training data, which is {self.X.shape[1]}, but you have {X.shape[1]}")

        # Slice the new X to make dimension the same
        new_X = X[:, self.use_features_idx]

        return LR.predict(self, new_X, **kwargs)
    
    def plot_tree(self, figsize = (14, 8), **kwargs):
        """
        Plot an image representing the structure of the decision tree.
        NOT Implemented. Will raise NOT Implemented Error. Since LRTW is NOT a tree.
        """
        raise NotImplementedError("Method `plot_tree()` is NOT implemented in LRTW. Since it is not a tree model.")
        
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
        return LR.summary(self)
    
    def __repr__(self):
        try:
            self.regressor._check_is_fit()
            return f"LRTW, (N = {self.regressor.original_X.shape[0]}, k = {self.regressor.original_X.shape[1]}, {'with intercept' if self.regressor.fit_intercept == True else 'without intercept'})."
        except:
            return f"LRTW, (Not fitted, {'with intercept' if self.regressor.fit_intercept == True else 'without intercept'})."


def regression_test():
    
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
    model = LRTW(
        use = "GD",
        fit_intercept=True,
        cov=None,
        tol=1e-8,
        lr=0.02,
        max_iter=10000,
        batch_size=None,
        shuffle=True,
        l1=0.0,
        l2=0.0001
    )

    # Train the model with training data
    model.fit(X_train, y_train,
              verbosity = 1, evalset = {"Test": (X_test, y_test)}, evalmetrics = ["mse", "rmse"])

    # Predict using test data and check if it returns a 2D array-like object of shape [-1, 1]
    predictions = model.predict(X_test)
    
    # Evaluate mse
    rmse_error = regm(y_test, predictions, "rmse").compute()
    print(f"RMSE Error: {rmse_error}")
    
    # Print the summary
    print(model.summary())
    
    # Plot Feature Importance
    model.plot_feature_importance(8, [14, 8])
    
    
def random_forest_test():
    
    backend = "numpy"
    
    import pandas as pd
    from scaling import Scaling
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    def scale_and_interact(X, interaction_only=False, include_bias=False, degree=2):
        """
        Scale the input features and generate interaction terms.
        
        Parameters:
            X: pd.DataFrame or np.ndarray
            interaction_only: bool, optional (default=False) Whether to include only interaction features (no powers).
            include_bias: bool, optional (default=False) Whether to include a bias (intercept) column.
            degree: int, optional (default=2) Degree of the polynomial features.
        
        Returns:
            X_out: np.ndarray
                Scaled features with interaction terms.
            feature_names: list of str
                Names of the generated features.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        X_interacted = poly.fit_transform(X_scaled)
        feature_names = poly.get_feature_names_out(X.columns)
    
        return X_interacted, feature_names
    
    X, y = load_diabetes(return_X_y=True)
    # X, names = scale_and_interact(X, False)
    # X = pd.DataFrame(X).fillna(0)
    
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
    
    from tree import CART
    from random_forest import RandomForest
    model = RandomForest(
                      task = "regression", agg_method = "mean",
                      tree_type = LRTW,
                      n_estimators = 10,
                      n_workers = 2,
                      max_features = 0.6,
                      bootstrap_ratio = 1,
                      random_state = None,
                      tree_kwargs = {"use": "Closed", "lr": 0.01, "l2": 0.0001,
                                     "max_iter": 20000, "max_depth": 10},
                      agg_kwargs = {})
    
    # Train the model with training data
    model.fit(X_train, y_train)

    # Predict using test data and check if it returns a 2D array-like object of shape [-1, 1]
    predictions = model.predict(X_test)
    
    # Evaluate mse
    mse_error = mse(y_test.flatten().data, predictions.flatten().data)
    print(f"RMSE Error: {mse_error ** 0.5}")
    
    # Plot average feature importance
    model.plot_feature_importances_average(8)
    
    