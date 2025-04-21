# linear.py
#
# Linear Regression and Classification Models
# From MML Library by Nathmath

import math
from typing import Any

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from objtyp import Object
from matrix import Matrix
from tensor import Tensor

from baseml import Regression, Classification

from metrices import RegressionMetrics
from metrices import ClassificationMetrics


# Base Linear Models
class BaseLinearModels(Regression, Classification):
    
    __attr__ = "MML.BaseLinearModels"
    
    def __init__(self, **kwargs):
        
        super().__init__()
          
    def fit(self, **kwargs):
        raise NotImplementedError("Fit is NOT implemented in the base class.")
        
    def predict(self, **kwargs):
        raise NotImplementedError("Predict is NOT implemented in the base class.")    
    
    @staticmethod
    def _make_intercept(X: Matrix | Tensor, fit_intercept: bool = True, /, **kwargs) -> Matrix | Tensor:
        """
        Add an intercept column to the feature matrix if required.
        
        Parameters:
            X: Matrix | Tensor, The input feature data.
            fit_intercept: bool, If true a constant term is added to the predictors by appending a column of ones. Default to True.
        
        Returns:
            Matrix or Tensor: The modified matrix with an intercept column if required.
        
        """
        if fit_intercept == True:
            one_shape = [X.shape[0], 1]
            ones_like = type(X).ones(one_shape, backend = X._backend).to(backend = X._backend, device = X.device, dtype = X.dtype)
            return ones_like.hstack(X)
        else:
            return X.copy()
    
    @staticmethod
    def _log_likelihood_gaussian(**kwargs):
        """
        Gaussian log‐likelihood with dispersion = σ². Sinve error terms follows centered gaussian.
        
        Parameters:
            -------
            y_true: Matrix | Tensor, of the true values, with shape [n_samples, n_outputs]
            y_pred: Matrix | Tensor, of the predicted values, with shape [n_samples, n_outputs]
            
            
        Returns:
            -------
            Matrix | Tensor, loglikelihood measure.
        """
        try:
            # dispersion = kwargs["dispersion"]
            y_true = kwargs["y_true"]
            y_pred = kwargs["y_pred"]
        except KeyError as e:
            raise 
            
        n = y_true.shape[0]
        resid = y_true - y_pred
        sigma2 = (resid ** 2).sum() / n
        return -(n / 2.0) * ((2.0 * math.pi * sigma2).log() + 1.0)

    @staticmethod
    def _log_likelihood_inverse_gaussian(**kwargs):
        """
        Inverse‐Gaussian log‐likelihood (mean=μ, shape=λ) with dispersion = λ > 0
        
        Parameters:
            -------
            dispersion: Matrix | Tensor of scalar, dispersion measurement, which is λ.
            y_true: Matrix | Tensor, of the true values, with shape [n_samples, n_outputs]
            y_pred: Matrix | Tensor, of the predicted values, with shape [n_samples, n_outputs]
            
        Returns:
            -------
            Matrix | Tensor, loglikelihood measure.
        """
        try:
            dispersion = kwargs["dispersion"]
            y_true = kwargs["y_true"]
            y_pred = kwargs["y_pred"]
        except KeyError as e:
            raise 
        
        term1 = 0.5 * (dispersion / (2.0 * math.pi * (y_true ** 3))).log()
        term2 = -dispersion * ((y_true - y_pred) ** 2) / (2.0 * (y_pred ** 2) * y_true)
        return (term1 + term2).sum()

    @staticmethod
    def _log_likelihood_logistic(**kwargs):
        """
        Bernoulli log‐likelihood for binary y ∈ {0,1},
        
        Parameters:
            -------
            dispersion: Matrix | Tensor of scalar, dispersion measurement, which is λ.
            y_true: Matrix | Tensor, of the true values, with shape [n_samples, n_outputs]
            y_pred: Matrix | Tensor, of the predicted values, with shape [n_samples, n_outputs]
            * Note. All of the `y`s above should be post-logistic function (after called sigmoid).
            
        Returns:
            -------
            Matrix | Tensor, loglikelihood measure.
        """
        try:
            dispersion = kwargs["dispersion"]
            y_true = kwargs["y_true"]
            y_pred = kwargs["y_pred"]
        except KeyError as e:
            raise 
        
        term1 = 0.5 * (dispersion / (2 * math.pi * (y_true ** 3))).log()
        term2 = -dispersion * ((y_true - y_pred) ** 2) / (2 * (y_pred ** 2) * y_true)
        return (term1 + term2).sum()
    
    @staticmethod
    def _log_likelihood_gamma(**kwargs):
        """
        Gamma log‐likelihood (shape = α, scale = μ/α) with dispersion = α > 0
        
        Parameters:
            -------
            dispersion: Matrix | Tensor of scalar, dispersion measurement, which is α.
            y_true: Matrix | Tensor, of the true values, with shape [n_samples, n_outputs]
            y_pred: Matrix | Tensor, of the predicted values, with shape [n_samples, n_outputs]

        Returns:
            -------
            Matrix | Tensor, loglikelihood measure.
        """
        try:
            dispersion = kwargs["dispersion"]
            y_true = kwargs["y_true"]
            y_pred = kwargs["y_pred"]
        except KeyError as e:
            raise 
        
        beta = y_pred / dispersion
        return ( dispersion * beta.log() - dispersion.loggamma() + (dispersion - 1.0) * y_true - y_true / beta).sum()    

    def __repr__(self):
        return "BaseLinearModels(Abstract Class)."
    
    
# Base Linear Single Value Regression Models
class BaseSingleValueLinearRegression(BaseLinearModels):

    __attr__ = "MML.BaseSingleValueLinearRegression"

    def __init__(self, family: str = "gaussian", fit_intercept: bool = True, **kwargs) -> None:
        """
        Initialize a Base Single Value Linear Regression model.
        
        Parameters:
            family: str, the estimation family. Can be {"gaussian", "logistic", "inverse_gaussian", "gamma"}
            fit_intercept: bool, if True, the intercept is learned during fitting. Default to True.
        
        """
        
        super().__init__()
        
        # Linear Regression Family.
        self.family = family
        
        # Dispersion Measure (for loglikelihood in GLM)
        self.dispersion = None  # Will be used by children
        
        # Basic components used in linear regressions.
        self.fit_intercept = fit_intercept  # boolean
        self.coefs = None                   # shape (n_features + 1 if True, 1), with intercept
        # shape[1] = 1
        # [Alpha_0,]
        # [Beta1_0,]
        # [Beta2_0,]
        # [...]
        self.betas = None                   # shape (n_features, 1), without intercept
        # shape[1] = 1
        # [Beta1_0,]
        # [Beta2_0,]
        # [...]
        self.intercept = None               # shape (1, )
        # shape[1] = 1
        # [Alpha_0,]
        
        # Copy the data used to train
        self.original_X = None              # shape (n_samples, n_features)
        self.original_y = None              # shape (n_samples, 1)
        self.intercepted_X = None           # shape (n_samples, 1 + n_features)

    def _check_is_fit(self, throw_if_not: bool = True) -> None | bool:
        """
        Check whether the model has been fitted.

        Parameters:
            throw_if_not (bool): If True and the model is not fitted, raise a RuntimeError. 
                                            If False, return a boolean indicating whether the model is fitted or not.
        
        Returns:
            None or bool: None if `throw_if_not` is True, otherwise returns a Boolean indicating whether the model is fitted.

        Raises:
            RuntimeError: If `throw_if_not` is True and the model has not been fitted.
        """
        if self.coefs is None:
            if throw_if_not:
                raise RuntimeError("Model not fitted. Call `.fit()` before using the estimator.")
            else:
                return False
        else:
            if throw_if_not:
                return None
            else:
                return True

    def _set_params(self, coefficients: Matrix | Tensor) -> None:
        """
        Set the parameters for the base single linear regression model.
        
        Parameters:
            coefficients: Matrix or Tensor, The coefficient matrix to be used in fitting.
        
        Raises:
            ValueError: If `coefficients` is not a 2D data with reshape([-1, 1]).
        
        Returns:
            None
        """
        if len(coefficients.shape) != 2:
            raise ValueError("The `coefficients` must also be a 2d data with reshape([-1, 1])")
                    
        if self.fit_intercept:
            self.coefs = coefficients.copy()
            self.intercept, self.betas = coefficients[0], coefficients[1:]
        else:
            self.coefs = coefficients.copy()
            self.intercept, self.betas = None, coefficients.copy()
        return None

    def _fit_prep(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs) -> None:
        """
        Helper function for fit() the linear regression model on training data.
        .1 Do type checks
        .2 Do data copies
        .3 Do data extensions (including a column of ones)
         
        Parameters:
            X: X: Matrix | Tensor, the input features, must in 2D.
            y: X: Matrix | Tensor, the target values, must in 2D (shape [-1,1]).
        
         Raises:
            ValueError: If `X` and `y` are not of type Matrix or Tensor.
            ValueError: If either `X` or `y` is not a 2D array-like.
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
            raise ValueError("Input target `y` must also be a 2d data. If only one value per sample, use data.reshape([-1, 1])")
        if y.shape[1] != 1:
            raise ValueError("Input target `y` must also be a 2d data but with only 1 column since it is a single-value linear model. Consider the multi-value one if having more to fit.")
                             
            
        # Copy Training data
        self.original_X = X.to(backend=X._backend, dtype = X.dtype, device=X.device)
        self.original_y = y.to(backend=y._backend, dtype = y.dtype, device=y.device)
        
        # Extend the data
        self.intercepted_X = self._make_intercept(X, self.fit_intercept)

    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs):
        """
        Fit the linear regression model on training data.
        
        Parameters:
            X: Matrix or Tensor, the input features, must be a 2D array-like.
            y: Matrix or Tensor, the target values, must also be a 2D array-like (shape [-1,1]).
        
        Raises:
            NotImplementedError: If the fit method is not implemented in this class.
        
        """
        
        # Call prep to prepare the data
        self._fit_prep(X, y)
        
        # You have to implement a specific algorithm to train the model.
        raise NotImplementedError("Fit is NOT implemented in the base single value linear regression class.")
        
    def predict(self, X: Matrix | Tensor, *, post: Any | None = None, **kwargs) -> Matrix | Tensor:
        """
        Predict the target values using the linear model for given input features.
        
        Parameters:
            X: Matrix or Tensor, The input feature data.
            Optional:
                post: Any | None, None, or callable to be applied after doing the matrix multiplication.
        
        Returns:
            Matrix or Tensor: The predicted output in 2D array ([n_samples, 1]).
        
        Raises:
            ValueError: If model is not fitted. Call `fit()` before using predict method.
            ValueError: If `post` is given but not callable.
        """
        # This simply do a (1 + X) @ coefs
        # The output will be a [-1, 1] shaped array (single value regression).
        
        # Check if fitted
        self._check_is_fit()
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False:
            raise ValueError("Input feature `X` must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(self.original_X):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if X.shape[1] != self.original_X.shape[1]:
            raise ValueError(f"Input feature `X` must have the same second dimension as the training features. Trained {self.original_X.shape}, but you have {X.shape} instead.")
        
        # Process based on if you have intercept or not
        X_news = self._make_intercept(X, self.fit_intercept)
        y_pred = X_news @ self.coefs
        
        # Do post-process or not
        if post is None:
            return y_pred
        else:
            if callable(post) == False:
                raise ValueError("If you choose to apply a post function, then `post` argument must be a callable object and takes Matrix or Tensor as the input, as well as the output.")
            else:
                return post(y_pred)
            
    def _log_likelihood(self, /, **kwargs) -> Matrix | Tensor:
        """
        Calculate the log-likelihood under Gaussian/Any specified noise at the fitted MLE.

        Parameters:
            None (other parameters are passed via `**kwargs` for compatibility).

        Raises:
            ValueError: If not all required fields have been set or if the model is not fitted.
        
        Returns:
            Matrix | Tensor: Log-likelihood value of the model under Gaussian noise.
            
        """
        
        # Check if fitted
        self._check_is_fit()

        # Calculate the prediction for the given X
        y_pred = self.predict(self.original_X)
        
        # Call the internal loglikelihood based on family
        family = self.family.lower()
        if family == "gaussian":
            return self._log_likelihood_gaussian(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        elif family == "inverse_gaussian":
            return self._log_likelihood(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        elif family == "logistic":
            return self._log_likelihood_logistic(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        elif family == "gamma":
            return self._log_likelihood_gamma(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        else:
            return ValueError(f'Unknown family. It should be one of ["gaussian", "logistic", "inverse_gaussian", "gamma"] but you have {family}.')

    def aic(self) -> Matrix | Tensor:
        """
        Calculate the Akaike Information Criterion (AIC).

        Parameters:
            None (other parameters are passed via `**kwargs` for compatibility).
        
        Returns:
            Matrix or Tensor: The AIC value of the model.
            
        """
        # Akaike Information Criterion:
        # AIC = 2*k - 2*loglik
        # where k = (number of features + intercept if any) + 1 for σ²
        n = self.original_X.shape[0]
        p = self.coefs.shape[0]
        k = p + 1  # include σ²
        ll = self._log_likelihood()
        return 2 * k - 2 * ll
    
    def bic(self) -> Matrix | Tensor:
        """
        Calculate the Bayesian Information Criterion (BIC).
        
        Parameters:
            None (other parameters are passed via `**kwargs` for compatibility).
        
        Returns:
            Matrix or Tensor: The BIC value of the model.
            
        """
        # Bayesian Information Criterion:
        # BIC = ln(n)*k - 2*loglik
        n = self.original_X.shape[0]
        p = self.coefs.shape[0]
        k = p + 1  # include σ²
        ll = self._log_likelihood()
        return type(self.original_X)(n, backend=ll._backend, device=ll.device, dtype=ll.dtype).log() * k - 2 * ll

    def metrics_r2(self, X: Matrix | Tensor, y: Matrix | Tensor, /, **kwargs) -> Matrix | Tensor:
        """
        Calculate R-squared (R^2) value for the single value linear regression model.

        Parameters:
            X: Matrix or Tensor, The input features.
            y: Matrix or Tensor, The target values.
        
        Raises:
            ValueError: If `X` and `y` are not of type Matrix or Tensor.
            ValueError: If either `X` or `y` is not a 2D array-like.
        
        Returns:
            Matrix or Tensor: R-squared (R²) value for the model.
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
            raise ValueError("Input target `y` must also be a 2d data. If only one value per sample, use data.reshape([-1, 1])")
                    
        # Check if fitted
        self._check_is_fit()
        
        # Calculate the prediction for the given X
        y_pred = self.predict(X)
        
        # Calculate the metrics
        return RegressionMetrics(y_pred, y, metric_type = "r2").compute()

    def metrics_r2_adjusted(self, X: Matrix | Tensor, y: Matrix | Tensor, /, **kwargs) -> Matrix | Tensor:
        """
        Calculate Adjusted R-squared (Adj R^2) value for the single value linear regression model.

        Parameters:
            X: Matrix or Tensor, The input features.
            y: Matrix or Tensor, The target values.
        
        Raises:
            ValueError: If `X` and `y` are not of type Matrix or Tensor.
            ValueError: If either `X` or `y` is not a 2D array-like.
        
        Returns:
            Matrix or Tensor: R-squared (R²) value for the model.
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
            raise ValueError("Input target `y` must also be a 2d data. If only one value per sample, use data.reshape([-1, 1])")
                    
        # Check if fitted
        self._check_is_fit()
        
        # Calculate the prediction for the given X
        y_pred = self.predict(X)
        
        # Calculate the metrics
        return RegressionMetrics(y_pred, y, metric_type = "adjusted r2", k = self.betas.shape[0]).compute()

    def summary(self) -> str:
        """
        Return a formatted string summary table with:
          - N            : number of samples
          - k            : number of predictors
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
        # ensure the model has been fitted
        self._check_is_fit()
    
        # Gather statistics using the methods to compute.
        N = self.original_X.shape[0]
        k = self.betas.shape[0]
        intercept = float(self.intercept.to_list()[0])
        betas = self.betas.flatten().to_list()
        # format each beta with up to 6 significant digits.
        betas_strs = ["{:.6g}".format(b) for b in betas]
        betas_repr =  ", ".join(betas_strs)
        # Metrics values
        aic_val = float(self.aic().to_list())
        bic_val = float(self.bic().to_list())
        r2_val = float(self.metrics_r2(self.original_X, self.original_y).to_list())
        adj_r2_val = float(self.metrics_r2_adjusted(self.original_X, self.original_y).to_list())
    
        # Prepare rows of the summary table.
        rows = [
            ("N",          str(N)),
            ("k",          str(k)),
            ("Intercept",  "{:.6g}".format(intercept)),
            ("Betas",      betas_repr),
            ("AIC",        "{:.6g}".format(aic_val)),
            ("BIC",        "{:.6g}".format(bic_val)),
            ("R\u00b2",    "{:.6g}".format(r2_val)),
            ("Adj R\u00b2", "{:.6g}".format(adj_r2_val)),
        ]
    
        # Compute column widths (+2 for padding).
        param_col_w = max(len("Parameter"), *(len(p) for p, _ in rows)) + 2
        value_col_w = max(len("Value"), *(len(v) for _, v in rows)) + 2
    
        # Build table lines.
        sep = "+" + "-" * param_col_w + "+" + "-" * value_col_w + "+"
        header = (
            "| "
            + "Parameter".ljust(param_col_w - 2)
            + " | "
            + "Value".ljust(value_col_w - 2)
            + " |"
        )
    
        lines = [sep, header, sep]
        for p, v in rows:
            line = (
                "| "
                + p.ljust(param_col_w - 2)
                + " | "
                + v.ljust(value_col_w - 2)
                + " |"
            )
            lines.append(line)
        lines.append(sep)

        return "Model Summary: " + self.__attr__  + " (family " + self.family + ")\n" + "\n".join(lines)

    def __repr__(self):
        return "BaseSingleValueLinearRegression(Abstract Class)."


# Base Linear Multi Value Regression Models
class BaseMultiValueLinearRegression(BaseLinearModels):

    __attr__ = "MML.BaseMultiValueLinearRegression"

    def __init__(self, family: str = "gaussian", fit_intercept: bool = True, n_outputs: int = 1, **kwargs) -> None:
        """
        Initialize a Base Multi Value Linear Regression model.
        
        Parameters:
            family: str, the estimation family. Can be {"gaussian", "logistic", "inverse_gaussian", "gamma"}
            fit_intercept: bool, if True, the intercept is learned during fitting. Default to True.
            n_outputs: int, the number of output values (greater than 1 to be multi-values). Default 1.
        
        """
        
        super().__init__()
        
        # Multi output dimensions
        if isinstance(n_outputs, int) == False:
            raise ValueError("Parameter n_outputs represents the dimension of your target. Must be a positive int.")
        if n_outputs < 1:
            raise ValueError("Parameter n_outputs represents the dimension of your target. Must be a positive int.")
        self.n_outputs = n_outputs
        
        # Linear Regression Family.
        self.family = family
        
        # Dispersion Measure (for loglikelihood in GLM)
        self.dispersion = None  # Will be used by children
        
        # Basic components used in linear regressions.
        self.fit_intercept = fit_intercept  # boolean
        self.coefs = None                   # shape (n_features + 1 if True, n_outputs), with intercept
        #  number of outputs ->
        # [Alpha_0, Alpha_1, ...]
        # [Beta1_0, Beta1_1, ...]
        # [Beta2_0, Beta2_1, ...]
        # [...]
        self.betas = None                   # shape (n_features, n_outputs), without intercept
        #  number of outputs ->
        # [Beta1_0, Beta1_1, ...]
        # [Beta2_0, Beta2_1, ...]
        # [...]
        self.intercept = None               # shape (n_outputs, )
        #  number of outputs ->
        # [Alpha_0, Alpha_1, ...]

        # Copy the data used to train
        self.original_X = None              # shape (n_samples, n_features)
        self.original_y = None              # shape (n_samples, n_outputs)
        self.intercepted_X = None           # shape (n_samples, 1 + n_features)

    def _check_is_fit(self, throw_if_not: bool = True) -> None | bool:
        """
        Check whether the model has been fitted.

        Parameters:
            throw_if_not (bool): If True and the model is not fitted, raise a RuntimeError. 
                                            If False, return a boolean indicating whether the model is fitted or not.
        
        Returns:
            None or bool: None if `throw_if_not` is True, otherwise returns a Boolean indicating whether the model is fitted.

        Raises:
            RuntimeError: If `throw_if_not` is True and the model has not been fitted.
        """
        if self.coefs is None:
            if throw_if_not:
                raise RuntimeError("Model not fitted. Call `.fit()` before using the estimator.")
            else:
                return False
        else:
            if throw_if_not:
                return None
            else:
                return True

    def _set_params(self, coefficients: Matrix | Tensor) -> None:
        """
        Set the parameters for the base multi linear regression model.
        
        Parameters:
            coefficients: Matrix or Tensor, The coefficient matrix to be used in fitting.
        
        Raises:
            ValueError: If `coefficients` is not a 2D data with reshape([n_features + 1 if intercept, n_outputs]).
        
        Returns:
            None
        """
        #  number of outputs ->
        # [Alpha_0, Alpha_1, ...]
        # [Beta1_0, Beta1_1, ...]
        # [Beta2_0, Beta2_1, ...]
        # [...]        
        if len(coefficients.shape) != 2:
            raise ValueError("The `coefficients` must also be a 2d data with reshape([-1, 1])")
                    
        if self.fit_intercept:
            self.coefs = coefficients.copy()
            self.intercept, self.betas = coefficients[0], coefficients[1:]
        else:
            self.coefs = coefficients.copy()
            self.intercept, self.betas = None, coefficients.copy()
        return None

    def _fit_prep(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs) -> None:
        """
        Helper function for fit() the linear regression model on training data.
        .1 Do type checks
        .2 Do data copies
        .3 Do data extensions (including a column of ones)
         
        Parameters:
            X: X: Matrix | Tensor, the input features, must in 2D.
            y: X: Matrix | Tensor, the target values, must in 2D (shape [n_samples, n_outputs]).
        
         Raises:
            ValueError: If `X` and `y` are not of type Matrix or Tensor.
            ValueError: If either `X` or `y` is not a 2D array-like.
            ValueError: If the second shape of `y` does NOT equal to n_outputs.
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
            raise ValueError("Input target `y` must also be a 2d data. If only one value per sample, use data.reshape([-1, 1])")
        if y.shape[1] != self.n_outputs:
            raise ValueError(f"Input target `y` must have {self.n_outputs} outputs per sample, as indicated by the constructor parameter.")
            
        # Copy Training data
        self.original_X = X.to(backend=X._backend, dtype = X.dtype, device=X.device)
        self.original_y = y.to(backend=y._backend, dtype = y.dtype, device=y.device)
        
        # Extend the data
        self.intercepted_X = self._make_intercept(X, self.fit_intercept)

    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs):
        """
        Fit the linear regression model on training data.
        
        Parameters:
            X: Matrix or Tensor, the input features, must be a 2D array-like.
            y: Matrix or Tensor, the target values, must also be a 2D array-like (shape [-1,1]).
        
        Raises:
            NotImplementedError: If the fit method is not implemented in this class.
        
        """
        
        # Call prep to prepare the data
        self._fit_prep(X, y)
        
        # You have to implement a specific algorithm to train the model.
        raise NotImplementedError("Fit is NOT implemented in the base multiple value linear regression class.")
        
    def predict(self, X: Matrix | Tensor, *, post: Any | None = None, **kwargs) -> Matrix | Tensor:
        """
        Predict the target values using the linear model for given input features.
        
        Parameters:
            X: Matrix or Tensor, The input feature data.
            Optional:
                post: Any | None, None, or callable to be applied after doing the matrix multiplication.
        
        Returns:
            Matrix or Tensor: The predicted output in 2D array ([n_samples, n_outputs]).
        
        Raises:
            ValueError: If model is not fitted. Call `fit()` before using predict method.
            ValueError: If `post` is given but not callable.
        """
        # This simply do a (1 + X) @ coefs
        # The output will be a [-1, n_outputs] shaped array (multi value regression).
        
        # Check if fitted
        self._check_is_fit()
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False:
            raise ValueError("Input feature `X` must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(self.original_X):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if X.shape[1] != self.original_X.shape[1]:
            raise ValueError(f"Input feature `X` must have the same second dimension as the training features. Trained {self.original_X.shape}, but you have {X.shape} instead.")
        
        # Process based on if you have intercept or not
        X_news = self._make_intercept(X, self.fit_intercept)
        y_pred = X_news @ self.coefs
        
        # Do post-process or not
        if post is None:
            return y_pred
        else:
            if callable(post) == False:
                raise ValueError("If you choose to apply a post function, then `post` argument must be a callable object and takes Matrix or Tensor as the input, as well as the output.")
            else:
                return post(y_pred)

    def _log_likelihood(self, /, **kwargs) -> Matrix | Tensor:
        """
        Calculate the log-likelihood under Gaussian/Any specified noise at the fitted MLE.

        Parameters:
            None (other parameters are passed via `**kwargs` for compatibility).

        Raises:
            ValueError: If not all required fields have been set or if the model is not fitted.
        
        Returns:
            Matrix | Tensor: Log-likelihood value of the model under Gaussian noise.
            
        """
        
        # Check if fitted
        self._check_is_fit()

        # Calculate the prediction for the given X
        y_pred = self.predict(self.original_X)
        
        # Call the internal loglikelihood based on family
        family = self.family.lower()
        if family == "gaussian":
            return self._log_likelihood_gaussian(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        elif family == "inverse_gaussian":
            return self._log_likelihood(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        elif family == "logistic":
            return self._log_likelihood_logistic(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        elif family == "gamma":
            return self._log_likelihood_gamma(dispersion = self.dispersion, y_true = self.original_y, y_pred = y_pred)
        else:
            return ValueError(f'Unknown family. It should be one of ["gaussian", "logistic", "inverse_gaussian", "gamma"] but you have {family}.')

    def aic(self) -> Matrix | Tensor:
        """
        Calculate the Akaike Information Criterion (AIC).

        Parameters:
            None (other parameters are passed via `**kwargs` for compatibility).
        
        Returns:
            Matrix or Tensor: The AIC value of the model.
            
        """
        # Akaike Information Criterion:
        # AIC = 2*k - 2*loglik
        # where k = (number of features + intercept if any) + 1 for σ²
        n = self.original_X.shape[0]
        p = self.coefs.shape[0]
        k = p + 1  # include σ²
        ll = self._log_likelihood()
        return 2 * k - 2 * ll
    
    def bic(self) -> Matrix | Tensor:
        """
        Calculate the Bayesian Information Criterion (BIC).
        
        Parameters:
            None (other parameters are passed via `**kwargs` for compatibility).
        
        Returns:
            Matrix or Tensor: The BIC value of the model.
            
        """
        # Bayesian Information Criterion:
        # BIC = ln(n)*k - 2*loglik
        n = self.original_X.shape[0]
        p = self.coefs.shape[0]
        k = p + 1  # include σ²
        ll = self._log_likelihood()
        return type(self.original_X)(n, backend=ll._backend, device=ll.device, dtype=ll.dtype).log() * k - 2 * ll

    def metrics_r2(self, X: Matrix | Tensor, y: Matrix | Tensor, /, **kwargs) -> Matrix | Tensor:
        """
        Calculate R-squared (R^2) value for the single value linear regression model.

        Parameters:
            X: Matrix or Tensor, The input features.
            y: Matrix or Tensor, The target values.
        
        Raises:
            ValueError: If `X` and `y` are not of type Matrix or Tensor.
            ValueError: If either `X` or `y` is not a 2D array-like.
        
        Returns:
            Matrix or Tensor: R-squared (R²) value for the model.
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
            raise ValueError("Input target `y` must also be a 2d data. If only one value per sample, use data.reshape([-1, 1])")
                    
        # Check if fitted
        self._check_is_fit()
        
        # Calculate the prediction for the given X
        y_pred = self.predict(X)
        
        # Calculate the metrics
        return RegressionMetrics(y_pred, y, metric_type = "r2").compute()

    def metrics_r2_adjusted(self, X: Matrix | Tensor, y: Matrix | Tensor, /, **kwargs) -> Matrix | Tensor:
        """
        Calculate Adjusted R-squared (Adj R^2) value for the single value linear regression model.

        Parameters:
            X: Matrix or Tensor, The input features.
            y: Matrix or Tensor, The target values.
        
        Raises:
            ValueError: If `X` and `y` are not of type Matrix or Tensor.
            ValueError: If either `X` or `y` is not a 2D array-like.
        
        Returns:
            Matrix or Tensor: R-squared (R²) value for the model.
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
            raise ValueError("Input target `y` must also be a 2d data. If only one value per sample, use data.reshape([-1, 1])")
                    
        # Check if fitted
        self._check_is_fit()
        
        # Calculate the prediction for the given X
        y_pred = self.predict(X)
        
        # Calculate the metrics
        return RegressionMetrics(y_pred, y, metric_type = "adjusted r2", k = self.betas.shape[0]).compute()

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
        # ensure the model has been fitted
        self._check_is_fit()
    
        # Gather statistics using the methods to compute.
        N = self.original_X.shape[0]
        k = self.betas.shape[0]
        outputs = self.n_outputs
        intercepts = self.intercept.flatten().to_list()
        betas = self.betas.transpose().to_list()
        # format each beta with up to 6 significant digits.
        intercept_strs = ["{:.6g}".format(b) for b in intercepts]
        betas_strs = [["{:.6g}".format(b) for b in bs] for bs in betas]
        betas_repr =  ""
        for i, bs in enumerate(betas_strs):
            betas_repr += (", ".join(betas_strs[i])) + "; "
        intercept_repr = ", ".join(intercept_strs)
        # Metrics values
        aic_val = float(self.aic().to_list())
        bic_val = float(self.bic().to_list())
        r2_val = float(self.metrics_r2(self.original_X, self.original_y).to_list())
        adj_r2_val = float(self.metrics_r2_adjusted(self.original_X, self.original_y).to_list())
    
        # Prepare rows of the summary table.
        rows = [
            ("N",          str(N)),
            ("k",          str(k)),
            ("outputs",    str(outputs)),
            ("Intercept",  intercept_repr),
            ("Betas",      betas_repr),
            ("AIC",        "{:.6g}".format(aic_val)),
            ("BIC",        "{:.6g}".format(bic_val)),
            ("R\u00b2",    "{:.6g}".format(r2_val)),
            ("Adj R\u00b2", "{:.6g}".format(adj_r2_val)),
        ]
    
        # Compute column widths (+2 for padding).
        param_col_w = max(len("Parameter"), *(len(p) for p, _ in rows)) + 2
        value_col_w = max(len("Value"), *(len(v) for _, v in rows)) + 2
    
        # Build table lines.
        sep = "+" + "-" * param_col_w + "+" + "-" * value_col_w + "+"
        header = (
               "| "
            + "Parameter".ljust(param_col_w - 2)
            + " | "
            + "Value".ljust(value_col_w - 2)
            + " |"
        )
    
        lines = [sep, header, sep]
        for p, v in rows:
            line = (
                   "| "
                + p.ljust(param_col_w - 2)
                + " | "
                + v.ljust(value_col_w - 2)
                + " |"
            )
            lines.append(line)
        lines.append(sep)

        return "Model Summary: " + self.__attr__  + " (family " + self.family + ")\n" + "\n".join(lines)

    def __repr__(self):
        return "BaseMultiValueLinearRegression(Abstract Class)."


# Closed Form Single Value Regression
class ClosedFormSingleValueRegression(BaseSingleValueLinearRegression):
    
    __attr__ = "MML.ClosedFormSingleValueRegression"

    def __init__(self, family: str = "gaussian", fit_intercept: bool = True,
                 cov: Matrix | Tensor | None = None, **kwargs) -> None:
        """
        Initialize a Closed Form Single Value Linear Regression model.
        The family must be `gaussian` or raised with a value error.
         
        Parameters:
            family: str, the estimation family. Must be `gaussian`. Or, consider Gradient Descend Implementations.
            fit_intercept: bool, if True, the intercept is learned during fitting. Default to True.
            cov: Matrix | Tensor | None
                 Error-term covariance matrix Σ (shape m×m, m = n_samples), or just the variance,
                 Provide for GLS; leave None for OLS.
        
        """
        
        super().__init__(family=family, fit_intercept=fit_intercept, **kwargs)
        
        # Record the covariance matrix if provided
        self.cov = cov.copy() if cov is not None else None
        
        # family check. In closed form version. 
        if self.family != "gaussian":
            raise ValueError("In Closed Form Single Value Linear Regression, you must set `family` to `gaussian`. Or you should attempt to use GradientDescendSingleValueRegression instead.")

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
        
        # Call prep to prepare the data.
        self._fit_prep(X, y)
        
        # Solve it by OLS or GLS with covariance matrix.
        X = self.intercepted_X.copy()
        y = self.original_y.copy()
        n, d = X.shape
        
        # Since single value, we just perform regression on 1 column of y.
        if self.cov is None:
            # OLS:  β̂ = (XᵀX)⁻¹Xᵀy
            Xt = X.transpose()
            XtX = Xt @ X
            XtX_inv = XtX.inverse()
            coefs = XtX_inv @ (Xt @ y)
        else:
            # GLS:  β̂ = (XᵀΣ⁻¹X)⁻¹XᵀΣ⁻¹y
            sigma_inv = self.cov.inverse()
            XtSi = X.transpose() @ sigma_inv
            XtSiX = XtSi @ X
            XtSiX_inv = XtSiX.inverse()
            coefs = XtSiX_inv @ (XtSi @ y)
        
        # Set the parameters with intercept.
        self._set_params(coefs)
        return self

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
        return super().predict(X, post = None, **kwargs)

    def __repr__(self):
        try:
            self._check_is_fit()
            return f"ClosedFormSingleValueRegression(N = {self.original_X.shape[0]}, k = {self.original_X.shape[1]}, {'with intercept' if self.fit_intercept == True else 'without intercept'})."
        except:
            return f"ClosedFormSingleValueRegression(Not fitted, {'with intercept' if self.fit_intercept == True else 'without intercept'})."


# Gradient Descend Single Value Regression
class GradientDescendSingleValueRegression(BaseSingleValueLinearRegression):
    
    __attr__ = "MML.GradientDescendSingleValueRegression"

    def __init__(self, family: str = "gaussian", fit_intercept: bool = True,
                 cov: Matrix | Tensor | None = None, *,
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
        You may need to scale the data before passing them in to avoid gradient crash.
         
        Parameters:
            family: str, the estimation family. Must be `gaussian`. Or, consider Gradient Descend Implementations.
            fit_intercept: bool, if True, the intercept is learned during fitting. Default to True.
            cov: Matrix | Tensor | None
                 Error-term covariance matrix Σ (shape m×m, m = n_samples), or just the variance.
                 Provide for GLS; leave None for OLS.
            tol: float, tolerence for convergence when optimizing.
            lr: float, learning rate. Default 1e-2.
            max_iter: int, maximum number of passes over the data. Default 10000.
            batch_size: int | None, minibatch size (None = full-batch GD).
            shuffle : bool, whether to reshuffle data at each epoch
            l1: float, optional Lasso penalty λ * sign(β) (0 = plain OLS).
            l2: float, optional Ridge penalty λ * ‖beta‖² (0 = plain OLS).
            random_state: int | None, random seed, can be None.
        """
        
        super().__init__(family=family, fit_intercept=fit_intercept, **kwargs)
        
        # Record the covariance matrix if provided
        self.cov = cov.copy() if cov is not None else None
        
        # family check. In closed form version. 
        if self.family != "gaussian":
            raise ValueError("In Closed Form Single Value Linear Regression, you must set `family` to `gaussian`. Or you should attempt to use GradientDescendSingleValueRegression instead.")

        # Record all gradient method arguments
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Record regularization term
        self.l1 = l1
        self.l2 = l2
        
        # Record the random state
        self.random_state = random_state

    def _fit_gls(self, X: Matrix | Tensor, y: Matrix | Tensor, **kwargs):
        """
        Fit the linear regression model WITH/WITHOUT covariance matrix using gradient descend.
        
        Parameters:
            X: Matrix or Tensor, the input features, must be a 2D array-like.
            y: Matrix or Tensor, the target values, must also be a 2D array-like (shape [-1,1]).
        
        Returns:
            -------
            self
        """
        
        # Gaussian mode. This fit() providing gaussian result.
        #
       
        # Call prep to prepare the data.
        self._fit_prep(X, y)
        
        # Whiten helper (decorrelate X by cholesky)
        def _whiten(X: Matrix | Tensor, y: Matrix | Tensor, cov: Matrix | Tensor) -> Matrix | Tensor:
            """
            Return (W X, W y) with  WᵀW = Σ⁻¹.  If Σ is None, W = I.
            """
            if cov is None:
                return X.copy(), y.copy()
            
            def _makediag(variance: Matrix | Tensor) -> Matrix | Tensor:
                var = variance.flatten()
                return (1.0 / (var ** 0.5)).diag()
            
            # If cov is only 1 dim or 1 in any dim, regard as variance matrix
            if len(cov.shape) == 1:
                W = _makediag(cov)
            elif len(cov.shape) == 2 and cov.shape[1] == 1:
                W = _makediag(cov.flatten())
            elif len(cov.shape) == 2 and cov.shape[0] == 1:
                W = _makediag(cov.flatten())
            
            # Full covariance case
            else:            
                # Cholesky:  Σ = L Lᵀ  →  Σ⁻¹ = L⁻ᵀ L⁻¹  (so W = L⁻¹)
                L = cov.cholesky()
                W = L.inverse()
                
            if W.shape[0] != X.shape[0]:
                raise ValueError("The given `cov` can either be a variance vector or a covariiance matrix of the error terms, hence n_samples * n_samples. Incorrect dimension!")
        
            # The covariance is the covariance of the error vector, not of the predictors
            # Σ = Cov[ε], y = Xβ + ε,ε∼(0,Σ).
            return W @ X, W @ y
        
        # Prepare values required, like extended X, y, and dimensionals
        X, y = _whiten(self.intercepted_X, y, self.cov)
        n, d = X.shape
        batch_size = n if self.batch_size is None else self.batch_size
        
        # Create a 1 dimensional coefs (including intercept if any) 
        coefs = type(X).zeros(d, backend=X._backend, dtype=X.dtype).to(backend=X._backend, dtype=X.dtype, device=X.device)
        # 2 dimensional conversion
        coefs = coefs.reshape([-1, 1])
        
        # If passed a random state, use it
        if self.random_state is not None:
            np.random.seet(self.random_state)
        
        # Iterate over the maximum iteration
        for itr in range(self.max_iter):
            # If shuffle, then re-assign the row indices.
            if self.shuffle:
                idx = np.random.choice(list(range(n)), n, replace = False)
                X_, y_ = X[idx], y[idx].reshape([-1, 1])
            else:
                X_, y_ = X, y
                
            # A list of gradient norms (float)
            grad_norms = []
            
            # Perform a batch-wise gradient update.
            for start in range(0, n, batch_size):
                
                Xb = X_[start : start + batch_size]
                yb = y_[start : start + batch_size]
                nXb = Xb.shape[0]
                
                # MSE Gradient L_λ(b) = (1/2)‖y - Xb‖² + λ‖b‖² ​+ λ_1‖b‖
                grad = (2.0 / nXb) * Xb.transpose() @ (Xb @ coefs - yb) + 2 * self.l2 * coefs + self.l1 * coefs.sign()
                # Vectorized gradient for squared‑error + Ridge to prevent singularity Hessian matrix
                
                # Compute the norm of gradient
                grad_norm = (grad ** 2).sum() ** 0.5
                grad_norms.append(grad_norm.to_list())
                
                # Use the plain gradient descend approach to update the parameters
                # @todo, maybe use nn.Optimizer I implemented to do this stuff
                coefs -= self.lr * grad
                
            # If the average L2 norm of the gradient is lower then tolerance, then stop training
            if np.array(grad_norms).mean() < self.tol:
                self._set_params(coefs)
                return self
        
        self._set_params(coefs)
        return self

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
        
        # Call _fit_gls for backend
        return self._fit_gls(X, y, **kwargs)

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
        return super().predict(X, post = None, **kwargs)

    def __repr__(self):
        try:
            self._check_is_fit()
            return f"GradientDescendSingleValueRegression(N = {self.original_X.shape[0]}, k = {self.original_X.shape[1]}, {'with intercept' if self.fit_intercept == True else 'without intercept'})."
        except:
            return f"GradientDescendSingleValueRegression(Not fitted, {'with intercept' if self.fit_intercept == True else 'without intercept'})."



###############################################################################
#
# Tests

def test_BaseSingleValueLinearRegression():
    
    # Create toy data
    np.random.seed(None)
    N, k = 100, 3
    X = np.random.randn(N, k)
    
    # true parameters: intercept=4.0, betas=[2.5, -1.2, 0.7]
    true_params = np.array([[4.0], [2.5], [-1.2], [0.7]])
    X_int = np.hstack([np.ones((N,1)), X])
    y = X_int.dot(true_params) + np.random.randn(N,1) * 0.7  # add small noise

    # Fit OLS via NumPy (for test purpose)
    coefs = np.linalg.inv(X_int.T @ X_int) @ (X_int.T @ y)
    
    # Matrix-ize
    backend = "torch"
    X = Matrix(X, backend=backend)
    y = Matrix(y, backend=backend)

    # Instantiate and manually inject fitted state
    lr = BaseMultiValueLinearRegression(fit_intercept=True, n_outputs=1)
    try:
        lr.fit(X, y)
    except:
        # NOT IMPLEMENTED exception is raised, but never mind.
        pass
    lr.coefs = Matrix(coefs, backend=backend)
    lr.intercept  = Matrix(float(coefs[0].tolist()[0]), backend=backend)
    lr.betas = Matrix(coefs[1:].reshape(-1,1), backend=backend)

    # Print the summary table
    print(lr.summary())
    
    
def test_ClosedFormSingleValueRegression():
    
    backend = "torch"
    import pytest
    
    # OLS with intercept on y = 2x + 1
    X1 = np.array([[0.0], [1], [2], [3], [4]])
    y1 = 2 * X1 + 1
    model1 = ClosedFormSingleValueRegression()
    model1.fit(Matrix(X1, backend = backend), Matrix(y1, backend = backend))
    preds1 = model1.predict(Matrix(X1, backend = backend)).to_numpy_array()
    assert np.allclose(preds1, y1), "OLS with intercept failed to recover y = 2x + 1"

    # OLS without intercept on y = 3x
    X2 = np.array([[1.0], [2], [3]])
    y2 = 3 * X2
    model2 = ClosedFormSingleValueRegression(fit_intercept=False)
    model2.fit(Matrix(X2, backend = backend), Matrix(y2, backend = backend))
    preds2 = model2.predict(Matrix(X2, backend = backend)).to_numpy_array()
    assert np.allclose(preds2, y2), "OLS without intercept failed to recover y = 3x"

    # GLS with Σ = I should match OLS on y = 4x + 2
    X3 = np.array([[0.0], [1], [2]])
    y3 = 4 * X3 + 2
    cov_identity = Matrix(np.eye(3))
    model3 = ClosedFormSingleValueRegression(cov=cov_identity)
    model3.fit(Matrix(X3, backend = backend), Matrix(y3, backend = backend))
    preds3 = model3.predict(Matrix(X3, backend = backend)).to_numpy_array()
    assert np.allclose(preds3, y3), "GLS with identity covariance did not match OLS result"

    # Invalid family argument must raise ValueError
    with pytest.raises(ValueError):
        ClosedFormSingleValueRegression(family="poisson")

    # See predicting before fitting must raise RuntimeError
    X5 = np.array([[1], [2]])
    model5 = ClosedFormSingleValueRegression()
    with pytest.raises(RuntimeError):
        model5.predict(Matrix(X5, backend = backend))

    # repr should include correct sample size (N) and feature count (k)
    X6 = np.random.RandomState(0).rand(5, 2)
    y6 = X6 @ np.array([[1.5], [-0.5]]) + 0.3
    model6 = ClosedFormSingleValueRegression()
    model6.fit(Matrix(X6, backend = backend), Matrix(y6, backend = backend))
    print(repr(model6))
    
    
def test_GradientDescendSingleValueRegression():
    
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
    model = GradientDescendSingleValueRegression(
        fit_intercept=True,
        cov=None,
        tol=1e-8,
        lr=1e-2,
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
    model.summary()
    
    
if __name__ == "__main__":
    
    test_BaseSingleValueLinearRegression()
    test_ClosedFormSingleValueRegression()
    test_GradientDescendSingleValueRegression()
    