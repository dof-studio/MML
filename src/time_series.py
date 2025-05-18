# time_series.py
#
# A time series model implementation
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any
    
from .matrix import Matrix
from .tensor import Tensor


# Base class
class BaseTimeSeriesModel:
    
    __attr__ = "MML.BaseTimeSeriesModel"
    
    def __init__(self, backend="numpy"):
        self.backend = backend.lower()
        
    def _is_matrix(self, data, *, raise_error: bool = False):
        """
        Checks whether the provided data is an instance of Matrix.
    
        Args:
            data (Any): The data to be checked.
            raise_error (bool, optional): If True and data is not a Matrix, raises ValueError. Defaults to False.
    
        Returns:
            bool: Whether the data is an instance of Matrix.
    
        Raises:
            ValueError: If `raise_error` is True and the data is not a Matrix.
        
        """
        if isinstance(data, Matrix):
            return True
        else:
            if raise_error == True:
                raise ValueError(f"Time series models require Matrix to be the input format. Try converting your data {type(data)} into a Matrix.")
            return False
    
    def _to_matrix(self, data):
        """
        Converts the input data into a Matrix object.
        
        Args:
            data: The input data which can be of various types including Matrix, Tensor, numpy.ndarray, and torch.Tensor.
        
        Returns:
            Matrix: A new Matrix object created from the input data.
        
        Raises:
            ValueError: If `data` is not an instance of Matrix (but Tensor).
        
        """
        if isinstance(data, Matrix):
            return data
        elif isinstance(data, Tensor):
            raise ValueError("Time series models only accept Matrix as the data type. Please convert to Matrix using data=Matrix(data.data)")
        elif isinstance(data, np.ndarray):
            return Matrix(data, backend=self.backend)
        elif isinstance(data, torch.Tensor):
            return Matrix(data, backend=self.backend)
        
        # Try to convert to Matrix anyway.
        return Matrix(data, backend=self.backend)
    
    @staticmethod
    def _create_lagged_matrices(series: Matrix, order: int) -> Matrix:
        """
        Constructs the regression design matrix X and target vector Y.
        For t from order to n-1, each row of X is [1, y[t-1], ..., y[t-order]].
        Y is the column vector [y[t]].
    
        Args:
            series (Matrix): The time series data with a 'data' attribute for values.
            order (int): Number of lags to include in the lagged features.
    
        Returns:
            tuple[Matrix, Matrix]: A tuple containing two matrices. 
                                  The first matrix is the feature matrix X with lagged values and possibly an intercept,
                                  and the second matrix is the target vector Y consisting of the original time series data.
        
        """
        
        # Series.data is a 1D array or 1D tensor.
        if len(series.shape) > 1:
            if len(series.shape) == 2 and series.shape[1] == 1:
                series = series.flatten()
            else:
                raise ValueError("To create a lagged matrix, input data must be a 1D array")
        n = series.shape[0]
        X_list, Y_list = [], []
        
        # Loop over time indices where lagged values are available.
        for t in range(order, n):
            # Build lag vector in reverse order so that the most recent observation is first.
            if series._is_numpy:
                lag = series.data[t-order:t][::-1]  # reverse order
                row = np.concatenate(([1], lag))
            else:
                # For torch, use tensor operations. We assume torch is available.
                lag = series.data[t-order:t].flip(0)  # flip to reverse the order
                row = torch.cat((torch.tensor([1], dtype=series.data.dtype), lag))
            X_list.append(row)
            Y_list.append(series.data[t])
            
        if series._is_numpy:
            X = Matrix(np.array(X_list), backend=series._backend, dtype=series.dtype, device=series.device)
            Y = Matrix(np.array(Y_list).reshape(-1, 1), backend=series._backend, dtype=series.dtype, device=series.device)
        else:
            X = Matrix(torch.stack(X_list), backend=series._backend, dtype=series.dtype, device=series.device)
            Y = Matrix(torch.tensor(Y_list, dtype=series.data.dtype).unsqueeze(1), backend=series._backend, dtype=series.dtype, device=series.device)
        return X, Y

    @staticmethod
    def _differencing(series: Matrix, d: int, axis: int = 0) -> Matrix:
        """
        Difference the input series d times.
        
        Args:
            series (Matrix): The original n-dimensional time series data.
            d (int): The number of times to difference the data.
            axis (int): The axis to difference the data.
            
        Returns:
            Matrix: The differenced time series. Lost elements will be padded to NaN.
        """
        diff_matrix = series.copy()
        
        # Only 1 dimension, regard as a single array.
        if len(diff_matrix.shape) == 1:

            # Differencing for d times.
            for i in range(d):
                # For a 1D time series, compute differences between consecutive elements.
                diff_matrix = diff_matrix[1:] - diff_matrix[:-1]
            
            # Padding elems.
            pad = Matrix(np.repeat(np.NaN, d), backend=series._backend, dtype=series.dtype, device=series.device)
            
            return pad.append(diff_matrix)
        
        # Perform differencing d times along the given axis.
        # Default: axis = 0, along the row sequence.
        else:
            for i in range(d):
                # Create slicers for the two parts of the array.
                slicer1 = [slice(None)] * len(diff_matrix.shape)
                slicer2 = [slice(None)] * len(diff_matrix.shape)
                slicer1[axis] = slice(1, None)     # from index 1 to the end along axis
                slicer2[axis] = slice(None, -1)    # from index 0 to one before the end along axis
                
                # Compute the difference along the axis.
                diff_matrix = diff_matrix[tuple(slicer1)] - diff_matrix[tuple(slicer2)]
            
            # Determine the shape for the padding: same as original shape except the given axis has length d.
            pad_shape = list(series.shape)
            pad_shape[axis] = d
            
            # Create a padding matrix filled with NaN.
            pad = Matrix(np.full(pad_shape, np.NaN), backend=series._backend, dtype=series.dtype, device=series.device)
            
            return pad.append(diff_matrix, axis=axis)


# ARIMAX implementation
class ARIMAX(BaseTimeSeriesModel):
    """
    ARIMAX model class that implements ARIMA with exogenous regressors.
    The model is defined as:
    
        φ(B)(1-B)^d y_t = β' x_t + θ(B) ε_t
    
    where φ(B) represents the AR part, (1-B)^d is the differencing operator,
    β' x_t includes exogenous variables, and θ(B) is the MA component.
    
    This class is modularized so that functions for differencing, AR estimation,
    and MA estimation (via iterative non-linear fitting, Kalman filter, and MLE)
    can be reused or extended in future subclasses.
    """
    
    __attr__ = "MML.ARIMAX"
    
    def __init__(self, p=1, d=0, q=0, constant=True, backend="numpy",* ,
                 lr=1e-3, tol=1e-8, max_iter=1000, delta=1e-8):
        """
        Initializes the ARIMAX model.
        
        Args:
            p (int): Order of the autoregressive part.
            d (int): Degree of differencing.
            q (int): Order of the moving average part.
            backend (str): Computational backend ("numpy" or "torch").
            lr (float): Learning rate when doing MA estimation.
            tol (float): Convergence threshold when doing MA estimation.
            max_iter (int): Max number of iterations allowed to be performed when doing MA estimation.
            delta (float): Micro-element to compute gradient offset to the original vector when doing MA estimation.
        """
        super().__init__(backend)
        
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.delta = delta
        # Model Realted
        self.p = p                  # AR order
        self.d = d                  # Differencing order
        self.q = q                  # MA order
        self.ar_params = None       # AR coefficients (including intercept)
        self.ma_params = None       # MA coefficients
        self.beta_params = None     # Coefficients for exogenous regressors
        # Data Related
        self.fitted = False         # Whether fitted or not
        self.residuals = None       # Residuals after ARX estimation
        self.original_y = None      # The original data used to fit
        self.differenced_y = None   # The data used to fit after differencing
        self.exogen_matrix = None   # The exogeneous data used to fit
        # Interim Data
        self.X_ar = None            # X_ar matrix computed during the AR estimation
        self.X_aligned = None       # X_aligned matrix computed during the AR estimation
        self.Y_ar = None            # Y_ar matrix computed during the AR estimation

    def _estimate_ar_beta(self, series: Matrix, exogenous: Matrix = None):
        """
        Estimate AR and beta coefficients via least squares on the differenced data.
        Uses lagged matrices for AR terms and appends exogenous regressors if provided.
        
        Args:
            series (Matrix): The (differenced) time series.
            exogenous (Matrix or None): Matrix of exogenous regressors.
        
        Returns:
            tuple(Matrix, Matrix, Matrix): A tuple containing the estimated 
            parameters for AR (including intercept), beta coefficients, and the 
            residuals from the ARX regression.
            
        Raises:
            ValueError: If the input data is not a Matrix.
            ValueError: If the exogenous matrix does not have the same 1st dimension with series.
            
        """
        # We check Matrix format strictly (raises error if mismatched)
        if self._is_matrix(series, raise_error=True) == False:
            pass
        if exogenous is not None:
            if self._is_matrix(exogenous, raise_error=True) == False:
                pass
        
        # Check dimension, if original input series and matrix does not have the same 1st dimension
        if exogenous is not None:
            exogenous = self._to_matrix(exogenous)
            if exogenous.shape[0] != series.shape[0]:
                raise ValueError(f"Time series data and exgenous matrix should have the same 1st dimension but are {exogenous.shape[0]} and {series.shape[0]} respectively")
        
        # If differenced, drop the first self.d elements.
        if self.d > 0:
            series = series[self.d:]
            if exogenous is not None:
                exogenous = exogenous[self.d:]
        
        # Create lagged matrix for AR estimation using BaseTimeSeriesModel's method.
        self.X_ar, self.Y_ar = self._create_lagged_matrices(series, order=self.p)
        
        # If exogenous regressors are provided, align and augment the design matrix.
        if exogenous is not None:
            # Align exogenous data with the lagged design (drop first self.p rows)
            exog_aligned = exogenous[self.p:]
            # Augment the AR design matrix with exogenous columns
            if len(exog_aligned.shape) == 1:
                exog_aligned = exog_aligned.reshape([-1,1])
            self.X_aligned = self.X_ar.hstack(exog_aligned)
        else:
            self.X_aligned = self.X_ar.copy()

        # Solve for parameters using the Matrix class least_square static method.
        params = Matrix.least_square(self.X_aligned, self.Y_ar, backend=self.backend, dtype=series.dtype)
        
        # Separate parameters: first (p+1) are for AR (including intercept) and the rest for beta.
        ar_params = params[:self.p+1]
        beta_params = None
        if exogenous is not None and params.shape[0] > (self.p+1):
            beta_params = params[self.p+1:]
            
        # Compute residuals: residuals = Y_ar - (X * params)
        X_params = self.X_aligned @ params
        residuals = self.Y_ar - X_params
        
        #      AR PARAMS, EXOG PARAMS, RESID
        return ar_params, beta_params, residuals

    def _estimate_ma_iterative(self, residuals: Matrix):
        """
        Estimate MA coefficients using an iterative non-linear fitting procedure.
        
        Args:
            residuals (Matrix): The residuals from ARX estimation.
        
        Returns:
            Matrix: The estimated MA coefficients.
        """
        n = residuals.shape[0]
        
        # Build the lag matrix.
        lagged, actual = self._create_lagged_matrices(residuals, order=self.q)
        
        # Get away the 1s in the lagged
        lagged = lagged[:,1:]
        
        # Transform it into standard normal
        lagged_std = lagged.std()
        lagged = lagged / lagged_std
        actual = actual / lagged_std
        
        # Initialize MA coefficients theta as zeros.
        theta = Matrix.zeros((self.q, 1), backend=self.backend).to(backend=self.backend, dtype=residuals.dtype, device=residuals.device)
        prev_loss = float('inf')
        
        # Iterative optimization loop.
        for iteration in range(self.max_iter):
            # Vectorized prediction: shape (n - q, 1) = lagged dot theta
            predictions = lagged @ theta
            
            # Compute the error and loss (mean squared error)
            error = actual - predictions
            loss = (error.transpose().__matmul__(error)).data[0, 0] / (n - self.q)
            
            # Check for convergence.
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            
            # Finite differences: estimate the gradient for each theta_i.
            # Instead of looping over each theta, we note that for each column i,
            # predictions_plus = predictions + delta * lagged[:, i]
            # so error_plus = error - delta * lagged[:, i]
            # We compute the loss for each parameter in a vectorized manner.
            error_array = error.data.reshape(-1)           # shape: (n - q,)
            # lagged.data has shape (n - q, q)
            # Compute loss_plus for each parameter i:
            loss_plus_array = np.mean((error_array.reshape(-1, 1) - self.delta * lagged.data)**2, axis=0)  # shape: (q,)
            grad_array = (loss_plus_array - loss) / self.delta     # finite-difference approximation (shape: (q,))
            
            # Wrap the gradient as a Matrix and update theta.
            grad = Matrix(grad_array.reshape(self.q, 1), backend=self.backend)
            theta = theta - (grad * self.lr)
        
        return theta * lagged_std

    def _estimate_ma_kalman(self, residuals: Matrix):
        """
        Estimate MA coefficients using a Kalman filter based approach.
        The ARIMA model is recast in state-space form and the filter is used to estimate the latent MA effects.
        
        Args:
            residuals (Matrix): The residuals from ARX estimation.
        
        Returns:
            Matrix: The estimated MA coefficients.
        """
        # Initialize state-space parameters.
        n = len(residuals.data)
        # For simplicity, we assume the state vector size equals q.
        state = Matrix.zeros((self.q, 1), backend=self.backend)
        covariance = Matrix.zeros((self.q, self.q), backend=self.backend)
        process_noise = 0.01
        measurement_noise = 0.1
        
        # Placeholder: accumulate information to infer MA coefficients.
        # In practice, one would define a proper state transition and observation model.
        theta_sum = Matrix.zeros((self.q, 1), backend=self.backend)
        count = 0
        
        for t in range(self.q, n):
            # Prediction step: Here, we assume identity transition for simplicity.
            state_pred = state  # state prediction (could be state transition matrix * state)
            covariance_pred = covariance  # plus process noise: covariance + process_noise * I
            if self.backend == "numpy":
                covariance_pred = covariance_pred + Matrix(process_noise * np.eye(self.q), backend=self.backend)
            else:
                # For torch, assume similar operation.
                covariance_pred = covariance_pred + Matrix(process_noise * np.eye(self.q), backend=self.backend)
            
            # Observation: actual residual at time t.
            innovation = Matrix([[residuals.data[t]]], backend=self.backend)  # scalar as a 1x1 matrix
            
            # Kalman gain calculation (using simplified scalar observation model)
            # Here, observation matrix H is assumed to be [1, 0, ..., 0]
            H = Matrix([[1] + [0]*(self.q-1)], backend=self.backend)
            S = H.__matmul__(covariance_pred).__matmul__(H.transpose()) + Matrix([[measurement_noise]], backend=self.backend)
            # Inverse S is computed via Matrix.inverse (assuming S is 1x1).
            K = covariance_pred.__matmul__(H.transpose()).__matmul__(S.inverse())
            
            # Update step: update state with innovation scaled by Kalman gain.
            state = state_pred + K.__matmul__(innovation - H.__matmul__(state_pred))
            covariance = (Matrix.zeros((self.q, self.q), backend=self.backend) - K.__matmul__(H)).__matmul__(covariance_pred)
            
            # Accumulate state estimates to infer MA coefficients.
            theta_sum = theta_sum + state
            count += 1
        
        # Average state estimates as the final MA coefficient estimates.
        theta_est = theta_sum * (1.0 / count)
        return theta_est

    def _estimate_ma_mle(self, residuals: Matrix):
        """
        Estimate MA coefficients using Maximum Likelihood Estimation (MLE).
        The likelihood function is optimized iteratively via a numerical method.
        
        Args:
            residuals (Matrix): The residuals from ARX estimation.
        
        Returns:
            Matrix: The estimated MA coefficients.
        """        
        # Build the lag matrix.
        lagged, actual = self._create_lagged_matrices(residuals, order=self.q)
        
        # Get away the 1s in the lagged
        lagged = lagged[:,1:]
        
        # Transform it into standard normal
        lagged_std = lagged.std()
        lagged = lagged / lagged_std
        actual = actual / lagged_std
                
        # Initialize MA coefficients theta as zeros.
        theta = Matrix.zeros((self.q, 1), backend=self.backend).to(backend=self.backend, dtype=residuals.dtype, device=residuals.device)
        prev_ll = -float('inf')
        
        # Iterative optimization loop.
        for iteration in range(self.max_iter):
            # Compute predicted values using vectorized multiplication.
            # predicted shape: (n - q, 1) = lagged dot theta
            predicted = lagged @ theta
            error = actual - predicted
            # Pseudo log-likelihood: negative sum of squared errors.
            ll = error.transpose() @ error
            ll = float(-ll.data[0, 0])
            
            # Check convergence.
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
            
            # Numerical gradient computation
            delta = self.delta
            # Instead of looping over each parameter, perturb all simultaneously.
            # Create a (q, q) identity matrix; each column will correspond to perturbing one parameter.
            I = Matrix.identity(self.q, backend=self.backend).to(backend=self.backend, dtype=residuals.dtype, device=residuals.device)
            
            # Broadcast theta: shape (q, 1) + delta * I => shape (q, q), where each column is a perturbed theta.
            theta_plus = theta.data + delta * I  # shape: (q, q)
            
            # Compute predictions for each perturbed theta in one go:
            # lagged.data: (n - q, q), theta_plus: (q, q) -> result: (n - q, q)
            predictions_plus = lagged.data @ theta_plus
            
            # Each column corresponds to predictions when a single parameter is perturbed.
            # Compute errors for each variant.
            error_plus = actual.reshape([-1, 1]) - predictions_plus  # shape: (n - q, q)
            
            # Compute the loss (negative SSE) for each perturbation.
            # Sum the squared errors along the time dimension.
            ll_plus = error_plus**2
            ll_plus = -ll_plus.sum(axis=0) # shape: (q,)
            
            # Finite-difference gradient for each parameter.
            grad_array = (ll_plus - ll) / delta  # shape: (q,)
            grad = grad_array.reshape([self.q, 1])
            
            # Gradient ascent step (to maximize the likelihood).
            theta = theta + (grad * self.lr)
        
        return theta * lagged_std

    def fit(self, y: Matrix | Any, exogenous: Matrix | Any = None, ma_method='mle'):
        """
        Fit the ARIMAX model on the provided time series and exogenous data.
        
        Args:
            y (Matrix or array-like): The time series data.
            exogenous (Matrix or array-like or None): The exogenous regressor data.
            ma_method (str): The method for MA estimation ('iterative', 'kalman', or 'mle').
        
        Returns:
            self: Fitted ARIMAX model with estimated parameters.
        """
        # For API methods, we allow inputs to be convertable.
        
        # Convert inputs to Matrix objects.
        y_matrix = self._to_matrix(y)
        self.original_y = y_matrix.copy()
        
        # Convert exog matrix to Matrix objects.
        if exogenous is not None:
            exog_matrix = self._to_matrix(exogenous)
        else:
            exog_matrix = None
        self.exogen_matrix = exog_matrix.copy() if exog_matrix is not None else None
        
        # Difference the series if required.
        if self.d > 0:
            diff_y = self._differencing(y_matrix, self.d)
        else:
            diff_y = y_matrix
        self.differenced_y = diff_y.copy()
        
        # Estimate AR and beta parameters via least squares.
        self.ar_params, self.beta_params, self.residuals = self._estimate_ar_beta(diff_y, exog_matrix)
        
        # Estimate MA parameters using the chosen method.
        if self.q > 0:
            if ma_method.lower() == 'iterative':
                self.ma_params = self._estimate_ma_iterative(self.residuals)
            elif ma_method.lower() == 'kalman':
                self.ma_params = self._estimate_ma_kalman(self.residuals)
            elif ma_method.lower() == 'mle':
                self.ma_params = self._estimate_ma_mle(self.residuals)
            else:
                raise ValueError("Unsupported MA estimation method. Choose one from 'iterative', 'kalman', or 'mle'.")
        else:
            self.ma_params = None
        
        self.fitted = True
        return self

    def predict(self, steps: int, exogenous_future: Matrix | Any = None):
        """
        Predict future values using the fitted ARIMAX model.
        
        The forecasting procedure iterates step-by-step. For each forecast step,
        it computes the AR component using the intercept and lagged values,
        adds the exogenous contribution (if available) and assumes the future 
        MA error contributions to be zero. If differencing was applied during fit,
        the forecasted differenced values are integrated back to the original scale
        using the Matrix.cumsum method.
        
        Args:
            steps (int): The number of steps ahead to forecast.
            exogenous_future (Matrix or array-like or None): Future values for exogenous regressors.
            
        Returns:
            Matrix: Forecasted values as a column vector.
            
        Raises:
            ValueError: If the model is not fitted or if exogenous_future dimensions do not match steps.
        """
        # Ensure the model has been fitted.
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        # Process future exogenous regressors if they were used in training.
        if self.beta_params is not None:
            if exogenous_future is None:
                raise ValueError("Exogenous regressors were used during training; exogenous_future must be provided for forecasting.")
            exog_future = self._to_matrix(exogenous_future)
            if exog_future.shape[0] != steps:
                raise ValueError(f"exogenous_future must have {steps} rows, got {exog_future.shape[0]}")
            if len(exog_future.shape) == 1:
                exog_future = exog_future.reshape([-1, 1])
        else:
            exog_future = None

        # Retrieve the last p values from the differenced series if d>0, else from original series.
        if self.d > 0:
            # For forecasting, we use the differenced series used during fit.
            history = self.differenced_y.data.tolist()  # converting to a list for easier indexing
        else:
            history = self.original_y.data.tolist()
            
        # Prepare an empty list to store residual terms
        residual_diff = None
        if self.q > 0:
            residual_diff = self.residuals[-self.q:]
        
        # Prepare an empty list to store forecasted values for the differenced series.
        forecast_diff = []

        # For each forecast step, compute the AR forecast iteratively.
        for step in range(steps):
            # Construct the regressor vector:
            # The vector comprises [1, y_{t-1}, y_{t-2}, ..., y_{t-p}].
            # Use historical data when available, otherwise use previously forecasted values.
            # We take the last p observations from 'history' (which includes prior forecasts).
            if len(history) < self.p:
                raise ValueError("Not enough historical data to perform prediction.")
            recent_vals = history[-self.p:]
            # Reverse the order to match the lag ordering (most recent first).
            recent_vals_reversed = recent_vals[::-1]
            # Create the base regressor vector with intercept.
            regressor_list = [1] + recent_vals_reversed
            regressor = Matrix(regressor_list, backend=self.backend)
            regressor = regressor.reshape((1, len(regressor_list)))  # shape: (1, p+1)
            
            # If exogenous regressors are used, append the corresponding future row.
            if exog_future is not None:
                # Extract the exogenous row for the current forecast step.
                exog_row = exog_future[step]
                # Ensure exog_row is a Matrix row.
                exog_row = Matrix(exog_row, backend=self.backend)
                exog_row = exog_row.reshape((1, exog_future.shape[1]))
                # Augment the regressor vector.
                regressor = regressor.hstack(exog_row)
            
            # Combine AR and exogenous parameters.
            # The estimated parameters are stored as:
            #   - self.ar_params: shape (p+1, 1)
            #   - self.beta_params: shape (n_exog, 1) if exogenous regressors exist.
            if self.beta_params is not None:
                params = self.ar_params.vstack(self.beta_params)
            else:
                params = self.ar_params

            # Forecasted AR value is given by regressor * params (matrix multiplication).
            forecast_value = (regressor @ params).data[0, 0]
            
            # Forecasted MA value is iteratively determined
            if self.q > 0 and residual_diff is not None:
                forecast_value += float((self.ma_params.transpose() @ residual_diff)[0,0])
                # Shift up 1 row
                residual_diff_shape = (1, residual_diff.shape[1])
                residual_diff = residual_diff[1:]
                residual_diff = Matrix.zeros(residual_diff_shape).append(residual_diff)
                
            # Append the forecasted differenced value.
            forecast_diff.append(forecast_value)
            # Update history with the forecast value so that it can be used in later steps.
            history.append(forecast_value)
        
        # If differencing was applied during training, integrate the forecasts.
        if self.d > 0:
            # Rebuild the chain of differenced series starting from the original series.
            chain = [self.original_y]
            for i in range(self.d):
                chain.append(self._differencing(chain[-1], 1))
            # Now, chain[0] is the original series, chain[1] is the first difference, ..., chain[d] is the dth difference.
            # The forecasts we produced (forecast_diff) are for the dth differenced series.
            forecast_matrix = Matrix(forecast_diff, backend=self.backend).reshape((steps, 1))
            # To integrate, iterate from the dth difference back to the original scale.
            # For each integration step, add the last observed value from the appropriate differenced series.
            for level in range(self.d):
                # For integration, use the last observed value from chain[d - level - 1].
                last_val = chain[self.d - level - 1].data[-1]
                forecast_matrix = forecast_matrix.cumsum() + Matrix([[last_val]], backend=self.backend)
            return forecast_matrix.reshape((steps, 1))
        else:
            return Matrix(forecast_diff, backend=self.backend).reshape((steps, 1))

    def summary(self):
        """
        Print a summary of the ARIMAX model, including parameter estimates and t-test results.
        """
        if not self.fitted:
            print("Model is not fitted yet.")
            return

        print("========== ARIMAX Model Summary ==========")
        print(f"Model Orders: AR(p={self.p}), d={self.d}, MA(q={self.q})")
        print("----- AR Parameters (including intercept) -----")
        print(self.ar_params.data)
        if self.beta_params is not None:
            print("----- Beta (Exogenous) Parameters -----")
            print(self.beta_params.data)
        if self.ma_params is not None:
            print("----- MA Parameters -----")
            print(self.ma_params.data)


# ===================== Test Cases =====================

def test_statsmodels():
    
    import statsmodels.api as sm
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA as smARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX as smSARIMAX
    
    np.random.seed(0)
    n = 100

    # ------------------- AR(2) Model -------------------
    # Generate an AR(2) process with true parameters: intercept=0.5, phi1=0.7, phi2=-0.2.
    y_ar = [0.0, 0.0]
    for t in range(2, n):
        new_val = 0.5 + 0.7 * y_ar[t-1] - 0.2 * y_ar[t-2] + np.random.normal(scale=0.1)
        y_ar.append(new_val)
    y_ar = np.array(y_ar)
    
    # Fit AR model using statsmodels AutoReg.
    print("=== Statsmodels AR(2) ===")
    ar_model_sm = AutoReg(y_ar, lags=2, old_names=False).fit()
    print("Estimated parameters (AR):")
    print(ar_model_sm.summary())
    
    # Forecast the next 5 time steps.
    forecast_ar_sm = ar_model_sm.predict(start=len(y_ar), end=len(y_ar)+4)
    print("Forecast (next 5 steps):")
    print(forecast_ar_sm)
    
    # ------------------- MA(1) Model -------------------
    # Generate synthetic data from a MA(1) process (here noise-only data for simplicity).
    np.random.seed(42)
    y_ma = np.array([np.random.normal() for _ in range(n)])
    
    # Fit MA model using ARIMA with order (0,0,1).
    print("\n=== Statsmodels MA(1) ===")
    ma_model_sm = smARIMA(y_ma, order=(0, 0, 1)).fit()
    print("Estimated parameters (MA):")
    print(ma_model_sm.summary())
    
    forecast_ma_sm = ma_model_sm.forecast(steps=5)
    print("Forecast (next 5 steps):")
    print(forecast_ma_sm)
    
    # ------------------- ARMA(1,1) Model -------------------
    # Generate synthetic ARMA(1,1) process with true parameters: intercept=0.2, phi1=0.5, theta1=-0.3.
    np.random.seed(117)
    y_arma = [0.0, 0.0, 0.0]
    for t in range(3, n):
        ar_term = 0.5 * y_arma[t-1]
        noise = np.random.normal(scale=0.2)
        new_val = 0.2 + ar_term + noise - 0.3 * noise
        y_arma.append(new_val)
    y_arma = np.array(y_arma)
    
    # Fit ARMA model using ARIMA with order (1,0,1).
    print("\n=== Statsmodels ARMA(1,1) ===")
    arma_model_sm = smARIMA(y_arma, order=(1, 0, 1)).fit()
    print("Estimated parameters (ARMA):")
    print(arma_model_sm.summary())
    
    forecast_arma_sm = arma_model_sm.forecast(steps=5)
    print("Forecast (next 5 steps):")
    print(forecast_arma_sm)
    
    # ------------------- ARIMA(1,1,1) Model -------------------
    # Generate an integrated series by cumulative summing white noise.
    y_arima = np.cumsum(np.random.normal(size=n))
    
    # Fit ARIMA model with order (1,1,1).
    print("\n=== Statsmodels ARIMA(1,1,1) ===")
    arima_model_sm = smARIMA(y_arima, order=(1, 1, 1)).fit()
    print("Estimated parameters (ARIMA):")
    print(arima_model_sm.params)
    
    forecast_arima_sm = arima_model_sm.forecast(steps=5)
    print("Forecast (next 5 steps):")
    print(forecast_arima_sm)
    
    # ------------------- SARIMA Model -------------------
    # Fit a seasonal ARIMA (SARIMA) model using seasonal_order=(1,0,0,12).
    print("\n=== Statsmodels SARIMA ===")
    sarima_model_sm = smSARIMAX(y_arima, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12)).fit()
    print("Estimated parameters (SARIMA):")
    print(sarima_model_sm.params)
    
    forecast_sarima_sm = sarima_model_sm.forecast(steps=5)
    print("Forecast (next 5 steps):")
    print(forecast_sarima_sm)
    
    # ------------------- SARIMAX Model -------------------
    # Generate dummy exogenous regressors.
    exog = np.random.normal(size=(n, 2))  # two exogenous regressors
    print("\n=== Statsmodels SARIMAX ===")
    sarimax_model_sm = smSARIMAX(y_arima, exog=exog, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12)).fit()
    print("Estimated parameters (SARIMAX):")
    print(sarimax_model_sm.params)
    
    # Generate forecast exogenous variables.
    exog_forecast = np.random.normal(size=(5, 2))
    forecast_sarimax_sm = sarimax_model_sm.forecast(steps=5, exog=exog_forecast)
    print("Forecast (next 5 steps):")
    print(forecast_sarimax_sm)


if __name__ == "__main__":
    
    import numpy as np

    np.random.seed(117)
    n = 100
    y_arma = [0.0, 0.0, 0.0]
    for t in range(3, n):
        ar_term = 0.5 * y_arma[t-1]
        noise = np.random.normal(scale=0.2)
        new_val = 0.2 + ar_term + noise - 0.3 * noise
        y_arma.append(new_val)
    y_arma = np.array(y_arma)
    
    # Fit ARMA model using ARIMA with order (1,0,1).
    print("\n=== My ARMA(1,1) ===")
    md = ARIMAX(1,0,1)
    md.fit(Matrix(y_arma), Matrix(np.random.normal(0,1,n)))
    md.summary()
    print(md.predict(5, Matrix(np.random.normal(0,1,5))) )
