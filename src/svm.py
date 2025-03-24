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

from scipy.optimize import minimize
from typing import Dict, Optional, Callable


class SVM(Classification):
    """
    A SVM classifier that uses the dual formulation and supports multiple kernels.
    This implementation relies on a custom Matrix class for all internal matrix operations.
    The dual optimization problem is solved using SciPy's SLSQP method or torch's Gradient method.
    """
    
    __kernel_linear__     = 1000
    __kernel_rbf__        = 1001
    __kernel_polynomial__ = 1002
    __kernel_sinh__       = 1003
    __kernel_tanh__       = 1004
    
    def __init__(self, kernel: str | Callable ='linear', 
                 scaling_method = "normalize", *, 
                 C: Optional=1.0, 
                 gamma: Optional=0.1, coef0: Optional=1.0, degree: Optional=3, 
                 lr: Optional = 0.001, tol: Optional = 1e-6, max_iters: Optional = 10000, penalty_coef: Optional = 1000.0,
                 cskernel_kwargs: Dict = {}):
        """
        Initializes the SVM classifier with the specified kernel and hyperparameters.
        The supported kernels are: 'linear', 'rbf', 'polynomial', 'sinh', and 'tanh', or customized callable.
        
        Args:
            kernel (str | Callable): The type of kernel to use or a kernel function (accepts two matrices).
            scaling_method (str): The method of applying scaling. Choose from ('centralize', 'normalize', or 'minmax')
            C (float): Regularization parameter.
            gamma (float): Kernel coefficient for 'rbf', 'polynomial', and 'sinh'.
            coef0 (float): Independent term in kernel functions (for 'polynomial' and 'sinh').
            degree (int): Degree of the polynomial kernel (if used).
            cskernel_kwargs (dict): Additional args for customized kernel function (other wise ignored).
        
        Torch Optimization Args: 
            lr (float): Learning rate for torch's gradient method for optimization problem. Default (0.001).
            tol (float): Tolerance of convergnce for optimization problem. Default (1e-6).
            max_iters (int): Number of maximum iterations allowed for optimization problem. Default (10000)
            penalty_coef (float): Penalty coefficient applied when doing torch optimization. Default (1000.0).
        """
        self.kernel = kernel.lower() if isinstance(kernel, str) else kernel
        self.reserved_kernel = None # Used when selected reserved kernel.
        if self.kernel == "linear":
            self.reserved_kernel = self.__kernel_linear__
        elif self.kernel == "rbf":
            self.reserved_kernel = self.__kernel_rbf__
        elif self.kernel == "polynomial":
            self.reserved_kernel = self.__kernel_polynomial__
        elif self.kernel == "sinh":
            self.reserved_kernel = self.__kernel_sinh__
        elif self.kernel == "tanh":
            self.reserved_kernel = self.__kernel_tanh__
        # Note:
        # If you pass kernel as a function, it should accept two Matrix or Tensor
        # in a sequence of (x, z, args ...)
        # and the output is a similar type Matrix or Tensor as a transformation
        self.C = C
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        # Torch Variables.
        self.lr = lr
        self.tol = tol
        self.max_iters = max_iters
        self.penalty_coef = penalty_coef
        self.cskernel_kwargs = cskernel_kwargs
        
        # Variables ------------------>
        self.support_indices = None  # Matrix | Tensor of Indices where support vectors exist
        self.support_vectors = None  # Matrix | Tensor of support vectors
        self.support_labels = None   # Matrix | Tensor of Corresponding labels for support vectors
        self.alphas = None           # Matrix | Tensor of Lagrange multipliers for support vectors
        self.support_alphas = None   # Matrix | Tensor of Accepted Alphas as support alphas
        self.bias = 0.0              # Scalar          of Bias term (b)
        
        # Training data (stored as Matrix objects).
        self.X = None
        self.y = None
        self.original_X = None  # Non-scaled Xs
        self.original_y = None  # Un shifted ys
        
        # Scaler applied to X.
        self.scaling = Scaling(method=scaling_method)
        
        # Label scheme (-1,1 scheme or 0,1 scheme).
        self.y_scheme = (-1, 1) # or (0, 1)
        
        # Precomputed kernel (Gram) matrix for training data.
        # Implemented by Nathmath Huang.
        self.K = None

    def _svm_normalize_labels(self, y):
        """
        Normalizes label vector to be in {-1, 1}. If labels are in {0, 1}, they are mapped
        using the transformation: y_new = 2*y - 1.
        
        Args:
            y (Matrix | Tensor): Original label vector.
            
        Returns:
            Matrix | Tensor: Normalized label vector with values in {-1, 1}.
        """
        # Check if the minimum is 0 and maximum is 1, indicating {0,1} labels.
        y_min = float(y.min().data)
        y_max = float(y.max().data)
        if int(round(y_min)) == 0 and int(round(y_max)) == 1:
            # Uses (0, 1) scheme
            self.y_scheme = (0, 1)
            
            # Transform each label: 0 -> -1 and 1 -> 1.
            new_labels = [1 if int(round(float(label))) == 1.0 else -1.0 for label in y.data]
            return type(y)(new_labels, backend=y._backend, dtype=y.dtype, device=y.device)
        # Still (-1, 1) scheme
        return y                 

    def _compute_kernel_matrix(self, X: Matrix | Tensor):
        """
        Computes the Gram (kernel) matrix for the training data X.
        Each entry K[i, j] is computed using the selected kernel function on row i and row j.
        
        Args:
            X (Matrix | Tensor): Training data of shape (n_samples, n_features).
            
        Returns:
            Matrix | Tensor: The computed Gram matrix/tensor of shape (n_samples, n_samples).
        """
        
        # Use optimized built-in methods.
        if self.reserved_kernel is not None:
        
            # Linear kernel
            if self.reserved_kernel == self.__kernel_linear__:
                return (X @ X.transpose()).to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # Polynomial
            elif self.reserved_kernel == self.__kernel_polynomial__:
                K_lin = X @ X.transpose()
                K_data = (self.gamma * K_lin + self.coef0) ** self.degree
                return K_data.to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # sinh
            elif self.reserved_kernel == self.__kernel_sinh__:
                K_lin = X @ X.transpose()
                K_data = (self.gamma * K_lin + self.coef0).sinh()
                return K_data.to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # tanh (sigmoid)
            elif self.reserved_kernel == self.__kernel_tanh__:
                K_lin = X @ X.transpose()
                K_data = (self.gamma * K_lin + self.coef0).tanh()
                return K_data.to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # RBF kernel:
            elif self.reserved_kernel == self.__kernel_rbf__:
                if X._backend == "numpy":
                    # Compute squared norms of each sample.
                    X2 = np.sum(X.data ** 2, axis=1).reshape(-1, 1)
                    dists = X2 + X2.T - 2 * X.data.dot(X.data.T)
                    K_data = np.exp(-self.gamma * dists)
                elif X._backend == "torch":
                    X2 = torch.sum(X.data ** 2, dim=1).reshape(-1, 1)
                    dists = X2 + X2.T - 2 * (X.data @ X.data.T)
                    K_data = torch.exp(-self.gamma * dists)
                else:
                    raise TypeError(f"Invalid backend {X._backend}! How can you reach here?!")
                return type(X)(K_data, backend=X._backend, dtype=X.dtype, device=X.device)
            
            else:
                raise ValueError("Unsupported kernel type: " + self.kernel + ", please use `linear`, `polynomial`, `rbf`, `sinh`, or a callable function!")

        else:
            # Use customized kernel.
            return self.kernel(X, X, **self.cskernel_kwargs).to(backend=X._backend, dtype=X.dtype, device=X.device)

    def _compute_cross_kernel_matrix(self, X: Matrix | Tensor, Y: Matrix | Tensor):
        """
        Computes the cross-kernel matrix between two sets of samples X and Y in a vectorized manner.
        The resulting matrix/tensor K has shape (n_samples_X, n_samples_Y) where K[i, j] = kernel(X[i], Y[j]).
        
        Args:
            X (Matrix | Tensor): Matrix or Tensor of shape (n_samples_X, n_features).
            Y (Matrix | Tensor): Matrix or Tensor of shape (n_samples_Y, n_features).
        
        Returns:
            Matrix | Tensor: Cross-kernel matrix/tensor computed using the selected kernel.
        """
        # Use optimized built-in methods.
        if self.reserved_kernel is not None:
        
            # Linear kernel
            if self.reserved_kernel == self.__kernel_linear__:
                return (X @ Y.transpose()).to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # Polynomial
            elif self.reserved_kernel == self.__kernel_polynomial__:
                K_lin = X @ Y.transpose()
                K_data = (self.gamma * K_lin + self.coef0) ** self.degree
                return K_data.to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # sinh
            elif self.reserved_kernel == self.__kernel_sinh__:
                K_lin = X @ Y.transpose()
                K_data = (self.gamma * K_lin + self.coef0).sinh()
                return K_data.to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # tanh (sigmoid)
            elif self.reserved_kernel == self.__kernel_tanh__:
                K_lin = X @ Y.transpose()
                K_data = (self.gamma * K_lin + self.coef0).tanh()
                return K_data.to(backend=X._backend, dtype=X.dtype, device=X.device)
            
            # RBF kernel:
            elif self.reserved_kernel == self.__kernel_rbf__:
                if X._backend == "numpy":
                    X2 = np.sum(X.data ** 2, axis=1).reshape(-1, 1)
                    Y2 = np.sum(Y.data ** 2, axis=1).reshape(1, -1)
                    dists = X2 + Y2 - 2 * X.data.dot(Y.data.T)
                    K_data = np.exp(-self.gamma * dists)
                elif X._backend == "torch":
                    X2 = torch.sum(X.data ** 2, dim=1).reshape(-1, 1)
                    Y2 = torch.sum(Y.data ** 2, dim=1).reshape(1, -1)
                    dists = X2 + Y2 - 2 * (X.data @ Y.data.T)
                    K_data = torch.exp(-self.gamma * dists)
                else:
                    raise TypeError(f"Invalid backend {X._backend}! How can you reach here?!")
                return type(X)(K_data, backend=X._backend, dtype=X.dtype, device=X.device)
            
            else:
                raise ValueError("Unsupported kernel type: " + self.kernel + ", please use `linear`, `polynomial`, `rbf`, `sinh`, or a callable function!")

        else:
            # Use customized cross-kernel.
            return self.kernel(X, Y, **self.cskernel_kwargs).to(backend=X._backend, dtype=X.dtype, device=X.device)

    def _objective_np(self, alpha, H):
        """
        Numpy implementation.
        
        Objective function for the dual SVM problem. 
        The function to minimize is defined as: 0.5 * alpha^T H alpha - sum(alpha)
        
        Args:
            alpha (np.ndarray): Current estimate of the dual variables (Lagrange multipliers).
            H (np.ndarray): Precomputed matrix H = (y_i y_j) * K, where K is the Gram matrix.
            
        Returns:
            (Matrix | Tensor): The objective function value.
        """
        return 0.5 * alpha.dot(H.dot(alpha)) - np.sum(alpha)

    def _constraint_np(self, alpha, y):
        """
        Numpy implementation.
        
        Equality constraint for the dual problem: sum(alpha_i * y_i) = 0.
        
        Args:
            alpha_np (numpy.ndarray): Dual variables.
            y_np (numpy.ndarray): Labels as a numpy array.
            
        Returns:
            (Matrix | Tensor): The constraint value (should be zero when satisfied).
        """
        return np.dot(alpha, y)

    def _objective_torch(self, alpha, H, y):
        """
        Torch implementation.
        
        Computes the dual objective function with an added penalty term to enforce
        the equality constraint: sum(alpha_i * y_i) = 0.
        The objective is defined as:
            0.5 * alpha^T H alpha - sum(alpha) + penalty_coef * (sum(alpha * y))^2
        
        Args:
            alpha (torch.tensor): Current estimate of dual variables.
            H (torch.tensor): Precomputed matrix H = (y_i y_j) * K.
            y (torch.tensor): Label vector (flattened).
        
        Returns:
            torch.tensor: The computed loss (scalar).
        """
        # Compute the quadratic term: alpha^T H alpha using torch.mv for matrix-vector multiplication.
        quad_term = torch.dot(alpha, torch.mv(H, alpha))
        
        # Standard dual objective function.
        loss = 0.5 * quad_term - alpha.sum()
        
        # Add a penalty for the equality constraint: (sum(alpha * y))^2
        constraint_penalty = self.penalty_coef * (torch.dot(alpha, y)) ** 2
        
        return loss + constraint_penalty

    def _optimizate(self, K: Matrix | Tensor, y: Matrix | Tensor):
        """
        Solves the dual optimization problem using SciPy's SLSQP method (numpy) or optimizer (torch).
        The dual problem is defined as minimizing:
          0.5 * alpha^T H alpha - sum(alpha)
        subject to:
          sum(alpha_i * y_i) = 0  and  0 <= alpha_i <= C for each i.

        Args:
            K (Matrix | Tensor): Precomputed kernel matrix/tensor.
            y (Matrix | Tensor): Label vector (1 dim).
            
        Returns:
            Matrix | Tensor: Optimized alpha values as a Matrix | Tensor.
            
        Warning:
            This method is the BOTTLENECK of this SVM implementation.
        """
        n_samples = K.shape[0]
        
        # Ensure shape (n_samples,)
        y = y.flatten()
            
        # Precompute the matrix H = (y_i * y_j) * K.
        # y.outer(y) yields a matrix [yi*yj].
        # y.outer(y) yields a matrix [(xi*xj)*(yi*yj)]
        H = y.outer(y) * K
        
        # Initial guess: zeros for all alpha (a) values.
        initial_alpha = type(K).zeros(n_samples, backend=K._backend).to(backend=K._backend, dtype=K.dtype, device=K.device)
        
        # Numpy backend optimization, using scipy.
        if K._backend == "numpy" and y._backend == "numpy":
            
            # Define bounds for each alpha: between 0 and C.
            bounds = [(0, self.C) for _ in range(n_samples)]
            
            # Define the equality constraint: sum(alpha_i * y_i) = 0.
            constraints = {
                'type': 'eq',
                'fun': lambda alpha: self._constraint_np(alpha, y.data)
            }
            
            # Call the SLSQP optimizer to minimize the dual objective.
            result = minimize(fun=lambda alpha: self._objective_np(alpha, H.data),
                              x0=initial_alpha.data,
                              bounds=bounds,
                              constraints=constraints,
                              method='SLSQP')
            
            if not result.success:
                raise RuntimeError("Scipy Optimization failed: " + result.message)
            
            # Return the optimized alpha values wrapped in a Matrix | Tensoe.
            return type(K)(result.x.copy(), backend=K._backend, dtype=K.dtype, device=K.device)
        
        # Torch backend optimization, using torch gradient method.
        elif K._backend == "torch" and y._backend == "torch":
            
            # Copy H_torch as a torch tensor.
            H_torch = H.data.clone().detach()
            
            # Copy flattened y as a torch tensor.
            y_flatten = y.data.flatten()
            
            # Set requires_grad=True for optimization.
            alpha = initial_alpha.data.clone().detach().requires_grad_(True)
            
            # Use Adam optimizer (this is perfect, do not adjust Adam).
            optimizer = torch.optim.Adam([alpha], lr = self.lr)
            
            # Legacy alpha for comparision.
            legacy_alpha = None
            
            # Run iterations until convergence.
            less_tol_count = 0
            for i in range(self.max_iters):
                optimizer.zero_grad()
                
                # Compute the loss with the torch objective.
                loss = self._objective_torch(alpha, H_torch, y_flatten)
                loss.backward()
                optimizer.step()
                
                # Project alpha to the feasible set [0, C].
                with torch.no_grad():
                    alpha.clamp_(0, self.C)
                    
                # If alpha has changed less than tol, converged.
                if i > 0:
                    difference = (alpha - legacy_alpha).abs().sum()
                    if difference.data < self.tol:
                        less_tol_count += 1
                        if less_tol_count > 20:
                            break
                    else:
                        less_tol_count = 0
                        legacy_alpha = alpha.clone().detach().requires_grad_(False)
                else:
                    legacy_alpha = alpha.clone().detach().requires_grad_(False)
                    
            # Return the optimized alphas wrapped in a Matrix | Tensor.
            return type(K)(alpha.clone().detach(), backend=K._backend, dtype=K.dtype, device=K.device)
        
        # Value Error, different backend.
        else:
            raise ValueError(f"Data `K` and `y` should have the same backend (numpy or torch), while they are {K._backend} and {y._backend}")
        
    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor, alpha_tol: float = 1e-12):
        """
        Trains the SVM classifier on the provided training data using the dual formulation.
        The method computes the Gram matrix, sets up the dual optimization problem,
        identifies support vectors, and computes the bias term.
        
        Args:
            X (Matrix | Tensor): Training data of shape (n_samples, n_features).
            y (Matrix | Tensor): Label vector (as a Matrix | Tensor) of shape (n_samples,).
            alpha_tol (float): The tolerance of alpha (if greater than this, we will accept it).
        """
        # Fit scaler using a copy of the input feature matrix X.
        self.scaling.fit(X.copy())
        self.original_X = X.copy() 
        self.X = self.scaling.transform(X).to(backend=X._backend, dtype=X.dtype, device=X.device)
        self.original_y = y.copy()
        self.y = self._svm_normalize_labels(y.flatten()) # Always flatten
        # By default, SVM uses -1, 1 scheme.
        # So, if the original y is in 0, 1 scheme, we should do a conversion.
        
        # Implemented by Nathmath Huang. MML Library.
        
        ###
        # Precompute the kernel matrix using the selected kernel function.
        self.K = self._compute_kernel_matrix(self.X)
        
        ###
        # Optimize the dual problem to obtain Lagrange multipliers (alphas).
        # The _optimizate method returns a Matrix containing the optimized alphas.
        alphas = self._optimizate(self.K, self.y)
        self.alphas = alphas.to(backend = X._backend, dtype = X.dtype, device = X.device)
        
        # Identify support vectors: indices where alpha > alpha_tolerance.
        allowed_alphas = alphas.data > alpha_tol
        # Bug fix for numpy: have to convert into a numpy.array
        # For torch, it is automatically a torch.tensor and no problem.
        if X._is_numpy == True:
            allowed_alphas = np.array(allowed_alphas)
        else:
            pass
        
        if X._is_numpy == True:
            self.support_indices = type(X)(allowed_alphas.nonzero()[0].flatten(),
                                        backend = X._backend, dtype = int, device = X.device)
        else:
            self.support_indices = type(X)(allowed_alphas.nonzero().flatten(),
                                        backend = X._backend, dtype = int, device = X.device)
        
        # Extract support vectors, corresponding labels, and their alpha values.
        self.support_vectors = self.X[self.support_indices.data]
        self.support_labels = self.y[self.support_indices.data]
        self.support_alphas = self.alphas[self.support_indices.data]
            
        ###
        # Compute the bias term using the Karush-Kuhn-Tucker (KKT) conditions.
        # Extract the submatrix of the kernel matrix corresponding to the support vectors.
        K_ss = self.K[self.support_indices.data, :][:, self.support_indices.data]
            
        # Compute the combined support coefficients (element-wise product of support alphas and support labels).
        support_coef = self.support_alphas * self.support_labels
        
        # Compute decision values for all support vectors at once.
        # Each entry corresponds to the sum over j: support_coef[j] * K_ss[i, j].
        decision_vector = K_ss @ support_coef
        
        # Compute the bias for each support vector: bias_i = true label - decision value.
        bias_vector = self.support_labels - decision_vector
        
        # Average the bias over all support vectors to obtain the final bias.
        self.bias = float(bias_vector.flatten().mean().data)
        return self

    def decision_function(self, X: Matrix | Tensor, decision_function_shape="ovo", *, 
                          with_bias: bool = False):
        """
        Evaluate the decision function for the samples in X.
        
        Args:
            X : Matrix | Tensor
                Input samples.
            decision_function_shape : str, optional
                'ovo' (one-vs-one) or 'ovr' (one-vs-rest). Default is 'ovo'.
            with_bias: bool, optional
                False to ignore the existance of bias, True to include bias. Defualt is False.
        
        Returns:
            decision_values : Matrix | Tensor
                If decision_function_shape='ovr', returns a shape of (n_samples, n_classes).
                If decision_function_shape='ovo', returns a shape of (n_samples, n_classes*(n_classes-1)/2)
                where each column corresponds to the decision value for a pair of classes.
        """
        # Transform the input X with the same scale.
        Scaled_X = self.scaling.transform(X).to(backend=X._backend, dtype=X.dtype, device=X.device)
        
        # Compute cross-kernel matrix between test data X and support vectors.
        # shape (n_test, n_sv).
        cross_K = self._compute_cross_kernel_matrix(Scaled_X, self.support_vectors) 
        sv_coef = self.support_alphas * self.support_labels
        sv_coef = sv_coef.reshape((sv_coef.shape[0], 1))
        
        # Compute decision for all test samples: decision = cross_K.dot(sv_coef) + bias.
        decisions = cross_K @ sv_coef
        if with_bias == True:
            decisions += self.bias
        
        # Identify unique classes present in the support set.
        unique_labels = self.support_labels.unique()
        n_classes = len(unique_labels.data)
        
        # If binary classification (2 classes)
        if n_classes == 2:            
            if decision_function_shape.lower() == "ovr":
                # In ovr, output one score per class. For binary SVM we can assign
                # the two classes the scores f(x) and -f(x) respectively.
                decision_matrix = decisions.vstack(-decisions).transpose()
                return type(X)(decision_matrix, backend=X._backend, dtype=X.dtype, device=X.device)
            else:
                # For ovo in binary, only one pair exists.
                return type(X)(decisions.reshape([-1, 1]).data, backend=X._backend, dtype=X.dtype, device=X.device)
        
        # DO NOT SUPPORT multiple classification
        else:
            raise ValueError("Error in fitting with non-binary classification data `y`!")

    def predict(self, X: Matrix | Tensor, *, 
                with_bias: bool = False,
                keep_pred: bool = False, 
                threshold: float = 0.5, 
                positive: float = 1, negative: float = -1):
        """
        Predicts the labels for the given input data X.
        For each sample, the decision function is computed as a weighted sum of kernel evaluations
        between the sample and the support vectors plus the bias. The sign of the decision function
        is used to determine the class label.
        
        Args:
            X (Matrix | Tensor): Input data matrix of shape (n_samples, n_features).
            with_bias (bool): To predict with bias added along with the SVM support vectors (might add precision). Defualt False.
            keep_pred (bool): To keep originally generated number for either -1,1 or 0,1 scheme. Default False.
            threshold (float): Decision threshold in 0,1 scheme to decide whether it belongs to positive or negative.
            positive (float): Label for positive data. Auto chosen.         
            negative (float): Label for positive data. Auto chosen.
            
        Returns:
            Matrix | Tensor: Predicted labels (typically 1 or -1) for each input sample.
            or 
            tuple(Matrix|Tensor, Matrix|Tensor): Predicted values (in -1,1 or 0,1 scheme), Predicted labels.
            if selected keep_pred 
            """
        # Scheme check
        if self.y_scheme == (-1, 1):
            positive = 1
            negative = -1
        elif self.y_scheme == (0, 1): 
            positive = 1
            negative = 0
        else:
            raise ValueError("Invalid y_scheme! We only support -1,1 or 0,1 binary classification")
        
        # Transform the input X with the same scale.
        Scaled_X = self.scaling.transform(X).to(backend=X._backend, dtype=X.dtype, device=X.device)
        
        # Compute cross-kernel matrix between test data X and support vectors.
        # shape (n_test, n_sv).
        cross_K = self._compute_cross_kernel_matrix(Scaled_X, self.support_vectors) 
        sv_coef = self.support_alphas * self.support_labels
        sv_coef = sv_coef.reshape((sv_coef.shape[0], 1))
        
        # Compute decision for all test samples: decision = cross_K.dot(sv_coef) + bias.
        decisions = cross_K @ sv_coef
        if with_bias == True:
            decisions += self.bias
        # Note:
        # decisions is with shape like [120, 1]
        # So you have to use d.data[0] to get the value out, or use flatten().
        
        # Map decision values to class labels.
        threshold_1 = float(Matrix(threshold, "numpy").logistic_inv().data)
        predictions = [positive if d.data[0] >= threshold_1 else negative for d in decisions]
        predictions = type(X)(predictions, backend=X._backend, dtype=X.dtype, device=X.device)
        
        # Only output labels.
        if keep_pred == False:
            return predictions
        
        # Output predicted values.
        else:
            if self.y_scheme == (-1, 1):
                # Directly return. It is naturally in -1,1 scheme.
                return decisions.flatten(), predictions
            elif self.y_scheme == (0, 1):
                # Apply logistic
                decisions = decisions.logistic().flatten()
                return decisions, predictions
            else:
                raise ValueError("Invalid y_scheme! We only support -1,1 or 0,1 binary classification")
                
 
            
def simple_test():
    # Example training data: two classes that are linearly separable.
    # Features and labels are chosen for demonstration.
    # Note: The Matrix class is used for all data operations.
    X_train = Matrix([[2, 3],
                      [3, 3],
                      [1, 1],
                      [2, 1]], backend="numpy", dtype=float)
    y_train = Matrix([1, 1, -1, -1], backend="numpy", dtype=float)
    
    # Create an SVM instance with a linear kernel.
    svm_classifier = SVM(kernel='rbf')
    
    # Train the SVM classifier.
    svm_classifier.fit(X_train, y_train)
    
    # Create test data.
    X_test = Matrix([[2.5, 3],
                     [1.5, 1]], backend="numpy", dtype=float)
    
    # Perform predictions.
    predictions = svm_classifier.predict(X_test)
    
    # Display the predictions.
    print("Predictions for the test data:", predictions)


def comparable_test():    
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    device = "torch"
    
    # Generate a non-linear dataset with two interleaving half circles.
    X, y = make_moons(n_samples=100, noise=0.2, random_state=None)
    # Convert labels: our custom SVM expects labels in {-1, 1} rather than {0, 1}.
    # y = np.where(y == 0, -1, 1)
    
    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Wrap the training and test data into our custom Matrix class.
    X_train_mat = Tensor(X_train.tolist(), backend="numpy", dtype=float)
    y_train_mat = Tensor(y_train.tolist(), backend="numpy", dtype=float)
    X_test_mat = Tensor(X_test.tolist(), backend="numpy", dtype=float)
    
    # Create an instance of our custom SVM classifier using the RBF kernel.
    custom_svm = SVM(kernel='rbf', C=1.0, gamma=0.1, tol = 1e-12)
    custom_svm.fit(
        X_train_mat.to(device, dtype=torch.float64, device="cpu"),
        y_train_mat.to(device, dtype=torch.float64, device="cpu"))
    
    # Get predictions from our custom SVM.
    pred_custom = custom_svm.predict(X_test_mat.to(device, dtype=torch.float64, device="cpu"),
                                     with_bias = True).to("numpy", dtype=int)
    # Calculate accuracy by comparing with the true labels.
    accuracy_custom = sum(1 for pred, true in zip(pred_custom, y_test) if pred == true) / len(y_test)
    
    # Calculate decision boundary
    decision_function = custom_svm.decision_function(X_test_mat.to(device, dtype=torch.float64, device="cpu"),
                                 with_bias = True)
    
    # Now, train sklearn's SVC on the same dataset for comparison.
    sklearn_svm = SVC(kernel='rbf', C=1.0, gamma=0.1)
    sklearn_svm.fit(X_train, y_train)
    pred_sklearn = sklearn_svm.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, pred_sklearn)
    
    # Display the comparison results.
    print("Custom SVM Accuracy:", accuracy_custom)
    print("Sklearn SVM Accuracy:", accuracy_sklearn)

        
if __name__ == '__main__':
    comparable_test()