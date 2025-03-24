# optimizer.py
#
# A function optimizer implementation
# From MML Library by Nathmath

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import unittest
from typing import Any, Callable, Dict, Optional, List


# Configure logging for production use.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =================== Optimizer ====================

# Optimzer class for torch Tensor
class GradientOptimizer:
    """
    A flexible gradient-based optimizer that uses PyTorch optimizers.

    The optimizer accepts a differentiable function to optimize and supports:
      - Minimizing the function (default)
      - Maximizing the function (by inverting the loss)
      - Driving the function’s output toward a specified target using a loss function (default: MSE)

    Parameters:
        func (Callable[[torch.Tensor], torch.Tensor]):
            The function to optimize. It should take a torch.Tensor (the parameters)
            and return a scalar torch.Tensor.
        init_params (torch.Tensor):
            The initial parameters. Must have requires_grad=True.
        **kwargs:
            dtype(type): Numeric type that is used in optimization. (default: torch.float64)
            optimizer_type (str): One of 'sgd', 'adam', 'adam+' (default: 'adam').
            lr (float): Learning rate (default: 0.001).
            max_iter (int): Maximum number of iterations (default: 1000).
            tolerance (float): Convergence tolerance (default: 1e-6).
            mode (str): One of 'minimize', 'maximize', or 'target' (default: 'minimize').
            target (Optional[torch.Tensor]): In 'target' mode, the desired target value.
            loss_fn (Optional[Callable]): Loss function to use when mode=='target'
                                            (default: nn.MSELoss()).
            optimizer_args (dict): Additional optimizer keyword arguments.
            verbose (bool): If True, logs progress at each iteration.
    """
    
    # Fast mode constants
    __minimize__ = 1
    __maximize__ = 2
    __target__   = 3

    # Init
    def __init__(self,
                 func: Callable[[torch.Tensor], torch.Tensor],
                 init_params: torch.Tensor,
                 **kwargs):
        self.func = func
        if not init_params.requires_grad:
            raise ValueError("init_params must have requires_grad=True")
        self.params = init_params

        # General optimization settings
        self.dtype: type = kwargs.get("dtype", torch.float64)
        self.optimizer_type: str = kwargs.get('optimizer_type', 'adam').lower()
        self.lr: float = kwargs.get('lr', 0.001)
        self.max_iter: int = kwargs.get('max_iter', 1000)
        self.tolerance: float = kwargs.get('tolerance', 1e-6)
        self.mode: str = kwargs.get('mode', 'minimize').lower()
        self.target: Optional[torch.Tensor] = kwargs.get('target', None)
        self.loss_fn: Callable = kwargs.get('loss_fn', nn.MSELoss())
        self.verbose: bool = kwargs.get('verbose', False)
        optimizer_extra_args: Dict = kwargs.get('optimizer_args', {})

        if self.mode == 'target' and self.target is None:
            raise ValueError("In 'target' mode, a target value must be provided.")
            
        # Fast mode
        self.fastmode_ = None
        if self.mode == "minimize":
            self.fastmode_ = self.__minimize__
        elif self.mode == "maximize":
            self.fastmode_ = self.__maximize__
        elif self.mode == "target":
            self.fastmode_ = self.__target__
        else:
            raise ValueError("mode must be one of 'minimize', 'maximize', or 'target'")
            
        # Optimizer initialization
        self.optimizer = self._init_optimizer(optimizer_extra_args)
        
    # Internal - Initialze the optimizer
    def _init_optimizer(self, optimizer_args: Dict) -> optim.Optimizer:
        """
        Initialize and return a PyTorch optimizer based on the chosen type.
        """
        if self.optimizer_type == 'sgd':
            return optim.SGD([self.params], lr=self.lr, **optimizer_args)
        elif self.optimizer_type == 'adam':
            return optim.Adam([self.params], lr=self.lr, **optimizer_args)
        elif self.optimizer_type in ('adam+', 'adam_plus'):
            # "Adam+" is implemented here as Adam with AMSGrad enabled.
            return optim.Adam([self.params], lr=self.lr, amsgrad=True, **optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    # API - Try to optimize
    def optimize(self) -> Dict[str, any]:
        """
        Run the optimization process.

        Returns:
            dict: A dictionary containing:
                - 'params': The optimized parameters (torch.Tensor).
                - 'loss': The optimized last loss (torch.Tensor).
                - 'loss_history': A list of loss values for each iteration.
        """
        loss_history: List[float] = []
        last_loss: Optional[float] = None

        for iteration in range(self.max_iter):
            self.optimizer.zero_grad()
            output = self.func(self.params)

            # Determine loss based on the selected mode.
            if self.fastmode_ == self.__target__:
                loss = self.loss_fn(output, self.target)
            else:
                loss = output

            # For maximization, invert the loss.
            if self.fastmode_ == self.__maximize__:
                loss = -loss

            loss_value = loss.item()
            loss_history.append(loss_value)

            loss.backward()
            self.optimizer.step()

            if self.verbose:
                logging.info(f"Iteration {iteration}, loss: {loss_value}")

            # Check for convergence.
            if last_loss is not None and abs(loss_value - last_loss) < self.tolerance and abs(loss_value) < self.tolerance:
                if self.verbose:
                    logging.info(f"Convergence reached at iteration {iteration}.")
                break

            last_loss = loss_value

        # Return a clone of parameters (detached) and the loss history.
        return {'params': self.params.clone().detach(), 
                'loss': loss_value,
                'loss_history': loss_history}

# Wrapper function
def optimize(func: Callable,
             init_val: Any,
             mode: str = 'minimize',
             lr: float = 0.001,
             dtype: type = torch.float64,
             optimizer_type: str = 'adam',
             max_iter: int = 1000,
             tolerance: float = 1e-6,
             target: Optional[Any] = None,
             loss_fn: Optional[Callable] = None,
             verbose: bool = False,
             extra_args: Optional[Dict] = None) -> Dict[str, Any]:
    """
    A wrapper that accepts ordinary Python functions and types,
    converts inputs to PyTorch objects as necessary, and performs
    gradient-based optimization.

    Parameters:
        func (Callable): The function to optimize. It should accept a torch.Tensor
                         and return a scalar torch.Tensor.
        init_val: The initial parameter value (e.g. an int or float). If not a torch.Tensor,
                  it will be converted to one with requires_grad=True.
        mode (str): Optimization mode: 'minimize', 'maximize', or 'target'. (default: 'minimize')
        lr (float): Learning rate (default: 0.001).
        dtype(type): Numeric type that is used in optimization. (default: torch.float64)
        optimizer_type (str): Optimizer to use: 'sgd', 'adam', or 'adam+'. (default: 'adam')
        max_iter (int): Maximum number of iterations (default: 1000).
        tolerance (float): Convergence tolerance (default: 1e-6).
        target: Target value for 'target' mode. Can be an ordinary type; will be converted to a tensor.
        loss_fn: Loss function for 'target' mode. Defaults to MSELoss if not provided.
        verbose (bool): If True, logs progress.
        extra_args (dict): Additional keyword arguments to pass to the optimizer.
    
    Returns:
        dict: A dictionary containing the optimized parameter(s) and the loss history.
    """
    # Ensure the initial value is a torch tensor with gradient enabled.
    if not isinstance(init_val, torch.Tensor):
        init_param = torch.tensor(init_val, dtype=dtype, requires_grad=True)
    else:
        init_param = init_val
        if not init_param.requires_grad:
            init_param.requires_grad = True

    # In target mode, convert target to tensor if needed.
    if mode.lower() == 'target':
        if target is None:
            raise ValueError("In 'target' mode, a target value must be provided.")
        if not isinstance(target, torch.Tensor):
            target_tensor = torch.tensor(target, dtype=dtype)
        else:
            target_tensor = target
    else:
        target_tensor = None

    # Default loss function is MSELoss in target mode.
    if mode.lower() == 'target' and loss_fn is None:
        loss_fn = nn.MSELoss()

    # Use an empty dict if no extra arguments are provided.
    if extra_args is None:
        extra_args = {}

    optimizer_instance = GradientOptimizer(
        func=func,
        init_params=init_param,
        dtype=dtype,
        optimizer_type=optimizer_type,
        lr=lr,
        max_iter=max_iter,
        tolerance=tolerance,
        mode=mode,
        target=target_tensor,
        loss_fn=loss_fn,
        verbose=verbose,
        optimizer_args=extra_args
    )
    result = optimizer_instance.optimize()
    return result


# =================== Unit Tests ===================

class TestGradientOptimizer(unittest.TestCase):

    def test_minimize_quadratic(self):
        # Minimize f(x) = (x - 3)^2.
        # The minimum is at x = 3.
        def f(x: torch.Tensor) -> torch.Tensor:
            return (x - 3) ** 2

        # Initialize parameter x close to 0.
        x = torch.tensor(0.0, requires_grad=True)
        optimizer = GradientOptimizer(f, x, optimizer_type='sgd', lr=0.1, max_iter=10000, mode='minimize', verbose=False)
        result = optimizer.optimize()
        optimized_x = result['params']

        self.assertTrue(torch.allclose(optimized_x, torch.tensor(3.0), atol=1e-2),
                        f"Expected optimized x ~ 3.0, got {optimized_x.item()}")

    def test_maximize_quadratic(self):
        # Maximize f(x) = -(x - 3)^2.
        # The maximum is at x = 3 (maximum value 0).
        def f(x: torch.Tensor) -> torch.Tensor:
            return -(x - 3) ** 2

        x = torch.tensor(0.0, requires_grad=True)
        optimizer = GradientOptimizer(f, x, optimizer_type='adam', lr=0.1, max_iter=10000, mode='maximize', verbose=False)
        result = optimizer.optimize()
        optimized_x = result['params']

        self.assertTrue(torch.allclose(optimized_x, torch.tensor(3.0), atol=1e-2),
                        f"Expected optimized x ~ 3.0, got {optimized_x.item()}")

    def test_target_mode(self):
        # Optimize f(x) = 2*x to get close to a target value of 10.
        # The loss is computed using MSE: loss = MSE(2*x, 10). Optimal x should be 5.
        def f(x: torch.Tensor) -> torch.Tensor:
            return 2 * x

        x = torch.tensor(0.0, requires_grad=True)
        target = torch.tensor(10.0)
        optimizer = GradientOptimizer(f, x, optimizer_type='adam+', lr=0.1, max_iter=10000, mode='target', target=target, verbose=False)
        result = optimizer.optimize()
        optimized_x = result['params']

        self.assertTrue(torch.allclose(optimized_x, torch.tensor(5.0), atol=1e-2),
                        f"Expected optimized x ~ 5.0, got {optimized_x.item()}")

class TestWrapperFunction(unittest.TestCase):

    def test_wrapper_minimize_quadratic(self):
        # Minimize f(x) = (x - 3)^2, expecting optimum x ~ 3.
        def f(x: torch.Tensor) -> torch.Tensor:
            return (x - 3) ** 2

        result = optimize(f, init_val=0.0, mode='minimize', lr=0.1, optimizer_type='sgd', max_iter=10000, verbose=False)
        optimized_x = result['params']
        self.assertTrue(torch.allclose(optimized_x, torch.tensor(3.0), atol=1e-2),
                        f"Expected optimized x ~ 3.0, got {optimized_x.item()}")

    def test_wrapper_maximize_quadratic(self):
        # Maximize f(x) = -(x - 3)^2, expecting optimum x ~ 3.
        def f(x: torch.Tensor) -> torch.Tensor:
            return -(x - 3) ** 2

        result = optimize(f, init_val=0.0, mode='maximize', lr=0.1, optimizer_type='adam', max_iter=10000, verbose=False)
        optimized_x = result['params']
        self.assertTrue(torch.allclose(optimized_x, torch.tensor(3.0), atol=1e-2),
                        f"Expected optimized x ~ 3.0, got {optimized_x.item()}")

    def test_wrapper_target_mode(self):
        # Optimize f(x) = 2*x to get close to target value 10. Optimal x should be 5.
        def f(x: torch.Tensor) -> torch.Tensor:
            return 2 * x

        result = optimize(f, init_val=0.0, mode='target', lr=0.1, optimizer_type='adam+', max_iter=10000,
                                     target=10.0, verbose=False)
        optimized_x = result['params']
        self.assertTrue(torch.allclose(optimized_x, torch.tensor(5.0), atol=1e-2),
                        f"Expected optimized x ~ 5.0, got {optimized_x.item()}")

if __name__ == "__main__":
    unittest.main()
    
    # Additional test
    def fx(r):
        return -500 + 100/(1+r) + 425/((1+r)**2)
    def fy(r):
        return -500 + 0/(1+r) + 525/((1+r)**2) - 415/((1+r)**3) + 525/((1+r)**4) 
    result = optimize(fy, 0.001, mode = "target", target=0, lr=1e-4, tolerance=1e-8)
    print(numpy.round(float(result['params']), 6))
    
    # Personal test
    import numpy as np
    class CrankNicolsonPricer:
        """
        Crank–Nicolson Finite Difference Pricer for pricing European and American
        Call/Put options.
        """
        
        # Constants
        __type_call__ =  0
        __type_put__  =  1
        __type_euro__ = 10
        __type_ame__  = 11
        
        def __init__(self, S0, K, T, r, sigma, S_max, M, N, option_type='call', exercise='European'):
            """
            Initialize pricer parameters.
            
            Parameters:
                S0         : float, current underlying asset price.
                K          : float, strike price.
                T          : float, time to maturity (in years).
                r          : float, risk-free interest rate.
                sigma      : float, volatility of the underlying asset.
                S_max      : float, maximum asset price considered in grid.
                M          : int, number of asset price grid steps.
                N          : int, number of time steps.
                option_type: str, 'call' or 'put'.
                exercise   : str, 'European' or 'American'.
            """
            self.S0 = S0
            self.K = K
            self.T = T
            self.r = r
            self.sigma = sigma
            self.S_max = S_max
            self.M = M
            self.N = N
            self.option_type = self.__type_call__ if option_type.lower() == "call" else self.__type_put__
            self.exercise = self.__type_euro__ if exercise.lower() == "european" else self.__type_ame__
            self.dS = S_max / M
            self.dt = T / N
            self.grid_S = np.linspace(0, S_max, M + 1)
        
        def intrinsic_value(self, S):
            """Return the intrinsic value (payoff) at asset price S."""
            if self.option_type == self.__type_call__:
                return np.maximum(S - self.K, 0)
            else:
                return np.maximum(self.K - S, 0)
        
        def _thomas_solver(self, a, b, c, d):
            """
            Solve a tridiagonal linear system using the Thomas algorithm.
            a, b, c: sub-diagonal, diagonal, and super-diagonal coefficients.
            d: right-hand side vector.
            Returns solution vector x.
            """
            n = len(d)
            cp = np.zeros(n)
            dp = np.zeros(n)
            cp[0] = c[0] / b[0]
            dp[0] = d[0] / b[0]
            for i in range(1, n):
                denom = b[i] - a[i] * cp[i-1]
                cp[i] = c[i] / denom if i < n - 1 else 0.0
                dp[i] = (d[i] - a[i] * dp[i-1]) / denom
            x = np.zeros(n)
            x[-1] = dp[-1]
            for i in range(n - 2, -1, -1):
                x[i] = dp[i] - cp[i] * x[i+1]
            return x

        def _psor_solver(self, A, B, C, d, payoff, omega=1.2, tol=1e-8, max_iter=10000):
            """
            Solve the linear complementarity problem for American options using PSOR.
            
            Parameters:
                A, B, C: tridiagonal coefficients arrays.
                d: right-hand side vector.
                payoff: intrinsic payoff vector.
                omega: relaxation parameter.
                tol: tolerance for convergence.
                max_iter: maximum iterations.
                
            Returns:
                solution vector for the current time step.
            """
            n = len(d)
            V = np.copy(payoff)
            for iteration in range(max_iter):
                error = 0.0
                for i in range(n):
                    left = A[i] * (V[i-1] if i > 0 else 0)
                    right = C[i] * (V[i+1] if i < n-1 else 0)
                    V_old = V[i]
                    V_new = (d[i] - left - right) / B[i]
                    V[i] = max(payoff[i], (1 - omega) * V_old + omega * V_new)
                    error = max(error, abs(V[i] - V_old))
                if error < tol:
                    break
            return V
        
        # API - Price it
        def price(self):
            """
            Compute the option price using the Crank–Nicolson finite difference method.
            
            Returns:
                Option price at S0.
            """
            # Initialize terminal condition.
            V = self.intrinsic_value(self.grid_S)
            
            # Time stepping backward using Crank-Nicolson scheme.
            for n in range(self.N):
                t = self.T - (n + 1) * self.dt
                M_inner = self.M - 1
                A = np.zeros(M_inner)
                B = np.zeros(M_inner)
                C = np.zeros(M_inner)
                d = np.zeros(M_inner)
                
                for i in range(1, self.M):
                    S = i * self.dS
                    alpha = 0.5 * self.sigma**2 * S**2 * self.dt / (self.dS**2)
                    beta  = 0.5 * self.r * S * self.dt / self.dS
                    
                    # Coefficients for the implicit part (n+1)
                    A[i-1] = -0.5 * (alpha - beta)
                    B[i-1] = 1 + alpha + 0.5 * self.r * self.dt
                    C[i-1] = -0.5 * (alpha + beta)
                    
                    # Right-hand side: combination of explicit and implicit terms from time level n.
                    # Note that the explicit part uses the same alpha and beta coefficients.
                    # For the explicit part, the coefficients are:
                    #   a' = 0.5*(alpha - beta), b' = 1 - alpha - 0.5*r*dt, c' = 0.5*(alpha + beta)
                    d[i-1] = (0.5*(alpha - beta)) * V[i-1] + (1 - alpha - 0.5 * self.r * self.dt) * V[i] + (0.5*(alpha + beta)) * V[i+1]
                
                # Boundary conditions for Crank-Nicolson.
                if self.option_type == self.__type_call__:
                    lower_bound = 0.0
                    upper_bound = self.S_max - self.K * np.exp(-self.r * (self.T - t))
                else:
                    lower_bound = self.K * np.exp(-self.r * (self.T - t))
                    upper_bound = 0.0
                d[0]    -= A[0] * lower_bound
                d[-1]   -= C[-1] * upper_bound
                
                # Solve the tridiagonal system.
                if self.exercise == self.__type_euro__:
                    V_inner = self._thomas_solver(A, B, C, d)
                else:
                    payoff = self.intrinsic_value(self.grid_S[1:self.M])
                    V_inner = self._psor_solver(A, B, C, d, payoff)
                
                # Update interior nodes.
                V[1:self.M] = V_inner
                V[0] = lower_bound
                V[-1] = upper_bound
            
            # Interpolate the price at S0.
            return np.interp(self.S0, self.grid_S, V)
        
    S0 = 45
    K = 50
    T = 0.75
    r = 0.1
    S_max = 150
    M = 100
    N = 100
    def price_(sigma_):
        sigma_ = float(sigma_.data)
        put = CrankNicolsonPricer(S0, K, T, r, sigma_, S_max, M, N, option_type='put', exercise='American')
        return torch.tensor(put.price()).clone().requires_grad_(True)
    # This will not yield a result since the grad does not exist
    results = optimize(price_, 0.25, mode = "target", target = 5.5)