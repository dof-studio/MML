# nn_optimizer.py
#
# Neural Network Optimizer Collection
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, List, Dict, Literal

from .objtyp import Object
from .tensor import Tensor

from .nn_base import nn_Base
from .nn_parameter import nn_Parameter
from .nn_module import nn_Module


# Base Class for All nn Optimizers
class nn_Optm_BaseOptimizer(nn_Base):
    """
    Base optimizer class.
    
    Any inherited optimizer takes all of the parameters as a reference and
    use a method connected with gradients to update trainable parameters.
    A typical __init__ at least requires a list/dict of parameters or a nn_Module.
    """
    
    __attr__ = "MML.nn_Optm_BaseOptimizer"
    
    def __init__(self, 
                 params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, 
                 **kwargs):
        """
        Initialize an optimizer. Call this and pass in parameters as
        a list/dict of nn_Parameter or a nn_Module which contains all parameters and submodules.

        Parameters:
            params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, if directly gives a list of parameters,
                    which we believe generates by .parameters(), we record them as a list;
                    If directly gives an nn_Module, we accepts and record all of the parameters as a list;

        Attributes:
            self.params: a List of nn_Parameters which may have gradients and needs to be updated.
        """
        
        super().__init__()
        
        # Record the unfiltered list of parameters
        if isinstance(params, list):
            self.params = params
        elif isinstance(params, dict):
            self.params = params.values()
        elif isinstance(params, nn_Module):
            self.params = params.parameters()
        else:
            raise ValueError(f"`params` for an optimizer can either be a list/dict of nn_Parameter or a nn_module, but you have {type(params)}")
            
        # Filter all parameters having gradients
        self.params = [p for p in self.params if p.requires_grad == True]

    def step(self):
        """
        Update all trainable parameters in one step (override in subclasses).
        
        Returns:
            self
        """
        raise NotImplementedError("step() is not implemented in the base optimizer class")

    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        
        Returns:
            self
        """
        for p in self.params:
            p.zero_grad()
        return self

    def __repr__(self):
        return f"nn_Optm_BaseOptimizer(with {len(self.params)} trainable parameters)."


# Implementation of Stochastic Gradient Descent Optimzier
class nn_Optm_SGD(nn_Optm_BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum.
    
    This optimizer uses stochastic gradient descent (SGD) to update model parameters 
    by minimizing a loss function. It supports optional momentum for faster convergence 
    and better stability in optimization.

    """
    
    __attr__ = "MML.nn_Optm_SGD"
    
    def __init__(self, 
                 params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, 
                 *,
                 lr: float = 0.01, 
                 momentum: float = 0.9,
                 **kwargs):
        
        """
        Initialize a SGD optimizer with momentum. Call this and pass in parameters as
        a list/dict of nn_Parameter or a nn_Module which contains all parameters and submodules.

        Parameters:
            params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, if directly gives a list of parameters,
                    which we believe generates by .parameters(), we record them as a list;
                    If directly gives an nn_Module, we accepts and record all of the parameters as a list;
            lr: float, learning rate, the step size functioned on the gradients to update parameters;
            momentum: float, momentum factor for acceleration in convergence.

        Attributes:
            self.params: a List of nn_Parameters which may have gradients and needs to be updated.
            self.n: int, the total number of steps performed.
            self.lr: float, the learning rate as a plain float.
            self.momentum: float, the momentum rate as a plain float.
        """
        
        # Call the base initialization to initialize all trainable parameters
        super().__init__(params, **kwargs)
        
        # Record the SGD parameters
        self.n = 0
        self.lr = lr
        self.momentum = momentum
        
        # Initialize velocity buffers if momentum is used
        if momentum != 0:
            self.velocities = [p.data.to_zeros() for p in self.params]
        else:
            self.velocities = None

    def step(self):
        """
        Update all trainable parameters in one step by SGD and momentum method.
        
        Returns:
            self
        """
        # For compatibility, n adds at the begining
        self.n += 1
        
        with torch.no_grad():
            # Iterate over trainable parameters
            for i, p in enumerate(self.params):
    
                # If uses momentum method
                if self.momentum != 0:
                    # Momentum update: v = momentum * v - lr * grad and update parameter: w = w + v
                    if p.autograd == False:
                        self.velocities[i] = self.velocities[i] * self.momentum - self.lr * p.grad
                        p.data += self.velocities[i]
                    else:
                        self.velocities[i].data = self.velocities[i].data * self.momentum - self.lr * p.data.data.grad
                        p.data.data.data += self.velocities[i].data
                        
                # Use plain vanilla SGD
                else:
                    # Vanilla SGD: w = w - lr * grad
                    if p.autograd == False:
                        p.data -= self.lr * p.grad
                    else:
                        p.data.data.data -= self.lr * p.data.data.grad
                        
        return self
    
    def __repr__(self):
        return f"nn_Optm_SGD(with {len(self.params)} trainable parameters)."


# Alias for nn_Optm_SGD
SGD = nn_Optm_SGD


# Implementation of Adaptive Momentum Optimzier
class nn_Optm_Adam(nn_Optm_BaseOptimizer):
    """
    Adaptive Momentum Estimate (Adam) Optimizer.

    Adam is an optimization algorithm that combines the advantages of both RMSProp and SGD. 
    It maintains two moving averages: one for the gradient (momentum) and another for the square 
    of the gradient, which are updated over time. The learning rate is adaptively adjusted based 
    on these estimates to achieve faster convergence and better stability.
    """
    
    __attr__ = "MML.nn_Optm_Adam"

    def __init__(self, 
                 params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, 
                 *,
                 lr: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 eps: float = 1e-16,
                 **kwargs):
        
        """
        Initialize an Adam optimizer with momentum. Call this and pass in parameters as
        a list/dict of nn_Parameter or a nn_Module which contains all parameters and submodules.

        Parameters:
            params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, if directly gives a list of parameters,
                    which we believe generates by .parameters(), we record them as a list;
                    If directly gives an nn_Module, we accepts and record all of the parameters as a list;
            lr: float, learning rate, the step size functioned on the gradients to update parameters;
            beta1: float, momentum term, the decay rate for the first moment estimate.
            beta2: float, adaptive scaling, it is the decay rate for the second moment estimate, 0.999 for 1000 steps averaging.

        Attributes:
            self.params: a List of nn_Parameters which may have gradients and needs to be updated.
            self.n: int, the total number of steps performed.
            self.lr: float, the learning rate as a plain float.
            self.beta1: float, the momentum factor stored in plain float.
            self.beta2: float, the adaptive scaling term stored in plain float.
            self.eps: float, epsilon to avoid dividing by 0.
        """
        
        # Call the base initialization to initialize all trainable parameters
        super().__init__(params, **kwargs)
        
        # Record the Adam parameters
        self.n = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize first and second moment estimates for each param
        self.m = [p.data.to_zeros() for p in self.params]  # first moment
        self.v = [p.data.to_zeros() for p in self.params]  # second moment

    def step(self):
        """
        Update all trainable parameters in one step by Adaptive Momentum Estimate (Adam) method.
        
        Returns:
            self
        """
        # For compatibility, n adds at the begining
        self.n += 1
        
        with torch.no_grad():
            # Iterate over trainable parameters
            for i, p in enumerate(self.params):
    
                if p.autograd == False:
                    # Update biased first moment: m = beta1*m + (1-beta1)*grad
                    self.m[i] = self.m[i] * self.beta1 + (1.0 - self.beta1) * p.grad
                    # Update biased second moment: v = beta2*v + (1-beta2)*grad^2
                    self.v[i] = self.v[i] * self.beta2 + (1.0 - self.beta2) * (p.grad ** 2)            
    
                    # Compute bias-corrected moments
                    m_hat = self.m[i] / (1.0 - (self.beta1 ** self.n))
                    v_hat = self.v[i] / (1.0 - (self.beta2 ** self.n))
                    
                    # Update parameter: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                    p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
                
                else:
                    # Update biased first moment: m = beta1*m + (1-beta1)*grad
                    self.m[i].data = self.m[i].data * self.beta1 + (1.0 - self.beta1) * p.data.data.grad
                    # Update biased second moment: v = beta2*v + (1-beta2)*grad^2
                    self.v[i].data = self.v[i].data * self.beta2 + (1.0 - self.beta2) * (p.data.data.grad ** 2)            
    
                    # Compute bias-corrected moments
                    m_hat = self.m[i] / (1.0 - (self.beta1 ** self.n))
                    v_hat = self.v[i] / (1.0 - (self.beta2 ** self.n))
                    
                    # Update parameter: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                    p.data.data.data -= self.lr * m_hat.data / (v_hat.data ** 0.5 + self.eps)
                        
        return self
    
    def __repr__(self):
        return f"nn_Optm_Adam(with {len(self.params)} trainable parameters, beta1 = {self.beta1}, beta2 = {self.beta2})."


# Alias for nn_Optm_Adam
Adam = nn_Optm_Adam


# Implementation of Adaptive Momentum Optimzier with Weight Decay
class nn_Optm_AdamW(nn_Optm_BaseOptimizer):
    """
    Adaptive Momentum Estimate (AdamW) Optimizer with Weight Decay.

    AdamW combines the Adam optimizer with a weight decay regularization term to prevent 
    overfitting and improve convergence. It applies weight decay to all parameters after 
    the gradient is computed, ensuring that large weights are penalized in the loss function. 
    The optimizer uses the same momentum terms as Adam but applies the weight decay during 
    the update step.

    Formula:
        For each parameter `p`:
        1. Compute gradient `g` and update momentum terms `v`.
        2. Apply weight decay: `p.data -= lr * (g + beta_2 * v)`.
        3. Update parameters using Adam's updates: `p.data -= lr * (g + beta_1 * v)`.
    """    
    
    __attr__ = "MML.nn_Optm_AdamW"
    
    def __init__(self,                 
                 params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, 
                 *, 
                 lr: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 eps: float = 1e-16,
                 weight_decay: float = 0.0001,
                 **kwargs):
        """
        Initialize an AdamW optimizer with weight decay. Call this and pass in parameters as
        a list/dict of nn_Parameter or a nn_Module which contains all parameters and submodules.

        Parameters:
            params: List[nn_Parameter] | Dict[Any, nn_Parameter] | nn_Module, if directly gives a list of parameters,
                    which we believe generates by .parameters(), we record them as a list;
                    If directly gives an nn_Module, we accepts and record all of the parameters as a list;
            lr: float, learning rate, the step size functioned on the gradients to update parameters;
            beta1: float, momentum term, the decay rate for the first moment estimate.
            beta2: float, adaptive scaling, it is the decay rate for the second moment estimate, 0.999 for 1000 steps averaging;
            weight_decay: float, L2 regularization coefficient (decoupled).

        Attributes:
            self.params: a List of nn_Parameters which may have gradients and needs to be updated.
            self.n: int, the total number of steps performed.
            self.lr: float, the learning rate as a plain float.
            self.beta1: float, the momentum factor stored in plain float.
            self.beta2: float, the adaptive scaling term stored in plain float.
            self.eps: float, epsilon to avoid dividing by 0.
            self.weight_decay: float, weight decay parameter.
        
        """

        # Call the base initialization to initialize all trainable parameters
        super().__init__(params, **kwargs)
        
        # Record the AdamW parameters
        self.n = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize first and second moment estimates for each param
        self.m = [p.data.to_zeros() for p in self.params]  # first moment
        self.v = [p.data.to_zeros() for p in self.params]  # second moment
        
    def step(self):
        """
        Update all trainable parameters in one step by Adaptive Momentum Estimate with Weight Decay (AdamW) method.
        
        Returns:
            self
        """
        # For compatibility, n adds at the begining
        self.n += 1
        
        with torch.no_grad():
            # Iterate over trainable parameters
            for i, p in enumerate(self.params):
    
                if p.autograd == False:
                    # Update biased first moment: m = beta1*m + (1-beta1)*grad
                    self.m[i] = self.m[i] * self.beta1 + (1.0 - self.beta1) * p.grad
                    # Update biased second moment: v = beta2*v + (1-beta2)*grad^2
                    self.v[i] = self.v[i] * self.beta2 + (1.0 - self.beta2) * (p.grad ** 2)            
    
                    # Compute bias-corrected moments
                    m_hat = self.m[i] / (1.0 - (self.beta1 ** self.n))
                    v_hat = self.v[i] / (1.0 - (self.beta2 ** self.n))
                    
                    # Apply decoupled weight decay
                    p.data *= (1.0 - self.lr * self.weight_decay)
                    
                    # Update parameter: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                    p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
                
                else:
                    # Update biased first moment: m = beta1*m + (1-beta1)*grad
                    self.m[i].data = self.m[i].data * self.beta1 + (1.0 - self.beta1) * p.data.data.grad
                    # Update biased second moment: v = beta2*v + (1-beta2)*grad^2
                    self.v[i].data = self.v[i].data * self.beta2 + (1.0 - self.beta2) * (p.data.data.grad ** 2)            
    
                    # Compute bias-corrected moments
                    m_hat = self.m[i] / (1.0 - (self.beta1 ** self.n))
                    v_hat = self.v[i] / (1.0 - (self.beta2 ** self.n))
                    
                    # Apply decoupled weight decay
                    p.data.data.data *= (1.0 - self.lr * self.weight_decay)
                    
                    # Update parameter: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                    p.data.data.data -= self.lr * m_hat.data / (v_hat.data ** 0.5 + self.eps)
                        
        return self

    def __repr__(self):
        return f"nn_Optm_AdamW(with {len(self.params)} trainable parameters, beta1 = {self.beta1}, beta2 = {self.beta2}, weight_decay = {self.weight_decay})."


# Alias for nn_Optm_AdamW
AdamW = nn_Optm_AdamW


# Test case of Neural Network Module
if __name__ == "__main__":

    from nn import Tensor
    from nn import Dense
    from nn import Sigmoid, ReLU
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import RMSE, MSE, MultiCrossEntropy
    from nn import Adam

    ##############################################
    #
    # Regression Test
    class reg_test(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="reg_test", **kwargs)
            self.dense = Dense(4, 16, True, **kwargs)
            self.actv = Sigmoid(**kwargs)
            self.sumover = Dense(16, 1, True, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.actv.forward(out)
            out = self.sumover.forward(out)
            return out

    class reg_test2(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="reg_test", **kwargs)
            self.dense = Dense(4, 96, True, **kwargs)
            self.actv = Sigmoid(**kwargs)
            self.dense2 = Dense(96, 256, True, **kwargs)
            self.actv2 = Sigmoid(**kwargs)
            self.sumover = Dense(256, 1, True, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.actv.forward(out)
            out = self.dense2.forward(out)
            out = self.actv2.forward(out)
            out = self.sumover.forward(out)
            return out

    # Sample Data 1
    inputs = Tensor([[0, 0.1, 0.5, 0.9], [1, 0.2, -0.1, -0.9]], backend="torch")
    label = Tensor([[0.002312], [0.991215]], backend="torch")
    # Sample Data 2
    inputs = Tensor.rand([10240, 4], backend="torch", device="cuda").logistic_inv()
    label = (inputs.logistic() + Tensor.rand([10240, 4], backend="torch", device="cuda") * 0.1).sum(axis = 1, keepdims=True).tanh()

    # Test forward
    x = reg_test2(backend="torch", device="cuda")
    x.train()
    
    crit = RMSE(backend="torch", device="cuda")
    optm = Adam(x.parameters(), lr=1E-3)
    
    for i in range(10000):
        out = x.forward(inputs)
        
        # Calculate loss
        loss = crit(out, label);
        print(loss.to_list())
        lossgrad = crit.backward()
    
        # Test backward
        x.backward(lossgrad)
        
        # Apply sgd
        optm.step()
        
        # Apply zero grad
        x.zero_grad()
    
    from nn import Tensor
    from nn import Dense
    from nn import Sigmoid, ReLU
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import RMSE, MSE, MultiCrossEntropy
    from nn import Adam
    
    ##############################################
    #
    # Classification Test
    class cls_test(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="cls_test", **kwargs)
            self.dense = Dense(4, 16, True, **kwargs)
            self.actv1 = Sigmoid(**kwargs)
            self.dense2 = Dense(16, 8, False, **kwargs)
            self.actv2 = Sigmoid(**kwargs)
            self.dense3 = Dense(8, 2, False, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.actv1.forward(out)
            out = self.dense2.forward(out)
            out = self.actv2.forward(out)
            out = self.dense3.forward(out)
            return out

    inputs = Tensor([[1, 2, 3, 4.], [-1,-7, 14, 9]], backend="torch")
    label = Tensor([[0, 1], [1, 0]], backend="torch")

    # Test forward
    x = cls_test()
    x.train()
    
    crit = MultiCrossEntropy()
    optm = Adam(x.parameters(), lr=1e-3)
    
    for i in range(10000):
        out = x.forward(inputs)
        
        # Calculate loss
        loss = crit(out, label)
        print(loss.to_list())
        lossgrad = crit.backward()
    
        # Test backward
        x.backward(lossgrad)
        
        # Apply sgd
        optm.step()
        
        # Apply zero grad
        x.zero_grad()
        