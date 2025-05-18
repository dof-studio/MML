# nn_parameter.py
#
# A Base Neural Network Data Class for Parameters
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None
    
from copy import deepcopy

from .objtyp import Object
from .tensor import Tensor

from .baseml import MLBase


# A Deep Neural Network Trainable Parameter Class
class nn_Parameter(MLBase):
    """
    A trainable parameter base data structure with gradient storage.
    Contains data and manually implemented gradients, while you can use pytorch autograd techniques.
    Optionally, you may set `autograd` = True to enable torch Autograd functionality instead of manual grads computation.
    """
    
    __attr__ = "MML.nn_Parameter"    
    
    def __init__(self, 
                 data: Tensor, 
                 requires_grad: bool = True, 
                 *, 
                 device: str | None = None, 
                 dtype: str | None = None, 
                 autograd: bool = False, 
                 **kwargs): 
        """
        Create a wrapped Neural Network Parameter Container, including gradients.
        
        Parameters:
            --------
            data: Tensor, The initial value for the parameter as a Tensor object.
            requires_grad: bool, A flag indicating whether gradients should be tracked for this parameter. Defaults to True.
            Optional:
                device: str | None, The device where the tensor should reside (e.g., "cpu", "cuda"). If None, uses the default device. Defaults to None.
                dtype: str | None, The data type of the tensor (e.g., "float32", "float64", or type like torch.float32). If None, uses the data type of the input `data`. Defaults to None.
                autograd: bool, A flag indicating whether to use PyTorch's autograd functionality for gradient computation.  If True, manual gradient tracking is disabled. Defaults to None.
        
        Raises:
            --------
            ValueError: If the input `data` is not a Tensor object.
    
        Attributes:
            --------
            self.autograd: bool, Indicates whether PyTorch's autograd is enabled.
            self.requires_grad: bool, A flag indicating whether gradients are tracked for this parameter.
            self.data: Tensor, The parameter data as a Tensor object, cloned and potentially moved to the specified device/dtype.
            self.grad: Tensor | None, The manually accumulated gradient (reserved for future evaluation); set to None initially.
        """
        
        # MLBase is for save/load purposes.
        super().__init__()
        
        # Record if it uses pytorch's autograd
        self.autograd = autograd
        
        # Record if it uses gradients (either autograd or manually calculated)
        self.requires_grad = requires_grad
        
        # Initialize the parameter with a tensor (ensure float dtype and device placement)
        if not isinstance(data, Tensor):
            raise ValueError("Input data MUST be a MML.Tensor! Please convert by calling Tensor(data, backend='torch')")
        # Parameter Data - a Tensor Object
        self.data = data if dtype is None and device is None else data.to(backend=data._backend, dtype=dtype, device=device)
        # Gradient manually accumulated during backprop
        # If uses autograd, then self.data.grad will record it
        self.grad = self.data.to_zeros() if requires_grad == True and autograd == False else None 
        
        # If uses pytorch autograd, then enable if requires grad
        if autograd == True and requires_grad == True:
            self.data.requires_grad_(True)
       
    def zero_grad(self):
        """
        Reset the gradient to zero.
        
        Returns:
            -------
            self
        """
        if self.requires_grad == True:
            # Use pytorch's autograd, then directly set to 0
            if self.autograd == True:
                self.data.data.grad.zero_()
            # Use manually calculated grads, then manually set to 0
            else:
                if self.grad is not None:
                    self.grad[...] = 0
                else:
                    self.grad = self.data.to_zeros()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """
        Set the attributes of `requires_grid` and enable/disable autograd if used.
        
        Returns:
            -------
            self
        """
        
        # If status conflicts, then create/disable grad
        if self.requires_grad != requires_grad:
            
            # Set the attribute of requiring grads or not
            self.requires_grad = requires_grad
            
            # If uses pytorch autograd, then enable if requires grad
            if self.autograd == True:
                self.data.requires_grad_(requires_grad)
            else:
                if requires_grad == True:
                    self.zero_grad()
                else:
                    self.grad = None

    def to(self, device: str | None = None):
        """
        Move the parameters and gradients to the specified device.
        
        Parameters:
            --------
            device: str | None, The device where the tensor should reside (e.g., "cpu", "cuda"). If None, do nothing.
        """
        if device is not None:
            self.data = self.data.to(backend = self.data._backend, device = device)
            if self.grad is not None:
                self.grad = self.grad.to(backend = self.grad._backend, device = device)
        return self

    def copy(self):
        """
        Create a deepcopy of the parameters and gradiets.
        """
        return deepcopy(self)

    def __repr__(self):
        return "nn_Parameter(Deep Neural Network Trainable Parameter Class)." + "\nData: " + self.data.__repr__() + "\nGrad: " + (self.grad.__repr__() if self.grad is not None else "")

